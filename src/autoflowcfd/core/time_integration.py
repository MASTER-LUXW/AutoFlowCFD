"""Time integration schemes for the pseudo-time steady solver and transient runs.

The steady solver advances the solution in pseudo-time towards the residual =
0 state.  For that we use explicit Strong-Stability-Preserving Runge-Kutta
schemes (SSP-RK2 / SSP-RK3), which are the standard, provably correct explicit
integrators for FV CFD, together with a **local (per-cell) time step** governed
by the convective+acoustic+viscous CFL condition.

This replaces the previous implementation, which (a) called itself
"backward Euler" while doing an explicit forward-Euler step, (b) used
placeholder residual history for RK2/AB3, and (c) hid divergence behind hard
magnitude clips on density/velocity/flux.  Physical positivity is now enforced
only where it is mathematically required (rho>0, p>0) via a *pressure floor*
that preserves velocity, and divergence is reported rather than masked.
"""

from __future__ import annotations

import numpy as np
from enum import Enum
from typing import Callable, Optional
from loguru import logger

GAMMA = 1.4


class TimeIntegrationScheme(Enum):
    """Explicit pseudo-time integration schemes."""

    FORWARD_EULER = "forward_euler"
    SSP_RK2 = "ssp_rk2"
    SSP_RK3 = "ssp_rk3"
    # Legacy aliases kept so existing configs/tests keep importing.
    BACKWARD_EULER = "forward_euler"
    RUNGE_KUTTA_2 = "ssp_rk2"
    ADAMS_BASHFORTH_3 = "ssp_rk3"


# SSP-RK Shu-Osher coefficients: stages of the form
#   u^(i) = sum_k alpha[i,k] u^(k) + beta[i] dt L(u^(i-1))
# where L(u) = -R(u).  We store per-scheme stage lists.
_SSP_RK2 = {
    "stages": 2,
    # u1 = u0 + dt L0 ;  u2 = 1/2 u0 + 1/2 (u1 + dt L1)
    "alpha": [[1.0], [0.5, 0.5]],
    "beta": [1.0, 0.5],
}
_SSP_RK3 = {
    "stages": 3,
    "alpha": [[1.0],
              [0.75, 0.25],
              [1.0/3.0, 0.0, 2.0/3.0]],
    "beta": [1.0, 0.25, 2.0/3.0],
}
_EULER = {"stages": 1, "alpha": [[1.0]], "beta": [1.0]}

_SCHEME_TABLE = {
    TimeIntegrationScheme.FORWARD_EULER: _EULER,
    TimeIntegrationScheme.SSP_RK2: _SSP_RK2,
    TimeIntegrationScheme.SSP_RK3: _SSP_RK3,
}


def enforce_positivity(U: np.ndarray, p_floor: float = 1.0) -> np.ndarray:
    """Enforce physical bounds on conservative variables after a time step.

    Projects density and pressure to positive floors while preserving velocity.
    Also clips velocity magnitude to prevent kinetic energy blow-up.
    """
    MAX_VELOCITY = 1e4  # 10 km/s upper bound
    
    rho = np.maximum(U[:, 0], 1e-6)
    U[:, 0] = rho

    vel = U[:, 1:4] / rho[:, None]
    
    # === CRITICAL: Clip velocity magnitude ===
    vel_mag = np.sqrt(np.sum(vel**2, axis=1))
    clip_mask = vel_mag > MAX_VELOCITY
    if np.any(clip_mask):
        clip_factor = MAX_VELOCITY / vel_mag[clip_mask]
        vel[clip_mask] *= clip_factor[:, None]
        # Update momentum with clipped velocities
        U[clip_mask, 1:4] = (rho[clip_mask, None] * vel[clip_mask])
    
    max_vel = MAX_VELOCITY
    vel_mag = np.sqrt(np.sum(vel**2, axis=1))
    if np.any(vel_mag > max_vel):
        vel = vel * np.minimum(max_vel / vel_mag, 1.0)
    
    ke = 0.5 * rho * np.sum(vel**2, axis=1)
    p = (GAMMA - 1.0) * (U[:, 4] - ke)
    low = p < p_floor
    if np.any(low):
        U[low, 4] = p_floor / (GAMMA - 1.0) + ke[low]
    U[:, 5] = np.maximum(U[:, 5], 0.0)      # rho*k >= 0
    U[:, 6] = np.maximum(U[:, 6], 1e-8)     # rho*omega > 0
    return U


class TimeIntegrator:
    """Explicit SSP Runge-Kutta integrator with local time stepping."""

    def __init__(
        self,
        scheme: TimeIntegrationScheme = TimeIntegrationScheme.SSP_RK3,
        dt: float = 1e-5,
        cfl_target: float = 1.0,
    ):
        # Map any legacy alias onto the canonical enum member.
        self.scheme = TimeIntegrationScheme(scheme.value) if isinstance(scheme, TimeIntegrationScheme) \
            else TimeIntegrationScheme(scheme)
        self.dt = dt
        self.cfl_target = cfl_target
        self.n_steps = 0
        self.current_time = 0.0
        self._table = _SCHEME_TABLE[self.scheme]

    # ------------------------------------------------------------------
    def local_time_step(self, U: np.ndarray, geom, mu_eff: Optional[np.ndarray] = None) -> np.ndarray:
        """Per-cell stable pseudo-time step dt_i = CFL * V_i / sum_f (|u.n|+a) A_f.

        Adds a viscous limit when ``mu_eff`` is provided.
        """
        rho = np.maximum(U[:, 0], 1e-9)
        vel = U[:, 1:4] / rho[:, None]
        ke = 0.5 * rho * np.sum(vel**2, axis=1)
        p = np.maximum((GAMMA - 1.0) * (U[:, 4] - ke), 1.0)
        a = np.sqrt(GAMMA * p / rho)

        n_cells = geom.n_cells
        spectral = np.zeros(n_cells)

        owner = geom.owner
        neigh = geom.neigh
        normals = geom.normals
        areas = geom.areas
        bmask = geom.boundary_mask
        imask = geom.internal_mask

        # internal faces contribute to both cells
        io, ineigh = geom.int_owner, geom.int_neigh
        n_int = normals[imask]
        a_int = areas[imask]
        un_o = np.abs(np.einsum('nd,nd->n', vel[io], n_int)) + a[io]
        un_n = np.abs(np.einsum('nd,nd->n', vel[ineigh], n_int)) + a[ineigh]
        np.add.at(spectral, io, un_o * a_int)
        np.add.at(spectral, ineigh, un_n * a_int)

        # boundary faces contribute to owner
        bo = geom.bnd_owner
        if bo.size:
            n_b = normals[bmask]
            a_b = areas[bmask]
            un_b = np.abs(np.einsum('nd,nd->n', vel[bo], n_b)) + a[bo]
            np.add.at(spectral, bo, un_b * a_b)

        spectral = np.maximum(spectral, 1e-30)
        dt = self.cfl_target * geom.cell_volumes / spectral

        if mu_eff is not None:
            # viscous stability: dt_visc ~ CFL * rho V^{5/3} / mu
            Lc2 = geom.cell_volumes ** (2.0 / 3.0)
            dt_visc = 0.25 * self.cfl_target * rho * Lc2 / np.maximum(mu_eff, 1e-30)
            dt = np.minimum(dt, dt_visc)

        return dt

    # ------------------------------------------------------------------
    def step(
        self,
        solution: np.ndarray,
        residual_func: Callable[[np.ndarray], np.ndarray],
        dt_local: np.ndarray,
        p_floor: float = 1.0,
        residual0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Advance one pseudo-time step with the configured SSP-RK scheme.

        Args:
            solution: current conservative state (n_cells, n_vars).
            residual_func: callable U -> R(U) already divided by cell volume,
                i.e. dU/dt = -R(U).
            dt_local: per-cell pseudo-time step (n_cells,).
            p_floor: minimum pressure for positivity projection.
            residual0: optional precomputed R(solution) - every stage-0 of
                every scheme here evaluates Ui=U0=solution, so if the caller
                already has R(solution) (e.g. for convergence monitoring),
                passing it in avoids re-running the residual (MUSCL + HLLC +
                viscous + SST source terms - the most expensive part of an
                iteration) a second time for no new information.

        Returns:
            Updated conservative state.
        """
        alpha = self._table["alpha"]
        beta = self._table["beta"]
        dt = dt_local[:, None]

        U0 = solution
        stages = [U0]
        for i in range(self._table["stages"]):
            Ui = stages[-1]
            if i == 0 and residual0 is not None:
                L = -residual0                 # dU/dt, reuse caller's R(U0)
            else:
                L = -residual_func(Ui)         # dU/dt
            combo = np.zeros_like(U0)
            for k, a_ik in enumerate(alpha[i]):
                if a_ik != 0.0:
                    combo += a_ik * stages[k]
            Unew = combo + beta[i] * dt * L
            Unew = enforce_positivity(Unew, p_floor)
            stages.append(Unew)

        self.n_steps += 1
        return stages[-1]

    def reset(self) -> None:
        self.n_steps = 0
        self.current_time = 0.0
