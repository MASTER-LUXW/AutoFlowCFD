"""Second-order viscous RANS residual for the steady-state solver.

This module replaces the first-order, inviscid-only residual path with a
physically complete cell-centred finite-volume residual that includes:

* **MUSCL reconstruction** (item 3): Green-Gauss gradients + Barth-Jespersen
  limiter give genuinely second-order left/right face states instead of the
  cell-centre values used before.
* **Inviscid flux** (HLLC Riemann solver) evaluated on the reconstructed states.
* **Viscous flux** (item 2): molecular + turbulent (eddy) shear stress and
  thermal/turbulent diffusion, using the compressible Newtonian stress tensor
  with Stokes' hypothesis.
* **SST k-omega source terms** (item 4): production, dissipation and
  cross-diffusion actually coupled into the k and omega equations, with the
  eddy viscosity feeding back into the momentum viscous flux.

The whole path is vectorised NumPy so it is deterministic and unit-testable.
Conservative variable ordering is ``[rho, rho u, rho v, rho w, E, rho k, rho w_sst]``
where the turbulence variables are carried in conservative (density-weighted)
form to match the transport in the inviscid flux.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from loguru import logger

from .fvm_gradients import FaceGeometry, green_gauss_gradient, barth_jespersen_limiter

GAMMA = 1.4
R_GAS = 287.058          # J/(kg K), dry air
PRANDTL_LAMINAR = 0.72
PRANDTL_TURBULENT = 0.90
CP = GAMMA * R_GAS / (GAMMA - 1.0)

# SST k-omega constants (Menter 2003).
SST_A1 = 0.31
SST_BETA_STAR = 0.09
SST_KAPPA = 0.41
SST_SIGMA_K1, SST_SIGMA_K2 = 0.85, 1.0
SST_SIGMA_W1, SST_SIGMA_W2 = 0.5, 0.856
SST_BETA1, SST_BETA2 = 0.075, 0.0828
SST_GAMMA1 = SST_BETA1 / SST_BETA_STAR - SST_SIGMA_W1 * SST_KAPPA**2 / np.sqrt(SST_BETA_STAR)
SST_GAMMA2 = SST_BETA2 / SST_BETA_STAR - SST_SIGMA_W2 * SST_KAPPA**2 / np.sqrt(SST_BETA_STAR)


def _blend(f1, a1_val, a2_val):
    """SST blend: f1*phi1 + (1-f1)*phi2."""
    return f1 * a1_val + (1.0 - f1) * a2_val


class ViscousRANSResidual:
    """Compute the full second-order viscous RANS residual.

    Parameters
    ----------
    geom:
        Oriented :class:`FaceGeometry`.
    mu_lam:
        Molecular dynamic viscosity (Pa s).
    wall_distance:
        Per-cell distance to the nearest viscous wall (m).  Required for the
        SST blending functions; if ``None`` a large value is assumed
        (free-stream behaviour everywhere).
    turbulent:
        If False, k/omega source terms and eddy viscosity are disabled
        (laminar Navier-Stokes).
    """

    def __init__(self, geom: FaceGeometry, mu_lam: float = 1.7894e-5,
                 wall_distance: np.ndarray | None = None,
                 turbulent: bool = True):
        self.geom = geom
        self.mu_lam = float(mu_lam)
        self.turbulent = turbulent
        n = geom.n_cells
        if wall_distance is None:
            self.wall_distance = np.full(n, 1.0e9, dtype=np.float64)
        else:
            self.wall_distance = np.maximum(np.asarray(wall_distance, np.float64), 1e-9)

        # Precompute owner->neighbour geometric quantities for internal faces.
        self._im = geom.internal_mask
        self._io = geom.int_owner
        self._in = geom.int_neigh
        d = geom.cell_centroids[self._in] - geom.cell_centroids[self._io]
        self._dist = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
        self._e_ON = d / self._dist[:, None]           # unit owner->neighbour

        # Precompute owner->ghost geometric quantities for boundary faces
        # (same role as _dist/_e_ON above, but the "neighbour" is the mirror
        # ghost state). Ghost states are constructed (see
        # BoundaryConditionHandler) so that the *face* value is the midpoint
        # average of owner and ghost - i.e. the ghost is assumed to sit at
        # the mirror point across the face, twice the owner->face distance
        # away, not at the face itself. Using the face distance directly
        # here would halve the true owner->ghost separation and double the
        # inferred near-wall gradient.
        self._bo = geom.bnd_owner
        if self._bo.size:
            db = geom.centers[geom.boundary_mask] - geom.cell_centroids[self._bo]
            self._bdist = np.maximum(2.0 * np.linalg.norm(db, axis=1), 1e-12)
            self._e_OB = db / (0.5 * self._bdist[:, None])
        else:
            self._bdist = np.zeros(0)
            self._e_OB = np.zeros((0, 3))

    # ------------------------------------------------------------------
    # primitive <-> conservative
    # ------------------------------------------------------------------
    @staticmethod
    def to_primitive(U: np.ndarray):
        """Return (rho, vel(n,3), p, T, k, omega) from conservative U."""
        rho = np.maximum(U[:, 0], 1e-9)
        vel = U[:, 1:4] / rho[:, None]
        ke = 0.5 * rho * np.sum(vel**2, axis=1)
        p = np.maximum((GAMMA - 1.0) * (U[:, 4] - ke), 1.0)
        T = p / (rho * R_GAS)
        k = np.maximum(U[:, 5] / rho, 0.0)
        omega = np.maximum(U[:, 6] / rho, 1e-6)
        return rho, vel, p, T, k, omega

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------
    def compute(self, U: np.ndarray, boundary_states: np.ndarray) -> np.ndarray:
        """Return dU/dt residual R such that V_i dU_i/dt = -R_i  (R already /V).

        Args:
            U: conservative solution, shape (n_cells, 7).
            boundary_states: ghost conservative states at boundary faces,
                shape (n_faces, 7); only rows for boundary faces are read.

        Returns:
            Residual array shape (n_cells, 7), already divided by cell volume,
            so the update is ``U -= dt * R``.
        """
        geom = self.geom
        n_cells = geom.n_cells
        rho, vel, p, T, k, omega = self.to_primitive(U)

        # Eddy viscosity (needs strain rate -> gradients of velocity).
        grad_vel = self._velocity_gradient(vel, U, boundary_states)
        mu_t = self._eddy_viscosity(rho, k, omega, grad_vel) if self.turbulent \
            else np.zeros(n_cells)

        flux_accum = np.zeros((n_cells, 7), dtype=np.float64)

        # --- inviscid flux via MUSCL + HLLC ---
        self._inviscid_flux(U, boundary_states, flux_accum)

        # --- viscous flux (molecular + turbulent) ---
        self._viscous_flux(rho, vel, T, k, omega, mu_t, grad_vel,
                           boundary_states, flux_accum)

        # Convert surface-integral of fluxes into a residual (divide by V).
        residual = flux_accum / geom.cell_volumes[:, None]

        # --- SST source terms (volumetric, added directly to residual) ---
        if self.turbulent:
            self._sst_sources(rho, k, omega, mu_t, grad_vel, residual)

        return residual

    # ------------------------------------------------------------------
    # velocity gradient (for strain rate & viscous stress)
    # ------------------------------------------------------------------
    def _velocity_gradient(self, vel, U, boundary_states):
        # boundary face velocities from ghost states
        bo = self.geom.bnd_owner
        if bo.size:
            rho_b = np.maximum(boundary_states[self.geom.boundary_mask, 0], 1e-9)
            vel_b = boundary_states[self.geom.boundary_mask, 1:4] / rho_b[:, None]
        else:
            vel_b = None
        # grad of each velocity component -> (n_cells, 3, 3): [cell, comp, dir]
        return green_gauss_gradient(vel, self.geom, vel_b)

    def _strain(self, grad_vel):
        """Symmetric strain-rate tensor S and its magnitude |S|=sqrt(2 SijSij)."""
        S = 0.5 * (grad_vel + np.transpose(grad_vel, (0, 2, 1)))
        Smag = np.sqrt(2.0 * np.einsum('nij,nij->n', S, S) + 1e-30)
        return S, Smag

    # ------------------------------------------------------------------
    # SST eddy viscosity  mu_t = rho a1 k / max(a1 omega, S F2)
    # ------------------------------------------------------------------
    def _eddy_viscosity(self, rho, k, omega, grad_vel):
        _, Smag = self._strain(grad_vel)
        nu = self.mu_lam / rho
        d = self.wall_distance
        
        # CRITICAL FIX: Protect against division by zero
        omega_safe = np.maximum(omega, 1e-8)  # Minimum physical omega (1/s)
        
        arg2 = np.maximum(2.0 * np.sqrt(np.maximum(k, 0.0)) / (SST_BETA_STAR * omega_safe * d),
                          500.0 * nu / (d**2 * omega_safe))
        F2 = np.tanh(arg2**2)
        denom = np.maximum(SST_A1 * omega_safe, Smag * F2)
        mu_t = rho * SST_A1 * np.maximum(k, 0.0) / np.maximum(denom, 1e-12)
        return np.clip(mu_t, 0.0, 1e5 * self.mu_lam)

    def _f1_blend(self, rho, k, omega, grad_k, grad_omega):
        """SST F1 blending function."""
        d = self.wall_distance
        nu = self.mu_lam / rho
        
        # CRITICAL FIX: Protect against division by zero in CDkw calculation
        omega_safe = np.maximum(omega, 1e-8)  # Minimum physical omega (1/s)
        
        CDkw = np.maximum(
            2.0 * rho * SST_SIGMA_W2 / omega_safe *
            np.einsum('nd,nd->n', grad_k, grad_omega),
            1e-10,
        )
        arg1 = np.minimum(
            np.maximum(np.sqrt(np.maximum(k, 0.0)) / (SST_BETA_STAR * omega_safe * d),
                       500.0 * nu / (d**2 * omega_safe)),
            4.0 * rho * SST_SIGMA_W2 * k / (CDkw * d**2),
        )
        return np.tanh(arg1**4), CDkw

    # ------------------------------------------------------------------
    # inviscid flux: MUSCL reconstruction + HLLC
    # ------------------------------------------------------------------
    def _inviscid_flux(self, U, boundary_states, flux_accum):
        geom = self.geom

        # --- second-order reconstruction on primitive variables ---
        rho, vel, p, T, k, omega = self.to_primitive(U)
        prim = np.column_stack([rho, vel[:, 0], vel[:, 1], vel[:, 2], p, k, omega])

        # BC primitive values for gradient boundary contribution.
        bo = geom.bnd_owner
        prim_b = None
        if bo.size:
            rb, vb, pb, tb, kb, wb = self.to_primitive(boundary_states[geom.boundary_mask])
            prim_b = np.column_stack([rb, vb[:, 0], vb[:, 1], vb[:, 2], pb, kb, wb])

        grad = green_gauss_gradient(prim, geom, prim_b)
        phi = barth_jespersen_limiter(prim, grad, geom)
        grad_lim = grad * phi[:, :, None]

        # Reconstruct to internal-face centroids.
        io, ineigh = geom.int_owner, geom.int_neigh
        fc = geom.centers[geom.internal_mask]
        rL = fc - geom.cell_centroids[io]
        rR = fc - geom.cell_centroids[ineigh]
        pL = prim[io] + np.einsum('nvd,nd->nv', grad_lim[io], rL)
        pR = prim[ineigh] + np.einsum('nvd,nd->nv', grad_lim[ineigh], rR)

        # Enforce positivity of reconstructed rho, p, k, omega.
        for col in (0, 4):
            pL[:, col] = np.maximum(pL[:, col], 1e-6)
            pR[:, col] = np.maximum(pR[:, col], 1e-6)
        pL[:, 5:] = np.maximum(pL[:, 5:], 0.0)
        pR[:, 5:] = np.maximum(pR[:, 5:], 0.0)

        n_int = geom.normals[geom.internal_mask]
        a_int = geom.areas[geom.internal_mask]
        f_int = self._hllc(pL, pR, n_int) * a_int[:, None]

        # R = (1/V) * sum_outward F.nA.  The face normal is outward for the
        # owner and inward for the neighbour, hence the opposite signs.
        np.add.at(flux_accum, io, f_int)
        np.add.at(flux_accum, ineigh, -f_int)

        # --- boundary faces: first order (owner state vs ghost state) ---
        if bo.size:
            pOwner = prim[bo]
            # ghost primitives already computed as prim_b
            n_b = geom.normals[geom.boundary_mask]
            a_b = geom.areas[geom.boundary_mask]
            f_b = self._hllc(pOwner, prim_b, n_b) * a_b[:, None]
            np.add.at(flux_accum, bo, f_b)

    def _hllc(self, primL: np.ndarray, primR: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """Vectorised HLLC flux for the 7-equation system.

        primL/primR columns: [rho, u, v, w, p, k, omega].
        Returns flux array (n, 7).
        """
        rhoL, uL, vL, wL, pL, kL, wkL = primL.T
        rhoR, uR, vR, wR, pR, kR, wkR = primR.T
        nx, ny, nz = normal[:, 0], normal[:, 1], normal[:, 2]

        # === NUMERICAL STABILITY: Clip velocity to prevent kinetic energy blow-up ===
        MAX_VELOCITY = 1e4  # 10 km/s, physically reasonable upper bound
        vel_mag_L = np.sqrt(uL**2 + vL**2 + wL**2)
        vel_mag_R = np.sqrt(uR**2 + vR**2 + wR**2)
        
        clip_factor_L = np.minimum(1.0, MAX_VELOCITY / np.maximum(vel_mag_L, 1e-12))
        clip_factor_R = np.minimum(1.0, MAX_VELOCITY / np.maximum(vel_mag_R, 1e-12))
        
        uL *= clip_factor_L; vL *= clip_factor_L; wL *= clip_factor_L
        uR *= clip_factor_R; vR *= clip_factor_R; wR *= clip_factor_R
        
        # Ensure positivity of density and pressure
        rhoL = np.maximum(rhoL, 1e-9)
        rhoR = np.maximum(rhoR, 1e-9)
        pL = np.maximum(pL, 1.0)
        pR = np.maximum(pR, 1.0)

        unL = uL * nx + vL * ny + wL * nz
        unR = uR * nx + vR * ny + wR * nz
        
        # Clamp sound speed to avoid division by zero or extreme values
        aL = np.sqrt(np.maximum(GAMMA * pL / rhoL, 1.0))
        aR = np.sqrt(np.maximum(GAMMA * pR / rhoR, 1.0))

        EL = pL / (GAMMA - 1.0) + 0.5 * rhoL * (uL**2 + vL**2 + wL**2)
        ER = pR / (GAMMA - 1.0) + 0.5 * rhoR * (uR**2 + vR**2 + wR**2)
        
        # Guard against energy overflow
        MAX_ENERGY = 1e12
        EL = np.minimum(EL, MAX_ENERGY)
        ER = np.minimum(ER, MAX_ENERGY)

        # Wave speed estimates (Davis / Einfeldt).
        SL = np.minimum(unL - aL, unR - aR)
        SR = np.maximum(unL + aL, unR + aR)
        denom = rhoL * (SL - unL) - rhoR * (SR - unR)
        denom = np.where(np.abs(denom) < 1e-12, np.sign(denom) * 1e-12 + 1e-12, denom)
        Sstar = (pR - pL + rhoL * unL * (SL - unL) - rhoR * unR * (SR - unR)) / denom

        def phys_flux(rho, u, v, w, p, E, kk, wk, un):
            return np.column_stack([
                rho * un,
                rho * u * un + p * nx,
                rho * v * un + p * ny,
                rho * w * un + p * nz,
                (E + p) * un,
                rho * kk * un,
                rho * wk * un,
            ])

        FL = phys_flux(rhoL, uL, vL, wL, pL, EL, kL, wkL, unL)
        FR = phys_flux(rhoR, uR, vR, wR, pR, ER, kR, wkR, unR)

        UL = np.column_stack([rhoL, rhoL*uL, rhoL*vL, rhoL*wL, EL, rhoL*kL, rhoL*wkL])
        UR = np.column_stack([rhoR, rhoR*uR, rhoR*vR, rhoR*wR, ER, rhoR*kR, rhoR*wkR])

        def star_state(rho, u, v, w, p, E, kk, wk, un, S):
            # Guard the (S - Sstar) denominator (S and Sstar can coincide).
            dS = S - Sstar
            dS = np.where(np.abs(dS) < 1e-12, np.sign(dS) * 1e-12 + 1e-12, dS)
            factor = rho * (S - un) / dS
            Ustar = np.empty((len(rho), 7))
            Ustar[:, 0] = factor
            Ustar[:, 1] = factor * (u + (Sstar - un) * nx)
            Ustar[:, 2] = factor * (v + (Sstar - un) * ny)
            Ustar[:, 3] = factor * (w + (Sstar - un) * nz)
            # Energy in the algebraically-cancelled Toro form: the (S-un) in the
            # p term cancels against factor, avoiding a 0/0 when S ~ un.
            Ustar[:, 4] = factor * (E / rho + (Sstar - un) * Sstar) \
                + (Sstar - un) * p / dS
            Ustar[:, 5] = factor * kk
            Ustar[:, 6] = factor * wk
            return Ustar

        F = np.empty_like(FL)
        # Region selection.
        left = SL >= 0
        right = SR <= 0
        starL = (~left) & (~right) & (Sstar >= 0)
        starR = (~left) & (~right) & (Sstar < 0)

        F[left] = FL[left]
        F[right] = FR[right]
        if np.any(starL):
            UsL = star_state(rhoL, uL, vL, wL, pL, EL, kL, wkL, unL, SL)
            F[starL] = FL[starL] + SL[starL, None] * (UsL[starL] - UL[starL])
        if np.any(starR):
            UsR = star_state(rhoR, uR, vR, wR, pR, ER, kR, wkR, unR, SR)
            F[starR] = FR[starR] + SR[starR, None] * (UsR[starR] - UR[starR])
        return F

    # ------------------------------------------------------------------
    # boundary-face helpers (owner -> face-centre, ghost state as target)
    # ------------------------------------------------------------------
    def _boundary_face_grad(self, cell_grad: np.ndarray, cell_val: np.ndarray,
                            ghost_val: np.ndarray) -> np.ndarray:
        """One-sided corrected gradient at boundary faces.

        Same over-relaxed correction as the internal-face treatment (see
        ``_viscous_flux``), except the "neighbour" is the ghost/wall value at
        the face centre rather than a real neighbour cell.

        Args:
            cell_grad: per-cell gradient, shape (n_cells, 3) for a scalar
                field or (n_cells, ncomp, 3) for a vector field.
            cell_val, ghost_val: owner-cell and ghost values at every
                boundary face, shape (n_bf,) / (n_bf,) or (n_bf, ncomp).

        Returns:
            Corrected face gradient, same trailing shape as ``cell_grad``.
        """
        bo = self._bo
        g_owner = cell_grad[bo]
        d_val = ghost_val - cell_val
        if g_owner.ndim == 2:
            proj = np.einsum('nd,nd->n', g_owner, self._e_OB)
            corr = d_val / self._bdist - proj
            return g_owner + corr[:, None] * self._e_OB
        proj = np.einsum('nij,nj->ni', g_owner, self._e_OB)
        corr = d_val / self._bdist[:, None] - proj
        return g_owner + corr[:, :, None] * self._e_OB[:, None, :]

    def wall_shear_stress(self, vel: np.ndarray, mu_t: np.ndarray,
                          grad_vel: np.ndarray, boundary_states: np.ndarray) -> np.ndarray:
        """Viscous stress tensor dotted with the outward normal, per boundary
        face: ``tau . n``, shape (n_boundary_faces, 3).

        Shared by :meth:`_viscous_flux` (the residual's own boundary viscous
        term) and by aerodynamic force integration (skin-friction drag),
        so both consistently use whatever force the momentum equation
        actually balances against.
        """
        geom = self.geom
        bo = self._bo
        if not bo.size:
            return np.zeros((0, 3))
        n_b = geom.normals[geom.boundary_mask]
        mu_eff = self.mu_lam + mu_t
        rho_b = np.maximum(boundary_states[geom.boundary_mask, 0], 1e-9)
        vel_ghost = boundary_states[geom.boundary_mask, 1:4] / rho_b[:, None]
        gv_face_b = self._boundary_face_grad(grad_vel, vel[bo], vel_ghost)
        return self._stress_dot_normal(gv_face_b, n_b, mu_eff[bo])

    # ------------------------------------------------------------------
    # viscous flux
    # ------------------------------------------------------------------
    def _viscous_flux(self, rho, vel, T, k, omega, mu_t, grad_vel,
                      boundary_states, flux_accum):
        geom = self.geom
        io, ineigh = geom.int_owner, geom.int_neigh
        n_int = geom.normals[geom.internal_mask]
        a_int = geom.areas[geom.internal_mask]

        mu_eff = self.mu_lam + mu_t                          # (n_cells,)

        # Face-averaged gradients with an over-relaxed correction along the
        # cell-connecting line for robustness on skewed meshes.
        gvL, gvR = grad_vel[io], grad_vel[ineigh]
        gv_face = 0.5 * (gvL + gvR)
        # directional derivative correction
        dvel = vel[ineigh] - vel[io]                         # (nif, 3)
        proj = np.einsum('nij,nj->ni', gv_face, self._e_ON) # (nif, 3)
        corr = (dvel / self._dist[:, None] - proj)
        gv_face = gv_face + corr[:, :, None] * self._e_ON[:, None, :]

        mu_f = 0.5 * (mu_eff[io] + mu_eff[ineigh])

        tau_n = self._stress_dot_normal(gv_face, n_int, mu_f)   # (nif, 3)

        # Temperature gradient for heat flux.
        gT = green_gauss_gradient(T[:, None], geom)[:, 0, :]    # (n_cells, 3)
        gT_face = 0.5 * (gT[io] + gT[ineigh])
        dT = T[ineigh] - T[io]
        gT_face = gT_face + (dT / self._dist - np.einsum('nd,nd->n', gT_face, self._e_ON))[:, None] * self._e_ON
        cond = CP * (self.mu_lam / PRANDTL_LAMINAR + 0.5*(mu_t[io]+mu_t[ineigh]) / PRANDTL_TURBULENT)
        qn = cond * np.einsum('nd,nd->n', gT_face, n_int)        # heat conduction

        vel_face = 0.5 * (vel[io] + vel[ineigh])
        work = np.einsum('nd,nd->n', tau_n, vel_face)

        fvisc = np.zeros((len(io), 7))
        fvisc[:, 1:4] = tau_n
        fvisc[:, 4] = work + qn

        # turbulent variable diffusion: (mu + sigma*mu_t) grad(k or omega).n
        # OPTIMIZED: Batch compute k and omega gradients together
        turb_vars = np.column_stack([k, omega])  # (n_cells, 2)
        gturb = green_gauss_gradient(turb_vars, geom)  # (n_cells, 2, 3)
        gk, gw = gturb[:, 0, :], gturb[:, 1, :]  # Each (n_cells, 3)
        
        gk_face = 0.5*(gk[io]+gk[ineigh])
        gw_face = 0.5*(gw[io]+gw[ineigh])
        # blend sigma with F1 (approx face value)
        mut_f = 0.5*(mu_t[io]+mu_t[ineigh])
        diff_k = (self.mu_lam + SST_SIGMA_K1 * mut_f) * np.einsum('nd,nd->n', gk_face, n_int)
        diff_w = (self.mu_lam + SST_SIGMA_W1 * mut_f) * np.einsum('nd,nd->n', gw_face, n_int)
        fvisc[:, 5] = diff_k
        fvisc[:, 6] = diff_w

        fvisc *= a_int[:, None]
        # Viscous flux enters the residual with the opposite sign to the
        # inviscid one: R = (1/V)[sum F_inv.nA - sum F_visc.nA].
        np.add.at(flux_accum, io, -fvisc)
        np.add.at(flux_accum, ineigh, fvisc)

        # --- boundary faces: molecular + turbulent viscous flux -----------
        # Previously missing entirely: with no boundary contribution here,
        # solid walls got zero shear stress, zero heat conduction, and zero
        # turbulent diffusion from the momentum/energy/k-omega equations, so
        # skin friction never actually entered the solved system despite the
        # wall ghost-state velocity mirror being specifically built to make
        # it possible (see BoundaryConditionHandler._wall_bc docstring).
        bo = self._bo
        if bo.size:
            n_b = geom.normals[geom.boundary_mask]
            a_b = geom.areas[geom.boundary_mask]

            rho_b = np.maximum(boundary_states[geom.boundary_mask, 0], 1e-9)
            vel_ghost = boundary_states[geom.boundary_mask, 1:4] / rho_b[:, None]
            tau_n_b = self.wall_shear_stress(vel, mu_t, grad_vel, boundary_states)

            E_ghost = boundary_states[geom.boundary_mask, 4]
            ke_ghost = 0.5 * rho_b * np.sum(vel_ghost**2, axis=1)
            p_ghost = np.maximum((GAMMA - 1.0) * (E_ghost - ke_ghost), 1.0)
            T_ghost = p_ghost / (rho_b * R_GAS)
            gT_face_b = self._boundary_face_grad(gT, T[bo], T_ghost)
            cond_b = CP * (self.mu_lam / PRANDTL_LAMINAR + mu_t[bo] / PRANDTL_TURBULENT)
            qn_b = cond_b * np.einsum('nd,nd->n', gT_face_b, n_b)

            vel_face_b = 0.5 * (vel[bo] + vel_ghost)
            work_b = np.einsum('nd,nd->n', tau_n_b, vel_face_b)

            fvisc_b = np.zeros((len(bo), 7))
            fvisc_b[:, 1:4] = tau_n_b
            fvisc_b[:, 4] = work_b + qn_b

            k_ghost = np.maximum(boundary_states[geom.boundary_mask, 5] / rho_b, 0.0)
            omega_ghost = np.maximum(boundary_states[geom.boundary_mask, 6] / rho_b, 1e-6)
            gk_face_b = self._boundary_face_grad(gk, k[bo], k_ghost)
            gw_face_b = self._boundary_face_grad(gw, omega[bo], omega_ghost)
            mut_b = mu_t[bo]
            diff_k_b = (self.mu_lam + SST_SIGMA_K1 * mut_b) * np.einsum('nd,nd->n', gk_face_b, n_b)
            diff_w_b = (self.mu_lam + SST_SIGMA_W1 * mut_b) * np.einsum('nd,nd->n', gw_face_b, n_b)
            fvisc_b[:, 5] = diff_k_b
            fvisc_b[:, 6] = diff_w_b

            fvisc_b *= a_b[:, None]
            np.add.at(flux_accum, bo, -fvisc_b)

    @staticmethod
    def _stress_dot_normal(grad_vel, normal, mu):
        """tau . n  with tau = mu(grad u + grad u^T - 2/3 div(u) I)."""
        divu = grad_vel[:, 0, 0] + grad_vel[:, 1, 1] + grad_vel[:, 2, 2]
        tau = mu[:, None, None] * (grad_vel + np.transpose(grad_vel, (0, 2, 1)))
        # subtract 2/3 mu divu on the diagonal
        for i in range(3):
            tau[:, i, i] -= (2.0/3.0) * mu * divu
        return np.einsum('nij,nj->ni', tau, normal)

    # ------------------------------------------------------------------
    # SST source terms
    # ------------------------------------------------------------------
    def _sst_sources(self, rho, k, omega, mu_t, grad_vel, residual):
        geom = self.geom
        S, Smag = self._strain(grad_vel)

        # OPTIMIZED: Batch compute k and omega gradients together
        turb_vars = np.column_stack([k, omega])  # (n_cells, 2)
        gturb = green_gauss_gradient(turb_vars, geom)  # (n_cells, 2, 3)
        gk, gw = gturb[:, 0, :], gturb[:, 1, :]  # Each (n_cells, 3)
        
        F1, CDkw = self._f1_blend(rho, k, omega, gk, gw)

        beta = _blend(F1, SST_BETA1, SST_BETA2)
        gamma = _blend(F1, SST_GAMMA1, SST_GAMMA2)
        sigma_w = _blend(F1, SST_SIGMA_W1, SST_SIGMA_W2)

        # Production of k, limited to 10*beta_star*rho*k*omega (Menter limiter).
        Pk = mu_t * Smag**2
        Pk = np.minimum(Pk, 10.0 * SST_BETA_STAR * rho * k * omega)
        Dk = SST_BETA_STAR * rho * k * omega

        Pw = gamma * rho * Smag**2  # = gamma/nu_t * Pk with mu_t=rho a1 k/...; use strain form
        Dw = beta * rho * omega**2
        
        # CRITICAL FIX: Protect cross-diffusion term from division by zero
        # When omega -> 0, the term blows up causing numerical divergence
        omega_safe = np.maximum(omega, 1e-8)  # Minimum physical omega (1/s)
        cross = 2.0 * (1.0 - F1) * rho * sigma_w / omega_safe * np.einsum('nd,nd->n', gk, gw)
        
        # Additional safety: clip cross-diffusion to prevent extreme values
        max_cross = 10.0 * np.maximum(np.abs(Pw), np.abs(Dw))
        cross = np.clip(cross, -max_cross, max_cross)

        # residual is dU/dt = -R ; sources enter with opposite sign (added to U).
        # For conservative rho*k, rho*omega equations:
        residual[:, 5] -= (Pk - Dk)
        residual[:, 6] -= (Pw - Dw + cross)


def estimate_wall_distance(geom: FaceGeometry, wall_face_mask: np.ndarray) -> np.ndarray:
    """Approximate nearest-wall distance for every cell centroid.

    Uses KD-Tree spatial indexing for O(N log M) complexity instead of
    brute-force O(N*M). For 2.8M cells and 130K wall faces, this reduces
    computation time from hours to seconds.
    
    Args:
        geom: Face geometry with cell centroids and face centers
        wall_face_mask: Boolean mask identifying wall boundary faces
        
    Returns:
        Array of minimum distances from each cell to nearest wall face
    """
    n_cells = geom.n_cells
    wall_faces = np.where(wall_face_mask)[0]
    
    if wall_faces.size == 0:
        logger.warning("No wall faces found, returning large default distance")
        return np.full(n_cells, 1.0e9)
    
    # Extract wall face center coordinates
    wall_pts = geom.centers[wall_faces]
    cc = geom.cell_centroids
    
    logger.info(f"Building KD-Tree for {len(wall_pts)} wall points...")
    
    # Build KD-Tree from wall points (O(M log M))
    tree = cKDTree(wall_pts)
    
    # Query nearest neighbor for all cell centroids (O(N log M))
    logger.info(f"Querying nearest wall distance for {n_cells} cells...")
    distances, _ = tree.query(cc, k=1, workers=-1)  # workers=-1 uses all CPUs
    
    logger.success(f"Wall distance computed: min={distances.min():.4e}, max={distances.max():.4e}, mean={distances.mean():.4e}")
    
    return np.maximum(distances, 1e-9)
