"""Time-accurate unsteady RANS/DES solver.

Marches the same viscous RANS residual used by the steady solver
(:class:`~autoflowcfd.core.fvm_viscous_residual.ViscousRANSResidual`) forward
in *physical* time with a fixed, uniform time step - reusing the steady
solver's face geometry, boundary-condition handling, and aerodynamic-force
integration instead of maintaining a second, independent flux/BC/force
pipeline.

DES/DDES/LES modes only implement the Spalart DES97 grid-based length-scale
limiter on the SST k-destruction term (see :meth:`TransientSolver._apply_des_correction`).
DDES's boundary-layer shielding function and a genuine resolved-turbulence
LES sub-grid model are not implemented; both currently fall back to DES97
with a startup warning.
"""

from __future__ import annotations

import time
import numpy as np
from typing import Optional
from loguru import logger

from ..grid.structures import VolumeMeshData
from ..config.solver_config import TransientConfig, TurbulenceModel, TimeIntegrationScheme as ConfigTimeScheme
from .fvm_core import FVMFaceExtractor
from .fvm_gradients import FaceGeometry
from .fvm_viscous_residual import ViscousRANSResidual, estimate_wall_distance
from .bc_handler import BoundaryConditionHandler
from .aero_coeffs import AeroCoefficientCalculator
from .checkpoint import CheckpointManager
from .time_integration import TimeIntegrator, TimeIntegrationScheme
from .transient_result import TransientResult

# Scheme names in TransientConfig/CLI predate the steady solver's SSP-RK
# rewrite (see time_integration.py's own module docstring on why
# "backward_euler"/"ab3" are legacy aliases for explicit schemes) - map them
# onto the real integrator this solver uses.
_SCHEME_MAP = {
    ConfigTimeScheme.BACKWARD_EULER: TimeIntegrationScheme.FORWARD_EULER,
    ConfigTimeScheme.RK2: TimeIntegrationScheme.SSP_RK2,
    ConfigTimeScheme.RK3: TimeIntegrationScheme.SSP_RK3,
    ConfigTimeScheme.AB3: TimeIntegrationScheme.SSP_RK3,
}

MU_LAM = 1.7894e-5  # Sutherland at 288 K, Pa s
SST_BETA_STAR = 0.09
C_DES = 0.61  # Spalart DES97 constant (SST-calibrated)


class TransientSolver:
    """Time-accurate unsteady RANS/DES solver coordinator."""

    def __init__(self, grid_data: VolumeMeshData, config: TransientConfig):
        self.grid_data = grid_data
        self.config = config
        self.solution: Optional[np.ndarray] = None

        logger.info("Initializing TransientSolver")
        logger.info(f"  Grid: {grid_data.node_count} nodes, {grid_data.cell_count} cells")

        if config.turbulence in (TurbulenceModel.DDES, TurbulenceModel.LES):
            logger.warning(
                f"turbulence={config.turbulence.value}: this solver only implements "
                "the Spalart DES97 grid-length-scale limiter (see _apply_des_correction). "
                "DDES's boundary-layer shielding function is not implemented, so thin "
                "near-wall cells may switch to LES-mode dissipation where a real DDES "
                "would not; 'les' is not a resolved sub-grid-scale model here and falls "
                "back to the same DES97 treatment as 'des'."
            )
        self.use_des = config.turbulence in (
            TurbulenceModel.DES, TurbulenceModel.DDES, TurbulenceModel.LES
        )
        self.turbulent = config.turbulence != TurbulenceModel.NONE

        self.face_extractor = FVMFaceExtractor()
        self.bc_handler = BoundaryConditionHandler(
            grid_data, self.face_extractor,
            rho_inf=config.rho_inf, p_inf=config.p_inf,
        )
        self.aero_calculator = AeroCoefficientCalculator(
            grid_data, self.face_extractor,
            rho_inf=config.rho_inf, vel_inf=config.vel_inf,
        )
        self.checkpoint_manager = CheckpointManager(
            config, output_dir=config.output_dir,
            checkpoint_interval=config.checkpoint_interval,
        )

        scheme = _SCHEME_MAP.get(config.time_scheme)
        if scheme is None:
            logger.warning(f"Unknown time_scheme {config.time_scheme}, defaulting to SSP-RK3")
            scheme = TimeIntegrationScheme.SSP_RK3
        # cfl_target is unused here (dt is fixed/uniform, not CFL-adapted) but
        # TimeIntegrator.local_time_step is never called in this solver.
        self.time_integrator = TimeIntegrator(scheme=scheme, dt=config.dt, cfl_target=1.0)

        self.current_time = 0.0
        self.n_steps = 0
        self.cd_history: list = []
        self.cl_history: list = []
        self.time_stamps: list = []
        self.checkpoint_path: Optional[str] = None

        self._geom: Optional[FaceGeometry] = None
        self._residual: Optional[ViscousRANSResidual] = None
        self._des_delta: Optional[np.ndarray] = None

        logger.info("TransientSolver initialization complete")

    # ------------------------------------------------------------------
    def _initialize_solution(self) -> None:
        """Uniform freestream initial condition (same form as FRSolver)."""
        n_cells = self.grid_data.cell_count
        rho_0, u_0, p_0 = self.config.rho_inf, self.config.vel_inf, self.config.p_inf
        gamma = 1.4
        E_0 = p_0 / (gamma - 1.0) + 0.5 * rho_0 * u_0 ** 2
        u_ref = max(u_0, 1.0)
        k_0 = 1.5 * (0.01 * u_ref) ** 2
        omega_0 = 5.0 * u_ref / 0.1

        self.solution = np.zeros((n_cells, 7), dtype=np.float64)
        self.solution[:, 0] = rho_0
        self.solution[:, 1] = rho_0 * u_0
        self.solution[:, 4] = E_0
        self.solution[:, 5] = rho_0 * k_0
        self.solution[:, 6] = rho_0 * omega_0
        logger.info(f"Solution initialized: {n_cells} cells")

    def _setup_boundary_conditions(self) -> None:
        """Wire freestream targets into bc_handler (no ramp: this is a real
        time-accurate march, not a pseudo-time startup - ramping the BC here
        would just be an unphysical time-dependent forcing)."""
        vel_inf = self.config.vel_inf
        for name in self.grid_data.boundaries.boundary_names:
            u = name.upper()
            if "INLET" in u or "INFLOW" in u:
                self.bc_handler.base_inlet_velocity = vel_inf
            elif "GROUND" in u or "FARFIELD" in u:
                self.bc_handler.base_farfield_velocity = vel_inf
            elif "BODY" in u or "CAR" in u or "OUTLET" in u or "SYMMETRY" in u or "TUNNEL" in u:
                # TUNNEL is classified as SYMMETRY (free-slip duct wall) by
                # bc_handler.py's _classify - it doesn't use
                # base_farfield_velocity at all, unlike a real open FARFIELD
                # boundary.
                pass
            else:
                self.bc_handler.base_farfield_velocity = vel_inf
        self.bc_handler.ramp_factor = 1.0
        logger.info(f"Boundary conditions setup: {len(self.grid_data.boundaries.boundary_names)} boundaries")

    def _setup(self) -> None:
        """Build face geometry / residual objects. Idempotent - safe to call
        from both solve() and load_checkpoint()."""
        if self._geom is not None:
            return

        self._setup_boundary_conditions()
        if self.solution is None:
            self._initialize_solution()

        logger.info("Building face geometry (optimized radix-sort)...")
        face_data_obj = self.grid_data.ensure_faces_exist()

        nodes_array = np.column_stack([
            self.grid_data.nodes.x, self.grid_data.nodes.y, self.grid_data.nodes.z,
        ])
        connectivity_int64 = self.grid_data.cells.connectivity.astype(np.int64)
        cell_centroids = nodes_array[connectivity_int64].mean(axis=1)
        boundary_flags = (face_data_obj.connectivity[:, 1] < 0).astype(np.int32)

        self.face_extractor.cell_centroids = cell_centroids
        self.face_extractor.face_connectivity = face_data_obj.connectivity
        self.face_extractor.face_normals = face_data_obj.normal
        self.face_extractor.face_areas = face_data_obj.area
        self.face_extractor.boundary_flags = boundary_flags

        cell_volumes = self.grid_data.get_cell_volumes()
        self._geom = FaceGeometry(
            connectivity=face_data_obj.connectivity,
            normals=face_data_obj.normal,
            areas=face_data_obj.area,
            centers=face_data_obj.center,
            boundary_flags=boundary_flags,
            cell_centroids=cell_centroids,
            cell_volumes=cell_volumes,
        )

        self.bc_handler._precompute_face_types()
        wall_face_mask = np.zeros(self._geom.n_faces, dtype=bool)
        for f, t in self.bc_handler._face_types.items():
            if t in ("WALL", "GROUND"):
                wall_face_mask[f] = True
        wall_distance = estimate_wall_distance(self._geom, wall_face_mask)

        # Reference Mach number for AUSM+up's low-Mach scaling function
        # f_a (see fvm_viscous_residual.py's _ausm_up / solver_steady.py's
        # matching comment) - same role here as in the steady solver.
        gamma_air = 1.4
        a_inf = np.sqrt(gamma_air * self.config.p_inf / self.config.rho_inf)
        mach_ref = self.config.vel_inf / max(a_inf, 1e-30)

        self._residual = ViscousRANSResidual(
            self._geom, mu_lam=MU_LAM, wall_distance=wall_distance, turbulent=self.turbulent,
            mach_ref=mach_ref,
        )
        self._des_delta = np.cbrt(cell_volumes)

        self._warn_if_dt_unstable()

    def _warn_if_dt_unstable(self) -> None:
        """Estimate the explicit-scheme stability limit (CFL=1, inviscid +
        SST source-term stiffness at the initial condition) and warn if the
        configured fixed dt exceeds it. Unlike the steady solver, this
        solver never adapts dt to CFL - an unstable choice here will just
        diverge, so surfacing it up front is the only warning the user gets
        before that happens. This is necessarily an initial-condition
        estimate only: omega evolves, so a dt that looks safe at t=0 can
        still become unstable later if omega grows near a wall."""
        omega0 = None
        if self.turbulent:
            rho0 = np.maximum(self.solution[:, 0], 1e-9)
            omega0 = np.maximum(self.solution[:, 6] / rho0, 1e-6)
        dt_stable = self.time_integrator.local_time_step(self.solution, self._geom, None, omega=omega0)
        min_dt_stable = float(np.min(dt_stable))
        if self.config.dt > min_dt_stable:
            logger.warning(
                f"--dt {self.config.dt:.3e}s exceeds the estimated explicit stability "
                f"limit ({min_dt_stable:.3e}s at t=0, CFL=1, inviscid+SST-stiffness) "
                f"for the smallest/stiffest cell in this mesh. The scheme has no CFL "
                f"adaptation in transient mode (dt is fixed for time accuracy) - this "
                f"will likely diverge."
            )

    # ------------------------------------------------------------------
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Resume from an HDF5 checkpoint (steady or transient)."""
        self._setup()
        solution, history, iteration, metadata = self.checkpoint_manager.load(checkpoint_path)
        if self.grid_data.cell_count != solution.shape[0]:
            raise ValueError(
                f"Grid cell count ({self.grid_data.cell_count}) does not match "
                f"checkpoint solution ({solution.shape[0]}). Provide the same "
                f"volume mesh used to produce this checkpoint."
            )
        self.solution = solution
        self.current_time = float(metadata.get('current_time', 0.0))
        self.n_steps = iteration
        if history:
            coeffs = history.get('coefficients', {})
            self.cd_history = list(coeffs.get('Cd', []))
            self.cl_history = list(coeffs.get('Cl', []))
            self.time_stamps = list(history.get('iterations', []))
        logger.info(f"Resumed transient state: t={self.current_time:.6f}s, step={self.n_steps}")

    def _apply_des_correction(self, U: np.ndarray, R: np.ndarray) -> None:
        """Spalart DES97 length-scale limiter on the SST k-destruction term.

        Scales the standard destruction Dk = beta_star*rho*k*omega by
        F_DES = max(l_RANS / (C_des*Delta), 1), where l_RANS is the modeled
        turbulence length scale and Delta the local grid spacing. Once the
        grid is fine enough to resolve the energy-containing scales directly
        (l_RANS > C_des*Delta), this increases k destruction so the model
        backs off and lets resolved turbulence take over - the one piece of
        physics that actually distinguishes DES mode from plain RANS here.
        """
        rho, vel, p, T, k, omega = self._residual.to_primitive(U)
        l_rans = np.sqrt(np.maximum(k, 0.0)) / np.maximum(SST_BETA_STAR * omega, 1e-12)
        l_les = C_DES * self._des_delta
        F_DES = np.maximum(l_rans / np.maximum(l_les, 1e-12), 1.0)
        Dk_std = SST_BETA_STAR * rho * k * omega
        # R already contains +Dk_std from _sst_sources; add the incremental
        # (F_DES - 1) part so the total matches Dk_std * F_DES.
        R[:, 5] += Dk_std * (F_DES - 1.0)

    def _compute_residual(self, U: np.ndarray) -> np.ndarray:
        bstates = self.bc_handler.build_boundary_states(U)
        R = self._residual.compute(U, bstates)
        if self.use_des:
            self._apply_des_correction(U, R)
        return R

    # ------------------------------------------------------------------
    def solve(self) -> TransientResult:
        """Advance the solution from ``current_time`` to ``config.total_time``."""
        self._setup()
        geom = self._geom
        residual = self._residual

        dt = self.config.dt
        total_time = self.config.total_time
        dt_array = np.full(geom.n_cells, dt, dtype=np.float64)

        logger.info(
            f"Starting transient solve: t={self.current_time:.6f}s -> "
            f"{total_time:.6f}s, dt={dt:.3e}s, turbulent={self.turbulent}, "
            f"DES={self.use_des}"
        )
        start = time.time()

        while self.current_time < total_time - 1e-12:
            R0 = self._compute_residual(self.solution)
            self.solution = self.time_integrator.step(
                self.solution, self._compute_residual, dt_array, residual0=R0
            )

            self.current_time += dt
            self.n_steps += 1

            if not np.all(np.isfinite(self.solution)):
                raise RuntimeError(
                    f"Transient solve diverged at step {self.n_steps} "
                    f"(t={self.current_time:.6f}s): non-finite state detected"
                )

            bstates = self.bc_handler.build_boundary_states(self.solution)
            rho_c, vel_c, p_c, T_c, k_c, w_c = residual.to_primitive(self.solution)
            gvel = residual._velocity_gradient(vel_c, self.solution, bstates)
            mu_t = residual._eddy_viscosity(rho_c, k_c, w_c, gvel) if self.turbulent \
                else np.zeros(geom.n_cells)
            Cd, Cl, Cd_p, Cd_f = self.aero_calculator.compute_coefficients(
                self.solution, self.n_steps,
                viscous_residual=residual, grad_vel=gvel, mu_t=mu_t,
                boundary_states=bstates,
            )
            self.cd_history.append(Cd)
            self.cl_history.append(Cl)
            self.time_stamps.append(self.current_time)

            if self.n_steps % max(1, self.config.sample_interval) == 0 or self.n_steps == 1:
                logger.info(
                    f"Step {self.n_steps:6d} | t={self.current_time:.6f}s | "
                    f"Cd={Cd:.4f} | Cl={Cl:.4f}"
                )
                # Second line: Cd breakdown (aligned with first |)
                prefix_len = len(f"Step {self.n_steps:6d}")
                logger.info(
                    f"{'':>{prefix_len + 2}s}  "
                    f"Cd breakdown: pressure={Cd_p:.4f}, friction={Cd_f:.4f}"
                )

            if self.checkpoint_manager.should_save(self.n_steps):
                history_dict = {
                    'iterations': self.time_stamps.copy(),
                    'coefficients': {'Cd': self.cd_history.copy(), 'Cl': self.cl_history.copy()},
                }
                ckpt = self.checkpoint_manager.save(
                    solution=self.solution, history=history_dict, iteration=self.n_steps,
                    metadata={'current_time': self.current_time},
                    extra_fields={'mu_t': mu_t},
                )
                if ckpt:
                    self.checkpoint_path = ckpt

        elapsed = time.time() - start
        logger.info(
            f"Transient solve finished: {self.n_steps} steps, "
            f"t={self.current_time:.6f}s, {elapsed:.1f}s"
        )

        return TransientResult(
            solution_final=self.solution.copy(),
            total_time=self.current_time,
            n_steps=self.n_steps,
            cd_history=self.cd_history,
            cl_history=self.cl_history,
            time_stamps=self.time_stamps,
            checkpoint_path=self.checkpoint_path,
        )
