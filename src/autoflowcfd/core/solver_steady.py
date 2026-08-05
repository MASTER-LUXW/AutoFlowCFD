"""Steady-state solver using Flux Reconstruction scheme.

This module implements the steady-state RANS solver coordinator,
orchestrating FVM algorithms and solver loop.

Key Components:
    - SteadyResult: Container for steady simulation results
    - FRSolver: Main steady-state solver class (coordinator)
"""

import time
import numpy as np
from typing import Dict, Optional, List, Union
from dataclasses import dataclass, field
from loguru import logger

from ..grid.structures import GridData, VolumeMeshData
from ..config.solver_config import SteadyConfig, TurbulenceModel, BackendType
from .time_integration import TimeIntegrator, TimeIntegrationScheme
from .backend import create_backend
from ..boundary.manager import BoundaryManager
from .fvm_core import FVMFaceExtractor
from .fvm_gradients import FaceGeometry
from .fvm_viscous_residual import ViscousRANSResidual, estimate_wall_distance
from .bc_handler import BoundaryConditionHandler
from .aero_coeffs import AeroCoefficientCalculator
from .checkpoint import CheckpointManager


@dataclass
class SteadyResult:
    """Container for steady-state simulation results."""
    
    converged: bool
    iterations: int
    final_residual: float
    cd_history: List[float] = field(default_factory=list)
    cl_history: List[float] = field(default_factory=list)
    residuals_history: List[float] = field(default_factory=list)
    solution_final: Optional[np.ndarray] = None
    checkpoint_path: Optional[str] = None
    
    def get_mean_coefficients(self) -> Dict[str, float]:
        """Compute mean aerodynamic coefficients."""
        if len(self.cd_history) == 0:
            return {"Cd": 0.0, "Cl": 0.0}
        
        n_samples = max(1, len(self.cd_history) // 10)
        cd_mean = float(np.mean(self.cd_history[-n_samples:]))
        cl_mean = float(np.mean(self.cl_history[-n_samples:]))
        
        return {"Cd": cd_mean, "Cl": cl_mean}
    
    def get_convergence_rate(self) -> float:
        """Compute average convergence rate."""
        if len(self.residuals_history) < 2:
            return 0.0
        
        initial_residual = self.residuals_history[0]
        final_residual = self.residuals_history[-1]
        
        if initial_residual <= 0:
            return 0.0
        
        total_reduction = np.log(initial_residual / max(final_residual, 1e-16))
        return total_reduction / self.iterations


class FRSolver:
    """Flux Reconstruction steady-state solver coordinator.
    
    Orchestrates FVM algorithms, boundary conditions, and solver loop.
    """
    
    def __init__(self, grid_data: Union[GridData, VolumeMeshData], config: SteadyConfig):
        """Initialize steady-state solver."""
        self.grid_data = grid_data
        self.config = config

        logger.info(f"Initializing FRSolver")
        logger.info(f"  Grid: {grid_data.node_count} nodes, {grid_data.cell_count} cells")

        # Backend selection. NOTE: the actual residual (ViscousRANSResidual)
        # is a fixed pure-NumPy CPU implementation - it does not currently
        # read from or dispatch through this backend object at all, so
        # --backend gpu silently runs on CPU. Warn loudly rather than let
        # a user believe they're getting GPU acceleration they aren't.
        try:
            self.backend = create_backend(
                backend_type=config.backend.value,
                n_threads=config.n_threads,
                device_id=config.gpu_device,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize backend: {e}")

        if config.backend == BackendType.GPU:
            logger.warning(
                "--backend gpu was requested, but the steady RANS residual "
                "(ViscousRANSResidual) is not yet wired to any GPU backend - "
                "the solve will run on CPU (NumPy) regardless. This is a "
                "known gap, not a silent failure: see backend/gpu_backend.py."
            )
        if config.order != 2:
            logger.warning(
                f"--order {config.order} was requested, but the residual's "
                "MUSCL reconstruction (Green-Gauss gradient + Barth-Jespersen "
                "limiter) is fixed at 2nd order - the FR order setting "
                "currently has no effect on the steady solve."
            )
        if config.turbulence not in (TurbulenceModel.SST_KW, TurbulenceModel.NONE):
            logger.warning(
                f"--turbulence {config.turbulence.value} was requested, but "
                "the residual's turbulence closure is a fixed SST k-omega "
                "model - only 'sst_kw' (on) or 'none' (off) currently apply."
            )

        # Time integrator (explicit SSP-RK3 in pseudo-time).
        self.time_integrator = TimeIntegrator(
            scheme=TimeIntegrationScheme.SSP_RK3,
            dt=1e-4,
            cfl_target=config.cfl_init,
        )

        # Boundary manager
        self.boundary_manager = BoundaryManager(grid_data.boundaries)

        # Solution vector
        self.solution = None

        # Convergence history restored from a checkpoint on resume (set by
        # the CLI before calling solve() - see resume path in
        # cli/solve_commands.py). Used by solve() to restore the adaptive
        # CFL state (cfl_history) instead of always resetting to
        # config.cfl_init on resume.
        self.convergence_history = None

        # FVM face data holder (build_from_tetrahedra() is not used - see
        # solve(): face data comes from grid_data.ensure_faces_exist(), the
        # Numba-accelerated path; this instance is kept only as the shared
        # data-holder that bc_handler/aero_calculator read face arrays from).
        self.face_extractor = FVMFaceExtractor()

        # Helper modules
        self.bc_handler = BoundaryConditionHandler(
            grid_data, self.face_extractor,
            rho_inf=config.rho_inf, p_inf=config.p_inf,
        )
        self.aero_calculator = AeroCoefficientCalculator(
            grid_data, self.face_extractor,
            rho_inf=config.rho_inf, vel_inf=config.vel_inf,
        )
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            config=config,
            output_dir=config.output_dir,
            checkpoint_interval=config.checkpoint_interval
        )

        logger.info("FRSolver initialization complete")
    
    def _get_cell_volumes(self) -> np.ndarray:
        """Get cell volumes."""
        if isinstance(self.grid_data, VolumeMeshData):
            return self.grid_data.get_cell_volumes()
        else:
            return self.cell_volumes
    
    def _initialize_solution(self):
        """Initialize solution field with freestream conditions.
        
        Uses freestream velocity as initial condition for numerical stability.
        Starting from rest can cause instability in HLLC flux computation.
        
        Solution variables (conservative form):
            [rho, rhou, rhov, rhow, E, k, omega]
        """
        logger.info("Initializing solution field...")
        
        n_cells = self.grid_data.cell_count

        # Freestream conditions for stable initialization (single source of
        # truth: self.config, shared with boundary conditions and Cd/Cl
        # normalization so all three always agree).
        rho_0 = self.config.rho_inf
        u_0 = self.config.vel_inf     # use freestream velocity for stability
        v_0 = 0.0
        w_0 = 0.0
        p_0 = self.config.p_inf
        gamma = 1.4
        
        # Compute conservative variables
        rhou_0 = rho_0 * u_0
        rhov_0 = rho_0 * v_0
        rhow_0 = rho_0 * w_0
        E_0 = p_0 / (gamma - 1.0) + 0.5 * rho_0 * (u_0**2 + v_0**2 + w_0**2)
        
        # Turbulence field: conservative (rho*k, rho*omega).
        # Freestream: 1% intensity, length scale ~0.1 m.
        u_ref = max(u_0, 1.0)
        k_0 = 1.5 * (0.01 * u_ref)**2
        omega_0 = 5.0 * u_ref / 0.1

        # Allocate and initialize solution array
        self.solution = np.zeros((n_cells, 7), dtype=np.float64)
        self.solution[:, 0] = rho_0   # density
        self.solution[:, 1] = rhou_0  # x-momentum
        self.solution[:, 2] = rhov_0  # y-momentum
        self.solution[:, 3] = rhow_0  # z-momentum
        self.solution[:, 4] = E_0     # total energy
        self.solution[:, 5] = rho_0 * k_0      # conservative turbulent KE
        self.solution[:, 6] = rho_0 * omega_0  # conservative specific dissipation

        logger.info(f"Solution initialized: {n_cells} cells")
        logger.info(f"  Initial conditions: rho={rho_0:.3f} kg/m^3, u={u_0:.1f} m/s, p={p_0:.0f} Pa")
        logger.info(f"  Turbulence: k={k_0:.4e} m^2/s^2, omega={omega_0:.2f} 1/s")

    def _setup_boundary_conditions(self):
        """Setup boundary conditions."""
        logger.info("Setting up boundary conditions...")

        boundary_names = self.grid_data.boundaries.boundary_names
        vel_inf = self.config.vel_inf
        p_inf = self.config.p_inf

        for boundary_name in boundary_names:
            name_upper = boundary_name.upper()

            if "INLET" in name_upper or "INFLOW" in name_upper:
                # Store base velocity for ramping (now handled by bc_handler)
                self.bc_handler.base_inlet_velocity = vel_inf
                self.boundary_manager.add_bc(
                    boundary_name, bc_type="INLET",
                    velocity_x=vel_inf, pressure=p_inf, temperature=288.15,
                )
            elif "OUTLET" in name_upper:
                self.boundary_manager.add_bc(boundary_name, bc_type="OUTLET", pressure=p_inf)
            elif "BODY" in name_upper or "CAR" in name_upper:
                self.boundary_manager.add_bc(boundary_name, bc_type="WALL")
            elif "GROUND" in name_upper:
                self.bc_handler.base_farfield_velocity = vel_inf
                self.boundary_manager.add_bc(boundary_name, bc_type="GROUND", moving_wall_velocity=vel_inf)
            elif "TUNNEL" in name_upper:
                # A named "tunnel" boundary is a physical (frictionless)
                # duct wall, not an open domain boundary - see
                # bc_handler.py's _classify for the matching live-path
                # reclassification (SYMMETRY = free-slip, zero-penetration).
                self.boundary_manager.add_bc(boundary_name, bc_type="SYMMETRY")
            elif "FARFIELD" in name_upper:
                self.bc_handler.base_farfield_velocity = vel_inf
                self.boundary_manager.add_bc(boundary_name, bc_type="FARFIELD", velocity_x=vel_inf, pressure=p_inf)
            elif "SYMMETRY" in name_upper:
                self.boundary_manager.add_bc(boundary_name, bc_type="SYMMETRY")
            else:
                self.bc_handler.base_farfield_velocity = vel_inf
                self.boundary_manager.add_bc(boundary_name, bc_type="FARFIELD", velocity_x=vel_inf, pressure=p_inf)
        
        logger.info(f"Boundary conditions setup: {len(boundary_names)} boundaries")
    
    def solve(self, max_iter: Optional[int] = None, start_iteration: int = 0):
        """Execute steady-state simulation.

        Drives an explicit SSP-RK pseudo-time march of the second-order viscous
        RANS residual, with local time stepping and a normalised multi-equation
        convergence criterion.

        Args:
            max_iter: Maximum TOTAL iteration count (overrides config if
                provided). This is an absolute target, not a count of
                additional steps - when resuming with start_iteration=50,
                max_iter=2500 means "run up to iteration 2500 total"
                (2450 more steps), matching the CLI's documented semantics.
            start_iteration: Iteration count already completed (e.g. loaded
                from a checkpoint). Iteration numbering, logging, and the
                inlet/farfield velocity ramp (BoundaryConditionHandler.
                update_ramp_factor) all continue from here instead of
                restarting at 1 - otherwise resuming would snap the ramp
                factor back down to ~0 and reintroduce a boundary-condition
                discontinuity at the resume point.

        Returns:
            SteadyResult with solution and history
        """

        # Initialize
        if self.solution is None:
            self._initialize_solution()

        self._setup_boundary_conditions()

        # CRITICAL FIX: Use optimized face extraction from VolumeMeshData instead of slow FVMFaceExtractor
        logger.info("Using pre-computed face data from VolumeMeshData (optimized radix-sort)...")
        t_face_start = time.perf_counter()
        
        # Ensure faces exist (uses optimized FaceExtractor with argsort)
        face_data_obj = self.grid_data.ensure_faces_exist()
        
        # Compute cell centroids FIRST (needed for gradient reconstruction)
        nodes_array = np.column_stack([
            self.grid_data.nodes.x,
            self.grid_data.nodes.y,
            self.grid_data.nodes.z,
        ])
        # Prism cells (if any - see VolumeMeshData.prism_cells) occupy the
        # front of the global cell-index space, tets the rest (same
        # convention grid_data.get_cell_volumes() below already follows) -
        # centroids must be built the same way, or they'd misalign against
        # cell_volumes/geom.cell_centroids for every BL cell once a prism
        # mesh is in play (a plain tets-only average silently produced only
        # n_tet rows, not n_prism+n_tet, for a mixed mesh here previously).
        tet_connectivity_int64 = self.grid_data.cells.connectivity.astype(np.int64)
        tet_centroids = nodes_array[tet_connectivity_int64].mean(axis=1)
        prism_cells_obj = getattr(self.grid_data, 'prism_cells', None)
        if prism_cells_obj is not None:
            prism_connectivity_int64 = prism_cells_obj.connectivity.astype(np.int64)
            prism_centroids = nodes_array[prism_connectivity_int64].mean(axis=1)
            cell_centroids = np.vstack([prism_centroids, tet_centroids])
        else:
            cell_centroids = tet_centroids
        
        # Store in face_extractor for later use
        self.face_extractor.cell_centroids = cell_centroids
        
        # Convert FaceData to the plain-dict format the rest of solve() uses
        face_data = {
            'connectivity': face_data_obj.connectivity,
            'normals': face_data_obj.normal,
            'areas': face_data_obj.area,
            'centers': face_data_obj.center,
            'boundary_flags': (face_data_obj.connectivity[:, 1] < 0).astype(np.int32),
            'cell_centroids': cell_centroids,
        }
        
        t_face_end = time.perf_counter()
        logger.success(f"Face data prepared in {t_face_end - t_face_start:.2f}s (optimized)")

        # Expose face data on face_extractor - bc_handler/aero_calculator
        # both read face arrays from this shared holder.
        self.face_extractor.face_connectivity = face_data['connectivity']
        self.face_extractor.face_normals = face_data['normals']
        self.face_extractor.face_areas = face_data['areas']
        self.face_extractor.boundary_flags = face_data['boundary_flags']

        # Assemble shared geometry bundle for the residual.
        cell_volumes = self._get_cell_volumes()
        geom = FaceGeometry(
            connectivity=face_data['connectivity'],
            normals=face_data['normals'],
            areas=face_data['areas'],
            centers=face_data['centers'],
            boundary_flags=face_data['boundary_flags'],
            cell_centroids=face_data['cell_centroids'],
            cell_volumes=cell_volumes,
        )

        # Wall distance from viscous-wall boundary faces (WALL/GROUND).
        try:
            self.bc_handler._precompute_face_types()
            wall_face_mask = np.zeros(geom.n_faces, dtype=bool)
            for f, t in self.bc_handler._face_types.items():
                if t in ("WALL", "GROUND"):
                    wall_face_mask[f] = True
            logger.info(f"Wall face mask computed: {np.sum(wall_face_mask)} wall faces")
            wall_distance = estimate_wall_distance(geom, wall_face_mask)
            logger.info(f"Wall distance estimated: min={wall_distance.min():.4e}, max={wall_distance.max():.4e}")
        except Exception as e:
            logger.error(f"Failed to compute wall distance: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

        # Molecular viscosity (Sutherland at 288 K ~ 1.79e-5 Pa s).
        mu_lam = 1.7894e-5
        turbulent = self.config.turbulence != TurbulenceModel.NONE

        # Reference (freestream) Mach number, used two ways:
        #  1. ViscousRANSResidual's inviscid flux: AUSM+up (see
        #     fvm_viscous_residual.py's _ausm_up), which replaced HLLC as
        #     the live flux specifically for its low-Mach robustness. It
        #     has a built-in low-Mach scaling function f_a that this
        #     reference Mach regularizes so f_a stays bounded away from
        #     zero at genuine stagnation points, instead of a wave-speed-
        #     bracket-based preconditioner (tried first, reverted - it
        #     narrowed HLLC's SL/SR margin around the star-state wave
        #     speed Sstar by ~10x at this case's M~0.09 everywhere in the
        #     domain at once, causing a much faster, more widespread
        #     numerical blow-up than the unpreconditioned scheme; see
        #     fvm_viscous_residual.py's _hllc docstring for the history).
        #  2. TimeIntegrator.local_time_step below: relaxes the pseudo-
        #     time CFL restriction that a density-based scheme otherwise
        #     inherits from the acoustic speed (~340 m/s) even though the
        #     physical flow here is far slower - this part never touches
        #     the flux itself, only how big a step is stable to take.
        gamma_air = 1.4
        a_inf = np.sqrt(gamma_air * self.config.p_inf / self.config.rho_inf)
        mach_ref = self.config.vel_inf / max(a_inf, 1e-30)

        residual = ViscousRANSResidual(
            geom, mu_lam=mu_lam, wall_distance=wall_distance, turbulent=turbulent,
            mach_ref=mach_ref,
            wall_face_mask=wall_face_mask if self.config.use_wall_functions else None,
        )
        if self.config.use_wall_functions:
            logger.info(
                "Wall functions enabled (Menter scalable/automatic wall treatment) "
                f"on {np.sum(wall_face_mask)} WALL/GROUND faces - near-wall mesh no "
                "longer needs y+~1 to be accurate."
            )

        # Aerodynamic reference area.
        body_face_indices = self.aero_calculator._identify_body_faces()
        ref_area = self.aero_calculator._compute_reference_area(body_face_indices)

        actual_max_iter = max_iter if max_iter is not None else self.config.max_iter
        if start_iteration >= actual_max_iter:
            logger.warning(
                f"start_iteration ({start_iteration}) >= max_iter ({actual_max_iter}); "
                f"nothing to do."
            )
        logger.info(
            f"Starting steady RANS solve (start_iteration={start_iteration}, "
            f"max_iter={actual_max_iter}, turbulent={turbulent})"
        )

        # Restore adaptive CFL state on resume. Without this, the CFL always
        # resets to config.cfl_init when resuming from a checkpoint,
        # discarding whatever the adaptive mechanism had converged to (e.g.
        # a CFL reduced far below cfl_init to survive a stiff region) - the
        # first post-resume iterations would then re-attempt the original,
        # already-known-risky cfl_init and could immediately diverge again.
        if self.convergence_history and self.convergence_history.get('cfl_history'):
            restored_cfl = float(self.convergence_history['cfl_history'][-1])
            if restored_cfl > 0:
                logger.info(
                    f"Restoring adaptive CFL from checkpoint history: "
                    f"{self.time_integrator.cfl_target:.4f} -> {restored_cfl:.4f}"
                )
                self.time_integrator.cfl_target = restored_cfl

        cd_history, cl_history, res_history = [], [], []
        initial_res_vec = None
        converged = False
        start = time.time()
        # Coordination state shared by the three CFL-adjustment mechanisms
        # below (divergence auto-recovery, explosive-growth guard, and the
        # residual-trend adaptive rule) so they can't fight each other -
        # e.g. the trend rule increasing CFL right back up in the same or
        # very next iteration after a safety mechanism just cut it, before
        # the lower CFL has had any chance to actually take effect.
        last_cfl_cut_iteration = -10**9
        cfl_cut_cooldown = 20
        # If start_iteration >= actual_max_iter the loop body below never
        # runs; keep `iteration` defined (as the last completed iteration)
        # so the post-loop logging/checkpoint code doesn't reference an
        # unbound local.
        iteration = start_iteration

        def residual_func(U):
            bstates = self.bc_handler.build_boundary_states(U)
            return residual.compute(U, bstates)

        for step in range(1, actual_max_iter - start_iteration + 1):
            iteration = start_iteration + step
            # Reset per-iteration flag: was CFL already cut by a safety
            # mechanism (divergence recovery / explosive-growth guard) this
            # iteration? If so, the trend-based rule below skips entirely
            # rather than potentially reversing the cut in the same step.
            cfl_cut_this_iter = False
            if self.bc_handler is not None:
                self.bc_handler.update_ramp_factor(iteration, actual_max_iter)

            # Boundary ghost states for this solution - computed once and
            # reused below (gradients, residual, aero coefficients) instead
            # of being rebuilt from scratch for each consumer.
            bstates = self.bc_handler.build_boundary_states(self.solution)

            # Effective viscosity for viscous time-step limit.
            rho_c, vel_c, p_c, T_c, k_c, w_c = residual.to_primitive(self.solution)
            
            # Diagnostic: log shapes before gradient computation
            if iteration <= 5 or iteration % 10 == 0:
                logger.debug(
                    f"[Iter {iteration}] Primitive variables shapes:\n"
                    f"  rho_c: {rho_c.shape}, vel_c: {vel_c.shape}\n"
                    f"  k_c: {k_c.shape}, w_c (omega): {w_c.shape}\n"
                    f"  wall_distance: {residual.wall_distance.shape}"
                )
            
            try:
                gvel = residual._velocity_gradient(vel_c, self.solution, bstates)
                
                if iteration <= 5 or iteration % 10 == 0:
                    logger.debug(f"[Iter {iteration}] Velocity gradient shape: {gvel.shape}")
                
                mu_t = residual._eddy_viscosity(rho_c, k_c, w_c, gvel) if turbulent \
                    else np.zeros(geom.n_cells)
                    
                if iteration <= 5 or iteration % 10 == 0:
                    logger.debug(f"[Iter {iteration}] Eddy viscosity shape: {mu_t.shape}")
            except Exception as e:
                logger.error(f"[Iter {iteration}] Failed to compute viscosity: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise

            # Compute local time step with current CFL. omega=w_c adds the
            # SST source-term stiffness limit (see local_time_step docstring)
            # - without it, near-wall cells with large omega can be unstable
            # for the k/omega equations even at a CFL that's comfortably
            # safe for the convective/viscous mean-flow terms.
            dt_local = self.time_integrator.local_time_step(
                self.solution, geom, mu_lam + mu_t, omega=(w_c if turbulent else None),
                mach_ref=mach_ref,
            )

            # One SSP-RK pseudo-time step. R is both the residual used for
            # convergence monitoring below and the RK scheme's own stage-0
            # residual (Ui=U0 at i=0) - computed once and reused via
            # residual0= instead of letting step() recompute the same
            # (expensive: MUSCL+HLLC+viscous+SST) evaluation a second time.
            R = residual.compute(self.solution, bstates)
            self.solution = self.time_integrator.step(
                self.solution, residual_func, dt_local, residual0=R
            )

            # Check for numerical divergence immediately after update
            if not np.all(np.isfinite(self.solution)):
                logger.error(f"Solver diverged at iteration {iteration}: non-finite state detected")
                logger.error("  Possible causes:")
                logger.error("    1. CFL number too high for current grid/solution")
                logger.error("    2. Boundary condition inconsistency")
                logger.error("    3. Turbulence model stiffness (try reducing CFL)")
                logger.error(f"  Current CFL: {self.time_integrator.cfl_target:.4f}")
                
                # === AUTOMATIC RECOVERY ATTEMPT ===
                if self.time_integrator.cfl_target > 0.01:
                    old_cfl = self.time_integrator.cfl_target
                    self.time_integrator.cfl_target = max(old_cfl * 0.1, 0.005)
                    last_cfl_cut_iteration = iteration
                    logger.warning(f"[AUTO-RECOVERY] Attempting automatic recovery by reducing CFL to {self.time_integrator.cfl_target:.4f}")

                    # Restore solution from previous step if available
                    if iteration > 1 and hasattr(self, '_last_stable_solution'):
                        self.solution = self._last_stable_solution.copy()
                        logger.info("[AUTO-RECOVERY] Restored solution from last stable state")
                    
                    continue  # Retry this iteration with lower CFL
                else:
                    raise RuntimeError(f"Solver diverged at iteration {iteration}: non-finite state")
            
            # Save stable solution for potential recovery
            if iteration % 5 == 0:
                self._last_stable_solution = self.solution.copy()

            # Normalised multi-equation residual (RMS over mass/momentum/energy).
            #
            # Volume-weighted, not a plain per-cell mean: R is already
            # per-unit-volume (residual.compute() divides by cell_volumes),
            # but an unweighted mean still counts every cell equally
            # regardless of how much of the domain it represents - a
            # boundary layer can hold thousands of tiny cells with locally
            # noisy residuals that would otherwise swamp the signal from
            # the much larger (but far fewer) far-field cells.
            cell_volumes = geom.cell_volumes
            total_volume = float(np.sum(cell_volumes))
            res_vec = np.sqrt(np.sum(R[:, :5]**2 * cell_volumes[:, None], axis=0) / total_volume)

            # Each of the 5 equations is normalised by ITS OWN initial-
            # iteration RMS before being combined into one scalar. Without
            # this, the energy equation's residual - intrinsically
            # ~rho*vel_inf^3 in scale, orders of magnitude larger than the
            # continuity/momentum residuals - dominates a plain combined
            # L2 norm, so "convergence" would really only track the energy
            # equation while mass/momentum are still moving.
            if initial_res_vec is None or iteration == 1:
                initial_res_vec = np.maximum(res_vec, 1e-30)
            rel_res = float(np.linalg.norm(res_vec / initial_res_vec)) / np.sqrt(len(res_vec))
            res_history.append(rel_res)
            
            # === EARLY DIVERGENCE WARNING ===
            if len(res_history) >= 3:
                recent_growth = res_history[-1] / max(res_history[-3], 1e-30)
                if recent_growth > 1e6:  # Explosive growth detected
                    logger.warning(
                        f"[DIVERGENCE WARNING] Residual grew by factor {recent_growth:.2e} in 3 steps! "
                        f"Current CFL={self.time_integrator.cfl_target:.4f}. "
                        f"Consider manual intervention."
                    )
                    # Force aggressive CFL reduction
                    self.time_integrator.cfl_target = max(self.time_integrator.cfl_target * 0.2, 0.005)
                    last_cfl_cut_iteration = iteration
                    cfl_cut_this_iter = True
                    logger.warning(f"[AUTO-FIX] Aggressively reduced CFL to {self.time_integrator.cfl_target:.4f}")

            # Adaptive CFL adjustment based on residual trend (IMPROVED)
            # Use longer window and log-scale for better stability. Skipped
            # entirely if a safety mechanism already cut CFL this iteration
            # (cfl_cut_this_iter) - otherwise this rule could immediately
            # increase CFL right back up in the very same step, since the
            # 8-iteration trend window doesn't yet reflect the cut's effect.
            if not cfl_cut_this_iter and iteration > 10 and len(res_history) >= 8:
                # Use last 8 iterations for trend analysis (smoother signal)
                n_window = min(8, len(res_history))
                recent = res_history[-n_window:]
                
                # Compute log-scale trend (better for exponential decay/growth)
                # trend = ln(res_final/res_initial) / n_steps
                # Negative = decreasing, Positive = increasing
                if recent[0] > 1e-30 and recent[-1] > 1e-30:
                    log_trend = np.log(recent[-1] / recent[0]) / (n_window - 1)
                else:
                    log_trend = 0.0
                
                # Add hysteresis to prevent oscillation
                # Only adjust if trend is significant AND sustained
                cfl_adjusted = False
                
                # Check for divergence or rapid increase
                if log_trend > 0.15:  # ~16% increase per step (aggressive threshold)
                    old_cfl = self.time_integrator.cfl_target
                    # More conservative reduction: ×0.6 instead of ×0.5
                    self.time_integrator.cfl_target = max(old_cfl * 0.6, 0.01)
                    last_cfl_cut_iteration = iteration
                    logger.warning(
                        f"  [CFL ADJUST] Residuals increasing (log_trend={log_trend:.3f}/step), "
                        f"reducing CFL: {old_cfl:.3f} -> {self.time_integrator.cfl_target:.3f}"
                    )
                    cfl_adjusted = True

                # Check for good convergence (can increase CFL) - only once
                # the CFL has been stable (no safety cut) for a cooldown
                # window, so an increase can't immediately undo a cut made
                # before the lower CFL has had a chance to prove itself.
                elif (log_trend < -0.25
                      and self.time_integrator.cfl_target < self.config.cfl_max
                      and iteration - last_cfl_cut_iteration >= cfl_cut_cooldown):
                    # Require sustained decrease over the window
                    # Check that most points are decreasing
                    decreases = sum(1 for i in range(len(recent)-1) 
                                   if recent[i+1] < recent[i])
                    decrease_ratio = decreases / (len(recent) - 1)
                    
                    if decrease_ratio > 0.7:  # At least 70% of steps decreasing
                        old_cfl = self.time_integrator.cfl_target
                        # Moderate increase: ×1.15 instead of ×1.2
                        self.time_integrator.cfl_target = min(old_cfl * 1.15, self.config.cfl_max)
                        logger.info(
                            f"  [CFL ADJUST] Residuals decreasing well (log_trend={log_trend:.3f}/step, "
                            f"decrease_ratio={decrease_ratio:.0%}), "
                            f"increasing CFL: {old_cfl:.3f} -> {self.time_integrator.cfl_target:.3f}"
                        )
                        cfl_adjusted = True
                
                if not cfl_adjusted and iteration % 50 == 0:
                    # Log status every 50 iterations even when not adjusting
                    logger.debug(
                        f"  [CFL STATUS] log_trend={log_trend:.3f}/step, "
                        f"CFL={self.time_integrator.cfl_target:.3f}, "
                        f"no adjustment needed"
                    )
            
            # Coefficients every iteration for accurate monitoring. Includes
            # skin-friction drag/lift via wall_shear_stress(), reusing this
            # iteration's gvel/mu_t/bstates (computed pre-step above) rather
            # than recomputing them for the post-step solution - a one-
            # iteration lag that's a good trade against doubling the
            # gradient+eddy-viscosity cost every iteration just for
            # monitoring output.
            
            # Diagnostic: log shapes before computing coefficients
            if iteration <= 5 or iteration % 10 == 0:
                logger.debug(
                    f"[Iter {iteration}] Pre-compute diagnostics:\n"
                    f"  Solution shape: {self.solution.shape}\n"
                    f"  gvel shape: {gvel.shape if gvel is not None else 'None'}\n"
                    f"  mu_t shape: {mu_t.shape if mu_t is not None else 'None'}\n"
                    f"  bstates shape: {bstates.shape if bstates is not None else 'None'}\n"
                    f"  Face normals shape: {self.face_extractor.face_normals.shape}\n"
                    f"  Face areas shape: {self.face_extractor.face_areas.shape}"
                )
            
            try:
                Cd, Cl, Cd_p, Cd_f = self.aero_calculator.compute_coefficients(
                    self.solution, iteration,
                    viscous_residual=residual, grad_vel=gvel, mu_t=mu_t,
                    boundary_states=bstates,
                )
            except Exception as e:
                logger.error(f"[Iter {iteration}] Failed to compute aerodynamic coefficients: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise
            
            cd_history.append(Cd)
            cl_history.append(Cl)

            # Output coefficients with improved formatting
            # Main line: iteration info
            logger.info(
                f"Iter {iteration:5d}/{actual_max_iter}  |  "
                f"Res(rel): {rel_res:.4e}  |  "
                f"Cd: {Cd:.4f}  |  "
                f"Cl: {Cl:.4f}"
            )
            
            # Second line: Cd breakdown (aligned with first |)
            # Calculate indentation to align 'Cd' with the first '|'
            # "Iter XXXX/XXXX  |  " = ~20 chars, so indent to position of first |
            prefix_len = len(f"Iter {iteration:5d}/{actual_max_iter}")
            logger.info(
                f"{'':>{prefix_len + 2}s}  "
                f"Cd breakdown: pressure={Cd_p:.4f}, friction={Cd_f:.4f}"
            )

            # Save checkpoint periodically
            if self.checkpoint_manager.should_save(iteration):
                history_dict = {
                    'iterations': list(range(1, iteration + 1)),
                    'residuals': {'continuity': res_history.copy()},
                    'coefficients': {'Cd': cd_history.copy(), 'Cl': cl_history.copy()},
                    'cfl_history': [self.time_integrator.cfl_target] * len(res_history),
                }
                
                ckpt_path = self.checkpoint_manager.save(
                    solution=self.solution,
                    history=history_dict,
                    iteration=iteration,
                    extra_fields={'mu_t': mu_t},
                )
                
                if ckpt_path:
                    logger.info(f"Checkpoint saved at iteration {iteration}")
                    
                    # Cleanup old checkpoints
                    self.checkpoint_manager.cleanup_old_checkpoints(keep_last=3)

            # Convergence: normalised residual below tolerance.
            if rel_res < self.config.convergence_tol:
                logger.success(f"Converged at iteration {iteration} (rel residual {rel_res:.3e})")
                converged = True
                break

        elapsed = time.time() - start
        logger.info(f"Solve finished: {len(res_history)} iters, {elapsed:.1f}s, converged={converged}")

        # Save final checkpoint
        try:
            logger.debug("Preparing final checkpoint data...")
            final_history = {
                'iterations': list(range(1, iteration + 1)),
                'residuals': {'continuity': res_history.copy()},
                'coefficients': {'Cd': cd_history.copy(), 'Cl': cl_history.copy()},
                'cfl_history': [self.time_integrator.cfl_target] * len(res_history),
            }
            
            logger.debug(f"Final history keys: {final_history.keys()}")
            logger.debug(f"Cd history length: {len(cd_history)}, Cl history length: {len(cl_history)}")
            
            final_ckpt = self.checkpoint_manager.save(
                solution=self.solution,
                history=final_history,
                iteration=iteration,
                extra_fields={'mu_t': mu_t},
            )
            
            if final_ckpt:
                logger.info(f"Final checkpoint saved: {final_ckpt}")
        except Exception as e:
            logger.error(f"Failed to save final checkpoint: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

        return SteadyResult(
            converged=converged,
            iterations=len(res_history),
            final_residual=res_history[-1] if res_history else 0.0,
            cd_history=cd_history,
            cl_history=cl_history,
            residuals_history=res_history,
            solution_final=self.solution.copy(),
            checkpoint_path=final_ckpt,
        )
