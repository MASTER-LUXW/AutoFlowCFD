"""Steady-state solver using Flux Reconstruction scheme.

This module implements the steady-state RANS solver coordinator,
orchestrating FVM algorithms and solver loop.

Key Components:
    - SteadyResult: Container for steady simulation results
    - FRSolver: Main steady-state solver class (coordinator)
"""

import numpy as np
from typing import Dict, Optional, List, Union
from dataclasses import dataclass, field
from loguru import logger

from ..grid.structures import GridData, VolumeMeshData
from ..config.solver_config import SteadyConfig, TurbulenceModel
from .fr_scheme import FRScheme, FROrder
from .turbulence import SSTKOmegaModel
from .convergence import ConvergenceMonitor
from .time_integration import TimeIntegrator, TimeIntegrationScheme
from .backend import create_backend
from ..boundary.manager import BoundaryManager
from .fvm_core import FVMFaceExtractor, FVMFluxCalculator, FVMResidualComputer
from .fvm_gradients import FaceGeometry
from .fvm_viscous_residual import ViscousRANSResidual, estimate_wall_distance
from .solver_loop import SteadySolverLoop
from .bc_handler import BoundaryConditionHandler
from .aero_coeffs import AeroCoefficientCalculator
from .solution_constraints import SolutionConstraintHandler


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
        
        # Initialize backend
        try:
            self.backend = create_backend(
                backend_type=config.backend.value,
                n_threads=config.n_threads,
                device_id=config.gpu_device,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize backend: {e}")
        
        # Initialize FR scheme
        self.fr_scheme = FRScheme(order=FROrder(config.order))
        
        # Initialize turbulence model
        self.turbulence_model = SSTKOmegaModel() if config.turbulence == TurbulenceModel.SST_KW else SSTKOmegaModel()
        
        # Initialize convergence monitor
        self.convergence_monitor = ConvergenceMonitor(
            max_iterations=config.max_iter,
            convergence_threshold=config.convergence_tol,
            cfl_initial=config.cfl_init,
            cfl_max=config.cfl_max,
        )
        
        # Initialize time integrator (explicit SSP-RK3 in pseudo-time).
        self.time_integrator = TimeIntegrator(
            scheme=TimeIntegrationScheme.SSP_RK3,
            dt=1e-4,
            cfl_target=config.cfl_init,
        )
        
        # Initialize boundary manager
        self.boundary_manager = BoundaryManager(grid_data.boundaries)
        
        # Initialize MUSCL reconstructor
        if config.order >= 2:
            from .reconstruction_v2 import MUSCLReconstructor, LimiterType
            self.muscl_reconstructor = MUSCLReconstructor(LimiterType.VAN_LEER)
        
        # Solution vector
        self.solution = None
        
        # Initialize FVM components
        self.face_extractor = FVMFaceExtractor()
        self.flux_calculator = FVMFluxCalculator(gamma=1.4)
        self.residual_computer = FVMResidualComputer(self.flux_calculator)
        
        # Initialize helper modules
        self.bc_handler = BoundaryConditionHandler(grid_data, self.face_extractor)
        self.aero_calculator = AeroCoefficientCalculator(grid_data, self.face_extractor)
        self.constraint_handler = SolutionConstraintHandler(gamma=1.4)
        
        # Initialize solver loop
        self.solver_loop = SteadySolverLoop(
            config=config,
            residual_computer=self.residual_computer,
            convergence_monitor=self.convergence_monitor,
            time_integrator=self.time_integrator,
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
        
        # Freestream conditions for stable initialization
        rho_0 = 1.225  # kg/m^3
        u_0 = 30.0     # m/s - use freestream velocity for stability
        v_0 = 0.0
        w_0 = 0.0
        p_0 = 101325.0 # Pa
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
        logger.info(f"  Initial conditions: rho={rho_0:.3f} kg/m³, u={u_0:.1f} m/s, p={p_0:.0f} Pa")
        logger.info(f"  Turbulence: k={k_0:.4e} m²/s², omega={omega_0:.2f} 1/s")

    def _setup_boundary_conditions(self):
        """Setup boundary conditions."""
        logger.info("Setting up boundary conditions...")
        
        boundary_names = self.grid_data.boundaries.boundary_names
        
        for boundary_name in boundary_names:
            name_upper = boundary_name.upper()
            
            if "INLET" in name_upper or "INFLOW" in name_upper:
                # Store base velocity for ramping (now handled by bc_handler)
                self.bc_handler.base_inlet_velocity = 30.0
                self.boundary_manager.add_bc(
                    boundary_name, bc_type="INLET",
                    velocity_x=30.0, pressure=101325.0, temperature=288.15,
                )
            elif "OUTLET" in name_upper:
                self.boundary_manager.add_bc(boundary_name, bc_type="OUTLET", pressure=101325.0)
            elif "BODY" in name_upper or "CAR" in name_upper:
                self.boundary_manager.add_bc(boundary_name, bc_type="WALL")
            elif "GROUND" in name_upper:
                self.bc_handler.base_farfield_velocity = 30.0
                self.boundary_manager.add_bc(boundary_name, bc_type="GROUND", moving_wall_velocity=30.0)
            elif "TUNNEL" in name_upper or "FARFIELD" in name_upper:
                self.bc_handler.base_farfield_velocity = 30.0
                self.boundary_manager.add_bc(boundary_name, bc_type="FARFIELD", velocity_x=30.0, pressure=101325.0)
            elif "SYMMETRY" in name_upper:
                self.boundary_manager.add_bc(boundary_name, bc_type="SYMMETRY")
            else:
                self.bc_handler.base_farfield_velocity = 30.0
                self.boundary_manager.add_bc(boundary_name, bc_type="FARFIELD", velocity_x=30.0, pressure=101325.0)
        
        logger.info(f"Boundary conditions setup: {len(boundary_names)} boundaries")
    
    def solve(self, max_iter: Optional[int] = None):
        """Execute steady-state simulation.

        Drives an explicit SSP-RK pseudo-time march of the second-order viscous
        RANS residual, with local time stepping and a normalised multi-equation
        convergence criterion.

        Args:
            max_iter: Maximum iterations (overrides config if provided)

        Returns:
            SteadyResult with solution and history
        """
        import time

        # Initialize
        if self.solution is None:
            self._initialize_solution()

        self._setup_boundary_conditions()

        # Build oriented face connectivity/geometry.
        nodes = np.column_stack([
            self.grid_data.nodes.x,
            self.grid_data.nodes.y,
            self.grid_data.nodes.z,
        ])
        face_data = self.face_extractor.build_from_tetrahedra(
            self.grid_data.cells.connectivity, nodes
        )

        # Expose face data on the flux calculator (used by aero/bc helpers).
        self.flux_calculator.face_connectivity = face_data['connectivity']
        self.flux_calculator.face_normals = face_data['normals']
        self.flux_calculator.face_areas = face_data['areas']
        self.flux_calculator.boundary_flags = face_data['boundary_flags']

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
        self.bc_handler._precompute_face_types()
        wall_face_mask = np.zeros(geom.n_faces, dtype=bool)
        for f, t in self.bc_handler._face_types.items():
            if t in ("WALL", "GROUND"):
                wall_face_mask[f] = True
        wall_distance = estimate_wall_distance(geom, wall_face_mask)

        # Molecular viscosity (Sutherland at 288 K ~ 1.79e-5 Pa s).
        mu_lam = 1.7894e-5
        turbulent = self.config.turbulence != TurbulenceModel.NONE

        residual = ViscousRANSResidual(
            geom, mu_lam=mu_lam, wall_distance=wall_distance, turbulent=turbulent
        )

        # Aerodynamic reference area.
        body_face_indices = self.aero_calculator._identify_body_faces()
        ref_area = self.aero_calculator._compute_reference_area(body_face_indices)

        actual_max_iter = max_iter if max_iter is not None else self.config.max_iter
        logger.info(f"Starting steady RANS solve (max_iter={actual_max_iter}, turbulent={turbulent})")

        cd_history, cl_history, res_history = [], [], []
        initial_res = None
        converged = False
        start = time.time()

        def residual_func(U):
            bstates = self.bc_handler.build_boundary_states(U)
            return residual.compute(U, bstates)

        for iteration in range(1, actual_max_iter + 1):
            if self.bc_handler is not None:
                self.bc_handler.update_ramp_factor(iteration, actual_max_iter)

            # Effective viscosity for viscous time-step limit.
            rho_c, vel_c, p_c, T_c, k_c, w_c = residual.to_primitive(self.solution)
            gvel = residual._velocity_gradient(vel_c, self.solution,
                                               self.bc_handler.build_boundary_states(self.solution))
            mu_t = residual._eddy_viscosity(rho_c, k_c, w_c, gvel) if turbulent \
                else np.zeros(geom.n_cells)
            dt_local = self.time_integrator.local_time_step(self.solution, geom, mu_lam + mu_t)

            # One SSP-RK pseudo-time step.
            R = residual_func(self.solution)
            self.solution = self.time_integrator.step(self.solution, residual_func, dt_local)

            if not np.all(np.isfinite(self.solution)):
                raise RuntimeError(f"Solver diverged at iteration {iteration}: non-finite state")

            # Normalised multi-equation residual (RMS over mass/momentum/energy).
            res_vec = np.sqrt(np.mean(R[:, :5]**2, axis=0))     # per-equation RMS
            res_norm = float(np.linalg.norm(res_vec))
            if initial_res is None or iteration == 1:
                initial_res = max(res_norm, 1e-30)
            rel_res = res_norm / initial_res
            res_history.append(rel_res)

            # Coefficients every few iterations.
            if iteration % 5 == 0 or iteration == 1:
                Cd, Cl = self.aero_calculator.compute_coefficients(self.solution, iteration)
                cd_history.append(Cd)
                cl_history.append(Cl)

            if iteration <= 10 or iteration % 20 == 0:
                logger.info(
                    f"Iter {iteration:5d}/{actual_max_iter} | "
                    f"Res(rel): {rel_res:.4e} | "
                    f"Cd: {cd_history[-1] if cd_history else 0.0:.4f} | "
                    f"Cl: {cl_history[-1] if cl_history else 0.0:.4f}"
                )

            # Convergence: normalised residual below tolerance.
            if rel_res < self.config.convergence_tol:
                logger.success(f"Converged at iteration {iteration} (rel residual {rel_res:.3e})")
                converged = True
                break

        elapsed = time.time() - start
        logger.info(f"Solve finished: {len(res_history)} iters, {elapsed:.1f}s, converged={converged}")

        return SteadyResult(
            converged=converged,
            iterations=len(res_history),
            final_residual=res_history[-1] if res_history else 0.0,
            cd_history=cd_history,
            cl_history=cl_history,
            residuals_history=res_history,
            solution_final=self.solution.copy(),
        )
