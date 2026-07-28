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
        
        # Initialize time integrator
        initial_dt = 1.0 / 30.0 * 0.1
        self.time_integrator = TimeIntegrator(
            scheme=TimeIntegrationScheme.BACKWARD_EULER,
            dt=initial_dt,
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
        
        # Allocate and initialize solution array
        self.solution = np.zeros((n_cells, 7), dtype=np.float64)
        self.solution[:, 0] = rho_0   # density
        self.solution[:, 1] = rhou_0  # x-momentum
        self.solution[:, 2] = rhov_0  # y-momentum
        self.solution[:, 3] = rhow_0  # z-momentum
        self.solution[:, 4] = E_0     # total energy
        self.solution[:, 5] = 0.001   # turbulent kinetic energy
        self.solution[:, 6] = 1.0     # specific dissipation rate
        
        logger.info(f"Solution initialized: {n_cells} cells")
        logger.info(f"  Initial conditions: rho={rho_0:.3f} kg/m³, u={u_0:.1f} m/s, p={p_0:.0f} Pa")

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
        
        Args:
            max_iter: Maximum iterations (overrides config if provided)
            
        Returns:
            SteadyResult with solution and history
        """
        # Initialize
        if self.solution is None:
            self._initialize_solution()
        
        self._setup_boundary_conditions()
        
        # Build face connectivity
        nodes = np.column_stack([
            self.grid_data.nodes.x,
            self.grid_data.nodes.y,
            self.grid_data.nodes.z
        ])
        
        face_data = self.face_extractor.build_from_tetrahedra(
            self.grid_data.cells.connectivity, nodes
        )
        
        # Update flux calculator
        self.flux_calculator.face_connectivity = face_data['connectivity']
        self.flux_calculator.face_normals = face_data['normals']
        self.flux_calculator.face_areas = face_data['areas']
        self.flux_calculator.boundary_flags = face_data['boundary_flags']
        
        # Run solver loop
        result_dict = self.solver_loop.run(
            solution=self.solution,
            grid_data=self.grid_data,
            get_cell_volumes_func=self._get_cell_volumes,
            apply_bc_func=lambda c, f: self.bc_handler.apply_boundary_condition(self.solution, c, f),
            compute_coeffs_func=lambda i: self.aero_calculator.compute_coefficients(self.solution, i),
            identify_body_faces_func=self.aero_calculator._identify_body_faces,
            compute_ref_area_func=self.aero_calculator._compute_reference_area,
            apply_constraints_func=lambda: self.constraint_handler.apply_constraints(self.solution),
            bc_handler=self.bc_handler,  # Pass bc_handler for ramp mechanism
            max_iter=max_iter,
        )
        
        # Create result object
        return SteadyResult(
            converged=result_dict['converged'],
            iterations=result_dict['iterations'],
            final_residual=result_dict['final_residual'],
            cd_history=result_dict['cd_history'],
            cl_history=result_dict['cl_history'],
            residuals_history=result_dict['residuals_history'],
            solution_final=result_dict['solution'],
        )
