"""AutoFlowCFD core solver module.

This module provides the core computational components for AutoFlowCFD,
including FR discretization, turbulence models, and solver backends.
"""

from .fr_scheme import FROrder, FRScheme
from .turbulence import SSTKOmegaModel
from .wall_functions import WallFunctionModel
from .time_integration import TimeIntegrator, TimeIntegrationScheme
from .convergence import ConvergenceMonitor, ConvergenceHistory
from .transient_result import TransientResult
from .transient_solver_loop import TransientSolver
from .solver_steady import FRSolver, SteadyResult
from .coupling import SteadyTransientCoupler, SyntheticTurbulenceGenerator
from .backend import create_backend, get_available_backends
from .backend.base import BackendBase
from .backend.cpu_backend import NumbaBackend
from .backend.gpu_backend import CUDABackend

# Reconstruction modules (modularized)
from .reconstruction_limiters import LimiterType, SlopeLimiters
from .reconstruction_gradients import GradientComputer
from .reconstruction_muscl import MUSCLReconstructor

# FVM Core modules (modularized)
from .fvm_faces import FVMFaceExtractor
from .fvm_flux import FVMFluxCalculator
from .fvm_residuals import FVMResidualComputer

# Other core modules
from .solver_loop import SteadySolverLoop
from .bc_handler import BoundaryConditionHandler
from .aero_coeffs import AeroCoefficientCalculator
from .solution_constraints import SolutionConstraintHandler


__all__ = [
    # FR Scheme
    "FRScheme",
    "FROrder",
    
    # Turbulence Models
    "SSTKOmegaModel",
    "WallFunctionModel",
    
    # Time Integration
    "TimeIntegrator",
    "TimeIntegrationScheme",
    
    # Convergence
    "ConvergenceMonitor",
    "ConvergenceHistory",
    
    # Steady Solver
    "FRSolver",
    "SteadyResult",
    
    # Transient Solver
    "TransientSolver",
    "TransientResult",
    
    # Coupling
    "SteadyTransientCoupler",
    "SyntheticTurbulenceGenerator",
    
    # Backends
    "create_backend",
    "get_available_backends",
    "BackendBase",
    "NumbaBackend",
    "CUDABackend",
    
    # Reconstruction and Limiters
    "MUSCLReconstructor",
    "SlopeLimiters",
    "LimiterType",
    "GradientComputer",
    
    # FVM Core modules
    "FVMFaceExtractor",
    "FVMFluxCalculator",
    "FVMResidualComputer",
    "SteadySolverLoop",
    "BoundaryConditionHandler",
    "AeroCoefficientCalculator",
    "SolutionConstraintHandler",
]
