"""AutoFlowCFD core solver module.

This module provides the core computational components for AutoFlowCFD,
including FR discretization, turbulence models, and solver backends.

LIVE vs. LEGACY: the actual production solve path (FRSolver.solve() /
TransientSolver.solve()) is built from BoundaryConditionHandler,
ViscousRANSResidual (fvm_viscous_residual.py), TimeIntegrator,
FVMFaceExtractor (used purely as a data holder, populated via
VolumeMeshData directly rather than its own build_from_tetrahedra), and
AeroCoefficientCalculator. SST k-omega, MUSCL reconstruction/limiting, and
residual assembly are re-implemented inline in fvm_viscous_residual.py /
bc_handler.py rather than through the classes below - confirmed by
`self.backend` (built via create_backend in FRSolver.__init__) never
having any of its methods called anywhere in solver_steady.py or
transient_solver_loop.py. The classes re-exported under "Legacy /
experimental (not on the live solve path)" below are a parallel,
unused-in-production implementation stack that predates the current
inline path; they still have their own standalone unit tests
(tests/unit/test_fr_scheme.py, test_muscl_reconstruction.py,
test_time_and_turbulence.py, tests/integration/test_iteration3_solver.py,
test_convergence_integration.py, test_end_to_end_steady.py) so they are
kept rather than deleted, but a fix made only here will NOT affect an
actual solve - fix the live modules listed above instead.
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

# Reconstruction modules - LEGACY/EXPERIMENTAL, see module docstring above.
from .reconstruction_limiters import LimiterType, SlopeLimiters
from .reconstruction_gradients import GradientComputer
from .reconstruction_muscl import MUSCLReconstructor

# FVM Core modules (modularized). FVMFaceExtractor IS live (see module
# docstring); FVMFluxCalculator/FVMResidualComputer are LEGACY/EXPERIMENTAL.
from .fvm_faces import FVMFaceExtractor
from .fvm_flux import FVMFluxCalculator
from .fvm_residuals import FVMResidualComputer

# Other core modules
from .solver_loop import SteadySolverLoop  # LEGACY/EXPERIMENTAL
from .bc_handler import BoundaryConditionHandler  # live
from .aero_coeffs import AeroCoefficientCalculator  # live
from .solution_constraints import SolutionConstraintHandler  # LEGACY/EXPERIMENTAL


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
