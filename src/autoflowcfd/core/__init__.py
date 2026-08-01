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
transient_solver_loop.py. The classes re-exported from the `legacy`
subpackage below are a parallel, unused-in-production implementation
stack that predates the current inline path; they still have their own
standalone unit tests (tests/unit/test_fr_scheme.py,
test_muscl_reconstruction.py, test_time_and_turbulence.py,
tests/integration/test_iteration3_solver.py, test_convergence_integration.py,
test_end_to_end_steady.py) so they are kept (see core/legacy/__init__.py)
rather than deleted, but a fix made only there will NOT affect an actual
solve - fix the live modules listed above instead.
"""

from .time_integration import TimeIntegrator, TimeIntegrationScheme
from .transient_result import TransientResult
from .transient_solver_loop import TransientSolver
from .solver_steady import FRSolver, SteadyResult
from .coupling import SteadyTransientCoupler, SyntheticTurbulenceGenerator
from .backend import create_backend, get_available_backends
from .backend.base import BackendBase

# Everything below this point lives under core/legacy/ - NOT on the live
# solve path, see that subpackage's own __init__.py docstring for why it's
# kept rather than deleted. Still re-exported here for backward
# compatibility with existing `from autoflowcfd.core import X` style imports.
from .legacy.fr_scheme import FROrder, FRScheme
from .legacy.turbulence import SSTKOmegaModel
from .legacy.wall_functions import WallFunctionModel
from .legacy.convergence import ConvergenceMonitor, ConvergenceHistory
from .backend.cpu_backend import NumbaBackend
from .backend.gpu_backend import CUDABackend
from .legacy.reconstruction_limiters import LimiterType, SlopeLimiters
from .legacy.reconstruction_gradients import GradientComputer
from .legacy.reconstruction_muscl import MUSCLReconstructor
from .legacy.fvm_flux import FVMFluxCalculator
from .legacy.fvm_residuals import FVMResidualComputer
from .legacy.solver_loop import SteadySolverLoop
from .legacy.solution_constraints import SolutionConstraintHandler

# FVM Core modules (modularized). FVMFaceExtractor IS live (see module
# docstring).
from .fvm_faces import FVMFaceExtractor

# Other core modules
from .bc_handler import BoundaryConditionHandler  # live
from .aero_coeffs import AeroCoefficientCalculator  # live


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
