"""AutoFlowCFD - High-performance CFD software for automotive aerodynamics.

AutoFlowCFD is an open-source Computational Fluid Dynamics (CFD) software
specialized for automotive external aerodynamics simulation. It provides
high-accuracy and high-speed CFD analysis with AI Agent integration capabilities.

Key Features:
    - Native NAS grid support (ANSA v22/v23/v24)
    - Hybrid CPU/GPU computing (Numba/CUDA)
    - High-order Flux Reconstruction solver
    - Advanced turbulence models (SST k-ω, DES/DDES, LES)
    - Dual interface (CLI + Python API)
    - Modular and extensible architecture

Example:
    >>> from autoflowcfd import AutoFlowCFDAPI
    >>> api = AutoFlowCFDAPI()
    >>> grid = api.load_grid("car_model.nas")
    >>> result = api.run_steady(grid, backend="gpu", order=3)
    >>> coeffs = api.calculate_coefficients(result)
    >>> print(f"Drag Coefficient: {coeffs['Cd']:.4f}")
"""

# ============================================================================
# CRITICAL: Set BLAS/linear algebra threading BEFORE importing NumPy
# This ensures maximum multi-core utilization for vectorized operations
# ============================================================================
import os
import multiprocessing

_cpu_count = multiprocessing.cpu_count()
os.environ.setdefault('MKL_NUM_THREADS', str(_cpu_count))
os.environ.setdefault('OPENBLAS_NUM_THREADS', str(_cpu_count))
os.environ.setdefault('NUMEXPR_NUM_THREADS', str(_cpu_count))
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', str(_cpu_count))
os.environ.setdefault('OMP_NUM_THREADS', str(_cpu_count))

__version__ = "0.1.0"
__author__ = "AutoFlowCFD Team"
__email__ = "contact@autoflowcfd.org"
__license__ = "Apache-2.0"

from typing import Any, Dict

# Import main API class
from .api import AutoFlowCFDAPI

# Module metadata
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "AutoFlowCFDAPI",
]


def get_version() -> str:
    """Get the current version of AutoFlowCFD.
    
    Returns:
        str: Version string in semver format (e.g., "0.1.0")
        
    Example:
        >>> import autoflowcfd
        >>> autoflowcfd.get_version()
        '0.1.0'
    """
    return __version__


def create_api(verbose: bool = False) -> AutoFlowCFDAPI:
    """Create AutoFlowCFD API instance.
    
    Convenience function to create API instance.
    
    Args:
        verbose: Enable verbose logging
        
    Returns:
        AutoFlowCFDAPI: API instance
        
    Example:
        >>> api = autoflowcfd.create_api()
        >>> grid = api.load_grid("model.nas")
    """
    return AutoFlowCFDAPI(verbose=verbose)
