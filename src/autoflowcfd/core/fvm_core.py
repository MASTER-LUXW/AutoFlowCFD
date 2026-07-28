"""FVM core algorithms for finite volume method.

This module provides backward compatibility by re-exporting from submodules.
For new code, import directly from:
    - autoflowcfd.core.fvm_faces
    - autoflowcfd.core.fvm_flux
    - autoflowcfd.core.fvm_residuals
"""

# Re-export from submodules for backward compatibility
from .fvm_faces import FVMFaceExtractor
from .fvm_flux import FVMFluxCalculator
from .fvm_residuals import FVMResidualComputer, _compute_residuals_kernel

__all__ = [
    'FVMFaceExtractor',
    'FVMFluxCalculator',
    'FVMResidualComputer',
    '_compute_residuals_kernel',
]
