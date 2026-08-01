"""FVM core algorithms for finite volume method.

This module provides backward compatibility by re-exporting from submodules.
For new code, import directly from:
    - autoflowcfd.core.fvm_faces
    - autoflowcfd.core.fvm_flux
    - autoflowcfd.core.fvm_residuals
"""

# Re-export from submodules for backward compatibility. FVMFluxCalculator/
# FVMResidualComputer live under core/legacy/ (not on the live solve path -
# see core/legacy/__init__.py); FVMFaceExtractor is live.
from .fvm_faces import FVMFaceExtractor
from .legacy.fvm_flux import FVMFluxCalculator
from .legacy.fvm_residuals import FVMResidualComputer, _compute_residuals_kernel

__all__ = [
    'FVMFaceExtractor',
    'FVMFluxCalculator',
    'FVMResidualComputer',
    '_compute_residuals_kernel',
]
