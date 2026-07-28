"""Backward compatibility module.

This module provides backward compatibility aliases for the postprocess module.
Use autoflowcfd.postprocess for new code.

Example:
    >>> from autoflowcfd.post import CoefficientCalculator  # Old style (deprecated)
    >>> from autoflowcfd.postprocess import CoefficientCalculator  # New style (recommended)
"""

# Import all public classes from postprocess module
from .postprocess import (
    CoefficientCalculator,
    AerodynamicCoefficients,
    AerodynamicForces,
    VTKExporter,
    ConvergenceAnalyzer,
    SimulationReport,
    TransientStatistics,
    PressurePSD,
    TransientResult,
)

__all__ = [
    "CoefficientCalculator",
    "AerodynamicCoefficients",
    "AerodynamicForces",
    "VTKExporter",
    "ConvergenceAnalyzer",
    "SimulationReport",
    "TransientStatistics",
    "PressurePSD",
    "TransientResult",
]

# Deprecation warning
import warnings
warnings.warn(
    "autoflowcfd.post is deprecated. Use autoflowcfd.postprocess instead.",
    DeprecationWarning,
    stacklevel=2
)
