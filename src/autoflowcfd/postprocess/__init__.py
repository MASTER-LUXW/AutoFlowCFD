"""Post-processing module.

This module provides tools for calculating aerodynamic coefficients,
exporting visualization data, and analyzing convergence history.

Key Components:
    - CoefficientCalculator: Cd, Cl, Cm calculation
    - VTKExporter: Field data export for ParaView
    - ConvergenceAnalyzer: Residual and coefficient history
    - TransientStatistics: Time-averaged fields, RMS, PSD (v0.2+)

Example:
    >>> from autoflowcfd.postprocess import CoefficientCalculator
    >>> calc = CoefficientCalculator(grid, ref_area=2.2)
    >>> cd = calc.compute_drag_coefficient(solution)
"""

from .coefficients import CoefficientCalculator, AerodynamicCoefficients, AerodynamicForces
from .vtk_export import VTKExporter
from .report import ConvergenceAnalyzer, SimulationReport
from .transient_stats import TransientStatistics, PressurePSD, TransientResult

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
