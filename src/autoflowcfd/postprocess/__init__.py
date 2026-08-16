"""后处理模块。

本模块提供计算气动系数、导出可视化数据、分析收敛历史的工具。

Key Components:
    - CoefficientCalculator: Cd、Cl、Cm 计算
    - VTKExporter: 供 ParaView 使用的场数据导出
    - ConvergenceAnalyzer: 残差与系数历史
    - TransientStatistics: 时间平均场、RMS、PSD

Example:
    >>> from autoflowcfd.postprocess import CoefficientCalculator
    >>> calc = CoefficientCalculator(grid, ref_area=2.2)
    >>> cd = calc.compute_drag_coefficient(solution)
"""

from .coefficients import CoefficientCalculator, AerodynamicCoefficients, AerodynamicForces
from .vtk_export import VTKExporter
from .report import ConvergenceAnalyzer, SimulationReport
from .transient_stats import TransientStatistics, TransientResult
from .pressure_psd import PressurePSD

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
