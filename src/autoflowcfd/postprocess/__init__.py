"""后处理模块。

本模块提供计算气动系数、导出可视化数据、分析收敛历史的工具。

Key Components:
    - fr_coefficients: FR 原生面通量点压力积分气动系数（Cd/Cl/Cm 等，唯一生产路径）
    - coefficients: 气动系数/气动力数据类（AerodynamicCoefficients/AerodynamicForces）
    - VTKExporter: 供 ParaView 使用的场数据导出
    - ConvergenceAnalyzer: 残差与系数历史
    - TransientStatistics: 时间平均场、RMS、PSD

Example:
    >>> from autoflowcfd.postprocess.fr_coefficients import (
    ...     compute_aerodynamic_coefficients_fr)
    >>> coeffs = compute_aerodynamic_coefficients_fr(solver, reference_area=2.2)
"""

from .coefficients import AerodynamicCoefficients, AerodynamicForces
from .vtk_export import VTKExporter
from .report import ConvergenceAnalyzer, SimulationReport
from .transient_stats import TransientStatistics, TransientResult
from .pressure_psd import PressurePSD

__all__ = [
    "AerodynamicCoefficients",
    "AerodynamicForces",
    "VTKExporter",
    "ConvergenceAnalyzer",
    "SimulationReport",
    "TransientStatistics",
    "PressurePSD",
    "TransientResult",
]
