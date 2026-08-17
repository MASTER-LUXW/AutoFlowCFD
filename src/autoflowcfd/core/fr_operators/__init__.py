"""
AutoFlowCFD Core - FR Operators Module

FR 算子与内核模块，包含面通量、逐点通量、梯度计算、体积项收缩等。
"""

from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.fr_operators.kernels import compute_ausm_up_flux
from autoflowcfd.core.fr_operators.gradients import compute_physical_gradient
from autoflowcfd.core.fr_operators.troubled_cell import suppress_residual_outliers

__all__ = [
    'get_flat_face_geometry',
    'compute_ausm_up_flux',
    'compute_physical_gradient',
    'suppress_residual_outliers',
]
