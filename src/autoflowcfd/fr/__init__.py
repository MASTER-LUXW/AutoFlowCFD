"""
AutoFlowCFD V2.0 - FR (Flux Reconstruction) 模块

本模块实现通量重构方法的核心组件，包括：
- 点集生成（Gauss-Legendre, Gauss-Lobatto）
- 算子生成（微分矩阵、插值矩阵、校正权重）
- 无粘/粘性通量计算
- 时间积分器

子模块：
    - quadrature_points: 求积点集生成
    - matrix_operators: 矩阵算子生成
    - operators: FR算子统一接口
    - inviscid_flux: 无粘通量计算
    - viscous_flux: 粘性通量计算
    - correction_kernel: 校正项内核
"""

from .operators import (
    FROperators,
    generate_fr_operators,
)

from .quadrature_points import (
    gauss_legendre,
    gauss_lobatto,
    generate_tensor_product_points,
)

from .matrix_operators import (
    compute_diff_matrix_1d,
    compute_diff_matrix_3d,
    compute_interpolation_matrix,
    compute_vandermonde,
    compute_correction_weights,
)

__all__ = [
    "FROperators",
    "generate_fr_operators",
    "gauss_legendre",
    "gauss_lobatto",
    "generate_tensor_product_points",
    "compute_diff_matrix_1d",
    "compute_diff_matrix_3d",
    "compute_interpolation_matrix",
    "compute_vandermonde",
    "compute_correction_weights",
]
