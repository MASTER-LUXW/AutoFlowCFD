"""
AutoFlowCFD V2.0 - FR (Flux Reconstruction) 离散的算子与几何

子模块 / 子包：
    - operators/: FR 算子容器 `FROperators` 与构造 `generate_fr_operators`
    - native_tet/: 原生四面体基（受限 PKD/Dubiner + Warp&Blend 节点）与过积分算子
    - native_prism/: 原生三棱柱基、面算子与过积分算子
    - face_flux_points/: 面通量点几何（定位、跨单元插值、精确法向）
    - native_padding.py: 原生基算子零填充到统一解点宽度、只统计真实自由度的归约
    - overintegration_order.py: 过积分阶数（2*order）
    - face_flux_trace.py: 面通量迹并进体积算子（离散守恒）
    - order_interp.py: 阶数延拓的解点插值
    - triangle_apex.py: 三角形面的坍缩顶点槽位
    - jacobi.py / quadrature_points.py / warp_blend_nodes.py: 正交多项式、求积点、节点集
    - diff_matrix_consistency.py: 微分矩阵常数零化校正
"""

from .operators import (
    FROperators,
    generate_fr_operators,
)

from .quadrature_points import (
    gauss_legendre,
    gauss_lobatto,
)

__all__ = [
    "FROperators",
    "generate_fr_operators",
    "gauss_legendre",
    "gauss_lobatto",
]
