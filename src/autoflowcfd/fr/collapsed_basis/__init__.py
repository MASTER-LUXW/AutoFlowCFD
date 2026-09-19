"""
AutoFlowCFD - 坍缩坐标单纯形（四面体/棱柱）专用模态基与微分矩阵 (V2.0)

背景：本代码库对四面体/棱柱单元统一沿用与六面体相同的张量积
Gauss-Legendre Solution Points（计算立方体 [-1,1]^3 上 (N+1)^3 个点），
通过 curved_mapping.py 的 Duffy 坍缩坐标变换映射到物理单元。这一路线
本身是合法的（Karniadakis & Sherwin《Spectral/hp Element Methods》Ch.2
"坍缩坐标"方法，并非虚构），但前提是**微分算子必须用与坍缩变换匹配的
模态基构造**，而不能像 fr/operators.py 里对六面体那样直接用朴素的
张量积 Lagrange 微分矩阵：真实网格数值验证发现，棱柱在 b=+1 退化边
附近（四面体在 b=+1、c=+1 两个退化边/面附近）所有单元的几何 Jacobian
行列式系统性地比其余区域小 1~2 个数量级（棱柱实测：全网格 107586 个
棱柱在 b≈+0.7746 处 Jacobian 中位数比 b≈-0.7746 处小约 8 倍，个别单元
低至 2e-10），残差公式里除以这个（数值上偏小、并非真正退化的）Jacobian
会把量级正常的通量散度舍入误差放大到 1e10 量级——这不是网格质量缺陷，
是当前"朴素张量积微分矩阵 + 坍缩坐标度量项"组合缺少解析奇异性抵消机制
导致的数值病态：Duffy 变换本身在 b→1（四面体另需 c→1）处度量项含
1/(1-b) 型奇异因子，正确的坍缩坐标谱方法必须用**模态基本身内建
(1-b)^i 这类权重因子**，使得基函数的微分在链式法则里与度量项的奇异
因子解析抵消——这是坍缩坐标谱/DG方法教科书级别的标准要求（Hesthaven &
Warburton《Nodal DG Methods》Ch.6；Karniadakis & Sherwin 同上），不是
可以绕开的细节。

本模块实现该模态基与对应的微分矩阵构造，SPs 位置完全不变（仍是现有的
张量积 Gauss-Legendre 点，经 Duffy 变换映射到物理空间）——只替换"如何
对这些点上的节点值求（参考坐标系）导数"这一步：
    D_ξ = V_ξ @ V^{-1}
其中 V 是模态基在 SPs 处取值的 Vandermonde 矩阵（(N+1)^3 × (N+1)^3
方阵，模态个数与现有 SPs 个数严格一致，因为沿用的正是 Karniadakis-
Sherwin"坍缩张量积"（i,j,k 各自独立取 0..N，共 (N+1)^3 个模态）的基，
而不是四面体真单纯形的总阶数截断），V_ξ 是各模态对参考坐标 ξ∈{a,b,c}
的解析导数在同一组 SPs 处取值。这个 D 矩阵与用哪组基构造在数学上无关
（差值多项式的导数是唯一确定的，与展开基无关），只要 V 可逆；用这个
"内建奇异抵消因子"的基构造出的 D，其自身在 b→1（或 c→1）附近保持良态，
不会重现朴素张量积基那样的病态。

微分矩阵只在这三处消费方（体积散度、梯度、几何 Jacobian 计算，均在
D_3d 的既有 3 个使用点）需要按单元类型替换；FR 的校正函数/Flux Points
外插机制完全是沿单个坍缩轴的一维边界插值（compute_1d_boundary_weights /
extrapolate_to_face），与坍缩坐标的 3D 体积微分奇异性无关，不需要
改动，也不受本次修复影响。

## 文件分工（2026-09-19 拆包）

    jacobi.py            Jacobi 正交多项式及导数（与基无关的基础设施）
    modal.py             坍缩坐标模态基与体积微分算子
    overintegration.py   去混叠三件套与 `OVERINTEGRATION_MAX_ORDER`
    boundary_extrap.py   体积 -> 边界面外插算子

本 `__init__.py` re-export 全部既有公开名，所以全仓库
`from autoflowcfd.fr.collapsed_basis import ...` 一个字都不用改。
"""

from .jacobi import (  # noqa: F401
    grad_jacobi_polynomial,
    jacobi_polynomial,
)
from .modal import (  # noqa: F401
    _collapsed_triangle_mode,  # noqa: F401  测试与诊断在用
    build_collapsed_diff_matrices,
    prism_modal_basis_and_grad,
    tet_modal_basis_and_grad,
)
from .overintegration import (  # noqa: F401
    OVERINTEGRATION_MAX_ORDER,
    build_overintegration_operators,
    resolve_overintegration_order_rule,
)
from .boundary_extrap import build_collapsed_boundary_extrap  # noqa: F401

__all__ = [
    "OVERINTEGRATION_MAX_ORDER",
    "build_collapsed_boundary_extrap",
    "build_collapsed_diff_matrices",
    "build_overintegration_operators",
    "grad_jacobi_polynomial",
    "jacobi_polynomial",
    "prism_modal_basis_and_grad",
    "resolve_overintegration_order_rule",
    "tet_modal_basis_and_grad",
]
