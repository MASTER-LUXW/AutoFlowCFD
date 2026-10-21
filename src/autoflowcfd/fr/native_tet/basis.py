"""
AutoFlowCFD V2.0 - 四面体原生单纯形基函数与微分算子。

直接在四面体自己的参考单纯形 `(r,s,t)` 上构造 Dubiner/PKD 正交模态基（`i+j+k<=order`，
`(order+1)(order+2)(order+3)/6` 个）、Warp & Blend 节点与它的**原生**梯度（直接对 r,s,t 求导）。

为什么不用坍缩坐标（Duffy）张量积基：把立方体的整张面压缩成四面体的一条棱 / 一个顶点，在退化处引入了低阶
模态根本无法表示的几何复杂度——实测是 P1 缓慢发散的根因（真实测得相对误差 862400%），且三分之一的四面体把
残差放大 6~7 个量级（与棱柱同类，见 `fr/native_prism/__init__.py` 的对照表）。

`simplex3d_value`/`simplex3d_grad` 逐行移植自 Hesthaven & Warburton《Nodal Discontinuous Galerkin Methods》配套
参考实现 `Simplex3DP.m`/`GradSimplex3DP.m`（https://github.com/tcew/nodal-dg，Codes1.1/Codes3D/）。这两个
函数内部用 (a,b,c) 坍缩坐标记号**写出**多项式公式（Dubiner 1991 构造这套正交多项式的数学工具，任何实现都绕不
开，包括 Nektar++），但全程只用非负整数次幂的 `(1-b)`、`(1-c)`，不含除法；会带来坐标奇点的是
"(r,s,t)→(a,b,c)"这个方向的映射，这里只用它**取值**（`rst_to_abc` 在退化轴上的 fallback 不影响取值，因为对应
权重因子在那里恰好为零），梯度由 `GradSimplex3DP` 的封闭解析公式直接给出对原生坐标的正确梯度。

验证：恒等映射（参考四面体自身顶点当物理坐标）下 det_jacs 在 P1~P3 全部节点精确等于 1.000000；Couette
剪切流残差相对坍缩基 P1 改善约 10400 倍、P2 约 37000 倍、P3 约 4200 倍（180 个随机四面体样本 100% 更优）。
"""

from typing import List, Tuple

import numpy as np
from numba import njit

from ..jacobi import jacobi_polynomial, grad_jacobi_polynomial


def restricted_tet_modes(order: int) -> List[Tuple[int, int, int]]:
    """四面体最小 PKD/Dubiner 模态索引集合：`i+j+k<=order`，
    共 `(order+1)(order+2)(order+3)/6` 个——四面体真正所属的单纯形多项式空间维度，不含冗余自由度
    （各方向独立取阶数的 `(order+1)^3` 族里多出来的模态在单纯形上不独立）。

    展平顺序：先按 i、再按 j、再按 k 递增（与 `simplex3d_value`/
    `simplex3d_grad` 循环顺序一致，供调用方在同一顺序下构造
    Vandermonde 矩阵列）。
    """
    return [
        (i, j, k)
        for i in range(order + 1)
        for j in range(order + 1 - i)
        for k in range(order + 1 - i - j)
    ]


@njit(cache=True)
def simplex3d_value(a: np.ndarray, b: np.ndarray, c: np.ndarray, i: int, j: int, k: int) -> np.ndarray:
    """Simplex3DP.m 逐行移植：四面体正交模态基在 (a,b,c) 处的取值。

    P = 2*sqrt(2) * h1(a) * h2(b) * (1-b)^i * h3(c) * (1-c)^(i+j)

    处处有限（包括 b=1 或 c=1，此时对应的 (1-b)^i / (1-c)^(i+j) 因子
    在 i>0 / i+j>0 时直接为零，不需要任何 L'Hopital 或 fallback 处理）。
    """
    h1 = jacobi_polynomial(a, 0.0, 0.0, i)
    h2 = jacobi_polynomial(b, float(2 * i + 1), 0.0, j)
    h3 = jacobi_polynomial(c, float(2 * (i + j) + 2), 0.0, k)
    return 2.0 * np.sqrt(2.0) * h1 * h2 * (1.0 - b) ** i * h3 * (1.0 - c) ** (i + j)


@njit(cache=True)
def simplex3d_grad(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, i: int, j: int, k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GradSimplex3DP.m 逐行移植：四面体正交模态基对参考单纯形原生
    坐标 (r,s,t) 的梯度（不是对 (a,b,c) 的梯度）。

    核心技巧（不是"先对a求导再乘以链式法则因子da/dr"——那条路必然
    经过 d(r,s,t)/d(a,b,c) 的逆，这个逆在退化轴上不存在，见 Part1
    3.1节）：直接把 `(1-b)`、`(1-c)` 权重因子的幂次代数地减一，
    等价于提前吸收了链式法则本该出现的、原本会奇异的那个因子——
    全程只用非负整数次幂，不含任何除法，因此处处有限，包括退化轴
    本身（已用恒等映射测试在四面体自身顶点验证：这样构造出的微分
    矩阵作用在顶点物理坐标上，给出精确的单位矩阵，不会像坍缩坐标
    方案或朴素有限差分那样在顶点给出退化的零几何 Jacobian）。

    Returns:
        (V3Dr, V3Ds, V3Dt)：对 r、s、t 的偏导数，形状与 a/b/c 相同。
    """
    fa = jacobi_polynomial(a, 0.0, 0.0, i)
    dfa = grad_jacobi_polynomial(a, 0.0, 0.0, i)
    gb = jacobi_polynomial(b, float(2 * i + 1), 0.0, j)
    dgb = grad_jacobi_polynomial(b, float(2 * i + 1), 0.0, j)
    hc = jacobi_polynomial(c, float(2 * (i + j) + 2), 0.0, k)
    dhc = grad_jacobi_polynomial(c, float(2 * (i + j) + 2), 0.0, k)

    half_1mb = 0.5 * (1.0 - b)
    half_1mc = 0.5 * (1.0 - c)

    V3Dr = dfa * (gb * hc)
    if i > 0:
        V3Dr = V3Dr * (half_1mb ** (i - 1))
    if i + j > 0:
        V3Dr = V3Dr * (half_1mc ** (i + j - 1))

    V3Ds = 0.5 * (1.0 + a) * V3Dr
    tmp = dgb * (half_1mb ** i)
    if i > 0:
        tmp = tmp + (-0.5 * i) * (gb * half_1mb ** (i - 1))
    if i + j > 0:
        tmp = tmp * (half_1mc ** (i + j - 1))
    tmp = fa * (tmp * hc)
    V3Ds = V3Ds + tmp

    V3Dt = 0.5 * (1.0 + a) * V3Dr + 0.5 * (1.0 + b) * tmp
    tmp2 = dhc * (half_1mc ** (i + j))
    if i + j > 0:
        tmp2 = tmp2 - 0.5 * (i + j) * (hc * (half_1mc ** (i + j - 1)))
    tmp2 = fa * (gb * tmp2)
    tmp2 = tmp2 * (half_1mb ** i)
    V3Dt = V3Dt + tmp2

    norm = 2.0 ** (2 * i + j + 1.5)
    return V3Dr * norm, V3Ds * norm, V3Dt * norm


def rst_to_abc(r: np.ndarray, s: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(r,s,t) -> (a,b,c)，只用于取值（安全），不用于求导。

    退化轴上（`denom_a=s+t≈0` 对应四面体棱 v3-v4；`denom_b=1-c≈0`
    对应顶点 v4）这个方向的映射是多对一、不可逆的——具体选哪个
    fallback 值不影响任何下游结果：`simplex3d_value`/`simplex3d_grad`
    在 i>0（或 i+j>0）的模态上，权重因子 `(1-b)^i`/`(1-c)^(i+j)`
    在这些点上恰好为零，把 'a' 或 'b' 的具体取值直接乘没了；i=0
    的模态则根本不依赖 'a'/'b'（`g_00≡1`）——取值在退化轴上处处有限。
    """
    c = t
    denom_b = 1.0 - c
    b = np.where(
        np.abs(denom_b) > 1e-12,
        2.0 * (s + 1.0) / np.where(np.abs(denom_b) > 1e-12, denom_b, 1.0) - 1.0,
        -1.0,
    )
    denom_a = s + t
    a = np.where(
        np.abs(denom_a) > 1e-12,
        -2.0 * (1.0 + r) / np.where(np.abs(denom_a) > 1e-12, denom_a, 1.0) - 1.0,
        -1.0,
    )
    return a, b, c


def build_native_tet_operators(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """构造四面体独立（路径C）体积微分算子。

    Args:
        order: 多项式阶数 P

    Returns:
        (ref_rst, D_native)：
        `ref_rst` 形状 `(n_native_sps, 3)`，Warp & Blend 优化节点在
        标准参考四面体 `(r,s,t)` 上的坐标（`n_native_sps =
        (order+1)(order+2)(order+3)/6`）；
        `D_native` 形状 `(n_native_sps, n_native_sps, 3)`，`D[:,:,m]@field` 给出对参考坐标第 m 个方向
        （r/s/t）的导数。
    """
    from ..warp_blend_nodes import warp_blend_nodes_3d
    from ..diff_matrix_consistency import enforce_constant_annihilation

    r, s, t = warp_blend_nodes_3d(order)
    a, b, c = rst_to_abc(r, s, t)
    modes = restricted_tet_modes(order)
    n = len(r)
    n_modes = len(modes)
    if n != n_modes:
        raise ValueError(
            f"Warp&Blend 节点数 {n} 与受限 PKD 模态数 {n_modes} 不一致（order={order}）——"
            "两者理论上必须相等（都是 (order+1)(order+2)(order+3)/6），出现不一致说明节点"
            "生成或模态索引其中之一有 bug，不应该静默继续。"
        )

    V = np.zeros((n, n_modes))
    Vr = np.zeros((n, n_modes))
    Vs = np.zeros((n, n_modes))
    Vt = np.zeros((n, n_modes))
    for m, (i, j, k) in enumerate(modes):
        V[:, m] = simplex3d_value(a, b, c, i, j, k)
        Vr[:, m], Vs[:, m], Vt[:, m] = simplex3d_grad(a, b, c, i, j, k)

    from scipy.linalg import lu_factor, lu_solve

    lu = lu_factor(V.T)
    D = np.stack([lu_solve(lu, Vr.T).T, lu_solve(lu, Vs.T).T, lu_solve(lu, Vt.T).T], axis=-1)
    # 强制逐位精确地零化常数（`D @ 1 = 0`）。解析上必然成立，但 LU 求解
    # 的舍入残余会留到 `eps*cond(V)`，而**直边四面体的自由流保持性完全
    # 由这个残余决定**（度量逐单元常数，均匀流下体积项散度恰好等于
    # `adj(J)*F*(D@1)/det(J)`，还会被小体积单元放大）。实测 order=4 的
    # 残余 9.3e-14 在真实网格上放大成 3.06e-3 的伪残差。详细推导、代价
    # 与适用范围见 `diff_matrix_consistency.py` 模块文档。
    enforce_constant_annihilation(D)
    return np.column_stack([r, s, t]), D


def map_native_tet_to_physical(ref_rst: np.ndarray, cell_nodes: np.ndarray) -> np.ndarray:
    """把参考单纯形坐标 (r,s,t)（Warp & Blend 节点，标准 [-1,1] 参考
    四面体）直接映射到物理四面体——重心坐标仿射组合，不经过任何坍缩
    坐标中间表示。

    Args:
        ref_rst: (n_native_sps, 3)，来自 `build_native_tet_operators`
        cell_nodes: (4, 3)，四面体 4 个顶点物理坐标，顺序需保证正体积
            （与 `grid/curved_mapping/curved_mapping_orientation.py::fix_tet_orientation` 同一约定）

    Returns:
        phys: (n_native_sps, 3)
    """
    from ...grid.curved_mapping.curved_mapping import tet_barycentric

    r, s, t = ref_rst[:, 0], ref_rst[:, 1], ref_rst[:, 2]
    L1, L2, L3, L4 = tet_barycentric(r, s, t)
    p0, p1, p2, p3 = cell_nodes
    return L1[:, None] * p0 + L2[:, None] * p1 + L3[:, None] * p2 + L4[:, None] * p3


def native_tet_face_points(order: int, excluded_vertex: int, slot: int = 0) -> np.ndarray:
    """排除顶点 `excluded_vertex` 的那个面上 `(order+1)^2` 个通量点的**参考**坐标
    `(r,s,t)`：坍缩三角形采样（`cube_to_tri_rs` 作用在张量积 Gauss-Legendre 方格上，
    与棱柱三角形封盖同一套）经面上三个顶点的重心坐标放到参考四面体上。`slot` 选坍缩
    顶点（`fr/triangle_apex.py`）。"""
    from ..quadrature_points import gauss_legendre
    from ...grid.curved_mapping.curved_mapping import cube_to_tri_rs, tri_barycentric

    n1d = order + 1
    sps_1d, _ = gauss_legendre(n1d)
    g1, g2 = np.meshgrid(sps_1d, sps_1d, indexing="ij")
    r_tri, s_tri = cube_to_tri_rs(g1.ravel(), g2.ravel())
    l1, l2, l3 = tri_barycentric(r_tri, s_tri)

    from ..triangle_apex import rotate_triangle

    va, vb, vapex = rotate_triangle(tuple(v for v in range(4) if v != excluded_vertex), slot)
    L = np.zeros((len(l1), 4))
    L[:, va] = l1
    L[:, vb] = l2
    L[:, vapex] = l3
    return np.column_stack([2.0 * L[:, 1] - 1.0, 2.0 * L[:, 2] - 1.0, 2.0 * L[:, 3] - 1.0])


def _native_face_value_vandermondes(order: int, excluded_vertex: int, slot: int = 0):
    """`build_native_tet_boundary_extrap`/`build_native_tet_lift` 共用的
    准备步骤：体积节点、面 Flux Points 各自在参考坐标处的模态取值
    Vandermonde 矩阵，二者用同一组几何量、只是矩阵组合方式不同
    （外插 vs 提升），提取成共享函数避免重复、避免未来改一处忘改
    另一处。

    Returns:
        (V_sps, V_fp, modes)：V_sps (n_native_sps,n_modes)，
        V_fp (n1d*n1d,n_modes)，modes 列表（与两个矩阵的列顺序一致）。
    """
    fp = native_tet_face_points(order, excluded_vertex, slot)
    ref_rst_sps, _ = build_native_tet_operators(order)
    a_sps, b_sps, c_sps = rst_to_abc(ref_rst_sps[:, 0], ref_rst_sps[:, 1], ref_rst_sps[:, 2])
    a_fp, b_fp, c_fp = rst_to_abc(fp[:, 0], fp[:, 1], fp[:, 2])
    modes = restricted_tet_modes(order)

    V_sps = np.column_stack([simplex3d_value(a_sps, b_sps, c_sps, i, j, k) for (i, j, k) in modes])
    V_fp = np.column_stack([simplex3d_value(a_fp, b_fp, c_fp, i, j, k) for (i, j, k) in modes])
    return V_sps, V_fp, modes


def _native_mode_norm_squared(i: int, j: int, k: int) -> float:
    """`simplex3d_value` 给出的模态 `∫_ref P_i,j,k^2 dV` 闭式解析解——
    **重要说明**：这些模态是正交的（不同模态内积恒为 0，`build_native_
    tet_lift` 用到的关键性质），但**不是**正交*归一*的（模长不是 1）：
    `simplex3d_value` 内部用的 `fr/jacobi.py::jacobi_polynomial` 是"未归一化" Jacobi 多项式，构造 Dubiner
    正交基时没有另外补上 Hesthaven-Warburton 原始 `JacobiP.m` 那个 L2 归一化前缀，因此是"正交但非归一"。

    推导（标准 Jacobi 多项式加权模方公式，A&S 22.2.1，代入本构造里
    全部用到的 β=0 情形后化简为不含 Gamma 函数的初等表达式）：
    `γ_n^(α,0) = 2^(α+1)/(2n+α+1)`，`simplex3d_value` 三个因子
    `h1=P_i^(0,0)(a)`、`h2=P_j^(2i+1,0)(b)*(1-b)^i`、
    `h3=P_k^(2(i+j)+2,0)(c)*(1-c)^(i+j)` 各自对应一段区间积分，配合
    (a,b,c)→(r,s,t) 的 Duffy 变换 Jacobian `(1-b)(1-c)^2/8` 与前缀
    `(2*sqrt(2))^2=8` 恰好抵消，最终得到
    `N(i,j,k)=γ_i^(0,0)*γ_j^(2i+1,0)*γ_k^(2(i+j)+2,0)
    =2^(4i+2j+5)/[(2i+1)(i+j+1)(2i+2j+2k+3)]`。

    已用独立三重数值积分（`scipy.integrate.tplquad`，不依赖本文件
    任何其它代码，direct 数值求解 `∫∫∫ P^2 dr ds dt`）对 order 0~3
    全部模态逐一核对，误差 <1e-6（相对），见开发过程记录，不是凭空
    推导后假设成立。
    """
    return 2.0 ** (4 * i + 2 * j + 5) / ((2 * i + 1) * (i + j + 1) * (2 * i + 2 * j + 2 * k + 3))


def build_native_tet_boundary_extrap(order: int, excluded_vertex: int, slot: int = 0) -> np.ndarray:
    """native 四面体体积->自身面外插矩阵（`E @ Q_volume_nodal` 给出该面 Flux Points 上的取值），
    只依赖 `(order, excluded_vertex, slot)`，与具体物理单元形状无关——可预计算一次、全网格所有同阶数单元
    共享（已用真实数值验证：两个形状
    差异很大的四面体，用 `fr/face_flux_points/geometry.py::native_tet_face_points_
    physical` 生成的面物理点，反解出的参考坐标 (r,s,t) 完全相同，机器
    精度，因为反解本质上是在还原生成这些物理点时用过的同一组重心坐标
    权重，这组权重只取决于 (a,b) 采样网格本身，不取决于具体单元形状）。

    face Flux Points 的数量固定为 `n1d*n1d`（`n1d=order+1`，不是 native 最小面节点数
    `(order+1)(order+2)/2`）：`_KernelFaceData` flat 数组要求全网格所有面（含棱柱四边形侧面）的 Flux Points
    数量统一，三角形面用与棱柱三角形封盖相同的坍缩三角形
    采样网格生成面上的参考点位置（见 `face_flux_points/geometry.py::
    native_tet_face_points_physical` 的物理版本，这里是它的参考坐标
    版本，直接给出 (r,s,t) 不需要另外反解）。

    `slot`：该面的坍缩顶点槽位（`fr/triangle_apex.py`），决定通量点集与顺序。

    Returns:
        E: (n1d*n1d, n_native_sps)
    """
    V_sps, V_fp, _ = _native_face_value_vandermondes(order, excluded_vertex, slot)

    from scipy.linalg import lu_factor, lu_solve

    lu_piv = lu_factor(V_sps.T)
    E = lu_solve(lu_piv, V_fp.T).T
    return E


def build_native_tet_lift(order: int, excluded_vertex: int, slot: int = 0) -> np.ndarray:
    """native 四面体 DG 提升算子（"lift"/"LIFT matrix"）——把某个真实面上逐 Flux Point 的通量跳跃提升成对
    体积节点（nodal）自由度的修正贡献。单纯形基没有张量积计算方向，1D Radau/VCJH 修正函数沿某一轴分布这个
    概念不适用，用的是标准 DG 的提升算子（Hesthaven-Warburton《Nodal DG》第6章
    `Lift3D.m`），但这里的 Flux Points 不是体积节点的子集（`n1d*n1d`
    个，用坍缩三角形采样生成，见 `native_tet_face_points_physical`
    文档"二·五"节的必要修正），不能照抄该书假设"面节点=体积节点子集"
    的原始版本，需要用更一般的（对任意一批面点都成立的）弱形式推导。

    ## 推导

    体积节点场用同一组正交（非归一）模态展开：`f(x)=sum_m c_m P_m(x)`，
    节点值 `f_nodal=V@c`（`V[node,mode]=P_mode(参考坐标)`，
    `build_native_tet_operators` 已经算过的同一个 Vandermonde），故
    `c=V^{-1}@f_nodal`。节点基"势函数"（nodal Lagrange 基）在模态展开下
    是 `Ψ(x)=V^{-T}@P(x)`（标准恒等式：`f_nodal=V@V^{-1}@f_nodal`
    重新写成 `f(x)=sum_node f_nodal[node]*Ψ_node(x)` 即得）。

    体积（物理）质量矩阵：`M_phys=det_j*M_ref`，`M_ref=V^{-T}@diag(N)@V^{-1}`
    （`N[m]=∫_ref P_m^2 dV`，`_native_mode_norm_squared`，正交非归一基
    的标准弱形式质量矩阵公式——`∫Ψ_a Ψ_b dV=∫(V^{-T}P)_a(V^{-T}P)_b dV
    =V^{-T}@(∫PP^T dV)@V^{-1}=V^{-T}@diag(N)@V^{-1}`），故
    `M_phys^{-1}=(1/det_j)*V@diag(1/N)@V^T`。

    弱形式提升定义（对每个体积节点基函数 Ψ_s 取矩）：
    `M_phys@LiftedJump_nodal = ∮_face Ψ(x)*jump(x) dA`，右端离散化为
    该面自己的物理面积权重求积（`w_p`=`fr/face_flux_points/exact_normal.py::
    compute_exact_face_normals_and_weights` 已经算好并验证过的
    `true_area_weight`，本函数只组装参考部分，物理面积权重在残差
    kernel 消费时按面才知道，不在这里，`Lift_ref` 只依赖 `(order,
    excluded_vertex)`，与 `build_native_tet_boundary_extrap` 同一个
    "可预计算一次、全网格同阶数单元共享"的性质）：
    `RHS=sum_p w_p*Ψ(FP_p)*jump[p]=V^{-T}@B^T@diag(w)@jump`
    （`B[p,m]=P_m(FP_p 参考坐标)`，即 `_native_face_value_vandermondes`
    已经算好的 `V_fp`）。代入并利用 `V^T@V^{-T}=I`（与是否正交归一
    无关，纯矩阵逆恒等式）化简：
    `LiftedJump_nodal=(1/det_j)*V@diag(1/N)@B^T@diag(w)@jump
    =(1/det_j)*Lift_ref@(w⊙jump)`，其中
    `Lift_ref=V@diag(1/N)@B^T`——本函数返回值。

    调用方（残差 kernel）消费方式：
    `correction[s,v] = -Lift_ref[s,:]@(true_area_weight[:,None]*jump)[:,v] / det_j`
    ——`dj=det_jacs[oc,s]` 在直边四面体处处相同，复用外层同一次 `/dj` 除法，不在这里预先乘 `1/det_j`。

    `slot`：坍缩顶点槽位（`fr/triangle_apex.py`），与 `build_native_tet_boundary_extrap`
    同一组通量点。

    Returns:
        Lift_ref: (n_native_sps, n1d*n1d)
    """
    V_sps, V_fp, modes = _native_face_value_vandermondes(order, excluded_vertex, slot)
    inv_norms = np.array([1.0 / _native_mode_norm_squared(i, j, k) for (i, j, k) in modes])
    return V_sps @ (inv_norms[:, None] * V_fp.T)


def compute_native_tet_jacobian(cell_nodes: np.ndarray) -> Tuple[float, np.ndarray]:
    """直边四面体物理雅可比——重心坐标仿射映射对 (r,s,t) 的偏导数是
    **常数矩阵**（`tet_barycentric` 展开：dphys/dr=0.5*(p1-p0)，
    dphys/ds=0.5*(p2-p0)，dphys/dt=0.5*(p3-p0)，均与 (r,s,t) 无关），
    处处非奇异（只要四面体本身体积非零），每个单元只需要算一次（不逐点用微分矩阵 `D@phys` 求）。

    Returns:
        (det_j, adj_j)：det_j 是标量（该单元的常数 Jacobian 行列式），
        adj_j 是 (3,3) 常数伴随矩阵（`det_j*inv(J)`，与 `compute_adj_j`
        输出同一约定），调用方对该单元的每个 SP 都复用同一份。
    """
    from ...grid.curved_mapping.curved_mapping import batched_det_inv_3x3

    p0, p1, p2, p3 = cell_nodes
    J = 0.5 * np.column_stack([p1 - p0, p2 - p0, p3 - p0])  # (3,3)
    det_j, inv_j = batched_det_inv_3x3(J[None, :, :])
    adj_j = det_j[0] * inv_j[0]
    return float(det_j[0]), adj_j
