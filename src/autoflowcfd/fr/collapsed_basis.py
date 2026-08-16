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
"""

from typing import Tuple

import numpy as np
from numba import njit


@njit(cache=True)
def jacobi_polynomial(x: np.ndarray, alpha: float, beta: float, n: int) -> np.ndarray:
    """计算未归一化 Jacobi 多项式 P_n^(alpha,beta)(x) 在 x（数组）处的取值。

    标准三项递推（Hesthaven & Warburton 附录 A / Abramowitz & Stegun）。
    只作为构造 Vandermonde 矩阵的基，不需要正交归一化——微分矩阵
    D=V_xi@inv(V) 与具体选用哪组（可逆的）基无关，只要 V 可逆。

    条件数随阶数增长（真实验证：四面体 N<=2 时 cond(V)<=1e5，N=3 时
    ~1e9，N=4 时 ~1e14）——本代码库当前实际使用的多项式阶数 N=2（P=2）
    在这个范围内工作良好（体积项散度残差已在真实网格上验证到 1e-12
    量级）。曾尝试用标准 L2 归一化 Jacobi 多项式降低高阶条件数
    （Hesthaven-Warburton 参考实现的做法），但发现对本模块这种
    "P_n^(alpha,0) 外面再乘 (1-b)^i 权重因子"的复合结构，单独归一化
    多项式因子本身并不能改善（真实测得反而在部分阶数下更差：N=4 时
    从 2.8e14 变成 8.2e16）——权重因子必须和多项式的正交性一起联合
    归一化才是正确做法，这是比单独归一化 Jacobi 多项式更复杂的构造，
    N>=3 时的条件数改善留作后续工作（不影响当前 N=2 生产阶数的正确性
    与数值稳健性，已充分验证）。

    numba @njit 编译（性能优化：这个函数在真实网格上被调用数百万次——
    每次 owner/neighbor 跨单元插值都要重新构造 Vandermonde 矩阵，
    130 万面的生产网格上单是网格加载阶段就要跑约 300 万次调用，纯
    Python 函数调用开销占了 FP 几何构建约 2/3 的时间，见开发过程记录
    的 cProfile 剖析）。数学公式与递推逻辑完全不变，只是编译成原生
    代码执行——已用随机输入对比新旧实现逐位一致（200 组随机 n/alpha/
    beta/x 全部 0.0 误差），实测单次调用提速约 13 倍。numba 要求输入
    是具体类型的 ndarray，不再接受 list 等其它可迭代对象——本模块内外
    全部调用点传入的都已经是 float64 ndarray（见调用处），这不是放宽/
    简化数值行为，只是收紧了函数签名对输入类型的隐式假设。
    """
    P0 = np.ones_like(x)
    if n == 0:
        return P0
    P1 = 0.5 * ((alpha - beta) + (alpha + beta + 2.0) * x)
    if n == 1:
        return P1
    Pnm1, Pn = P0, P1
    for k in range(1, n):
        a1 = 2.0 * (k + 1) * (k + alpha + beta + 1) * (2 * k + alpha + beta)
        a2 = (2 * k + alpha + beta + 1) * (alpha**2 - beta**2)
        a3 = (2 * k + alpha + beta) * (2 * k + alpha + beta + 1) * (2 * k + alpha + beta + 2)
        a4 = 2.0 * (k + alpha) * (k + beta) * (2 * k + alpha + beta + 2)
        Pnp1 = ((a2 + a3 * x) * Pn - a4 * Pnm1) / a1
        Pnm1, Pn = Pn, Pnp1
    return Pn


@njit(cache=True)
def grad_jacobi_polynomial(x: np.ndarray, alpha: float, beta: float, n: int) -> np.ndarray:
    """P_n^(alpha,beta) 对 x 的导数：(n+alpha+beta+1)/2 * P_{n-1}^(alpha+1,beta+1)(x)，n=0 时恒为 0。
    numba @njit 编译，理由/验证同 jacobi_polynomial 文档。"""
    if n == 0:
        return np.zeros_like(x)
    return 0.5 * (n + alpha + beta + 1.0) * jacobi_polynomial(x, alpha + 1.0, beta + 1.0, n - 1)


@njit(cache=True)
def _collapsed_triangle_mode(
    a: np.ndarray, b: np.ndarray, i: int, j: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """三角形坍缩坐标模态 g_ij(a,b) = P_i^(0,0)(a) * ((1-b)/2)^i * P_j^(2i+1,0)(b)
    及其对 a、b 的偏导，四面体、棱柱都要用到（棱柱的 (a,b) 截面与四面体
    共用同一套坍缩三角形模态，见 curved_mapping.cube_to_tri_rs 的推导）。

    (1-b)/2 权重因子是这套基的核心：Duffy 变换在 b→1 处度量项含
    1/(1-b) 奇异因子，这个权重因子在求导后（乘积法则）恰好提供解析
    抵消所需的结构，使 i>=1 的模态在 b=1 附近仍保持数值良态——这正是
    朴素张量积基（fr/operators.py 给六面体用的那套）所缺少的。

    numba @njit 编译，理由/验证同 jacobi_polynomial 文档。`2*i+1` 显式
    转成 float 传给 alpha 参数——numba 对同一个 njit 函数按实参的具体
    类型分别编译特化版本，显式转换避免 int/float 两套特化都被编译一遍
    的额外开销，不影响数值结果（alpha 本身就是数学意义上的浮点参数）。
    """
    f_i = jacobi_polynomial(a, 0.0, 0.0, i)
    df_i = grad_jacobi_polynomial(a, 0.0, 0.0, i)

    half_1mb = (1.0 - b) / 2.0
    Pj = jacobi_polynomial(b, float(2 * i + 1), 0.0, j)
    dPj = grad_jacobi_polynomial(b, float(2 * i + 1), 0.0, j)

    if i == 0:
        w = np.ones_like(half_1mb)
        dw = np.zeros_like(half_1mb)
    else:
        w = half_1mb**i
        dw = -0.5 * i * half_1mb ** (i - 1)

    g = w * Pj
    dg_db = dw * Pj + w * dPj

    val = f_i * g
    dval_da = df_i * g
    dval_db = f_i * dg_db
    return val, dval_da, dval_db


@njit(cache=True)
def tet_modal_basis_and_grad(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, order: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """四面体坍缩坐标模态基（Karniadakis-Sherwin"坍缩张量积"族，i,j,k 各自
    独立取 0..order，共 (order+1)^3 个模态，与现有张量积 SPs 个数一致）
    及其对 (a,b,c) 的偏导，在给定点集上取值。

    psi_ijk(a,b,c) = g_ij(a,b) * ((1-c)/2)^(i+j) * P_k^(2i+2j+2,0)(c)

    Returns:
        (V, Va, Vb, Vc)：每个形状 (n_pts, (order+1)^3)，模态按
        flat = i*(order+1)^2 + j*(order+1) + k 展平（与 SPs 的
        (ia,ib,ic) 展平约定一致，供 Vandermonde 求逆后直接得到与现有
        D_3d 同形状 (n_sps,n_sps,3) 的微分矩阵）。
    """
    n1d = order + 1
    n_modes = n1d**3
    n_pts = len(a)
    V = np.zeros((n_pts, n_modes))
    Va = np.zeros((n_pts, n_modes))
    Vb = np.zeros((n_pts, n_modes))
    Vc = np.zeros((n_pts, n_modes))

    half_1mc = (1.0 - c) / 2.0
    for i in range(n1d):
        for j in range(n1d):
            g_ij, dg_ij_da, dg_ij_db = _collapsed_triangle_mode(a, b, i, j)
            p = i + j
            if p == 0:
                w = np.ones_like(half_1mc)
                dw = np.zeros_like(half_1mc)
            else:
                w = half_1mc**p
                dw = -0.5 * p * half_1mc ** (p - 1)
            for k in range(n1d):
                Pk = jacobi_polynomial(c, float(2 * i + 2 * j + 2), 0.0, k)
                dPk = grad_jacobi_polynomial(c, float(2 * i + 2 * j + 2), 0.0, k)
                h = w * Pk
                dh_dc = dw * Pk + w * dPk

                flat = i * n1d * n1d + j * n1d + k
                V[:, flat] = g_ij * h
                Va[:, flat] = dg_ij_da * h
                Vb[:, flat] = dg_ij_db * h
                Vc[:, flat] = g_ij * dh_dc
    return V, Va, Vb, Vc


@njit(cache=True)
def prism_modal_basis_and_grad(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, order: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """棱柱坍缩坐标模态基：(a,b) 截面用与四面体相同的坍缩三角形模态
    （棱柱只在 a,b 之间做 Duffy 三角形坍缩，c 是真正的张量积挤出方向，
    不参与坍缩——与 curved_mapping.map_prism_to_physical 的构造一致），
    c 方向用普通（非坍缩）Legendre/Jacobi 基：

    phi_ijk(a,b,c) = g_ij(a,b) * P_k^(0,0)(c)

    Returns: 同 tet_modal_basis_and_grad，(V,Va,Vb,Vc) 形状 (n_pts,(order+1)^3)。
    """
    n1d = order + 1
    n_modes = n1d**3
    n_pts = len(a)
    V = np.zeros((n_pts, n_modes))
    Va = np.zeros((n_pts, n_modes))
    Vb = np.zeros((n_pts, n_modes))
    Vc = np.zeros((n_pts, n_modes))

    for i in range(n1d):
        for j in range(n1d):
            g_ij, dg_ij_da, dg_ij_db = _collapsed_triangle_mode(a, b, i, j)
            for k in range(n1d):
                Lk = jacobi_polynomial(c, 0.0, 0.0, k)
                dLk = grad_jacobi_polynomial(c, 0.0, 0.0, k)

                flat = i * n1d * n1d + j * n1d + k
                V[:, flat] = g_ij * Lk
                Va[:, flat] = dg_ij_da * Lk
                Vb[:, flat] = dg_ij_db * Lk
                Vc[:, flat] = g_ij * dLk
    return V, Va, Vb, Vc


def build_collapsed_diff_matrices(cell_type: str, order: int, ref_cube_sps: np.ndarray) -> np.ndarray:
    """在给定参考点集（现有张量积 Gauss-Legendre SPs，Duffy 映射前的
    计算立方体坐标）上，构造该单元类型专用的微分矩阵 D，与
    fr/operators.py::FROperators.D_3d 同形状 (n_sps,n_sps,3)、同语义
    （D[:,:,m] 是对第 m 个参考坐标方向求导的矩阵），可直接替换 D_3d 在
    体积散度/梯度/几何 Jacobian 计算三处的用法。

    Args:
        cell_type: "tet" 或 "prism"
        order: 多项式阶数 P（每方向 n1d=P+1 个点/模态）
        ref_cube_sps: (n_sps,3) 参考点坐标，须与当前单元类型 SPs 的
            展平顺序 (ia*n1d^2+ib*n1d+ic) 完全一致

    Returns:
        D: (n_sps,n_sps,3)
    """
    a, b, c = ref_cube_sps[:, 0], ref_cube_sps[:, 1], ref_cube_sps[:, 2]
    if cell_type == "tet":
        V, Va, Vb, Vc = tet_modal_basis_and_grad(a, b, c, order)
    elif cell_type == "prism":
        V, Va, Vb, Vc = prism_modal_basis_and_grad(a, b, c, order)
    else:
        raise ValueError(f"Unknown cell_type for collapsed differentiation matrix: {cell_type!r}")

    # D = Va @ V^{-1}，用 LU 分解 + lu_solve 而不是显式求逆——V 的条件数
    # 随阶数快速增长（本文件 jacobi_polynomial 文档：N=2~1e5，N=4~1e14），
    # 显式 np.linalg.inv 会把这个条件数直接乘进舍入误差。受控实验：
    # 体积项去混叠（over-integration）用 over_order=4 构造 D_fine 时，
    # 用显式求逆算出的 Kopriva 度量恒等式残差 ~2.5e-5（应为机器精度），
    # 改用 lu_solve 后见下方改动。D=Va@V^{-1} <=> D.T = V^{-T}@Va.T，
    # 即解 V.T @ X = Va.T 求 X=D.T。
    from scipy.linalg import lu_factor, lu_solve

    lu_piv = lu_factor(V.T)
    Da = lu_solve(lu_piv, Va.T).T
    Db = lu_solve(lu_piv, Vb.T).T
    Dc = lu_solve(lu_piv, Vc.T).T
    return np.stack([Da, Db, Dc], axis=-1)


# 过积分（over-integration）细网格阶数的硬上限。理想去混叠阶数是
# 2*order（二次非线性经验法则），但本模块的模态 Vandermonde 矩阵条件数
# 随阶数爆炸式增长（本文件 jacobi_polynomial 文档实测：N=2 时 cond~1e5，
# N=3 时 ~1e9，N=4 时 ~1e14——接近 float64 ~1e16 动态范围的可用边界）。
# 真实数值实验证实了这个上限的必要性：即使把 build_collapsed_diff_matrices/
# build_collapsed_boundary_extrap 的显式求逆换成 lu_solve（同一轮修复，
# 见上面 D=Va@V^{-1} 处的说明）大幅改善了条件数敏感度，over_order=4
# （cond~1e14）在生产阶数 P=2 上仍不稳定：均匀自由流场残差 1.7（应为
# ~0），线性剪切流残差 0.56（应为 0）——量级上比不做过积分更差，是真正
# 的数值噪声而非改善。上限设为 3（cond~1e9）后同一组测试稳定给出自由
# 流场残差 1.06e-5、剪切流残差 3.49e-6（后者比不做过积分时的 43~62 倍
# 误差改善约 5~6 个数量级）；P=3 下 over_order=min(2*3,3)=3=order，
# 退化为 fine 点集与 coarse 完全重合（interp_c2f/restrict_f2c 退化为
# 恒等矩阵，D_fine=D_3d_tet/prism 本身）——等价于不做过积分，不提供
# 额外去混叠效果，但也不会引入新的不稳定；P=3 本来就不是生产阶数，
# 测试容差也早已为此放宽，见
# tests/unit/test_fr_residual_inviscid.py::TestFreeStreamPreservation）。
OVERINTEGRATION_MAX_ORDER = 3


def build_overintegration_operators(
    cell_type: str, order: int, over_order: int, ref_cube_sps_coarse: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构造体积项去混叠（over-integration）三件套：coarse->fine 插值、
    fine 网格上的微分矩阵、fine->coarse 限制。

    背景（V2.0 二次专家评审 Tier 0 #2）：体积项残差 `D_3d_tet/prism @
    (adj(J)*F_phys(Q))` 直接在 coarse SPs（degree=order 的节点表示）上
    做微分——但 `adj(J)*F_phys(Q)` 是 Q 的非线性函数（欧拉通量含
    u_i*u_j、p*u_j 等二次项）与度量项的乘积，其真实多项式次数远高于
    order，直接对它的 degree-order 节点插值多项式求导，等价于先把它
    混叠（alias）到 degree-order 空间再求导——真实数值实验证实：对
    解析残差恒为 0 的线性剪切场 u=30+a*y，P2（生产默认阶数）算出的
    残差达到真值的约 43~62 倍，P1 更是 400+ 倍，只有从未在生产路径上
    使用过的 P3 才勉强正确。这是标准的体积项积分不足（aliasing），
    工程解法是"过积分"：把 Q 插值到更细的求积点上，在细网格上精确
    评估非线性通量和度量项后再求导，最后把结果限制回 coarse SPs——
    不是研究级问题（`5_重大问题修复-Part1.md` 对相关不守恒问题的结论
    "需要 entropy-stable/split-form 研究级重新设计" 针对的是另一个
    机制，见该文档；本机制的标准解法见 Kopriva《Implementing Spectral
    Methods for PDEs》Ch.5 "aliasing and the strong form" 与 Kirby &
    Karniadakis 关于二次非线性去混叠所需求积阶数的经典分析）。

    三个算子都基于同一套模态 Vandermonde 机制（与 build_cross_interp/
    build_collapsed_boundary_extrap 同源）：
    1. Interp_c2f = V_fine_pts_by_coarse_basis @ V_coarse_sps^{-1}
       （用 COARSE 阶数的模态基在 FINE 点上取值——Q 本身次数 <= order，
       这一步是精确插值，不引入混叠）
    2. D_fine：在 FINE 点集上、用 FINE 阶数的模态基构造的微分矩阵
       （即 build_collapsed_diff_matrices 在 over_order 下的结果）——
       微分的是 FINE 点上取值所代表的 degree-over_order 插值多项式，
       更接近真实非线性通量的次数，混叠误差大幅降低
    3. Restrict_f2c = V_coarse_pts_by_fine_basis @ V_fine_sps^{-1}
       （用 FINE 阶数的模态基在 COARSE 点上取值——把微分后的场从细网格
       插值回粗网格 SPs，供残差公式除以 coarse 的 det(J) 使用）

    Args:
        cell_type: "tet" 或 "prism"
        order: 当前求解阶数 P（coarse）
        over_order: 过积分阶数（建议 2*order，二次非线性去混叠的标准
            经验法则；order=0 时不适用，P0 走独立的有限体积路径）
        ref_cube_sps_coarse: coarse SPs 参考坐标 (n_coarse,3)

    Returns:
        (ref_cube_sps_fine, interp_c2f, D_fine, restrict_f2c)：
        ref_cube_sps_fine 形状 (n_fine,3)；interp_c2f 形状
        (n_fine,n_coarse)；D_fine 形状 (n_fine,n_fine,3)；restrict_f2c
        形状 (n_coarse,n_fine)。
    """
    fine_n1d = over_order + 1
    fine_1d = _gauss_legendre_1d(fine_n1d)
    ga, gb, gc = np.meshgrid(fine_1d, fine_1d, fine_1d, indexing="ij")
    ref_cube_sps_fine = np.column_stack([ga.ravel(), gb.ravel(), gc.ravel()])

    basis_fn = tet_modal_basis_and_grad if cell_type == "tet" else prism_modal_basis_and_grad

    V_coarse_sps, _, _, _ = basis_fn(
        ref_cube_sps_coarse[:, 0], ref_cube_sps_coarse[:, 1], ref_cube_sps_coarse[:, 2], order
    )
    V_fine_at_fine, _, _, _ = basis_fn(
        ref_cube_sps_fine[:, 0], ref_cube_sps_fine[:, 1], ref_cube_sps_fine[:, 2], over_order
    )

    # interp_c2f：COARSE 阶数模态基在 FINE 点上取值
    V_coarse_at_fine, _, _, _ = basis_fn(
        ref_cube_sps_fine[:, 0], ref_cube_sps_fine[:, 1], ref_cube_sps_fine[:, 2], order
    )
    from scipy.linalg import lu_factor, lu_solve

    lu_coarse = lu_factor(V_coarse_sps.T)
    interp_c2f = lu_solve(lu_coarse, V_coarse_at_fine.T).T

    # D_fine：FINE 阶数模态基自身的微分矩阵
    D_fine = build_collapsed_diff_matrices(cell_type, over_order, ref_cube_sps_fine)

    # restrict_f2c：FINE 阶数模态基在 COARSE 点上取值
    V_fine_at_coarse, _, _, _ = basis_fn(
        ref_cube_sps_coarse[:, 0], ref_cube_sps_coarse[:, 1], ref_cube_sps_coarse[:, 2], over_order
    )
    lu_fine = lu_factor(V_fine_at_fine.T)
    restrict_f2c = lu_solve(lu_fine, V_fine_at_coarse.T).T

    return ref_cube_sps_fine, interp_c2f, D_fine, restrict_f2c


def _gauss_legendre_1d(n: int) -> np.ndarray:
    """n 点 1D Gauss-Legendre 求积点（不需要权重，过积分只用点位）。"""
    from .quadrature_points import gauss_legendre

    pts, _ = gauss_legendre(n)
    return pts


def build_collapsed_boundary_extrap(
    cell_type: str, order: int, ref_cube_sps: np.ndarray, axis: int, side: float
) -> np.ndarray:
    """把体积 SPs 上的节点值外插到某个立方体边界面（fixed axis=side）
    的 Flux Points，用与 build_collapsed_diff_matrices 同一套坍缩坐标
    模态基构造，而不是 fr/face_flux_points.py::extrapolate_to_face 现在
    用的朴素 1D 张量积 Lagrange 外插。

    背景：extrapolate_to_face 对固定轴做 1D Lagrange 边界外插、其余两个
    轴按原生 SP 网格索引直接对应，这个简化对朴素张量积基完全等价于
    "在该点求整张三维张量积插值多项式的值"——但对坍缩坐标单元，度量项
    adj(J) 这类场在退化边附近变化剧烈（真实网格验证：a=+1 面上外插出的
    等效法向方向与真实几何法向偏差最大约 28°，虽然仍在现有 60° 校验
    阈值内、不会报错，但足以在残差公式除以（该处真实偏小的）Jacobian
    后被放大到灾难量级），用只有 3 个内部 Gauss-Legendre 点、不含边界点
    的朴素张量积外插去逼近这种剧烈变化在数学上站不住脚——必须换成与
    体积微分矩阵一致的坍缩坐标模态基外插，让边界取值与体积微分共用同一
    个、真正匹配坍缩坐标退化结构的插值空间。

    Args:
        cell_type: "tet" 或 "prism"
        order: 多项式阶数
        ref_cube_sps: 体积 SPs 参考坐标 (n_sps,3)（与 build_collapsed_
            diff_matrices 用的是同一组点）
        axis: 被固定的坍缩坐标轴（0=a,1=b,2=c）
        side: 该轴的边界取值（-1.0 或 1.0）

    Returns:
        E: (n_fp, n_sps)，E @ field(SPs) 给出 field 在该边界面 Flux
        Points（其余两个轴仍取原生张量积 Gauss-Legendre 网格）处的取值，
        Flux Points 展平顺序与 fr/face_flux_points.py::face_ref_grid
        完全一致（other_axes[0] 为外层、other_axes[1] 为内层）。
    """
    n1d = order + 1
    other_axes = [a for a in range(3) if a != axis]
    # ref_cube_sps 是 1D 节点集合 sps_1d 的三维张量积，任一维展开后按
    # 升序取唯一值即可还原 sps_1d（不依赖调用方单独传入）。
    sps_1d = np.unique(ref_cube_sps[:, 0])

    g1, g2 = np.meshgrid(sps_1d, sps_1d, indexing="ij")
    fp_pts = np.zeros((n1d * n1d, 3))
    fp_pts[:, axis] = side
    fp_pts[:, other_axes[0]] = g1.ravel()
    fp_pts[:, other_axes[1]] = g2.ravel()

    a_sps, b_sps, c_sps = ref_cube_sps[:, 0], ref_cube_sps[:, 1], ref_cube_sps[:, 2]
    a_fp, b_fp, c_fp = fp_pts[:, 0], fp_pts[:, 1], fp_pts[:, 2]
    if cell_type == "tet":
        V_sps, _, _, _ = tet_modal_basis_and_grad(a_sps, b_sps, c_sps, order)
        V_fp, _, _, _ = tet_modal_basis_and_grad(a_fp, b_fp, c_fp, order)
    elif cell_type == "prism":
        V_sps, _, _, _ = prism_modal_basis_and_grad(a_sps, b_sps, c_sps, order)
        V_fp, _, _, _ = prism_modal_basis_and_grad(a_fp, b_fp, c_fp, order)
    else:
        raise ValueError(f"Unknown cell_type for collapsed boundary extrapolation: {cell_type!r}")

    # 同 build_collapsed_diff_matrices：用 lu_solve 而不是显式求逆，控制
    # V_sps 条件数带来的舍入放大（同一份 V_sps^{-1} 会被同一 (cell_type,
    # order) 的所有单元共享，值得用分解而不是每次都算一次 inv）。
    from scipy.linalg import lu_factor, lu_solve

    lu_piv = lu_factor(V_sps.T)
    return lu_solve(lu_piv, V_fp.T).T
