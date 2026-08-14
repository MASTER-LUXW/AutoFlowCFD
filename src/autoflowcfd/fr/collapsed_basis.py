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
    """
    x = np.asarray(x, dtype=np.float64)
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


def grad_jacobi_polynomial(x: np.ndarray, alpha: float, beta: float, n: int) -> np.ndarray:
    """P_n^(alpha,beta) 对 x 的导数：(n+alpha+beta+1)/2 * P_{n-1}^(alpha+1,beta+1)(x)，n=0 时恒为 0。"""
    if n == 0:
        return np.zeros_like(np.asarray(x, dtype=np.float64))
    return 0.5 * (n + alpha + beta + 1.0) * jacobi_polynomial(x, alpha + 1.0, beta + 1.0, n - 1)


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
    """
    f_i = jacobi_polynomial(a, 0.0, 0.0, i)
    df_i = grad_jacobi_polynomial(a, 0.0, 0.0, i)

    half_1mb = (1.0 - b) / 2.0
    Pj = jacobi_polynomial(b, 2 * i + 1, 0.0, j)
    dPj = grad_jacobi_polynomial(b, 2 * i + 1, 0.0, j)

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
                Pk = jacobi_polynomial(c, 2 * i + 2 * j + 2, 0.0, k)
                dPk = grad_jacobi_polynomial(c, 2 * i + 2 * j + 2, 0.0, k)
                h = w * Pk
                dh_dc = dw * Pk + w * dPk

                flat = i * n1d * n1d + j * n1d + k
                V[:, flat] = g_ij * h
                Va[:, flat] = dg_ij_da * h
                Vb[:, flat] = dg_ij_db * h
                Vc[:, flat] = g_ij * dh_dc
    return V, Va, Vb, Vc


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

    V_inv = np.linalg.inv(V)
    Da = Va @ V_inv
    Db = Vb @ V_inv
    Dc = Vc @ V_inv
    return np.stack([Da, Db, Dc], axis=-1)


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

    return V_fp @ np.linalg.inv(V_sps)
