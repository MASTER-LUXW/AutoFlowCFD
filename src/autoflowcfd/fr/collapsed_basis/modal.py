"""AutoFlowCFD V2.0 - 坍缩坐标（Duffy）模态基与体积微分算子。

从 `fr/collapsed_basis.py`（原 610 行）拆出（2026-09-19）。纯搬家，逻辑
未改；背景（为什么需要坍缩坐标模态基、退化边附近的实测收益与局限）见
`__init__.py` 的模块文档。

**两半都已不在生产残差路径的默认档上**：坍缩四面体基已于 2026-09-03
删除，`tet_modal_basis_and_grad` / `build_collapsed_diff_matrices("tet",
...)` 现在只被面通量点的跨单元插值（`fr/face_flux_points/`）与测试消费；
棱柱那一半自 2026-09-20 起也不再是默认（`AFCFD_PRISM_BASIS` 默认值改为
`native`，三份证据见 `fr/native_prism/mode.py`），只在显式指定
`collapsed` 时才进入残差路径 —— 而那半边的终态是整套删除。
"""

from typing import Tuple

import numpy as np
from numba import njit

from .jacobi import grad_jacobi_polynomial, jacobi_polynomial


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
    D = np.stack([Da, Db, Dc], axis=-1)
    # 强制逐位精确地零化常数（`D @ 1 = 0`）。上一段已经把"显式求逆 ->
    # lu_solve"这一层舍入压下去了，但残余仍是 `eps*cond(V)` 量级，而它
    # 直接进自由流保持性：坍缩基 over_order=3 的 D_fine 实测
    # `max|D@1|` = 5.6e-14，经 restrict 放大到 5.8e-13，在真实网格上表现
    # 为棱柱段均匀自由流残差 4.72e-3。
    #
    # 对棱柱这个修正只是**部分**的：度量逐点变化，自由流保持需要完整的
    # 离散 GCL `sum_m D_m(adj(J)_m) = 0`，行和修正只消掉"度量恒定部分"
    # 那一项。适用范围与实测见 `diff_matrix_consistency.py` 模块文档。
    from ..diff_matrix_consistency import enforce_constant_annihilation

    enforce_constant_annihilation(D)
    return D
