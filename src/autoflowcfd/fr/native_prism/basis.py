"""AutoFlowCFD V2.0 - 棱柱原生基：三角形 PKD/Dubiner ⊗ 直线 Legendre。

## 为什么需要它

棱柱至今用**坍缩坐标基**（`collapsed_basis.py::prism_modal_basis_and_grad`：
在 (a,b) 截面上做 Duffy 三角形坍缩、节点取张量积 Gauss-Legendre 方格）。
实测三条后果，完整数据见 `native_triangle_basis.py` 模块文档：

  1. 微分矩阵元素量级随阶数爆炸（`max|D_3d_prism|` P1 2.05 -> P3 560.1，
     每阶约 x25；native 四面体对照 0.50 -> 4.05）；
  2. 自由流保持性只有 ~1e-9，且实测严格等于 `eps * max|D| / det(J)`
     —— 是舍入被算子量级放大，**不降低 max|D| 就改不掉**；
  3. **伪横流**：只有被三角化的两个参考轴方向会长出非物理横流速度
     （把展向换成挤出轴后保持机器零，差 11 个数量级），它在 P1 上饱和、
     在 P2 上无界增长 —— 1152 单元干净结构化棱柱网格上 P2 第 75 步发散，
     这与真实网格 plate_demo 363k 的 P2 第 68 步发散是同一特征。

四面体当年有完全同类的病理，解法就是整套换成 native PKD 基。本模块是
棱柱侧的同一条路。

## 构造

棱柱参考单元 = 参考三角形 `(r,s)` × 区间 `t ∈ [-1,1]`：

    节点：三角形 Warp & Blend 节点 ⊗ 一维 Gauss-Legendre 点
    模态：三角形受限 PKD 模态 {i+j<=order} ⊗ Legendre {k<=order}
    基：  phi_{ijk}(r,s,t) = psi_{ij}(r,s) * P_k^{0,0}(t)

自由度数 `(order+1)^2(order+2)/2`，比坍缩基的 `(order+1)^3` 少
`(order+1)/((order+2)/2*...)`，例如 P2 是 18 而不是 27、P3 是 40 而不是 64。

**挤出方向 `t` 保持 Legendre + Gauss-Legendre 不变**：那个方向本来就
不坍缩、是真正的张量积方向，现有实现在它上面是精确的（直壁挤出下物理
`y` 是 `t` 的精确线性函数，近壁剪切解复合后是精确多项式、插值截断误差
为零 —— 见 `tests/validation/_channel_mesh.py` 与项目记忆
`tet_collapsed_coord_anisotropy`）。只换被坍缩的那两个轴。

## 节点排列约定

节点按 **"三角形点外层、挤出点内层"** 排列：

    flat = i_tri * n_1d + k_t      （i_tri = 0..n_tri-1, k_t = 0..order）

这样"同一条挤出线上的点"在内存里连续，与挤出方向是张量积这一事实一致，
也便于把挤出方向的一维算子写成对内层的作用。模态用同一套排列。
"""

from typing import List, Tuple

import numpy as np

from ..collapsed_basis import grad_jacobi_polynomial, jacobi_polynomial
from .triangle_basis import (
    eval_tri_modes,
    restricted_tri_modes,
    warp_blend_nodes_2d,
)
from ..quadrature_points import gauss_legendre


def native_prism_n_sps(order: int) -> int:
    """棱柱原生基的自由度数 `(order+1)^2 (order+2) / 2`。"""
    n_tri = (order + 1) * (order + 2) // 2
    return n_tri * (order + 1)


def restricted_prism_modes(order: int) -> List[Tuple[int, int, int]]:
    """棱柱模态索引 `(i, j, k)`：`(i,j)` 是三角形受限 PKD 模态
    （`i+j<=order`），`k` 是挤出方向的 Legendre 次数（`0..order`）。

    排列与节点一致：三角形模态外层、挤出模态内层。
    """
    tri = restricted_tri_modes(order)
    return [(i, j, k) for (i, j) in tri for k in range(order + 1)]


def build_native_prism_nodes(order: int) -> np.ndarray:
    """棱柱参考节点 `(n_sps, 3)`，列为 `(r, s, t)`。

    `(r,s)` 是参考三角形上的 Warp & Blend 节点，`t` 是一维
    Gauss-Legendre 点（与现有实现在挤出方向的取点一致）。
    """
    r_tri, s_tri = warp_blend_nodes_2d(order)
    t_1d, _ = gauss_legendre(order + 1)
    n_tri = len(r_tri)
    n_t = len(t_1d)
    out = np.empty((n_tri * n_t, 3))
    for a_i in range(n_tri):
        for k in range(n_t):
            flat = a_i * n_t + k
            out[flat, 0] = r_tri[a_i]
            out[flat, 1] = s_tri[a_i]
            out[flat, 2] = t_1d[k]
    return out


def build_native_prism_vandermonde(order: int, ref_rst: np.ndarray):
    """在给定参考坐标上构造 `(V, Vr, Vs, Vt)`。

    Args:
        order: 多项式阶数
        ref_rst: `(n_pts, 3)`，列为 `(r, s, t)`

    Returns:
        四个 `(n_pts, n_modes)` 数组。`Vr`/`Vs` 是对参考**三角形**坐标
        的偏导（由 `simplex2d_grad` 直接给出闭式，不经过坍缩坐标的逆），
        `Vt` 是对挤出坐标的偏导。
    """
    ref_rst = np.asarray(ref_rst, dtype=np.float64)
    r, s, t = ref_rst[:, 0], ref_rst[:, 1], ref_rst[:, 2]

    # 三角形那一半整块复用 `eval_tri_modes`（同一份公式，不抄第二遍）。
    # 它一次给出全部 `n_tri` 个三角形模态在**全部**棱柱点上的值。
    tri_V, tri_Vr, tri_Vs = eval_tri_modes(order, r, s)

    # 挤出方向的 Legendre 只有 order+1 个，逐 k 求一次即可。
    n_t = order + 1
    leg = np.empty((len(t), n_t))
    dleg = np.empty_like(leg)
    for k in range(n_t):
        leg[:, k] = jacobi_polynomial(t, 0.0, 0.0, k)
        dleg[:, k] = grad_jacobi_polynomial(t, 0.0, 0.0, k)

    # 模态排列是"三角形模态外层、挤出模态内层"（见模块文档），所以
    # 第 m 个棱柱模态的三角形列号恰好是 `m // n_t`、Legendre 次数是
    # `m % n_t` —— 不需要再查表。
    n_tri = tri_V.shape[1]
    tri_col = np.repeat(np.arange(n_tri), n_t)
    leg_col = np.tile(np.arange(n_t), n_tri)

    V = tri_V[:, tri_col] * leg[:, leg_col]
    Vr = tri_Vr[:, tri_col] * leg[:, leg_col]
    Vs = tri_Vs[:, tri_col] * leg[:, leg_col]
    Vt = tri_V[:, tri_col] * dleg[:, leg_col]
    return V, Vr, Vs, Vt


def build_native_prism_operators(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """构造棱柱原生体积微分算子。

    Returns:
        `(ref_rst, D_native)`：
        `ref_rst` 形状 `(n_sps, 3)`，节点在参考棱柱 `(r,s,t)` 上的坐标；
        `D_native` 形状 `(n_sps, n_sps, 3)`，`D[:,:,m] @ field` 给出对
        参考坐标第 m 个方向（r/s/t）的导数 —— 与 `D_3d_prism` 同样的
        消费方式，但参考坐标是**原生**的 `(r,s,t)` 而不是坍缩的 `(a,b,c)`。

    与 `native_simplex_basis.build_native_tet_operators` 同一套做法，
    包括最后那步 `enforce_constant_annihilation`：解析上 `D @ 1 = 0`
    必然成立，但 LU 求解的舍入残余会留到 `eps*cond(V)`，而**直边单元的
    自由流保持性完全由这个残余决定**（度量逐单元常数，均匀流下体积项
    散度恰好正比于 `D @ 1`，还会被小体积单元的 `1/det(J)` 放大）。
    """
    from scipy.linalg import lu_factor, lu_solve

    from ..diff_matrix_consistency import enforce_constant_annihilation

    ref_rst = build_native_prism_nodes(order)
    V, Vr, Vs, Vt = build_native_prism_vandermonde(order, ref_rst)
    n, n_modes = V.shape
    if n != n_modes:
        raise ValueError(
            f"棱柱节点数 {n} 与模态数 {n_modes} 不一致（order={order}）——"
            f"两者理论上必须相等（(order+1)^2 (order+2)/2），不一致说明"
            f"节点生成或模态索引有 bug，不应当静默继续。")

    lu = lu_factor(V.T)
    D = np.stack([lu_solve(lu, Vr.T).T,
                  lu_solve(lu, Vs.T).T,
                  lu_solve(lu, Vt.T).T], axis=-1)
    enforce_constant_annihilation(D)
    return ref_rst, D

# ---------------------------------------------------------------------------
# 几何映射与雅可比（原生参考坐标版本）
# ---------------------------------------------------------------------------
#
# 与坍缩版本（`grid/curved_mapping/curved_mapping.py::map_prism_to_physical`
# 与 `curved_mapping_exact_jacobian.py::prism_exact_jacobian`）是**同一个
# 几何映射**，只是自变量换成原生 `(r,s,t)`：
#
#     x(r,s,t) = (1-t)/2 * bottom(r,s) + (1+t)/2 * top(r,s)
#     bottom   = l1 p0 + l2 p1 + l3 p2,   top = l1 p3 + l2 p4 + l3 p5
#
# 重心坐标 `l1 = -(r+s)/2`、`l2 = (1+r)/2`、`l3 = (1+s)/2` 是 `(r,s)` 的
# **仿射**函数，所以 `d(bottom)/dr` 等是**常向量**。坍缩版本里对应的偏导
# 带着 `(1-b)/4` 这个因子（因为 `(r,s)` 本身是 `(a,b)` 的多项式），它在
# 退化边 `b -> 1` 上趋零 —— 这正是坍缩度量在那一带病态的来源。
#
# 直棱柱（top = bottom + h，h 为常向量）在原生坐标下雅可比**逐单元恒定**：
# `d/dr`、`d/ds` 两列的上下面贡献相同、与 t 无关，`d/dt = h/2`。于是均匀
# 流下体积项散度恰好正比于 `D @ 1`，而那是机器零（见
# `build_native_prism_operators` 里 `enforce_constant_annihilation` 的说明）
# —— 这就是"自由流保持性从 ~1e-9 回到机器零"的结构性依据。


def map_native_prism_to_physical(ref_rst: np.ndarray,
                                 cell_nodes: np.ndarray) -> np.ndarray:
    """把原生参考棱柱坐标 `(r,s,t)` 映射到物理棱柱单元。

    Args:
        ref_rst: `(n_pts, 3)`，列为 `(r, s, t)`
        cell_nodes: `(6, 3)` 顶点物理坐标，顺序 `(v0,v1,v2,w0,w1,w2)`
            —— 与 `map_prism_to_physical` **完全相同**的约定（v 为底面
            三角形，w_i 在 v_i 正上方）。顺序不一致会静默给出翻转/扭曲
            的单元，所以这里刻意复用同一个顶点约定与同一个重心坐标实现。

    Returns:
        `(n_pts, 3)` 物理坐标。
    """
    from autoflowcfd.grid.curved_mapping.curved_mapping import tri_barycentric

    ref_rst = np.asarray(ref_rst, dtype=np.float64)
    if cell_nodes.shape != (6, 3):
        raise ValueError(
            f"棱柱顶点数组形状 {cell_nodes.shape} 应为 (6, 3)")
    r, s, t = ref_rst[:, 0], ref_rst[:, 1], ref_rst[:, 2]
    l1, l2, l3 = tri_barycentric(r, s)
    bottom = (l1[:, None] * cell_nodes[0][None, :]
              + l2[:, None] * cell_nodes[1][None, :]
              + l3[:, None] * cell_nodes[2][None, :])
    top = (l1[:, None] * cell_nodes[3][None, :]
           + l2[:, None] * cell_nodes[4][None, :]
           + l3[:, None] * cell_nodes[5][None, :])
    tt = t[:, None]
    return 0.5 * (1.0 - tt) * bottom + 0.5 * (1.0 + tt) * top


def native_prism_exact_jacobian(ref_rst: np.ndarray,
                                cell_nodes: np.ndarray) -> np.ndarray:
    """直边棱柱在**原生**参考坐标下的解析精确雅可比。

    Returns:
        `(n_pts, 3, 3)`，`J[:, :, m] = d(phys)/d(xi_m)`，m=0,1,2 对应
        `r, s, t`。

    解析求导而不是用谱微分矩阵作用在 `sps_coords` 上：后者会把算子的
    舍入放大进度量，而度量的误差直接进自由流保持性（坍缩棱柱那 ~1e-9
    实测严格等于 `eps * max|D| / det(J)`）。与
    `curved_mapping_exact_jacobian.py::prism_exact_jacobian` 同一条理由、
    同一个几何映射，只是自变量是原生坐标（见本节顶部说明）。
    """
    from autoflowcfd.grid.curved_mapping.curved_mapping import tri_barycentric

    ref_rst = np.asarray(ref_rst, dtype=np.float64)
    if cell_nodes.shape != (6, 3):
        raise ValueError(
            f"棱柱顶点数组形状 {cell_nodes.shape} 应为 (6, 3)")
    r, s, t = ref_rst[:, 0], ref_rst[:, 1], ref_rst[:, 2]
    p0, p1, p2, p3, p4, p5 = cell_nodes

    # 重心坐标对 (r,s) 的偏导是常数：
    #   dl1/dr = -1/2, dl2/dr = +1/2, dl3/dr = 0
    #   dl1/ds = -1/2, dl2/ds = 0,    dl3/ds = +1/2
    d_bottom_dr = 0.5 * (p1 - p0)
    d_top_dr = 0.5 * (p4 - p3)
    d_bottom_ds = 0.5 * (p2 - p0)
    d_top_ds = 0.5 * (p5 - p3)

    l1, l2, l3 = tri_barycentric(r, s)
    bottom = (l1[:, None] * p0[None, :] + l2[:, None] * p1[None, :]
              + l3[:, None] * p2[None, :])
    top = (l1[:, None] * p3[None, :] + l2[:, None] * p4[None, :]
           + l3[:, None] * p5[None, :])

    n = ref_rst.shape[0]
    jac = np.empty((n, 3, 3))
    half_m = 0.5 * (1.0 - t)[:, None]
    half_p = 0.5 * (1.0 + t)[:, None]
    jac[:, :, 0] = half_m * d_bottom_dr[None, :] + half_p * d_top_dr[None, :]
    jac[:, :, 1] = half_m * d_bottom_ds[None, :] + half_p * d_top_ds[None, :]
    jac[:, :, 2] = 0.5 * (top - bottom)
    return jac


def build_native_prism_modal_filter(order: int) -> np.ndarray:
    """原生棱柱模态滤波矩阵，形状 `(n_sps, n_sps)`。

    归一化判据 `eta = max(i+j, k) / order`：棱柱基是**张量积**
    `psi_ij(r,s) * P_k(t)`，两个因子各自有自己的"逼近上限"——三角形因子
    的阶数是总阶数 `i+j`（受限 PKD 模态集就是按 `i+j<=order` 定义的），
    挤出因子的阶数是 `k`。某一个因子逼近 `order` 就意味着该模态处于那个
    方向插值多项式的最高阶、数值噪声主导区间。

    与坍缩棱柱 `build_prism_modal_filter` 的 `max(i,j,k)/order` 是**同一个
    语义**（"某一根轴自己的索引逼近 order"），只是原生基里三角形那两个
    方向不是独立的轴、它们合起来受一个总阶数约束，所以取 `i+j`。

    **不能用总阶数 `(i+j+k)/(2*order)`**：那样顶模态的 eta 只有 1，但
    `i+j=order, k=0` 这种"三角形方向已经到顶、挤出方向还是常数"的模态
    eta 只有 0.5，会被当成充分解析的低阶模态放过——而它恰恰是被三角化
    的那两个轴上的最高阶模态，也就是伪横流所在的那一支。同时这个归一化
    会让 `legacy`/`project` 档不再"恰好削掉最高阶"，把那两档已经标定好的
    语义（见 `modal_filter.py` 模块文档）一起改掉。

    衰减公式、`off` 档短路与 sigma 的选取全部复用
    `modal_filter.py::assemble_modal_filter`（一份实现服务全部基）。
    """
    if order == 0:
        return np.eye(1)

    from ..modal_filter import assemble_modal_filter

    ref = build_native_prism_nodes(order)
    V, _, _, _ = build_native_prism_vandermonde(order, ref)
    modes = restricted_prism_modes(order)
    etas = [max(i + j, k) / order for (i, j, k) in modes]
    return assemble_modal_filter(V, etas)
