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

from .collapsed_basis import grad_jacobi_polynomial, jacobi_polynomial
from .native_triangle_basis import (
    restricted_tri_modes,
    rs_to_ab,
    simplex2d_grad,
    simplex2d_value,
    warp_blend_nodes_2d,
)
from .quadrature_points import gauss_legendre


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
    a, b = rs_to_ab(r, s)
    modes = restricted_prism_modes(order)
    n_pts = len(r)
    V = np.empty((n_pts, len(modes)))
    Vr = np.empty_like(V)
    Vs = np.empty_like(V)
    Vt = np.empty_like(V)

    # 三角形部分按 (i,j) 缓存：每个 (i,j) 对应 order+1 个 k，值相同。
    tri_cache = {}
    for m, (i, j, k) in enumerate(modes):
        key = (i, j)
        if key not in tri_cache:
            psi = simplex2d_value(a, b, i, j)
            dpsi_dr, dpsi_ds = simplex2d_grad(a, b, i, j)
            tri_cache[key] = (psi, dpsi_dr, dpsi_ds)
        psi, dpsi_dr, dpsi_ds = tri_cache[key]
        lk = jacobi_polynomial(t, 0.0, 0.0, k)
        dlk = grad_jacobi_polynomial(t, 0.0, 0.0, k)
        V[:, m] = psi * lk
        Vr[:, m] = dpsi_dr * lk
        Vs[:, m] = dpsi_ds * lk
        Vt[:, m] = psi * dlk
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

    from .diff_matrix_consistency import enforce_constant_annihilation

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
