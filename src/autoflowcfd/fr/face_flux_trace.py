"""AutoFlowCFD V2.0 - 体积通量多项式在面通量点上的法向迹（FR 修正项的本侧通量）。

## 为什么需要

FR 修正项在每个面通量点上取 `F_common - F_D`，`F_D` 必须是**体积散度所微分的那个
通量多项式**在该点的法向值：参考单元上 `∫ div F_D = ∮ F_D·n`，修正项减去的正是这个
边界积分，两者抵消后单元积分只剩 `∮ F_common·n`，相邻单元相消——离散守恒。若本侧
改用"外插状态上重算的通量" `a·F(Q_fp)`，它与体积通量多项式的迹只差混叠量，守恒就
只成立到混叠量级（封闭对称盒子能量积分相对 6e-9，P1）。

## 迹算子

过积分下体积散度微分的是细层逆变通量 `F~ = adj(J) F` 在细点上的插值多项式。按
Nanson 公式，物理面积矢量 `a = adj^T a_ref`（`a_ref` 是参考单元上的面积矢量，J=I 时
就是 adj 行本身），所以

    a·F = a_ref·F~     ->     迹 F_D·a 在第 i 个通量点 = Σ_q Σ_m a_ref[i,m] L_q(ξ_i) F~[q,m]

`L_q` 是细层节点 Lagrange 基、`ξ_i` 是该面通量点的参考坐标（与面外插矩阵同一组点、
同一顺序）。`a_ref` 与生产几何里 `owner_adj_row_exact` 同一套公式在参考单元上求得：
四面体用同一个叉积函数（参考顶点），棱柱用同一组外向余向量与 Duffy 因子——约定只有
一个来源。`a_ref` 只依赖通量点的坍缩坐标 `b`，与坍缩顶点槽位无关（平面三角面，循环
轮换不改变面积元）。

## 并进体积算子（`build_lifted_divergence`）

每个单元的每个面恰有一条"本单元为 primary 侧"的面记录（构造期核查，见
`core/fr_operators/face_kernels.py`），所以修正项里减去的本侧迹
`Σ_面 lift (w ⊙ Tn F~)` 是纯单元内的线性运算，与体积散度合成一个算子

    K = f2c · D_fine - Σ_面 lift_面 · diag(w) · Tn_面          (n_sps, n_fine, 3)

体积项 `-(K F~)/det J`，界面核只施加公共通量 `lift (w ⊙ |a| F_common)`。

## 按坍缩顶点槽位组合各一份

三角形面的通量点集随坍缩顶点槽位（`fr/triangle_apex.py`）变化，迹与提升都在**该面
实际使用的通量点**上求积，所以一个单元的 `K` 由它各三角形面的槽位组合决定：四面体
`3^4 = 81` 种、棱柱（两个封盖）`3^2 = 9` 种，组合编号见
`triangle_apex.k_combo_ids`（P3 全部约 13 MB）。

**不能**把迹项改成与通量点无关的精确面积分（那样每类单元只要一个 `K`，就是 DG
弱形式体积项）：体积分部积分用精确面积分、公共通量却用通量点求积，两次分部积分的
面求积不一致，离散分部积分恒等式不再成立。实测（棱柱槽道 SST、隐式 P3，140 步）：

    强形式（迹在通量点上求积）   105 步收敛到 1e-9
    精确面积分（弱形式）         140 步只降到 0.44，残差振荡

封闭盒子守恒两者都到机器精度——守恒只要求两侧通量点重合，稳定性要求迹与公共通量
同一组求积点。
"""

import numpy as np

from .quadrature_points import gauss_legendre
from .triangle_apex import N_TRI_SLOTS, k_combo_slots

#: 参考四面体顶点（`map_native_tet_to_physical` 的 p0..p3 顺序）。
_REF_TET_VERTICES = np.array([[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])


def build_tet_face_flux_trace(order: int, over_order: int) -> np.ndarray:
    """四面体四个面、三个槽位的迹算子 `(4, 3, (order+1)^2, n_fine, 3)`。"""
    from .face_flux_points.exact_normal import _native_tet_adj_row_batched
    from .native_tet.basis import build_native_tet_operators, native_tet_face_points, restricted_tet_modes
    from .native_tet.overintegration import _native_modal_vandermonde

    ref_fine, _ = build_native_tet_operators(over_order)
    modes = restricted_tet_modes(over_order)
    V_inv = np.linalg.inv(_native_modal_vandermonde(ref_fine, modes))
    sps_1d, _ = gauss_legendre(order + 1)
    out = np.empty((4, N_TRI_SLOTS, (order + 1) ** 2, ref_fine.shape[0], 3))
    for v in range(4):
        a_ref = _native_tet_adj_row_batched(v, _REF_TET_VERTICES[None], order + 1, sps_1d)[0]
        for s in range(N_TRI_SLOTS):
            L = _native_modal_vandermonde(native_tet_face_points(order, v, s), modes) @ V_inv   # (n_fp, n_fine)
            out[v, s] = L[:, :, None] * a_ref[:, None, :]
    return out


def build_prism_face_flux_trace(order: int, over_order: int) -> np.ndarray:
    """棱柱五个面的迹算子 `(5, 3, (order+1)^2, n_fine, 3)`；侧四边形只有槽位 0，其余槽位
    填 NaN（误用立刻暴露）。"""
    from .native_prism.basis import build_native_prism_operators, build_native_prism_vandermonde
    from .native_prism.face import PRISM_FACE_IDS, native_prism_face_points, reference_face_area_vectors

    ref_fine, _ = build_native_prism_operators(over_order)
    V_inv = np.linalg.inv(build_native_prism_vandermonde(over_order, ref_fine)[0])
    out = np.full((5, N_TRI_SLOTS, (order + 1) ** 2, ref_fine.shape[0], 3), np.nan)
    for fid in PRISM_FACE_IDS:
        a_ref = reference_face_area_vectors(order, fid)
        for s in (range(N_TRI_SLOTS) if fid < 2 else (0,)):
            L = build_native_prism_vandermonde(over_order, native_prism_face_points(order, fid, s))[0] @ V_inv
            out[fid, s] = L[:, :, None] * a_ref[:, None, :]
    return out


def face_reference_weights(order: int) -> np.ndarray:
    """面通量点的参考求积权重 `(n1d^2,)`：`[-1,1]^2` 张量积 Gauss-Legendre（与面通量点
    同一组 `gauss_legendre(order+1)`；面几何的 `ref_area_weight` 就是它）。"""
    _, w = gauss_legendre(order + 1)
    return np.ascontiguousarray(np.outer(w, w).ravel())


def build_lifted_divergence(f2c: np.ndarray, D_fine: np.ndarray, lift_by_slot, traces: np.ndarray,
                            order: int, kind: str) -> np.ndarray:
    """全部槽位组合的 `K = f2c·D_fine - Σ_面 lift diag(w) Tn`，`(n_combo, n_sps, n_fine, 3)`。

    Args:
        f2c: `(n_sps, n_fine)` 细->粗 L2 投影（粗轴已填充）。
        D_fine: `(n_fine, n_fine, 3)` 细层微分矩阵。
        lift_by_slot: `lift_by_slot(面, 槽位)` -> 已填充的 `(n_sps, n_fp)` DG 提升矩阵。
        traces: `build_*_face_flux_trace` 的结果 `(n_faces, 3, n_fp, n_fine, 3)`。
        kind: `"tet"` / `"prism"`（决定组合编号的含义，`triangle_apex.k_combo_slots`）。
    """
    w = face_reference_weights(order)
    base = np.einsum("sq,qtm->stm", f2c, D_fine)
    n_faces = traces.shape[0]
    # 逐（面, 槽位）的提升量只算一次，各组合按自己的槽位取用
    C = {}
    combos = k_combo_slots(kind)                      # (n_combo, n_faces)
    for f in range(n_faces):
        for s in np.unique(combos[:, f]):
            C[f, int(s)] = np.einsum("si,iqm->sqm", lift_by_slot(f, int(s)) * w[None, :], traces[f, int(s)])
    K = np.empty((combos.shape[0],) + base.shape)
    for c, slots in enumerate(combos):
        K[c] = base - sum(C[f, int(s)] for f, s in enumerate(slots))
    return np.ascontiguousarray(K)
