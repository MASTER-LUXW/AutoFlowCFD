"""AutoFlowCFD V2.0 - 原生棱柱基的面算子：面通量点、体积->面外插、DG 提升。

从 `native_prism_basis.py` 拆出（单文件 400 行上限的项目约定）。与
`native_simplex_basis.py` 里四面体那三个同名构造是**同一套推导**，只是
参考单元换成 `三角形 × 直线`：

    build_native_tet_boundary_extrap  ->  build_native_prism_boundary_extrap
    build_native_tet_lift             ->  build_native_prism_lift
    _native_mode_norm_squared         ->  native_prism_mode_norm_squared

## 五个面与面通量点

参考棱柱 = 参考三角形 `{r>=-1, s>=-1, r+s<=0}` × `t in [-1,1]`，三角形
顶点 `V0=(-1,-1)`、`V1=(1,-1)`、`V2=(-1,1)`（与
`grid/curved_mapping/curved_mapping.py::tri_barycentric` 的
`l1/l2/l3 <-> V0/V1/V2` 约定一致，复用它、不另立一套）。

    face_id  几何                 通量点构造
      0      底三角形 t=-1        坍缩三角形采样 (r,s) x {t=-1}
      1      顶三角形 t=+1        同上 x {t=+1}
      2      侧四边形（排除 V0）  三角形边 V1-V2（斜边 r+s=0） x t
      3      侧四边形（排除 V1）  三角形边 V0-V2（r=-1 边）    x t
      4      侧四边形（排除 V2）  三角形边 V0-V1（s=-1 边）    x t

**每个面恒有 `n1d^2 = (order+1)^2` 个通量点**，与坍缩棱柱方案、与 native
四面体方案完全一致 —— `_KernelFaceData` 的 flat 数组假设全网格所有面的
通量点数统一（见 `native_simplex_basis.py::build_native_tet_boundary_
extrap` 文档"二·五"节记录的那条必要修正）。这里不需要任何填充：
  * 两个三角形封盖用**与 native 四面体三角形面相同**的坍缩三角形采样
    网格（`cube_to_tri_rs` 作用在张量积 Gauss-Legendre 方格上），
    天然 `n1d^2` 个点；
  * 三个侧四边形本来就是张量积面，`n1d x n1d` 也恰好是 `n1d^2`。

侧面用"排除的三角形顶点"当键，与四面体用"排除的局部顶点"同一个命名
习惯；封盖用 `t` 的符号区分。两者合成 0~4 的单一整数键，让消费方
（残差 kernel 的面分派）只需要一个查找表。

## 为什么要这一套（病根）

坍缩棱柱基的 `max|D|` 随阶数爆炸（P1 2.05 -> P3 560.1），实测后果是
自由流保持性只有 ~1e-9、以及只在被三角化的两个参考轴上长出的伪横流
（P1 饱和、P2 无界增长导致发散）。完整数据见
`native_triangle_basis.py` 与 `native_prism_basis.py` 的模块文档。
"""

from typing import Dict, List, Tuple

import numpy as np

from .native_prism_basis import (
    build_native_prism_nodes,
    build_native_prism_vandermonde,
    restricted_prism_modes,
)
from .quadrature_points import gauss_legendre

__all__ = [
    "PRISM_FACE_IDS",
    "native_prism_face_points",
    "native_prism_face_points_physical",
    "build_native_prism_boundary_extrap",
    "build_native_prism_lift",
    "native_prism_mode_norm_squared",
]

#: 五个面的整数键（见模块文档的表）。
PRISM_FACE_IDS: Tuple[int, ...] = (0, 1, 2, 3, 4)

#: 参考三角形顶点，下标与 `tri_barycentric` 的 `l1/l2/l3` 对应。
_TRI_VERTS = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]])

#: 侧面 face_id -> 被排除的三角形顶点下标。
_SIDE_EXCLUDED = {2: 0, 3: 1, 4: 2}


def native_prism_face_points(order: int, face_id: int) -> np.ndarray:
    """某个面上 `(order+1)^2` 个通量点的**参考棱柱坐标** `(r,s,t)`。

    Args:
        order: 多项式阶数
        face_id: 0~4，见模块文档的表

    Returns:
        `(n1d*n1d, 3)`，列为 `(r, s, t)`。

    Raises:
        ValueError: `face_id` 不在 0~4。
    """
    if face_id not in PRISM_FACE_IDS:
        raise ValueError(
            f"face_id={face_id} 不合法：参考棱柱只有 5 个面（0/1 是两个"
            f"三角形封盖，2/3/4 是三个侧四边形），见模块文档的表")

    n1d = order + 1
    g_1d, _ = gauss_legendre(n1d)

    if face_id in (0, 1):
        # 三角形封盖：与 native 四面体的三角形面**同一套**坍缩三角形
        # 采样（一份实现、一个事实来源）。
        from ..grid.curved_mapping.curved_mapping import cube_to_tri_rs

        g1, g2 = np.meshgrid(g_1d, g_1d, indexing="ij")
        r, s = cube_to_tri_rs(g1.ravel(), g2.ravel())
        t = np.full(r.shape, -1.0 if face_id == 0 else 1.0)
        return np.column_stack([r, s, t])

    # 侧四边形：三角形的一条边 x 挤出方向，张量积。
    excluded = _SIDE_EXCLUDED[face_id]
    edge = [v for v in range(3) if v != excluded]
    va, vb = _TRI_VERTS[edge[0]], _TRI_VERTS[edge[1]]
    lam, tt = np.meshgrid(g_1d, g_1d, indexing="ij")
    lam = lam.ravel()
    tt = tt.ravel()
    # 边上的线性参数化（lam=-1 -> va, lam=+1 -> vb）
    w_a = 0.5 * (1.0 - lam)
    w_b = 0.5 * (1.0 + lam)
    rs = w_a[:, None] * va[None, :] + w_b[:, None] * vb[None, :]
    return np.column_stack([rs[:, 0], rs[:, 1], tt])


def native_prism_face_points_physical(order: int, face_id: int,
                                      cell_nodes: np.ndarray) -> np.ndarray:
    """某个面上通量点的**物理**坐标。

    只是 `native_prism_face_points` 与
    `native_prism_basis.map_native_prism_to_physical` 的合成 —— 单独给出
    是为了让面几何构造方（`fr/face_flux_points.py`）有一个与
    `native_tet_face_points_physical` 对称的入口，而不是各处自己拼。
    """
    from .native_prism_basis import map_native_prism_to_physical

    return map_native_prism_to_physical(
        native_prism_face_points(order, face_id), cell_nodes)


def _face_vandermondes(order: int, face_id: int):
    """`build_native_prism_boundary_extrap`/`build_native_prism_lift` 共用的
    准备步骤：体积节点与该面通量点各自的模态取值 Vandermonde。

    两者用同一组几何量、只是矩阵组合方式不同（外插 vs 提升），提取成
    共享函数避免"改一处忘改另一处"—— 与四面体侧
    `_native_face_value_vandermondes` 同一个理由。

    Returns:
        `(V_sps, V_fp, modes)`：`V_sps (n_sps, n_modes)`、
        `V_fp (n1d^2, n_modes)`、`modes` 与两个矩阵的列序一致。
    """
    ref_sps = build_native_prism_nodes(order)
    V_sps, _, _, _ = build_native_prism_vandermonde(order, ref_sps)
    fp = native_prism_face_points(order, face_id)
    V_fp, _, _, _ = build_native_prism_vandermonde(order, fp)
    return V_sps, V_fp, restricted_prism_modes(order)


def build_native_prism_boundary_extrap(order: int,
                                       face_id: int) -> np.ndarray:
    """体积 -> 自身某个面的外插矩阵，`E @ Q_volume_nodal` 给出该面通量点
    上的取值。

    与 `collapsed_basis.py::build_collapsed_boundary_extrap` 同样的用途与
    消费方式，但只依赖 `(order, face_id)`、与具体物理单元形状无关：外插是
    **参考空间**里的运算，同阶数全网格共享一份（与 native 四面体那条
    同一个性质）。

    Returns:
        `E`：`(n1d^2, n_sps)`，`n1d=order+1`，
        `n_sps=(order+1)^2(order+2)/2`。
    """
    from scipy.linalg import lu_factor, lu_solve

    V_sps, V_fp, _ = _face_vandermondes(order, face_id)
    lu = lu_factor(V_sps.T)
    return lu_solve(lu, V_fp.T).T


def native_prism_mode_norm_squared(i: int, j: int, k: int) -> float:
    """棱柱模态的参考体积模方 `N(i,j,k) = ∫_ref phi_ijk^2 dV`（闭式）。

    棱柱基是张量积 `phi_ijk(r,s,t) = psi_ij(r,s) * P_k^{0,0}(t)`，所以模方
    **因式分解**成"三角形部分 × 直线部分"：

        N(i,j,k) = N_tri(i,j) * N_line(k)

    直线部分是标准 Legendre 模方 `N_line = 2/(2k+1)`。

    三角形部分（与 `_native_mode_norm_squared` 的三维推导同一条路，降一维）：
    `psi_ij = sqrt(2) P_i^{0,0}(a) P_j^{2i+1,0}(b) (1-b)^i`，Duffy 变换的
    Jacobian 是 `dr ds = (1-b)/2 da db`，代入后前缀 `(sqrt2)^2 * 1/2 = 1`
    恰好抵消，两个因子各自成为标准加权 Jacobi 模方：

        N_tri = gamma_i^{0,0} * gamma_j^{2i+1,0}
              = [2/(2i+1)] * [2^{2i+1}/(i+j+1)]
              = 2^{2i+2} / [(2i+1)(i+j+1)]

    其中 `gamma_n^{alpha,0} = ∫ P_n^{alpha,0}(x)^2 (1-x)^alpha dx
    = 2^{alpha+1}/(2n+alpha+1)`（A&S 22.2.1 在 beta=0 下化简）。合起来：

        N(i,j,k) = 2^{2i+3} / [(2i+1)(i+j+1)(2k+1)]

    **已用独立三重数值积分逐模态核对**（见
    `tests/unit/test_native_prism_face.py::TestModeNormsAgainstQuadrature`），
    不是推导完就假设成立 —— 与四面体那份闭式解同一条验收标准。
    """
    return 2.0 ** (2 * i + 3) / ((2 * i + 1) * (i + j + 1) * (2 * k + 1))


def build_native_prism_lift(order: int, face_id: int) -> np.ndarray:
    """某个面的 DG 提升算子（"lift"）：把该面逐通量点的通量跳跃
    （`F_common - F_own`）提升成对体积节点自由度的修正贡献。

    这是坍缩坐标方案里"1D Radau/VCJH 修正函数 `g_left`/`g_right` +
    `_distribute_point`"对原生棱柱基的**唯一正确推广**。参考空间的代数
    推导与四面体侧 `build_native_tet_lift` **逐字相同**（那边有完整推导），
    这里只重述结论与本单元的特殊之处：

        Lift_ref = V @ diag(1/N) @ B^T

    `V[node, mode]` 是体积节点 Vandermonde，`B[p, mode]` 是面通量点
    Vandermonde，`N[mode]` 是模态参考模方（`native_prism_mode_norm_
    squared`）。物理面积权重与 `1/det_j` **不在这里**：前者按面才知道
    （`true_area_weight`），后者由调用方复用同一次除法，所以 `Lift_ref`
    只依赖 `(order, face_id)`、同阶数全网格共享一份。

    ## 棱柱与四面体的一处实质差别（以及它为什么不引入新近似）

    直边四面体的 `det_j` 是**逐单元常数**，于是 `M_phys = det_j * M_ref`
    精确成立、`Lift_ref/det_j` 就是精确的 `M_phys^{-1}` 作用结果。

    棱柱不一样：即便六个顶点都是直边，雅可比一般**随点变化** —— 只有当
    顶面是底面的**纯平移**（右棱柱）时三列偏导才全部恒定。真实边界层网格
    沿壁面法向逐层挤出，相邻层法向不严格平行，所以一般情形下 `det_j`
    随点变化，`M_phys != det_j * M_ref`。

    **这不意味着这里要存逐单元质量矩阵**（P2 下 18x18/单元 x 36 万单元
    接近 1 GB，不可接受）。FR 的强形式**不需要**质量矩阵：格式是

        du/dt = -(1/det_j(s)) [ sum_m D_m (adj_m . F) + sum_f C_f(xi) * (w_f ⊙ jump) ]

    其中 `C_f` 是**参考空间**里固定的修正函数、`w_f` 是物理面积权重、
    `1/det_j` **逐 SP**。生产中的坍缩棱柱路径用的正是这个形式（参考空间的
    `g_left`/`g_right` + 逐 SP 的 `det_jacs[oc, s]` 除法），所以把
    `Lift_ref` 当 `C_f` 用、配逐 SP 的 `1/det_j`，与已长期验证的现有实现
    是**同一类形式**，没有引入新的近似 —— 换的只是修正函数本身（1D
    Radau/VCJH -> DG 提升），因为原生基没有"坍缩计算方向"、1D 修正函数
    沿某一轴分布这个概念不适用。

    常度量棱柱（右棱柱）上它额外地**恰好**等于精确的 `M_phys^{-1}`。

    接线时必须逐点核对的一条：调用方对棱柱要用**逐 SP** 的 `det_j`，不能
    像四面体那样取一个标量广播。

    消费方式：
        `correction[s, v] = -Lift_ref[s, :] @ (w[:, None] * jump)[:, v]
                            / det_j[s]`

    Returns:
        `Lift_ref`：`(n_sps, n1d^2)`。
    """
    V_sps, V_fp, modes = _face_vandermondes(order, face_id)
    inv_norms = np.array(
        [1.0 / native_prism_mode_norm_squared(i, j, k) for (i, j, k) in modes])
    return V_sps @ (inv_norms[:, None] * V_fp.T)


def build_all_native_prism_face_operators(
        order: int) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """一次构造全部 5 个面的 `(外插, 提升)` 两组算子。

    Returns:
        `(extrap, lift)`，两个以 `face_id` 为键的字典。
    """
    extrap = {f: build_native_prism_boundary_extrap(order, f)
              for f in PRISM_FACE_IDS}
    lift = {f: build_native_prism_lift(order, f) for f in PRISM_FACE_IDS}
    return extrap, lift


# ---------------------------------------------------------------------------
# 立方体面 <-> 原生棱柱面 的对应表
# ---------------------------------------------------------------------------
#
# 现有面连接/残差 kernel 用 `(axis, side)`（"立方体面"）给单元的面编号，
# 因为坍缩棱柱的参考单元就是个立方体（其中 `b=+1` 退化为一条侧棱、不携带
# 独立通量信息，见 `grid/curved_mapping/curved_mapping.py::PRISM_CUBE_FACES`）。
# 原生棱柱的参考单元不是立方体，它的 5 个面用本模块的 `face_id` 编号。
# 两者之间是**精确双射**：
#
#     立方体面        物理面（PRISM_CUBE_FACES 的局部顶点）   face_id
#     c=-1 (2,-1)     (0,1,2)  底面三角形                      0
#     c=+1 (2,+1)     (3,4,5)  顶面三角形                      1
#     a=+1 (0,+1)     (1,2,5,4) 侧面（对边 v1v2，排除 v0）      2
#     a=-1 (0,-1)     (0,2,5,3) 侧面（对边 v0v2，排除 v1）      3
#     b=-1 (1,-1)     (0,1,4,3) 侧面（对边 v0v1，排除 v2）      4
#     b=+1 (1,+1)     退化（侧棱）                             无
#
# 这张表是**唯一事实来源**：任何"按 (axis,side) 取原生棱柱算子"的消费点
# 都必须走 `cube_face_to_native_prism_face`，不许各处自己判断 —— 面 id 配错
# 不会报错，只会静默地对某个面用错外插/提升矩阵（与当年多 GPU"四面体拿到
# 棱柱矩阵"完全同一类缺陷）。对应关系已用几何方式验证（见
# `tests/unit/test_native_prism_face.py::TestCubeFaceMapping`：把原生面
# 通量点映射到物理空间，检验它们确实落在该立方体面所对应的那组物理顶点
# 张成的平面上）。

#: `(axis, side)` -> `face_id`。`(1, +1.0)` 不在表里：它是退化面。
_CUBE_FACE_TO_NATIVE: Dict[Tuple[int, float], int] = {
    (2, -1.0): 0,
    (2, 1.0): 1,
    (0, 1.0): 2,
    (0, -1.0): 3,
    (1, -1.0): 4,
}

#: 反向表，供诊断/测试用。
NATIVE_PRISM_FACE_TO_CUBE_FACE: Dict[int, Tuple[int, float]] = {
    v: k for k, v in _CUBE_FACE_TO_NATIVE.items()
}


def cube_face_to_native_prism_face(axis: int, side: float) -> int:
    """把 `(axis, side)` 立方体面编号换成原生棱柱的 `face_id`。

    Raises:
        ValueError: `(1, +1)` —— 坍缩棱柱参考立方体上那个**退化**面
            （坍缩成一条侧棱）。它不携带独立通量信息，原生棱柱里根本
            没有对应的面。静默返回某个 face_id 会让一条不存在的面参与
            界面项组装，所以这里硬失败。
    """
    key = (int(axis), float(side))
    if key not in _CUBE_FACE_TO_NATIVE:
        raise ValueError(
            f"立方体面 (axis={axis}, side={side:+.0f}) 没有对应的原生棱柱面。"
            f"合法的 5 个面见本模块的对应表；(1, +1) 是坍缩参考立方体上的"
            f"退化面（坍缩成一条侧棱），不携带独立通量信息。")
    return _CUBE_FACE_TO_NATIVE[key]
