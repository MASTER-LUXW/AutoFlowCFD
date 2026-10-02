"""AutoFlowCFD V2.0 - 三角形面通量点的坍缩顶点槽位（共享面两侧取同一组物理点）。

## 为什么需要

三角形面（四面体的面、原生棱柱的上下封盖）的 `(order+1)^2` 个通量点由坍缩三角形采样
生成：`(r,s) = cube_to_tri_rs(a, b)`、重心坐标 `(l1, l2, l3)` 依次乘面上三个顶点
`(Pa, Pb, Papex)`，`b -> 1` 时坍缩到第三个顶点 `Papex`。交换前两个顶点只是把 Gauss 网格在
`a` 方向镜像（Gauss 点关于 0 对称），点集不变；**换坍缩顶点则点集整体不同**。

此前坍缩顶点取"面上局部编号排第三的顶点"，它取决于单元自己的局部顶点编号——共享一个
三角面的两个单元常常选了不同的顶点（plate_demo 抽样 6.6% 的内部面、四面体通道约 1/5），
两侧的通量点物理位置不重合。各侧在自己的点上对非线性公共通量求积，两侧积分对不上，离散
守恒只到面求积量级（四面体 P1 能量相对 1e-7）。

## 规则

坍缩顶点取**面上全局节点编号最大的顶点**——只依赖面本身，两侧必然一致。面上三个顶点按
单元的局部面顶点顺序 `fv = (v0, v1, v2)` 给出（四面体：排除顶点之外的三个局部顶点升序；
棱柱封盖：`(0,1,2)` / `(3,4,5)`），槽位 `slot` 表示循环轮换

    (Pa, Pb, Papex) = (fv[slot], fv[slot+1], fv[slot+2])     （下标模 3）

`slot = 0` 即旧约定。循环轮换保持定向（棱柱封盖的外向余向量与 Duffy 因子逐点不变）。
四边形面（棱柱侧面）的张量积 Gauss 点集对四边形的全部对称都不变，恒取槽位 0。

槽位只依赖单元连接关系（`cell_triangle_slots`），是每个"单元 x 局部面"的固有属性；
面记录上的槽位、面算子索引都由它派生，不另存第二份规则。
"""

import numpy as np

#: 三角形面的槽位数。
N_TRI_SLOTS = 3


def rotate_triangle(fv, slot: int):
    """`(Pa, Pb, Papex)`：面顶点三元组按槽位循环轮换。"""
    s = int(slot) % N_TRI_SLOTS
    return fv[s], fv[(s + 1) % 3], fv[(s + 2) % 3]


def apex_slots(face_global_ids: np.ndarray) -> np.ndarray:
    """逐面槽位 `(n,)`：使坍缩顶点（轮换后的第三个）是全局编号最大的那个。

    Args:
        face_global_ids: `(n, 3)`，按单元局部面顶点顺序给出的全局节点编号。
    """
    p = np.argmax(np.asarray(face_global_ids), axis=1)
    return ((p + 1) % N_TRI_SLOTS).astype(np.int64)


#: 单元局部面列数：四面体按排除顶点 0~3、棱柱按 face_id 0~4（面编码减去该类单元
#: 编码区间起点）。
N_CELL_FACE_COLUMNS = 5


def cell_triangle_slots(tet_conn, prism_conn) -> np.ndarray:
    """逐单元、逐局部面的槽位 `(n_prism + n_tet, 5)`（单元序：棱柱在前、四面体在后）。

    四面体面 `v` 的顶点是排除 `v` 后的三个局部顶点升序；棱柱封盖 0/1 是 `(0,1,2)` /
    `(3,4,5)`；棱柱侧面与四面体的第 5 列恒 0。连接数组里的节点号必须是**全局**编号
    （分布式下各 rank 的局部编号会让共享面两侧选出不同的坍缩顶点）。
    """
    tet = np.empty((0, 4), np.int64) if tet_conn is None else np.asarray(tet_conn, np.int64)
    prism = np.empty((0, 6), np.int64) if prism_conn is None else np.asarray(prism_conn, np.int64)
    out = np.zeros((prism.shape[0] + tet.shape[0], N_CELL_FACE_COLUMNS), np.int64)
    out[:prism.shape[0], 0] = apex_slots(prism[:, 0:3])
    out[:prism.shape[0], 1] = apex_slots(prism[:, 3:6])
    for v in range(4):
        out[prism.shape[0]:, v] = apex_slots(tet[:, [u for u in range(4) if u != v]])
    return out


def face_record_slots(cell_slots: np.ndarray, cell, code) -> np.ndarray:
    """面记录一侧的槽位：`cell_slots[cell, 局部面列]`（`code` 为原生面编码 6~14）。
    `cell < 0`（边界面的 neighbor 侧）给 0。"""
    cell = np.asarray(cell, np.int64)
    code = np.asarray(code, np.int64)
    col = np.where(code < 10, code - 6, code - 10)
    valid = cell >= 0
    out = np.zeros(cell.shape, np.int64)
    out[valid] = cell_slots[cell[valid], col[valid]]
    return out


#: 无粘体积算子 `K` 的槽位组合数（`fr/face_flux_trace.py`）：四面体 4 个三角面、棱柱
#: 2 个三角封盖。
N_K_COMBOS = {"tet": N_TRI_SLOTS ** 4, "prism": N_TRI_SLOTS ** 2}


def k_combo_slots(kind: str) -> np.ndarray:
    """组合编号 -> 各局部面槽位，`(n_combo, n_faces)`。编号 `Σ_f slot_f 3^f`（只数三角形
    面：四面体面 0~3、棱柱封盖 0~1；棱柱侧面恒 0）。`k_combo_ids` 是它的逆。"""
    n_tri, n_faces = (4, 4) if kind == "tet" else (2, 5)
    combo = np.arange(N_K_COMBOS[kind])
    out = np.zeros((combo.size, n_faces), np.int64)
    for f in range(n_tri):
        out[:, f] = (combo // N_TRI_SLOTS ** f) % N_TRI_SLOTS
    return out


def k_combo_ids(cell_slots: np.ndarray, n_prism: int):
    """逐单元的 `K` 组合编号 `(prism_ids (n_prism,), tet_ids (n_tet,))`（`cell_triangle_slots`
    的结果按 `k_combo_slots` 的编号规则折叠）。"""
    cs = np.asarray(cell_slots, np.int64)
    weights = N_TRI_SLOTS ** np.arange(4)
    prism = cs[:n_prism, :2] @ weights[:2]
    tet = cs[n_prism:, :4] @ weights
    return np.ascontiguousarray(prism), np.ascontiguousarray(tet)
