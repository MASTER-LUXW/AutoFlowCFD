"""体单元与三角形面的退化判据（唯一来源：导入外部体网格与生成体网格末尾的硬检查共用）。

判据相对单元 / 面自身尺度（量纲无关），与单元朝向无关（加载高阶网格时 `fix_tet_orientation` /
`fix_prism_orientation` 统一朝向），c = DEGENERATE_RELATIVE_MEASURE，L 为该单元 / 面的最长边：

- 四面体：|V| <= c L^3 即退化；
- 棱柱：六个角点的 Jacobian 行列式（角点出发的两条底 / 顶边与竖边张成的三重积）必须同号且 |det| > c L^3——
  只看体积会漏掉扭曲 / 翻折的棱柱（三个子四面体体积取模后之和照样为正）；
- 三角形面：A <= c L^2 即退化。

正常单元远离这个界：cube_demo 生成的 37.4 万单元混合网格上四面体 |V|/L^3 最小 2.6e-3、棱柱同号角点
|det|/L^3 最小 1.0e-2、面 A/L^2 最小 2.1e-2。
"""

import numpy as np

DEGENERATE_RELATIVE_MEASURE = 1e-12


def _longest_edge(xyz: np.ndarray) -> np.ndarray:
    """(n, nv, 3) 顶点坐标 -> 每个单元 / 面任意两顶点间的最大距离。"""
    nv = xyz.shape[1]
    longest = np.zeros(len(xyz))
    for a in range(nv):
        for b in range(a + 1, nv):
            np.maximum(longest, np.linalg.norm(xyz[:, a] - xyz[:, b], axis=1), out=longest)
    return longest


def tet_relative_measure(node_xyz: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """四面体 |V| / L^3。"""
    p = node_xyz[tets]
    vol = np.einsum('ij,ij->i', p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])) / 6.0
    return np.abs(vol) / _longest_edge(p) ** 3


def prism_relative_measure(node_xyz: np.ndarray, prisms: np.ndarray) -> np.ndarray:
    """棱柱六个角点 Jacobian 行列式 / L^3 的最小值（按第一个角点的符号取向；角点异号时为负）。

    棱柱顶点 (v0, v1, v2, w0, w1, w2)：角点 v_i 的行列式为 det(v_j - v_i, v_k - v_i, w_i - v_i)，角点 w_i 为
    det(w_j - w_i, w_k - w_i, w_i - v_i)，(i, j, k) 取 (0, 1, 2) 的轮换。
    """
    p = node_xyz[prisms]
    v, w = p[:, :3], p[:, 3:]
    dets = []
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        rise = w[:, i] - v[:, i]
        for cap in (v, w):
            dets.append(np.einsum('ij,ij->i', np.cross(cap[:, j] - cap[:, i], cap[:, k] - cap[:, i]), rise))
    dets = np.stack(dets, axis=1)
    return (dets * np.sign(dets[:, :1])).min(axis=1) / _longest_edge(p) ** 3


def degenerate_cells(node_xyz: np.ndarray, conn: np.ndarray) -> np.ndarray:
    """退化（棱柱还包括扭曲 / 翻折）单元的下标；`conn` 每行 4 个节点为四面体、6 个为棱柱。NaN 坐标同样判为退化。"""
    measure = (tet_relative_measure if conn.shape[1] == 4 else prism_relative_measure)(node_xyz, conn)
    return np.flatnonzero(~(measure > DEGENERATE_RELATIVE_MEASURE))


def degenerate_triangles(corners: np.ndarray) -> np.ndarray:
    """(n, 3, 3) 三角形角点 -> 面积不大于 c L^2 的三角形下标。"""
    area = 0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    return np.flatnonzero(~(area > DEGENERATE_RELATIVE_MEASURE * _longest_edge(corners) ** 2))


def require_nondegenerate_cells(node_xyz: np.ndarray, prism_cells: np.ndarray, tet_cells: np.ndarray) -> None:
    """混合网格（棱柱 + 四面体）不含退化单元，否则报错并给出每类前 5 个单元（全局混合编号：棱柱
    [0, n_prism)、四面体其后）的质心与相对量度。

    Raises:
        ValueError: 存在退化单元
    """
    found = []
    for kind, conn, offset, measure in (
            ("棱柱", np.asarray(prism_cells, dtype=np.int64).reshape(-1, 6), 0, prism_relative_measure),
            ("四面体", np.asarray(tet_cells, dtype=np.int64).reshape(-1, 4), len(prism_cells), tet_relative_measure)):
        bad = degenerate_cells(node_xyz, conn)
        if bad.size:
            rel = measure(node_xyz, conn[bad[:5]])
            found.append(f"{kind} {bad.size} 个：" + "、".join(
                f"{offset + int(c)}（质心 {np.round(node_xyz[conn[c]].mean(axis=0), 6).tolist()}，"
                f"相对量度 {rel[i]:.3e}）" for i, c in enumerate(bad[:5])))
    if found:
        raise ValueError(f"体网格含退化单元（相对量度 <= {DEGENERATE_RELATIVE_MEASURE:.0e}）：" + "；".join(found))
