"""三棱柱侧面四边形的对角线与 3 四面体拆分（唯一实现：面提取、BL 层转四面体、棱柱降级与空腔修补共用）。

规则（Dompierre 等，"How to subdivide pyramids, prisms and hexahedra into tetrahedra"）：每个侧面四边形沿过其
四个角点中**全局编号最小**的顶点的对角线拆分。它只依赖四边形自己的四个编号，与棱柱的底 / 顶指定、存储顺序都
无关，共享侧面的两个单元必然选同一条对角线；对棱柱这三条对角线总能给出 3 个四面体的分解（全局最小顶点同时
是两条对角线的端点）。

2026-10-20 以前的规则是"底面顶点按编号排序后，小号底点连大号对应顶点"：只在相邻棱柱对共享侧面的底 / 顶分配
一致时才协调（本项目挤出的 BL 满足：上层编号恒大于下层），导入的外部体网格（棱柱底 / 顶方向任意）上相邻棱柱
可能选不同对角线，共享侧面变成四个单侧"边界"三角形。在上层编号大于下层的网格上两条规则逐位相同
（cube_demo 生成网格实测 0 处不同），输出的面与四面体顺序也与此前相同。

规范化：顶面最小编号严格小于底面时上下互换（只是换个方向看这个棱柱，体积与拆分不变），再把底面按编号排序
（顶面随之重排，w_i 仍在 v_i 正上方），于是 v0 是全局最小顶点：四边形 (v0,v1,w1,w0)、(v0,v2,w2,w0) 的对角线
是 v0-w1、v0-w2；四边形 (v1,v2,w2,w1) 在 min(v1,w2) <= min(v2,w1) 时取 v1-w2，否则 v2-w1（取等只在 BL 停长、
上下层节点重合时出现，取 v1-w2 与此前一致）。
"""

from typing import Tuple

import numpy as np


def _canonical(prisms: np.ndarray) -> Tuple[np.ndarray, ...]:
    """(n, 6) 棱柱 -> 规范化后的 v0, v1, v2, w0, w1, w2 与四边形 (v1,v2,w2,w1) 取 v1-w2 对角线的掩码。"""
    prisms = np.asarray(prisms, dtype=np.int64).reshape(-1, 6)
    bottom, top = prisms[:, :3], prisms[:, 3:]
    flip = (top.min(axis=1) < bottom.min(axis=1))[:, None]
    bottom, top = np.where(flip, top, bottom), np.where(flip, bottom, top)
    order = np.argsort(bottom, axis=1)
    rows = np.arange(len(prisms))[:, None]
    v, w = bottom[rows, order], top[rows, order]
    v0, v1, v2, w0, w1, w2 = v[:, 0], v[:, 1], v[:, 2], w[:, 0], w[:, 1], w[:, 2]
    diag_v1w2 = np.minimum(v1, w2) <= np.minimum(v2, w1)
    return v0, v1, v2, w0, w1, w2, diag_v1w2


def prism_face_triangles(prisms: np.ndarray) -> np.ndarray:
    """(n, 6) 棱柱 -> (n, 8, 3) 边界三角形：底、顶、三个侧面各拆成两个（侧面顺序 (v0,v1)、(v1,v2)、(v0,v2)）。

    停长 / 坍缩的棱柱（v_i 与 w_i 编号相同）产生编号重复的退化三角形，由调用方按需过滤。
    """
    v0, v1, v2, w0, w1, w2, d = _canonical(prisms)
    return np.stack([
        np.stack([v0, v1, v2], axis=1),
        np.stack([w0, w1, w2], axis=1),
        np.stack([v0, v1, w1], axis=1),
        np.stack([v0, w0, w1], axis=1),
        np.stack([v1, v2, np.where(d, w2, w1)], axis=1),
        np.stack([np.where(d, v1, v2), w1, w2], axis=1),
        np.stack([v0, v2, w2], axis=1),
        np.stack([v0, w0, w2], axis=1),
    ], axis=1)


def split_prisms_to_tets(prisms: np.ndarray) -> np.ndarray:
    """(n, 6) 棱柱 -> (3n, 4) 四面体，按块排列（前 n 行是每个棱柱的第一个子四面体，依此类推），未定向。

    T3 = (v0, w0, w1, w2)；其余是以 v0 为顶点、底为四边形 (v1, v2, w2, w1) 的四棱锥沿其对角线拆成的两个：
    v1-w2 时 (v0, v1, v2, w2)、(v0, v1, w1, w2)，v2-w1 时 (v0, v1, v2, w1)、(v0, w1, v2, w2)。
    """
    v0, v1, v2, w0, w1, w2, d = _canonical(prisms)
    t1 = np.stack([v0, v1, v2, np.where(d, w2, w1)], axis=1)
    t2 = np.where(d[:, None], np.stack([v0, v1, w1, w2], axis=1), np.stack([v0, w1, v2, w2], axis=1))
    t3 = np.stack([v0, w0, w1, w2], axis=1)
    return np.concatenate([t1, t2, t3], axis=0)
