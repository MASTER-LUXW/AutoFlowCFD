"""BL 前沿自相交检测：广相位候选对 + 精确三角形相交/跨状态检测。

从 mesh_front_collision.py 拆分出来，是该模块"事后检测"这一半的底层
实现：`find_self_colliding_faces`（同一快照自碰撞）和
`find_cross_state_colliding_faces`（新旧两个快照之间的跨状态碰撞，捕捉
单步推进过快导致的"穿透"）。`clamp_budget_for_convergence`（事前预算裁剪）
和 `freeze_self_colliding_nodes`（事后冻结）仍留在 mesh_front_collision.py。

广相位是 validation/overlap_geometry.py::candidate_face_pairs（与网格重叠检查同一份）：查询面 i 与树侧面 j 的
质心距离不超过 r_i + r_j。两个检测函数取 r = 外接半径（质心到顶点的最大距离）——两三角形相交必须外接球相交，
所以候选完备。2026-10-20 以前半径是
2 x sqrt(面积)：细长三角形（BL 前沿上常见）顶点到质心的距离远大于 sqrt(面积)（长 1、宽 0.01 的针形三角形
sqrt(A) = 0.07，顶点离质心 0.67），两个细长面在尖端附近相交时两侧都查不到对方。
"""

import numpy as np

from ...validation.overlap_geometry import candidate_face_pairs, triangle_triangle_intersect


def _face_geometry(nodes: np.ndarray, faces: np.ndarray):
    """(三角形角点, 质心, sqrt(面积), 单位法向, 外接半径)。"""
    tri = nodes[faces]  # (n_faces, 3, 3)
    centroids = tri.mean(axis=1)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    cross_norm = np.linalg.norm(cross, axis=1)
    face_size = np.sqrt(np.maximum(0.5 * cross_norm, 1e-300))
    normal = cross / np.maximum(cross_norm, 1e-300)[:, np.newaxis]
    circumradius = np.linalg.norm(tri - centroids[:, np.newaxis, :], axis=2).max(axis=1)
    return tri, centroids, face_size, normal, circumradius


def _mark_intersecting(colliding, tri_a, tri_b, row_idx, col_idx) -> None:
    a, b = tri_a[row_idx], tri_b[col_idx]
    hit = triangle_triangle_intersect(a[:, 0], a[:, 1], a[:, 2], b[:, 0], b[:, 1], b[:, 2])
    if np.any(hit):
        colliding[row_idx[hit]] = True
        colliding[col_idx[hit]] = True


def find_self_colliding_faces(
    nodes: np.ndarray,
    faces: np.ndarray,
    chunk_size: int = 2000,
) -> np.ndarray:
    """`faces` 中每个参与真实（精确、非基于接近度——此处无"接近度"
    阈值可调）自相交的面的索引，与另一个非相邻面在同一个三角形集合中。

    广相位按外接球相交取候选（完备，见模块文档），窄相位对每个候选执行精确
    triangle_triangle_intersect。无"接近"情况——仅报告真实相交；
    见 clamp_budget_for_convergence 了解互补的、基于接近度的、
    事前步骤机制。

    Args:
        nodes: (n_nodes, 3) 当前节点位置
        faces: (n_faces, 3) 三角形连接关系（整型）
        chunk_size: 每个 KD 树批处理的面数

    Returns:
        int64 面索引数组，包含至少一个真实相交的面
        （若无则为空数组，永不返回 None）
    """
    n_faces = len(faces)
    if n_faces == 0:
        return np.array([], dtype=np.int64)

    tri, centroids, _size, _normal, circ = _face_geometry(nodes, faces)
    colliding = np.zeros(n_faces, dtype=bool)
    for row_idx, col_idx in candidate_face_pairs(faces, centroids, circ, centroids, circ, chunk_size,
                                                 symmetric=True):
        _mark_intersecting(colliding, tri, tri, row_idx, col_idx)
    return np.flatnonzero(colliding)


def find_cross_state_colliding_faces(
    new_nodes: np.ndarray,
    current_nodes: np.ndarray,
    faces: np.ndarray,
    chunk_size: int = 2000,
) -> np.ndarray:
    """每个新位置真实相交其他非相邻面当前位置的面索引。

    仅 find_self_colliding_faces——将 `new_nodes` 与自身比较——
    会遗漏一个在 cube_demo 上直接确认的真实失败模式：一个快速
    推进的三角形 A 和一个慢速（或不同曲率）的邻居三角形 B 可以
    在两个快照上各自看起来正常（A-new vs B-new 不相交，按定义
    当 B 上一层自身为"new"时已通过此检查），而 A 自身的大步长
    在这一层将其扫过 B 仍在占据的空间——无论是同层检查（仅比较
    同一快照状态）还是 clamp_budget_for_convergence（在步长开始
    时评估的一阶/瞬时线性近似——见 CONVERGENCE_CLOSING_RATE_
    THRESHOLD 的注释）都不能保证在单步相对于局部特征尺寸过大
    时捕获此问题（Stage 2 过渡层可以每层增长到 4 倍——见
    extrude_layers 的 target_handoff_size 求解）。这是与
    CONVERGENCE_SAFETY_FRACTION 对单对存在的穿透关注相同的
    跨三角形推广；已在 cube_demo 上直接验证能找到另外两个机制
    未发现的真实案例（涉及最多约 20% 的表面三角形，跨越大部分
    BL 堆栈深度——"整体邻近柱体扫过较慢柱体"模式，而非孤立碎片）。

    面自身的新-当前对（跨步长将三角形与自身比较）以与自对总是
    被排除的相同方式排除——三角形自身的扫过包含其自身的优先
    位置正是棱柱本身，不是缺陷。候选为新状态面与当前状态面外接球相交的面对（完备）。

    Args:
        new_nodes: (n_nodes, 3) 本层的试探节点位置
        current_nodes: (n_nodes, 3) 前一层（已接受）的位置
        faces: (n_faces, 3) 三角形连接关系（整型）
        chunk_size: 每个 KD 树批处理的面数

    Returns:
        int64 面索引数组（若无则为空，永不返回 None）
    """
    n_faces = len(faces)
    if n_faces == 0:
        return np.array([], dtype=np.int64)

    tri_new, c_new, _s1, _n1, circ_new = _face_geometry(new_nodes, faces)
    tri_cur, c_cur, _s2, _n2, circ_cur = _face_geometry(current_nodes, faces)
    colliding = np.zeros(n_faces, dtype=bool)
    for row_idx, col_idx in candidate_face_pairs(faces, c_new, circ_new, c_cur, circ_cur, chunk_size):
        _mark_intersecting(colliding, tri_new, tri_cur, row_idx, col_idx)
    return np.flatnonzero(colliding)
