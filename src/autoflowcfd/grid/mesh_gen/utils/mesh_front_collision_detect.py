"""BL 前沿自相交检测：广相位候选对 + 精确三角形相交/跨状态检测。

从 mesh_front_collision.py 拆分出来，是该模块"事后检测"这一半的底层
实现：`find_self_colliding_faces`（同一快照自碰撞）和
`find_cross_state_colliding_faces`（新旧两个快照之间的跨状态碰撞，捕捉
单步推进过快导致的"穿透"）。`clamp_budget_for_convergence`（事前预算裁剪）
和 `freeze_self_colliding_nodes`（事后冻结）仍留在 mesh_front_collision.py。
"""

import numpy as np

from ...validation.overlap_geometry import triangle_triangle_intersect


def _face_geometry(nodes: np.ndarray, faces: np.ndarray):
    tri = nodes[faces]  # (n_faces, 3, 3)
    centroids = tri.mean(axis=1)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    cross_norm = np.linalg.norm(cross, axis=1)
    face_size = np.sqrt(np.maximum(0.5 * cross_norm, 1e-300))
    normal = cross / np.maximum(cross_norm, 1e-300)[:, np.newaxis]
    return tri, centroids, face_size, normal


def _iter_candidate_pairs(
    faces: np.ndarray,
    centroids: np.ndarray,
    face_size: np.ndarray,
    search_multiplier: float,
    chunk_size: int,
):
    """产出 (row_idx, col_idx) int64 数组，每个块一对块本地候选索引数组：
    非自对、非节点共享的面片对，其质心在 `search_multiplier * 自身
    sqrt(面积)` 范围内。find_self_colliding_faces 和
    clamp_budget_for_convergence 共享的宽相位——见任一函数的文档
    了解为何半径是每面的（局部网格尺度）而非单一域常量，以及为何
    分块处理限制内存而非一次物化所有候选对（与 mesh_overlap_check.py
    相同模式的理由，在真实数百万面网格上验证）。
    """
    from scipy.spatial import cKDTree

    n_faces = len(faces)
    tree = cKDTree(centroids)
    search_radius = search_multiplier * face_size

    for start in range(0, n_faces, chunk_size):
        end = min(start + chunk_size, n_faces)
        idx_chunk = np.arange(start, end)
        neighbor_lists = tree.query_ball_point(
            centroids[idx_chunk], r=search_radius[idx_chunk], workers=-1
        )
        counts = np.fromiter(
            (len(lst) for lst in neighbor_lists), dtype=np.int64, count=len(neighbor_lists)
        )
        if counts.sum() == 0:
            continue

        row_idx = np.repeat(idx_chunk, counts)
        col_idx = np.concatenate(
            [np.asarray(lst, dtype=np.int64) for lst in neighbor_lists if len(lst) > 0]
        )
        keep = row_idx != col_idx
        row_idx, col_idx = row_idx[keep], col_idx[keep]
        if len(row_idx) == 0:
            continue

        # 共享节点的面是普通相邻拓扑，不是缺陷
        # （与 mesh_overlap_check.py 的节点共享过滤规则相同）。
        fi, fj = faces[row_idx], faces[col_idx]
        shares_node = np.zeros(len(row_idx), dtype=bool)
        for a in range(3):
            for b in range(3):
                shares_node |= fi[:, a] == fj[:, b]
        keep = ~shares_node
        if not np.any(keep):
            continue
        row_idx, col_idx = row_idx[keep], col_idx[keep]

        yield row_idx, col_idx


def find_self_colliding_faces(
    nodes: np.ndarray,
    faces: np.ndarray,
    search_multiplier: float = 2.0,
    chunk_size: int = 2000,
) -> np.ndarray:
    """`faces` 中每个参与真实（精确、非基于接近度——此处无"接近度"
    阈值可调）自相交的面的索引，与另一个非相邻面在同一个三角形集合中。

    仅窄相位：对 _iter_candidate_pairs 的每个宽相位候选执行精确
    triangle_triangle_intersect。无"接近"情况——仅报告真实相交；
    见 clamp_budget_for_convergence 了解互补的、基于接近度的、
    事前步骤机制。

    Args:
        nodes: (n_nodes, 3) 当前节点位置
        faces: (n_faces, 3) 三角形连接关系（整型）
        search_multiplier: 宽相位 KD 树查询半径，作为每个面
            自身 sqrt(面积) 的倍数
        chunk_size: 每个 KD 树批处理的面数

    Returns:
        int64 面索引数组，包含至少一个真实相交的面
        （若无则为空数组，永不返回 None）
    """
    n_faces = len(faces)
    if n_faces == 0:
        return np.array([], dtype=np.int64)

    tri, centroids, face_size, _normal = _face_geometry(nodes, faces)
    colliding = np.zeros(n_faces, dtype=bool)

    for row_idx, col_idx in _iter_candidate_pairs(
        faces, centroids, face_size, search_multiplier, chunk_size
    ):
        a_nodes, b_nodes = tri[row_idx], tri[col_idx]
        intersects = triangle_triangle_intersect(
            a_nodes[:, 0], a_nodes[:, 1], a_nodes[:, 2],
            b_nodes[:, 0], b_nodes[:, 1], b_nodes[:, 2],
        )
        if np.any(intersects):
            colliding[row_idx[intersects]] = True
            colliding[col_idx[intersects]] = True

    return np.flatnonzero(colliding)


def find_cross_state_colliding_faces(
    new_nodes: np.ndarray,
    current_nodes: np.ndarray,
    faces: np.ndarray,
    search_multiplier: float = 2.0,
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
    位置正是棱柱本身，不是缺陷。

    Args:
        new_nodes: (n_nodes, 3) 本层的试探节点位置
        current_nodes: (n_nodes, 3) 前一层（已接受）的位置
        faces: (n_faces, 3) 三角形连接关系（整型）
        search_multiplier: 宽相位 KD 树查询半径，作为查询
            （新状态）面自身 sqrt(面积) 的倍数
        chunk_size: 每个 KD 树批处理的面数

    Returns:
        int64 面索引数组（若无则为空，永不返回 None）
    """
    n_faces = len(faces)
    if n_faces == 0:
        return np.array([], dtype=np.int64)

    from scipy.spatial import cKDTree

    tri_new, centroids_new, face_size_new, _n1 = _face_geometry(new_nodes, faces)
    tri_cur, centroids_cur, _face_size_cur, _n2 = _face_geometry(current_nodes, faces)

    tree = cKDTree(centroids_cur)
    search_radius = search_multiplier * face_size_new

    colliding = np.zeros(n_faces, dtype=bool)

    for start in range(0, n_faces, chunk_size):
        end = min(start + chunk_size, n_faces)
        idx_chunk = np.arange(start, end)
        neighbor_lists = tree.query_ball_point(
            centroids_new[idx_chunk], r=search_radius[idx_chunk], workers=-1
        )
        counts = np.fromiter(
            (len(lst) for lst in neighbor_lists), dtype=np.int64, count=len(neighbor_lists)
        )
        if counts.sum() == 0:
            continue

        row_idx = np.repeat(idx_chunk, counts)  # 索引新状态（查询侧）
        col_idx = np.concatenate(
            [np.asarray(lst, dtype=np.int64) for lst in neighbor_lists if len(lst) > 0]
        )  # 索引当前状态（树侧）
        keep = row_idx != col_idx  # 面自身的扫过不是缺陷
        row_idx, col_idx = row_idx[keep], col_idx[keep]
        if len(row_idx) == 0:
            continue

        fi, fj = faces[row_idx], faces[col_idx]
        shares_node = np.zeros(len(row_idx), dtype=bool)
        for a in range(3):
            for b in range(3):
                shares_node |= fi[:, a] == fj[:, b]
        keep = ~shares_node
        if not np.any(keep):
            continue
        row_idx, col_idx = row_idx[keep], col_idx[keep]

        a_nodes = tri_new[row_idx]
        b_nodes = tri_cur[col_idx]
        intersects = triangle_triangle_intersect(
            a_nodes[:, 0], a_nodes[:, 1], a_nodes[:, 2],
            b_nodes[:, 0], b_nodes[:, 1], b_nodes[:, 2],
        )
        if np.any(intersects):
            colliding[row_idx[intersects]] = True
            colliding[col_idx[intersects]] = True

    return np.flatnonzero(colliding)
