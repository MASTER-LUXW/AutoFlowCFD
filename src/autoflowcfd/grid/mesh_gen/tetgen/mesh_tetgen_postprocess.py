"""tetgen 核心域填充：填充后的清理与修复。

从 mesh_tetgen_core.py 拆分出来，负责 fill_core_volume 产出的四面体在
拼接/导出前需要做的收尾处理：重合点合并、超大四面体重填、非流形面修复、
以及从 tetgen 自带的 facet marker 反推每个单元所属的边界分组。
"""

from typing import Callable, Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from loguru import logger


def _dedupe_coincident_points(
    points: np.ndarray,
    faces: np.ndarray,
    tolerance: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """合并重合点（在容差内）并重映射 faces（或四面体单元——
    `faces` 只是一个 (n, k) 索引数组，通过普通花式索引重映射，
    所以 k=3 或 k=4 均可直接使用）。

    同时返回 `remap`（shape=(len(points),)，旧索引 -> 新索引），
    这样持有指向同一原始 `points` 的任意其他索引数组的调用方
    （例如同一网格的棱柱连接关系）可以应用相同的重映射并保持一致
    ——传 `remap[some_other_array]` 即可。
    `remap` 在没有重合点时为恒等映射。

    完全传递闭包（使用 scipy connected_components 处理重合图，
    而不是单跳合并），与本包其他地方旧的 `merge_conforming_meshes`
    节点去重逻辑不同。

    两个调用场景：一是 fill_core_volume 中的原始回退，用于 tetgen
    未返回完全保形边界的情况；二是 mesh_background.generate_hybrid_mesh
    对整体合并网格的最终防御性传递，用于一个在真实案例中发现的
    不同失败模式——当 BL 挤出的收敛预算在大比例表面顶点上同时
    收紧时，许多节点的
    `remaining_budget` 在相同的几层内恰好达到零，将它们冻结在
    后续每层的相同坐标上——但每个仍然获得自己的独立全局节点
    索引（每层无条件一个新索引），所以结果是在不同索引下产生
    大量几何重合的点。这不会触发 repair_nonmanifold_cells（它按
    精确节点索引而非几何匹配），也不能可靠地触发退化体积过滤器
    （混合冻结节点和仍在增长的邻居的四面体可以有小但不可忽略的
    体积）——这是一个静默的拓扑撕裂（两组几何相同的面在不同索引
    下，各自独立计为法向边界面的），而不是崩溃，所以上游没有任何
    地方能捕获它。
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    pairs = tree.query_pairs(tolerance)

    n_points = len(points)
    if not pairs:
        return points, faces, np.arange(n_points, dtype=np.int64)

    rows = [p[0] for p in pairs]
    cols = [p[1] for p in pairs]
    # 确保数据数组为整数类型以避免 connected_components 产生浮点数标签
    graph = coo_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(n_points, n_points))
    n_components, labels = connected_components(graph, directed=False)

    # 确保 labels 为整数类型以用于索引
    labels = labels.astype(np.int64)

    # 使用每个分量中最小的原始索引作为代表。
    representative = np.full(n_components, n_points, dtype=np.int64)
    np.minimum.at(representative, labels, np.arange(n_points))

    new_index_of_label = np.arange(n_components)
    unique_points = points[representative]
    remap = new_index_of_label[labels]

    new_faces = remap[faces]
    logger.warning(
        f"Coincident-point fallback stitch: {n_points} -> {len(unique_points)} points "
        f"({n_points - len(unique_points)} merged)"
    )
    return unique_points, new_faces, remap


def _tet_volumes(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """四面体无符号体积（与朝向无关）。"""
    p0 = nodes[cells[:, 0]]
    p1 = nodes[cells[:, 1]]
    p2 = nodes[cells[:, 2]]
    p3 = nodes[cells[:, 3]]
    return np.abs(np.einsum('ij,ij->i', p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0


# fill_without_oversized_tetrahedra 的重填轮数上限：每轮在体内超大四面体的并集里插入目标间距的点阵点（放不下时
# 取质心），tetgen 的 Delaunay 插入必然把它们拆开；离群的超大四面体一轮即消除，此后最大超限倍数不再下降即停止。
_MAX_REFILL_ROUNDS = 6


def interior_lattice_points(nodes: np.ndarray, tets: np.ndarray, spacing: float) -> np.ndarray:
    """落在 `tets`（一组四面体，下标指向 `nodes`）并集内部的全域体心立方（BCC）点阵点 `(k, 3)`。

    - BCC 而不是简单立方：简单立方点阵每个立方体 8 点共球，是 Delaunay 的退化情形，四面体化必然出薄片；BCC 点阵
      的 Delaunay 是 Sommerville 四面体（二面角只有 60 与 90 度）。立方体边长 `2^(1/3) spacing`，点密度
      1/spacing^3，最近点距 0.89 spacing。
    - 点阵原点固定在坐标原点：相邻四面体、前后各轮取到的是同一套点阵，不会在共享面两侧或轮与轮之间出现几乎重合的点。
    - 只对并集的**外表面**（不被两个所选四面体共享的面）留 `spacing / 2` 的间隙，并与已有节点保持 `spacing / 2`；
      并集内部的共享面不需要间隙（2026-10-17 初版对每个四面体的四个面都留间隙，多数四面体只放得下质心，需要多轮
      重填，各轮的点互不对齐，同样出薄片）。
    - 一个点都放不下的四面体取其质心，保证每轮每个超大四面体至少被拆开。
    """
    from scipy.spatial import cKDTree

    nodes = np.asarray(nodes, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    s = float(spacing)
    a = 2.0 ** (1.0 / 3.0) * s                                   # BCC 立方体边长
    opp = np.array([(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)])
    face_key = np.sort(tets[:, opp], axis=2).reshape(-1, 3)      # (4m, 3)，第 4k+i 行为四面体 k 的第 i 个面
    _, inv, cnt = np.unique(face_key, axis=0, return_inverse=True, return_counts=True)
    outer = (cnt[inv.ravel()] == 1).reshape(-1, 4)
    tree = cKDTree(nodes)
    out = []
    for k, cell in enumerate(tets):
        P = nodes[cell]
        T = (P[1:] - P[0]).T                                     # 列为三条棱
        vol6 = abs(np.linalg.det(T))
        areas = np.array([0.5 * np.linalg.norm(np.cross(P[j] - P[i], P[l] - P[i])) for i, j, l in opp])
        # 顶点 i 到对面的高 h_i = 3V / A_i；离外表面 i >= s/2 等价于重心坐标 lambda_i >= s / (2 h_i) = s A_i / vol6
        lam_min = np.where(outer[k], s * areas / vol6, 0.0)
        lo = np.floor(P.min(axis=0) / a).astype(int) - 1
        hi = np.ceil(P.max(axis=0) / a).astype(int) + 1
        corner = np.stack(np.meshgrid(*[np.arange(lo[d], hi[d] + 1) * a for d in range(3)], indexing="ij"),
                          axis=-1).reshape(-1, 3)
        grid = np.vstack([corner, corner + 0.5 * a])
        lam123 = np.linalg.solve(T, (grid - P[0]).T).T
        lam = np.column_stack([1.0 - lam123.sum(axis=1), lam123])
        # 内部共享面要求严格 > 0：恰好落在共享面上的点（测度为零）两侧都不收，避免同一点收两次
        inside = np.all(np.where(outer[k], lam >= lam_min, lam > 0.0), axis=1)
        pts = grid[inside]
        if len(pts):
            pts = pts[tree.query(pts)[0] >= 0.5 * s]
        if not len(pts):
            pts = P.mean(axis=0)[None]
        out.append(pts)
    return np.vstack(out) if out else np.zeros((0, 3))


def boundary_limited_tetrahedra(nodes: np.ndarray, tets: np.ndarray, max_volume: float) -> np.ndarray:
    """`(m,)` bool：贴在区域边界面上、且该边界面上的正四面体体积就已超过 `max_volume` 的四面体。

    区域边界面 = 只属于一个四面体的面。面积 A 的面上最小的良态四面体是以它为底的正四面体，体积
    (4A/sqrt(3))^1.5 / (6 sqrt(2))；超过上限时只有加密边界才能缩小它，在体内插点只会造出贴着该面的薄片。
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    opp = np.array([(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)])
    faces = tets[:, opp]                                          # (m, 4, 3)
    key = np.sort(faces, axis=2).reshape(-1, 3)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    on_boundary = (cnt[inv.ravel()] == 1).reshape(-1, 4)
    P = nodes[faces]                                              # (m, 4, 3, 3)
    area = 0.5 * np.linalg.norm(np.cross(P[..., 1, :] - P[..., 0, :], P[..., 2, :] - P[..., 0, :]), axis=-1)
    v_regular = (4.0 * area / np.sqrt(3.0)) ** 1.5 / (6.0 * np.sqrt(2.0))
    return np.any(on_boundary & (v_regular > max_volume), axis=1)


def fill_without_oversized_tetrahedra(
    fill: Callable[[Optional[np.ndarray]], Tuple[np.ndarray, np.ndarray]],
    background_points: Optional[np.ndarray],
    max_volume: float,
    target_edge_length: float,
    max_rounds: int = _MAX_REFILL_ROUNDS,
) -> Tuple[np.ndarray, np.ndarray]:
    """调用 `fill(背景点)`（`fill_core_volume` 的闭包）做四面体化；体内有体积超过 `max_volume` 的四面体时，在这些
    四面体的并集内按目标边长插入点阵点（`interior_lattice_points`），并入背景点重新四面体化，直到全部达标。

    存在原因：tetgen 的体积精化（`regions`/`varvolume`）队列是形状质量优先、体积其次，四个顶点都在 PLC 边界上、
    附近没有任何触发插入的四面体可以完全不被处理（cube_demo 上一个 14.15 m^3 的四面体从入口跨到出口，约为区域
    目标的 16000 倍）。新加的点与 `generate_core_background_points` 的背景点走同一条路径（tetgen 的初始 Delaunay
    原样纳入自由点），重新四面体化时区域体积精化与质量精化照常进行。只要体内最大超限倍数还在下降就继续下一轮，
    不再下降时退回最好的一轮、给出警告（每轮都要对整个核心区重跑 tetgen；nobisect 下边界比目标尺寸粗时，边界附近
    tetgen 自然留下的偏大四面体靠体内插点消不掉，反复重跑只会此消彼长）。

    贴在边界面上、尺寸由该边界面决定的超大四面体（`boundary_limited_tetrahedra`）不处理、给出警告：在体内插点
    缩不小它们，只会造出薄片；要缩小只能加密边界（nobisect 下 tetgen 不能剖分边界面）。

    2026-10-17 以前对全部超大四面体做质心 1 拆 4 细分：贴原四面体某面的子四面体底面不变、高度每层缩为 1/4，
    k 层后纵横比约 4^k（8 层 65536），用必然制造薄片的方式修"体积过大"（I-145）。

    """
    extra = None if background_points is None else np.asarray(background_points, dtype=np.float64)
    best = None                                   # (体内最大超限倍数, nodes, tets)
    for rnd in range(max_rounds + 1):
        nodes, tets = fill(extra)
        vols = _tet_volumes(nodes, tets)
        over = vols > max_volume
        limited = np.zeros_like(over)
        if np.any(over):
            limited[over] = boundary_limited_tetrahedra(nodes, tets[over], max_volume)
        interior = over & ~limited
        worst = float(vols[interior].max()) / max_volume if np.any(interior) else 0.0
        if best is not None and worst >= best[0]:
            break                                 # 不再改善：退回上一轮的结果
        best = (worst, nodes, tets, int(interior.sum()), limited, vols)
        if worst == 0.0 or rnd == max_rounds:
            break
        new_points = interior_lattice_points(nodes, tets[interior], target_edge_length)
        logger.info(
            f"体内超大四面体 {int(interior.sum())} 个（最大为上限的 {worst:.1f} 倍）："
            f"插入 {len(new_points)} 个点阵点后重新四面体化（第 {rnd + 1} 轮）")
        extra = new_points if extra is None else np.vstack([extra, new_points])
    if best is None:                              # 第一轮必然记下结果
        raise RuntimeError("四面体化没有产生任何结果")
    worst, nodes, tets, n_interior, limited, vols = best
    if np.any(limited):
        logger.warning(
            f"{int(limited.sum())} 个贴边界面的四面体超过体积上限 {max_volume:.4g}（最大为上限的 "
            f"{float(vols[limited].max()) / max_volume:.1f} 倍）：尺寸由所贴边界面决定，体内插点缩不小、只会"
            f"造出薄片，保留原样；要缩小需加密该处边界")
    if n_interior:
        logger.warning(
            f"重填不再改善后体内仍有 {n_interior} 个四面体超过体积上限 {max_volume:.4g}（最大为上限的 {worst:.1f} 倍）："
            f"通常是 nobisect 下边界比目标尺寸粗、tetgen 不能在边界附近插点所致，保留原样（单元形状由 tetgen "
            f"质量精化保证，网格质量门另行检查）")
    return nodes, tets


def repair_nonmanifold_cells(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """检测使某些三角面被超过 2 个单元共享（非流形）的四面体，
    并标记多余的单元以便移除。

    主要已知原因已在源头修复：隔离的嵌入实体（例如车身）需要
    tetgen hole 种子，否则其内部会被填充与已占据该空间的 BL 棱柱
    重叠的伪四面体（见 mesh_domain_classify.find_point_inside_closed_shell
    和 fill_core_volume 的 `holes` 参数）。本函数作为安全网保留，
    用于捕获该机制未覆盖的情况（例如一个非常非凸的实体找不到
    hole 点，或者一个真正紧密的 BL seam 产生的近退化边界面上
    tetgen 的 `nobisect=True` 核心填充无法通过插入边界点来解决）。
    如果不修复，这是真实守恒违反：有限体积面提取只能将一个共享面
    归属到 3+ 个接触单元中的 2 个，静默丢弃通过该面的通量（见
    face_extractor.py 在恰好这种情况下的硬失败）。

    一个三角面在每一侧最多有一个合法的邻居单元（四面体的第 4 个
    顶点，即"顶点"，位于该面平面的那一侧）。当多个单元共享同一
    侧时，它们是物理重叠的副本——只保留体积最大的一个，丢弃其余，
    每个超共享面独立处理。

    Args:
        nodes: (n_nodes, 3) 节点坐标
        cells: (n_cells, 4) 四面体连接关系

    Returns:
        布尔保留掩码，shape=(n_cells,)；False 标记要移除的单元
    """
    n_cells = len(cells)
    keep = np.ones(n_cells, dtype=bool)
    if n_cells == 0:
        return keep

    face_templates = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
    ], dtype=np.int64)
    apex_of_face = np.array([3, 2, 1, 0], dtype=np.int64)

    all_faces = cells[:, face_templates].reshape(-1, 3)
    apex_nodes = cells[:, apex_of_face].reshape(-1)
    cell_of_face = np.repeat(np.arange(n_cells), 4)

    sorted_faces = np.sort(all_faces, axis=1)
    face_dtype = np.dtype((np.void, sorted_faces.dtype.itemsize * 3))
    face_voids = np.ascontiguousarray(sorted_faces).view(face_dtype).reshape(-1)

    order = np.argsort(face_voids, kind='stable')
    sorted_voids = face_voids[order]
    sorted_cells = cell_of_face[order]
    sorted_apex = apex_nodes[order]
    sorted_face_nodes = sorted_faces[order]

    change = np.flatnonzero(sorted_voids[1:] != sorted_voids[:-1]) + 1
    group_starts = np.concatenate([[0], change])
    group_ends = np.concatenate([change, [len(sorted_voids)]])
    counts = group_ends - group_starts

    invalid_groups = np.flatnonzero(counts > 2)
    if len(invalid_groups) == 0:
        return keep

    volumes = _tet_volumes(nodes, cells)
    n_removed = 0

    for gi in invalid_groups:
        s, e = group_starts[gi], group_ends[gi]
        face_cells = sorted_cells[s:e]
        n0, n1, n2 = sorted_face_nodes[s]
        p0 = nodes[n0]
        normal = np.cross(nodes[n1] - p0, nodes[n2] - p0)

        apexes = sorted_apex[s:e]
        signed_dist = (nodes[apexes] - p0) @ normal

        for side_mask in (signed_dist > 0, signed_dist <= 0):
            side_cells = face_cells[side_mask]
            if len(side_cells) <= 1:
                continue
            best = side_cells[np.argmax(volumes[side_cells])]
            for c in side_cells:
                if c != best and keep[c]:
                    keep[c] = False
                    n_removed += 1

    if n_removed:
        logger.warning(
            f"Repaired {len(invalid_groups)} non-manifold faces by removing "
            f"{n_removed} redundant overlapping tetrahedra"
        )
    return keep
