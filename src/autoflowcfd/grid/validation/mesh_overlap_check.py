"""体网格单元重叠 / 近似接触面检测。

检测本项目现有检查都没有直接覆盖的一类网格缺陷：两个不同的、拓扑上不
相邻的单元，其面在三维空间中物理重叠，或者靠得足够近，只要参数稍微一变
就会重叠（例如两个 BL 挤出前沿隔着一道窄缝相向而行——见
mesh_tetgen_core.compute_local_thickness_limit，它只是在生成阶段*尽量
避免*这种情况，其自身文档也明确说明这只是启发式方法，不是保证）。

本模块是对以下检查的补充，而不是替代：
    - repair_nonmanifold_cells（mesh_tetgen_core.py）事后检测的是重叠的
      *症状*（一个面被超过 2 个单元共享）——如果重叠没有恰好产生这个特定
      拓扑特征（例如两个单元相互穿插但没有任何面真正重合），它就看不见。
    - fill_core_volume 里 tetgen 的自相交错误只在核心区域填充*之前*检查
      BL 外表面（单张二维壳体）本身是否自相交——对最终的三维体网格什么
      都不能说明。
    - 本模块直接检查实际生成的最终单元集合，用精确的三角形-三角形相交/
      距离检测（overlap_geometry.py），而不是间接信号。

只有两个面完全不共享节点时才会被拿来比较——共享节点的面（一条边、一个
顶点，或者同一个面从两侧看）是正常、正确的网格拓扑，不是缺陷，在任何
几何检测开始之前就已被排除。
"""

import time
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from loguru import logger

from .overlap_geometry import triangle_triangle_intersect, triangle_triangle_min_distance
from .mesh_overlap_report import OverlapProximityReport

if TYPE_CHECKING:
    from ..schema.grid_faces import FaceData

# 单个异常巨大的边界面（例如比典型边界面大很多倍的粗远场/域壳
# 面板——见 check_face_overlap_and_proximity 的 search_radius 文档
# 了解为何半径随每个面自身尺寸缩放）会得到一个按该巨大尺寸
# 缩放的宽相位搜索半径，其 query_ball_point 调用可能返回多达
# 数十万的候选列表——纯粹是因为其自身大小，而非真实的接近风险。
# 在真实案例上测量（cube_demo 的粗域壳/远场面板）：单个这样的面
# 返回了 142,944 个候选，将其所在的 500 面块扩展到 558 万候选对，
# 使整个检查耗时 6+ 分钟和数 GB 内存。真实的接近或重叠总是出现在
# 最近的几个候选中——缺陷阈值（proximity_fraction * min(size_i, size_j)）
# 远紧于 search_radius（search_multiplier 的文档解释了两者之间所需
# 的余量）——因此将任何单个面的候选集限制在其最近的 CAP 个邻居
# （而非其超大半径内的每个点）保留了所有真实候选，仅丢弃那些
# 远在半径内但远未达到实际阈值的、无论如何都不会被标记的多余项。
CANDIDATE_CAP_PER_FACE = 2000


def _extract_faces(nodes: np.ndarray, cells: np.ndarray) -> 'FaceData':
    # Lazy-imported: mesh_gen -> validation is a one-way dependency
    # elsewhere in this package (see quality_validator.py's identical
    # _extract_faces) - importing the other direction only at call time
    # avoids ever needing to reason about import order.
    from ..mesh_gen.extraction.face_extractor import FaceExtractor
    from ..schema.grid_nodes import NodeArray

    node_arr = NodeArray.from_array(nodes)
    return FaceExtractor.extract_faces(cells.astype(np.int32), node_arr)


def check_face_overlap_and_proximity(
    nodes: np.ndarray,
    cells: np.ndarray,
    faces: Optional['FaceData'] = None,
    proximity_fraction: float = 0.1,
    search_multiplier: float = 3.0,
    max_examples: int = 20,
    chunk_size: int = 500,
    boundary_faces_only: bool = True,
) -> OverlapProximityReport:
    """检测不同、非相邻单元之间真实重叠和接近接触的面。

    使用 FaceData 已去重面列表（每个独立三角形面一个条目，无论内部
    还是边界——见 grid_faces.py）作为候选集，而非重新推导所有
    4*n_cells 个原始四面体面：合法共享面的两个单元已折叠为单个
    FaceData 条目，同时拥有 owner 和 neighbour，因此无需单独处理
    "两面实际上是同一面"的情况。

    默认情况下（`boundary_faces_only=True`）只有网格的真正边界面
    （无邻居单元）是候选。这是刻意的范围限制，不仅是性能捷径：
    此检查存在的缺陷（见模块文档字符串——两个 BL 挤出前沿交叉、
    核心填充拼接伪影）都表现在各自区域碰撞的外表面，从不埋在
    已正确形成的 BL 层堆内部。也试过检查内部面，但在真实汽车
    网格上发现既错误又不切实际地昂贵：BL 堆栈自身的连续层按
    设计彼此堆积得远比面的自身横向尺寸近（第一层厚度可以只有
    几毫米），因此 `proximity_fraction` 缩放的"接近度"将几乎每个
    普通的层到层过渡都标记为假阳性——在 240 万单元案例上有数百万
    个，这也是使检查本身慢/内存密集的原因，且无诊断收益。仅边界
    面在相同案例上少约 36 倍（4,886,259 个中的 135,914 个），且正是
    真实跨区域碰撞会出现的地方。

    宽相位：对每个面，在该面质心 `search_multiplier * sqrt(自身面积)`
    范围内查询面质心的 KD 树——局部的、每面半径（非单一域尺度常量）。
    这在 BL 挤出的汽车网格上很重要，其中近壁单元可以比远场核心
    单元小几个数量级（见 mesh_tetgen_core.compute_local_thickness_limit
    的文档了解完全相同的教训，从测量的多分钟回归中学到，当该函数
    的早期版本无条件使用域缩放半径时）。从每个面（不仅是小面）
    用自身半径查询，然后取所有找到对的并集，自然地也能捕获靠近
    更大面的小面，即使小面自身的半径不足以到达大面的质心——
    大面自身的（更大的）查询从另一个方向捕获它。罕见异常巨大面
    的候选列表被限制在 CANDIDATE_CAP_PER_FACE 个最近邻居而非无界——
    见该常量的模块级文档了解为何这不会丢失任何真实缺陷。

    窄相位，对每个幸存的候选对（排除共享节点的对——见模块文档
    字符串）：先精确 triangle_triangle_intersect；若不相交，则
    triangle_triangle_min_distance，低于 `proximity_fraction *
    min(sqrt(area_i), sqrt(area_j))` 时标记为"接近"（每对局部
    缩放的阈值，非单一全局距离）。

    Args:
        nodes: (n_nodes, 3) float64 节点坐标
        cells: (n_cells, 4) int32/int64 四面体连接关系
        faces: 可选预计算的 FaceData——若调用方已拥有则复用
            （本项目的网格生成/修复管线通常如此），否则内部推导
        proximity_fraction: "接近"阈值，作为两个候选面中较小者
            自身特征尺寸的比例
        search_multiplier: 宽相位 KD 树查询半径，作为每个面自身
            特征尺寸的倍数——越大捕获越多候选（更安全、更慢）；
            必须超过 proximity_fraction 才能使接近对阈值可达，
            且实际上需要更多余量，因为两个面在最近边仍在范围内
            时质心距离可能远超其接近距离
        max_examples: 保留多少具体 (cell, cell[, distance]) 示例
            供人类可读报告——计数从无限制，仅示例列表
        chunk_size: KD 树查询和候选对去重每次批处理这么多面
            （与 mesh_tetgen_core.compute_local_thickness_limit 的
            分块理由相同——在细网格上一次性物化每个面的完整候选
            列表或所有见过的候选对不会缩放；之前试过全局跨块
            `set()` 存储所有见过的对，在真实 490 万面网格上增长
            到数十 GB）。去重仅在每个块内（通过 np.unique 向量化，
            非 Python 级 set）——因此真实缺陷对若从两个不同块
            可达可能被找到并进行几何测试两次，有界的 2x 计算成本
            换取有界（非无界）内存；最终的重叠/接近对列表无论如何
            在最后再次去重（廉价，受实际发现数而非候选对数限制）
        boundary_faces_only: 见上方——False 检查每个面（内部 +
            边界），在 BL 挤出网格上更慢且更嘈杂；仅对已确认
            自身网格完全没有 BL 区域的调用方有意义（例如裸 tetgen
            背景填充），其中"内部"不携带此默认值围绕的 BL 堆叠
            密度问题。

    Returns:
        OverlapProximityReport
    """
    from scipy.spatial import cKDTree

    start = time.perf_counter()

    if faces is None:
        faces = _extract_faces(nodes, cells)

    if faces.node_connectivity is None:
        raise ValueError(
            "faces.node_connectivity is required (see FaceExtractor.extract_faces) "
            "to determine which faces share a node"
        )

    owner_full = faces.connectivity[:, 0]
    neighbor_full = faces.connectivity[:, 1]

    if boundary_faces_only:
        face_idx = faces.get_boundary_face_indices().astype(np.int64)
    else:
        face_idx = np.arange(faces.count, dtype=np.int64)

    n_faces = len(face_idx)
    centroids = faces.center[face_idx]
    face_nodes = faces.node_connectivity[face_idx]
    face_size = np.sqrt(np.maximum(faces.area[face_idx], 1e-300))
    owner = owner_full[face_idx]
    neighbor = neighbor_full[face_idx]

    tree = cKDTree(centroids)
    search_radius = search_multiplier * face_size

    overlap_pairs: List[Tuple[int, int]] = []
    close_pairs: List[Tuple[int, int, float]] = []
    n_candidate_pairs = 0
    min_gap_found: Optional[float] = None
    n_chunks = (n_faces + chunk_size - 1) // chunk_size
    progress_every = max(1, n_chunks // 20)

    for chunk_num, start_idx in enumerate(range(0, n_faces, chunk_size)):
        end_idx = min(start_idx + chunk_size, n_faces)
        idx_chunk = np.arange(start_idx, end_idx)
        neighbor_lists = tree.query_ball_point(
            centroids[idx_chunk], r=search_radius[idx_chunk], workers=-1
        )

        # 全向量化候选对构造——不是 Python 级循环遍历每个
        # (面, 候选) 组合，也不是不断增长的跨块 Python `set()`
        # 存储所有见过的对。两者都试过，在真实数百万面网格
        # （Ahmed Body, 490 万面）上，set 增长到数十 GB，循环
        # 10+ 分钟看不到任何进展就被杀掉——同类无界积累性能
        # 陷阱本项目之前已遇到过（见本文档 Part4, P1/P2）。
        # 去重改为仅在每个块内通过 np.unique 对小（块本地）数组
        # 进行；一对若从两个不同块的两个方向都被找到仍可能被
        # 测试两次（有界的 2x 额外工作，非无界内存）——刻意的、
        # 廉价的权衡，不是疏忽。
        counts = np.fromiter(
            (len(lst) for lst in neighbor_lists), dtype=np.int64, count=len(neighbor_lists)
        )

        # 见 CANDIDATE_CAP_PER_FACE 的模块级文档字符串。这仅对
        # 罕见的异常巨大面触发——正常规模面（绝大多数）的计数
        # 很小，完全不受影响。
        over_cap = np.flatnonzero(counts > CANDIDATE_CAP_PER_FACE)
        if len(over_cap):
            k = min(CANDIDATE_CAP_PER_FACE, n_faces)
            for local_i in over_cap:
                face_i = int(idx_chunk[local_i])
                original_count = int(counts[local_i])
                nn_dists, nn_idx = tree.query(
                    centroids[face_i], k=k, distance_upper_bound=search_radius[face_i]
                )
                valid = np.isfinite(nn_dists)
                capped = nn_idx[valid].tolist()
                neighbor_lists[local_i] = capped
                counts[local_i] = len(capped)
                logger.warning(
                    f"Overlap check: face {face_i} had an oversized broad-phase candidate set "
                    f"({original_count}+ within radius {search_radius[face_i]:.3e}); capped to the "
                    f"{len(capped)} nearest to keep the check tractable - likely a large outlier "
                    f"face (size={face_size[face_i]:.3e}) relative to the mesh's typical boundary "
                    f"face scale."
                )

        if counts.sum() == 0:
            if chunk_num % progress_every == 0:
                logger.debug(f"Overlap check: {chunk_num}/{n_chunks} chunks, 0 candidates so far")
            continue

        row_idx = np.repeat(idx_chunk, counts)
        col_idx = np.concatenate(
            [np.asarray(lst, dtype=np.int64) for lst in neighbor_lists if len(lst) > 0]
        )
        keep_self = row_idx != col_idx
        row_idx, col_idx = row_idx[keep_self], col_idx[keep_self]
        if len(row_idx) == 0:
            continue

        pairs = np.stack([np.minimum(row_idx, col_idx), np.maximum(row_idx, col_idx)], axis=1)
        pairs = np.unique(pairs, axis=0)
        i_idx, j_idx = pairs[:, 0], pairs[:, 1]

        # Exclude any pair sharing a node (legitimate topology, not overlap).
        shares_node = np.zeros(len(i_idx), dtype=bool)
        ni, nj = face_nodes[i_idx], face_nodes[j_idx]
        for a in range(3):
            for b in range(3):
                shares_node |= ni[:, a] == nj[:, b]
        keep = ~shares_node
        if chunk_num % progress_every == 0:
            logger.debug(
                f"Overlap check: {chunk_num}/{n_chunks} chunks, "
                f"{n_candidate_pairs:,} candidate pairs tested so far"
            )
        if not np.any(keep):
            continue

        i_idx, j_idx = i_idx[keep], j_idx[keep]
        n_candidate_pairs += len(i_idx)

        a_nodes = nodes[face_nodes[i_idx]]  # (M, 3, 3)
        b_nodes = nodes[face_nodes[j_idx]]

        intersects = triangle_triangle_intersect(
            a_nodes[:, 0], a_nodes[:, 1], a_nodes[:, 2],
            b_nodes[:, 0], b_nodes[:, 1], b_nodes[:, 2],
        )

        for k in np.flatnonzero(intersects):
            fi, fj = int(i_idx[k]), int(j_idx[k])
            cells_i = [owner[fi]] + ([neighbor[fi]] if neighbor[fi] >= 0 else [])
            cells_j = [owner[fj]] + ([neighbor[fj]] if neighbor[fj] >= 0 else [])
            for ci in cells_i:
                for cj in cells_j:
                    overlap_pairs.append((int(ci), int(cj)))

        non_intersecting = np.flatnonzero(~intersects)
        if len(non_intersecting):
            ni_idx, nj_idx = i_idx[non_intersecting], j_idx[non_intersecting]
            an, bn = a_nodes[non_intersecting], b_nodes[non_intersecting]
            dists = triangle_triangle_min_distance(
                an[:, 0], an[:, 1], an[:, 2], bn[:, 0], bn[:, 1], bn[:, 2]
            )
            threshold = proximity_fraction * np.minimum(face_size[ni_idx], face_size[nj_idx])
            close_mask = dists < threshold
            for k in np.flatnonzero(close_mask):
                fi, fj = int(ni_idx[k]), int(nj_idx[k])
                d = float(dists[k])
                min_gap_found = d if min_gap_found is None else min(min_gap_found, d)
                cells_i = [owner[fi]] + ([neighbor[fi]] if neighbor[fi] >= 0 else [])
                cells_j = [owner[fj]] + ([neighbor[fj]] if neighbor[fj] >= 0 else [])
                for ci in cells_i:
                    for cj in cells_j:
                        close_pairs.append((int(ci), int(cj), d))

    # 真实缺陷对可能被发现两次（从每个面所在的块各一次——
    # 见上方仅块内去重的说明）；在此折叠为唯一的 (cell_a, cell_b)
    # 条目，而非重复报告/重复计数。廉价：受实际发现数限制，
    # 而非（可能巨大的）总候选对数限制。
    overlap_pairs = list(dict.fromkeys(overlap_pairs))
    close_pairs = list({(a, b): (a, b, d) for a, b, d in close_pairs}.values())

    overlap_cell_ids = np.unique(np.array([p for pair in overlap_pairs for p in pair], dtype=np.int64)) \
        if overlap_pairs else np.array([], dtype=np.int64)
    close_cell_ids = np.unique(np.array([p for pair in close_pairs for p in pair[:2]], dtype=np.int64)) \
        if close_pairs else np.array([], dtype=np.int64)

    elapsed = time.perf_counter() - start
    report = OverlapProximityReport(
        n_faces_checked=n_faces,
        n_candidate_pairs=n_candidate_pairs,
        n_overlapping_pairs=len(overlap_pairs),
        n_close_pairs=len(close_pairs),
        overlapping_cell_ids=overlap_cell_ids,
        close_cell_ids=close_cell_ids,
        min_gap_found=min_gap_found,
        overlap_examples=overlap_pairs[:max_examples],
        close_examples=close_pairs[:max_examples],
        elapsed_seconds=elapsed,
    )

    if report.has_overlaps:
        logger.warning(
            f"Overlap check: {report.n_overlapping_pairs} overlapping face pair(s) "
            f"found across {len(overlap_cell_ids)} cells ({elapsed:.2f}s)"
        )
    else:
        logger.debug(
            f"Overlap check: no overlapping faces found among {n_faces} faces "
            f"({n_candidate_pairs} candidate pairs tested, {elapsed:.2f}s)"
        )

    return report
