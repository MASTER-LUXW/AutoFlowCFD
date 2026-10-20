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

from .overlap_geometry import (
    candidate_face_pairs, triangle_triangle_intersect, triangle_triangle_min_distance,
)
from .mesh_overlap_report import OverlapProximityReport

if TYPE_CHECKING:
    from ..schema.grid_faces import FaceData

def _extract_faces(nodes: np.ndarray, cells: np.ndarray) -> 'FaceData':
    # 延迟导入：本包其余地方 mesh_gen -> validation 是单向依赖（见 quality_validator.py 里相同的
    # _extract_faces），反方向只在调用时导入，不必考虑导入顺序。
    from ..mesh_gen.extraction.face_extractor import FaceExtractor
    from ..schema.grid_nodes import NodeArray

    node_arr = NodeArray.from_array(nodes)
    return FaceExtractor.extract_faces(cells.astype(np.int32), node_arr)


def check_face_overlap_and_proximity(
    nodes: np.ndarray,
    cells: np.ndarray,
    faces: Optional['FaceData'] = None,
    proximity_fraction: float = 0.1,
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

    宽相位（validation/overlap_geometry.py::candidate_face_pairs，与 BL 前沿碰撞检测同一份）：质心距离不超过
    (r_i + proximity_fraction * sqrt(面积_i)) + r_j 的面对，r 为外接半径（质心到顶点的最大距离）——对"相交"与
    "间隙小于阈值"都完备；每对只从外接半径较小的一侧查询一次。2026-10-20 以前半径是 3 x sqrt(自身面积)：细长面
    （顶点离质心远大于 sqrt(面积)）可能漏检；异常巨大的面因自身半径大一次取回数十万候选，于是另有一个"截断到
    最近 2000 个"的上限（CANDIDATE_CAP_PER_FACE），截断本身就可能丢掉真实缺陷。现在大面只查同档或更大的面，
    候选数有界，不再截断。
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
        max_examples: 保留多少具体 (cell, cell[, distance]) 示例
            供人类可读报告——计数从无限制，仅示例列表
        chunk_size: KD 树查询每次批处理这么多面（在细网格上一次性物化全部候选对不会缩放；
            每对只出现一次，不需要跨块去重）
        boundary_faces_only: 见上方——False 检查每个面（内部 +
            边界），在 BL 挤出网格上更慢且更嘈杂；仅对已确认
            自身网格完全没有 BL 区域的调用方有意义（例如裸 tetgen
            背景填充），其中"内部"不携带此默认值围绕的 BL 堆叠
            密度问题。

    Returns:
        OverlapProximityReport
    """

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

    tri = nodes[face_nodes]
    circumradius = np.linalg.norm(tri - centroids[:, None, :], axis=2).max(axis=1)

    overlap_pairs: List[Tuple[int, int]] = []
    close_pairs: List[Tuple[int, int, float]] = []
    n_candidate_pairs = 0
    min_gap_found: Optional[float] = None

    for i_idx, j_idx in candidate_face_pairs(
        face_nodes, centroids, circumradius + proximity_fraction * face_size, centroids, circumradius,
        chunk_size, symmetric=True,
    ):
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

    # 不同的面对可能落在同一对单元上（一个单元的几个面都与另一个单元相交）；在此折叠为唯一的
    # (cell_a, cell_b) 条目。廉价：受实际发现数限制。
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
