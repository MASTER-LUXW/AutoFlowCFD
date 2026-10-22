"""真 3D 顶点节点（价 >= 3，多条独立尖锐边方向汇聚于一点）的静态挤出厚度上限。

`mesh_extrusion.extrude_layers` 在挤出开始前把它并入 `remaining_budget`（与外部传入的地面间隙上限
`thickness_limit` 取 `np.minimum`），由 `mesh_layer_step.py` 的硬停止在某个完整的层边界停下——决定"长几层"；
每层实际走多远、兜住真实碰撞由 `clamp_budget_for_convergence` 负责。

为什么是静态上限而不是逐层连续衰减：单边视角的衰减（按节点最尖锐的单条二面角、按到最近尖锐边的距离）既不识别
"多条尖锐边在同一节点汇聚、组合几何比任何单边都更紧"的角点信号，又让停止点落在某个被压薄的中间状态，产生大量
退化程度不一的薄棱柱（cube_demo 上与连续衰减并存时相邻体积比一位不变）；ANSA 的 Collapse 机制同样是逐层精确
回退而非渐进逼近。两个连续衰减启发式已删除。
"""

from collections import defaultdict
from typing import List

import numpy as np
from loguru import logger

from ..utils.mesh_corner_split_geometry import implied_edge_radius
from ..utils.mesh_utils import compute_face_normals, face_edge_incidence, two_sided_edges
from ..tetgen.mesh_tetgen_seam import _smooth_thickness_limit

SHARP_DIHEDRAL_THRESHOLD_DEG = 150.0
DIRECTION_CLUSTER_TOL_DEG = 25.0
VERTEX_RADIUS_SAFETY_FACTOR = 1.0


def compute_vertex_corner_thickness_limit(
    nodes: np.ndarray,
    extrude_faces: np.ndarray,
    safety_factor: float = VERTEX_RADIUS_SAFETY_FACTOR,
    sharp_angle_threshold_deg: float = SHARP_DIHEDRAL_THRESHOLD_DEG,
    direction_cluster_tol_deg: float = DIRECTION_CLUSTER_TOL_DEG,
) -> np.ndarray:
    """为真 3D 顶点节点（>=2 条方向不同的尖锐边汇聚于一点）计算基于其
    组合隐含曲率半径的静态累积 BL 厚度上限。返回 (n_nodes,) float 数组，
    未受限节点为 np.inf——与 mesh_tetgen_seam.compute_local_thickness_
    limit 返回值的约定完全一致，调用方用 np.minimum 组合两者。
    """
    n_nodes = len(nodes)
    limit = np.full(n_nodes, np.inf, dtype=np.float64)
    if len(extrude_faces) == 0:
        return limit

    face_normals = compute_face_normals(nodes, extrude_faces)
    edge_vertices, edge_of_slot, face_of_slot, counts = face_edge_incidence(extrude_faces)
    edge_ids, fa, fb = two_sided_edges(edge_of_slot, face_of_slot, counts)
    normal_angle = np.arccos(np.clip(np.einsum("ij,ij->i", face_normals[fa], face_normals[fb]), -1.0, 1.0))
    sharp = 180.0 - np.degrees(normal_angle) < sharp_angle_threshold_deg
    # 按边在面表里首次出现的顺序（面号 x 3 + 面内边号）处理：下面的方向聚类是贪心的，结果与遍历顺序有关
    n_faces = len(extrude_faces)
    slot_rank = face_of_slot * 3 + np.arange(len(face_of_slot)) // n_faces
    first_rank = np.full(len(counts), np.iinfo(np.int64).max)
    np.minimum.at(first_rank, edge_of_slot, slot_rank)
    sharp_edges = edge_ids[sharp]
    sharp_angle = normal_angle[sharp]
    by_appearance = np.argsort(first_rank[sharp_edges], kind="stable")
    sharp_edges, sharp_angle = sharp_edges[by_appearance], sharp_angle[by_appearance]

    node_dirs: dict = defaultdict(list)
    node_min_radius: dict = {}

    for e, normal_angle_rad in zip(sharp_edges, sharp_angle):
        a, b = int(edge_vertices[e, 0]), int(edge_vertices[e, 1])
        edge_length = float(np.linalg.norm(nodes[a] - nodes[b]))
        radius = implied_edge_radius(edge_length, float(normal_angle_rad))
        d = nodes[b] - nodes[a]
        d = d / max(float(np.linalg.norm(d)), 1e-12)
        node_dirs[a].append(d)
        node_dirs[b].append(d)
        node_min_radius[a] = min(node_min_radius.get(a, np.inf), radius)
        node_min_radius[b] = min(node_min_radius.get(b, np.inf), radius)

    if not node_dirs:
        return limit

    cluster_cos_thresh = np.cos(np.radians(direction_cluster_tol_deg))

    def _n_distinct_directions(dirs: List[np.ndarray]) -> int:
        reps: List[np.ndarray] = []
        for d in dirs:
            is_new = True
            for r in reps:
                if abs(float(np.dot(d, r))) > cluster_cos_thresh:
                    is_new = False
                    break
            if is_new:
                reps.append(d)
        return len(reps)

    n_vertex_nodes = 0
    for node_id, dirs in node_dirs.items():
        if _n_distinct_directions(dirs) >= 2:
            limit[node_id] = node_min_radius[node_id] * safety_factor
            n_vertex_nodes += 1

    if n_vertex_nodes:
        logger.info(
            f"Vertex-corner thickness limit: {n_vertex_nodes} true 3D-vertex "
            f"node(s) (>=2 distinct sharp-edge directions converging) capped "
            f"by their own combined implied curvature radius (min cap "
            f"{float(np.min(limit[np.isfinite(limit)])):.4e} m)"
        )
        limit = _smooth_thickness_limit(limit, extrude_faces)

    return limit
