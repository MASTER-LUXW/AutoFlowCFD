"""真 3D 顶点节点（价>=3，多条独立尖锐边方向汇聚于一点）的静态挤出
厚度上限——供 mesh_extrusion.extrude_layers 在挤出开始前，将其并入
`remaining_budget`，与既有的地面间隙厚度上限（外部传入的
`thickness_limit`）用 `np.minimum` 组合。

V2.0 专项攻关记录（cube_demo BL 质量campaign 第十六轮，"协同循环"
重设计）：这是第十二次尝试同一个检测算法的第二次实现——第十二次
把它接到 `remaining_budget`，但当时挤出主循环仍然对每一层额外乘一个
连续衰减系数（`_compute_sharp_angle_attenuation`/`_compute_edge_
distance_field`），导致停止点仍然落在某个被衰减压薄的中间状态，不是
干净的层边界，实测相邻单元体积比一位都没变——真正卡住主指标的不是
"顶点没有厚度上限"，是"停止发生的方式本身不够干净"。第十六轮把连续
衰减机制整体移除（见 extrude_layers 自身改动），只保留这个静态上限
（决定"长几层"）和 clamp_budget_for_convergence（决定"这一层实际
走多远，不多不少，反应式兜底真实碰撞"）两个机制——本函数负责前者。

现有的两个衰减启发式（已移除）都是"单边独立"视角：`_compute_sharp_
angle_attenuation` 按节点自身最尖锐的单条二面角，`_compute_edge_
distance_field` 按到最近单条尖锐边的欧氏距离——都不识别"多条尖锐边在
同一节点汇聚，组合几何比任何单边算出的都更紧"这个真 3D 角点信号，且
两者都是连续压薄而非离散停止，与 ANSA 文档描述的 Collapse 机制
（"从外层往内逐层减少层数，上层节点收缩回下层节点"——精确回退，不是
渐进逼近）不符。本函数直接为这些顶点节点预先计算一个基于其自身组合
物理半径的目标厚度上限，让 remaining_budget 的既有硬停止机制（见
mesh_layer_step.py 自身注释）在某个干净的层边界精确停止，而不是依赖
连续衰减把每一层都压薄一点。
"""

from collections import defaultdict
from typing import List

import numpy as np
from loguru import logger

from ..utils.mesh_corner_split_geometry import _implied_edge_radius
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

    v0 = nodes[extrude_faces[:, 0]]
    v1 = nodes[extrude_faces[:, 1]]
    v2 = nodes[extrude_faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    face_normals = face_normals / np.maximum(
        np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-300
    )

    edge_map = {}
    for i, face in enumerate(extrude_faces):
        for j in range(3):
            a, b = int(face[j]), int(face[(j + 1) % 3])
            key = (min(a, b), max(a, b))
            edge_map.setdefault(key, []).append(i)

    node_dirs: dict = defaultdict(list)
    node_min_radius: dict = {}

    for (a, b), face_idx in edge_map.items():
        if len(face_idx) != 2:
            continue
        fa, fb = face_idx
        cosang = np.clip(np.dot(face_normals[fa], face_normals[fb]), -1.0, 1.0)
        normal_angle_rad = np.arccos(cosang)
        dihedral_deg = 180.0 - np.degrees(normal_angle_rad)
        if dihedral_deg >= sharp_angle_threshold_deg:
            continue

        edge_length = float(np.linalg.norm(nodes[a] - nodes[b]))
        radius = _implied_edge_radius(edge_length, normal_angle_rad)
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
