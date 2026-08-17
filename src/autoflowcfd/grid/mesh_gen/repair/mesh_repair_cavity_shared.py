"""阶段 B' 局部重铺（cavity retile）用到的共享底层工具。

从 mesh_repair_cavity.py 拆分出来，供 remesh_core_cavity（同目录
mesh_repair_cavity.py）和 patch_nonmanifold_cavity（同目录
mesh_repair_nonmanifold_patch.py）两个局部重新四面体化流程共用：cavity
（待重铺区域）的环形扩张、cavity 自身边界面提取，以及重铺后的质量评分。
"""

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from ...validation.quality_validator import MeshQualityValidator

# 正定向四面体 (v0,v1,v2,v3) 的外向三角形面，每行省略一个顶点——见
# mesh_prism_to_tet.orient_tetrahedra 了解假设的正定向约定。已对照参考
# 单位四面体 (0,0,0)-(1,0,0)-(0,1,0)-(0,0,1) 验证：每行的叉积法向指向
# 远离四面体自身质心的方向，即外向。
_CAVITY_FACE_TEMPLATES = np.array([
    [1, 2, 3],
    [0, 3, 2],
    [0, 1, 3],
    [0, 2, 1],
], dtype=np.int64)


def _grow_cavity_rings(
    seed_mask: np.ndarray,
    owner: np.ndarray,
    neighbor: np.ndarray,
    blocked_mask: np.ndarray,
    n_rings: int,
) -> np.ndarray:
    """将种子单元掩码向外扩展 `n_rings` 次面邻接跳，永不进入
    `blocked_mask` 单元（BL 单元/接触物理边界面的单元——见 remesh_core_cavity）。
    缓冲环的存在是为了让 cavity 自身的新边界落在已经好的单元上，
    而不是已经退化的单元上。

    Args:
        owner, neighbor: (n_interior_faces,) 仅每个内部面两侧的单元索引
            （边界面没有远侧可连接，所以它根本不在这个邻接图中）

    Returns:
        布尔单元掩码，与 seed_mask 形状相同，被阻止的单元保证为假
        即使可达。
    """
    cavity = seed_mask & ~blocked_mask
    for _ in range(n_rings):
        touches = cavity[owner] | cavity[neighbor]
        if not np.any(touches):
            break
        newly = np.zeros_like(cavity)
        newly[owner[touches]] = True
        newly[neighbor[touches]] = True
        newly &= ~blocked_mask
        if np.array_equal(newly | cavity, cavity):
            break
        cavity |= newly
    return cavity


def _cavity_boundary_faces(cells: np.ndarray, cavity_cell_idx: np.ndarray) -> np.ndarray:
    """单元子集的外向定向边界面的全局节点索引——两个 cavity 单元共享的
    面纯粹是内部的（tetgen 会将其重铺掉）并被排除；与子集外的单元共享的
    面，或与任何东西共享的面（真实的物理边界），在子集自己的面中出现恰好
    一次并成为 cavity 固定 PLC 的一部分。
    """
    cav_cells = cells[cavity_cell_idx]
    all_faces = cav_cells[:, _CAVITY_FACE_TEMPLATES].reshape(-1, 3)
    sorted_faces = np.sort(all_faces, axis=1)
    face_dtype = np.dtype((np.void, sorted_faces.dtype.itemsize * 3))
    voids = np.ascontiguousarray(sorted_faces).view(face_dtype).reshape(-1)
    _, inverse, counts = np.unique(voids, return_inverse=True, return_counts=True)
    boundary_mask = counts[inverse] == 1
    return all_faces[boundary_mask]


def _count_bad_cells(validator: 'MeshQualityValidator', nodes: np.ndarray, cells: np.ndarray) -> int:
    """有多少 `单元` 触发偏斜度、非正交或相邻体积比——与
    mesh_repair.py 自身 `_bad_cell_mask` 对整个网格使用的相同三项
    判据，此处在小重铺空腔上评估，使 remesh_core_cavity 的接受门控
    （参见其调用点）对 `bad_cell_mask` 的"坏"定义进行同类比较，
    而非仅偏斜度。新的局部重铺是几个到几千个单元（受 max_cavity_cells
    限制）——完全重新提取面很便宜，不像重新验证整个网格。
    """
    from ..extraction.face_extractor import FaceExtractor
    from ...schema.grid_nodes import NodeArray

    bad = validator.compute_cell_skewness(nodes, cells) > validator.thresholds['max_skewness']

    node_arr = NodeArray.from_array(nodes)
    # face_extractor 每次调用都无条件记录多个 INFO/SUCCESS 行
    # （那里没有 verbose= 开关，不像 fill_core_volume）——对于正常的
    # 每网格一次调用没问题，但这里每个 cavity 候选运行一次（最多
    # max_clusters_attempted 个，大部分被拒绝），所以在有多个小 cavity
    # 的真实案例上，每次修复传递会乘以数万行常规噪音（已直接确认：
    # 单次 Stage B' 传递产生了 70K+ 行日志）。只有这个模块自己的
    # 每 cavity/摘要行（由 remesh_core_cavity 自己单独记录）在这个
    # 粒度上实际上有用。
    logger.disable("autoflowcfd.grid.mesh_gen.face_extractor")
    try:
        faces = FaceExtractor.extract_faces(cells.astype(np.int32), node_arr)
    finally:
        logger.enable("autoflowcfd.grid.mesh_gen.face_extractor")
    diag = validator.compute_face_diagnostics(nodes, cells, faces)
    if len(diag['angle_deg']) > 0:
        face_bad = (
            (diag['angle_deg'] > validator.thresholds['max_orthogonality_angle'])
            | (diag['volume_ratio'] > validator.thresholds['max_adjacent_volume_ratio'])
        )
        bad[diag['owner'][face_bad]] = True
        bad[diag['neighbor'][face_bad]] = True

    return int(np.sum(bad))
