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

    退化面过滤（顶点索引在自身 3 个槽位里有重复，零面积三角形）：必须
    先剔除再做"恰好出现一次"的边界判定，否则一个退化的输入单元（例如
    折叠角棱柱经 _split_prisms_to_tets 拆分产生的那个恰好重复引用同一
    节点的第 3 个子四面体——patch_nonmanifold_cavity_mixed 会把这类
    prism 在被 demote_invalid_prisms_to_tets 真正处理之前，先按极端
    长细比送进这里做局部重铺）会贡献一个零面积三角形：如果这个空腔里
    没有其他单元也贡献同一个退化三角形，它会被误判为"出现恰好一次"
    的合法边界面，混进传给 tetgen 的固定 PLC 边界——已用真实 cube_demo
    数据实测确认（V2.0 专家组评审）：这正是把 n_buffer_rings 从默认 1
    调大后暴露出的崩溃根因（局部 tetgen 调用的固定边界本身包含退化
    三角形，产出的重铺结果在该退化三角形附近变得不可预测，表现为若干
    输出四面体的多个面坍缩成同一个三角形，最终在下游全网格
    face_extractor 上产生"面被 2 个以上单元引用"的拓扑异常甚至孤立
    单元）。与本文件同一模块里 patch_nonmanifold_cavity_mixed 自身
    在种子/聚类阶段已经在做的退化面过滤（`degenerate = (faces[:,0]==
    faces[:,1])|...`）是同一件事，这里之前遗漏了。
    """
    cav_cells = cells[cavity_cell_idx]
    all_faces = cav_cells[:, _CAVITY_FACE_TEMPLATES].reshape(-1, 3)
    degenerate = (
        (all_faces[:, 0] == all_faces[:, 1])
        | (all_faces[:, 0] == all_faces[:, 2])
        | (all_faces[:, 1] == all_faces[:, 2])
    )
    all_faces = all_faces[~degenerate]
    sorted_faces = np.sort(all_faces, axis=1)
    face_dtype = np.dtype((np.void, sorted_faces.dtype.itemsize * 3))
    voids = np.ascontiguousarray(sorted_faces).view(face_dtype).reshape(-1)
    _, inverse, counts = np.unique(voids, return_inverse=True, return_counts=True)
    boundary_mask = counts[inverse] == 1
    return all_faces[boundary_mask]


def _weld_near_coincident_boundary_points(
    local_points: np.ndarray,
    local_faces: np.ndarray,
    global_pts: np.ndarray,
    tolerance_fraction: float,
):
    """在交给 tetgen 之前，焊接一个空腔自身边界点集里彼此距离小于局部
    特征尺度某个比例的近重合点对。

    tetgen 做约束 Delaunay 时必须精确尊重给定的边界点集（只能加内部
    Steiner 点，不能移动/合并边界点）——如果边界点集本身含有一对近
    重合点（例如一个被 BL 挤出前沿撕裂的三角形留下的一对相差仅几毫米
    的角点，见 mesh_front_collision.py 模块文档字符串了解撕裂如何产生），
    任何重铺结果都绕不开在这两点之间产生退化/近退化的薄片单元——纯粹
    "重铺完再拒绝、回退"的质量门（见 _count_bad_cells 的调用方）解决不了
    这个根因，因为对这个具体边界输入，tetgen 能给出的每一个合法重铺
    结果都同样退化。必须在调用 tetgen 之前就把这类点对焊接掉。

    容差用**局部**特征尺度（这个空腔自身边界面的中位边长）的一个比例，
    不是固定的全局常量——与 mesh_tetgen_core.compute_local_thickness_limit
    用局部而非全局常量的既有先例一致：不同区域的网格尺寸差异很大（BL
    近壁 sub-mm 尺度 vs. core 区域可以到 cm 量级），固定容差要么在细密
    区域太松（合并真正不同的顶点，撕开与外部网格的缝合缝），要么在
    粗糙区域太紧（漏掉这个函数本该焊接的撕裂对）。

    焊接是纯粹的索引合并（保留每个组里*最小局部索引*那个点的原始坐标
    作为代表，不取质心平均）——不移动任何幸存点的坐标。这样被合并组
    里"消失"的那个索引，如果确实是这个空腔与外部保留单元共享的真实
    缝合点，外部单元仍然引用它自己的原始（未变的）全局索引和坐标，不
    受影响；只有这个空腔自己的重铺不再把它当独立点处理。合并后按
    mesh_repair_cavity_shared 自己 `_cavity_boundary_faces` 的既有先例
    过滤退化（重复顶点索引）面。

    Args:
        local_points: (n, 3) 空腔边界点局部坐标（`nodes[global_pts]`）
        local_faces: (m, 3) 局部索引三角形（索引到 `local_points`）
        global_pts: (n,) 与 `local_points` 平行的原始全局节点索引，
            用于把重铺结果的边界部分映射回调用方的全局节点数组
        tolerance_fraction: 焊接容差占本空腔边界面中位边长的比例；
            <= 0 时直接原样返回，不做任何事

    Returns:
        (new_local_points, new_local_faces, new_global_pts) - 未发生
        焊接时是输入的（非副本）原始数组
    """
    n = len(local_points)
    if tolerance_fraction <= 0.0 or n == 0 or len(local_faces) == 0:
        return local_points, local_faces, global_pts

    edges = np.vstack([local_faces[:, [0, 1]], local_faces[:, [1, 2]], local_faces[:, [2, 0]]])
    edge_len = np.linalg.norm(local_points[edges[:, 0]] - local_points[edges[:, 1]], axis=1)
    edge_len = edge_len[edge_len > 1e-300]
    if len(edge_len) == 0:
        return local_points, local_faces, global_pts
    local_scale = float(np.median(edge_len))
    tol = tolerance_fraction * local_scale
    if tol <= 0.0:
        return local_points, local_faces, global_pts

    from scipy.spatial import cKDTree

    tree = cKDTree(local_points)
    pairs = tree.query_pairs(r=tol, output_type='ndarray')
    if len(pairs) == 0:
        return local_points, local_faces, global_pts

    # 并查集，合并时总是把较大索引的根接到较小索引的根上——保证每组
    # 最终收敛到该组*最小*原始局部索引，无论 pairs 的处理顺序如何。
    parent = np.arange(n)

    def _find(i: int) -> int:
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    for a, b in pairs:
        ra, rb = _find(int(a)), _find(int(b))
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    root = np.array([_find(i) for i in range(n)])
    survivors, remap_compact = np.unique(root, return_inverse=True)
    if len(survivors) == n:
        return local_points, local_faces, global_pts

    new_local_points = local_points[survivors]
    new_global_pts = global_pts[survivors]

    new_faces = remap_compact[local_faces]
    degenerate = (
        (new_faces[:, 0] == new_faces[:, 1])
        | (new_faces[:, 0] == new_faces[:, 2])
        | (new_faces[:, 1] == new_faces[:, 2])
    )
    n_degenerate = int(np.sum(degenerate))
    new_faces = new_faces[~degenerate]

    logger.info(
        f"Cavity boundary weld: merged {n - len(survivors)} near-coincident "
        f"point(s) (tolerance {tol:.4e} m = {tolerance_fraction:.1%} of local "
        f"median edge length {local_scale:.4e} m) before local retile"
        + (f", dropped {n_degenerate} degenerate face(s)" if n_degenerate else "")
    )

    return new_local_points, new_faces.astype(local_faces.dtype), new_global_pts


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
