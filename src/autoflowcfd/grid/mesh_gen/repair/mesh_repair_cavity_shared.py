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


def _repeated_node_rows(cells: np.ndarray) -> np.ndarray:
    s = np.sort(cells, axis=1)
    return (s[:, 1:] == s[:, :-1]).any(axis=1)


def retile_is_conformal(retiled_tets: np.ndarray, n_boundary_pts: int, local_faces: np.ndarray) -> bool:
    """重剖分的外表面（只出现一次的非退化面）是否与输入边界 `local_faces` 逐面相同（按排序后的节点三元组）。

    全部空腔重剖分（Stage B'、非流形补丁、混合网格补丁）接受前都必须通过：tetgen 在空腔边界上插 Steiner 点时
    重剖分的外表面是边界三角形的细分，与空腔外单元的面对不上、网格在这里留缝——"边界点原样保留在节点数组
    最前面"那一项检查看不出来（2026-10-09 以前只有那一项）。
    """
    faces = retiled_tets[:, _CAVITY_FACE_TEMPLATES].reshape(-1, 3)
    faces = faces[~_repeated_node_rows(faces)]
    s = np.sort(faces, axis=1)
    uniq, counts = np.unique(s, axis=0, return_counts=True)
    outer = uniq[counts == 1]
    if len(outer) != len(local_faces) or (outer >= n_boundary_pts).any():
        return False
    want = np.unique(np.sort(np.asarray(local_faces, dtype=outer.dtype), axis=1), axis=0)
    return len(want) == len(outer) and np.array_equal(want, outer)


def _glued_face_pairs(faces: np.ndarray) -> np.ndarray:
    """焊接后成对重合、朝向相反的边界面（掩码，两份都标记）。

    撕裂缝两侧的面 (A,C,D) 与 (B,C,D) 在 A 并入 B 之后是同一个三角形、朝向相反——缝隙闭合，它成了缝两侧
    空腔外单元之间的内部面，不再是空腔边界，必须成对移出交给 tetgen 的边界（留着就是重复面，tetgen 无法
    原样保留边界）。同向重合是折叠、出现超过两次是非流形，都不标记：留给 tetgen 失败、该空腔不修补。
    """
    if len(faces) < 2:
        return np.zeros(len(faces), dtype=bool)
    key = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    glued = np.zeros(len(faces), dtype=bool)
    for g in np.flatnonzero(counts == 2):
        i, j = np.flatnonzero(inverse == g)
        # 同一三元组的两种循环序：把 i 旋转到与 j 同起点后比较方向
        fi, fj = list(faces[i]), list(faces[j])
        k = fi.index(fj[0])
        rotated = fi[k:] + fi[:k]
        if rotated[1] == fj[2] and rotated[2] == fj[1]:
            glued[i] = glued[j] = True
    return glued


def _weld_near_coincident_boundary_points(
    local_points: np.ndarray,
    local_faces: np.ndarray,
    global_pts: np.ndarray,
    tolerance_fraction: float,
    protected: np.ndarray,
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

    焊接是纯粹的索引合并（保留每组里一个点的原始坐标作为代表，不取质心平均）——不移动任何幸存点的坐标。
    合并结果以 `(removed, survivor)` 全局索引对返回，**调用方必须把它施加到全部单元上**（空腔外引用被合并点
    的单元一律改引用幸存点），否则空腔外单元的面与重铺结果的边界面对不上，网格在这里不再封闭。
    2026-10-09 以前这里写的是"外部单元仍然引用它自己的原始全局索引，不受影响"——这正是缺陷：重铺的边界面
    引用幸存点、外部单元的面引用被合并点，两侧各剩一个外部面；被丢弃的退化面在外部单元一侧同样成了外部面。
    cube_demo 实测：体网格 39574 个外部面，面网格只有 39488 个三角形，多出的 86 个就出在这里
    （`patch_nonmanifold_cavity_mixed` 照样报告"修补成功"）。

    受保护点（`protected[global]` 为真：域外边界上的点、已被接受的重铺引用的点）不会被合并掉：一组里有受保护
    点时它当幸存者，两个受保护点不合并（并查集保证一组里至多一个受保护点）——移动域外边界点会改变物体表面
    几何，移动已接受重铺的边界点会改变那份已经过质量门的重铺。合并后按 `_cavity_boundary_faces` 的既有先例
    过滤退化（重复顶点索引）面。

    Args:
        local_points: (n, 3) 空腔边界点局部坐标（`nodes[global_pts]`）
        local_faces: (m, 3) 局部索引三角形（索引到 `local_points`）
        global_pts: (n,) 与 `local_points` 平行的原始全局节点索引，
            用于把重铺结果的边界部分映射回调用方的全局节点数组
        tolerance_fraction: 焊接容差占本空腔边界面中位边长的比例；
            <= 0 时直接原样返回，不做任何事
        protected: (n_global_nodes,) bool，受保护的全局节点（见上）

    Returns:
        (new_local_points, new_local_faces, new_global_pts, removed, survivor) - 未发生焊接时前三项是输入的
        （非副本）原始数组、后两项为空；`removed[i]` 合并到 `survivor[i]`（全局索引）
    """
    no_weld = (local_points, local_faces, global_pts,
               np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    n = len(local_points)
    if tolerance_fraction <= 0.0 or n == 0 or len(local_faces) == 0:
        return no_weld

    edges = np.vstack([local_faces[:, [0, 1]], local_faces[:, [1, 2]], local_faces[:, [2, 0]]])
    edge_len = np.linalg.norm(local_points[edges[:, 0]] - local_points[edges[:, 1]], axis=1)
    edge_len = edge_len[edge_len > 1e-300]
    if len(edge_len) == 0:
        return no_weld
    local_scale = float(np.median(edge_len))
    tol = tolerance_fraction * local_scale
    if tol <= 0.0:
        return no_weld

    from scipy.spatial import cKDTree

    tree = cKDTree(local_points)
    pairs = tree.query_pairs(r=tol, output_type='ndarray')
    if len(pairs) == 0:
        return no_weld

    # 并查集：受保护点总是当根；两个根都受保护时不合并；都不受保护时较大索引的根接到较小索引的根上
    # ——结果与 pairs 的处理顺序无关。
    is_protected = np.asarray(protected, dtype=bool)[global_pts]
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
        if ra == rb or (is_protected[ra] and is_protected[rb]):
            continue
        if is_protected[rb] or (not is_protected[ra] and rb < ra):
            ra, rb = rb, ra
        parent[rb] = ra

    root = np.array([_find(i) for i in range(n)])
    survivors, remap_compact = np.unique(root, return_inverse=True)
    if len(survivors) == n:
        return no_weld
    merged = np.flatnonzero(root != np.arange(n))

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
    glued = _glued_face_pairs(new_faces)
    new_faces = new_faces[~glued]
    # 只在粘合面上出现的点已不在空腔边界上（它现在夹在空腔外两个单元之间），不能交给 tetgen
    used = np.unique(new_faces)
    if len(used) < len(new_local_points):
        compact = -np.ones(len(new_local_points), dtype=np.int64)
        compact[used] = np.arange(len(used))
        new_faces = compact[new_faces]
        new_local_points = new_local_points[used]
        new_global_pts = new_global_pts[used]

    logger.info(
        f"Cavity boundary weld: merged {n - len(survivors)} near-coincident "
        f"point(s) (tolerance {tol:.4e} m = {tolerance_fraction:.1%} of local "
        f"median edge length {local_scale:.4e} m) before local retile"
        + (f", dropped {n_degenerate} degenerate face(s)" if n_degenerate else "")
        + (f", {int(glued.sum()) // 2} glued face pair(s)" if glued.any() else "")
    )

    return (new_local_points, new_faces.astype(local_faces.dtype), new_global_pts,
            global_pts[merged].astype(np.int64), global_pts[root[merged]].astype(np.int64))


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
    # 真实 bug（V2.0 专家组盲审发现）：FaceExtractor 实际所在模块是
    # autoflowcfd.grid.mesh_gen.extraction.face_extractor（见上面
    # import 语句），少了 ".extraction." 这一级；loguru 的
    # disable/enable 按模块名精确/前缀匹配，路径不对时静默不生效——
    # 这条日志抑制此前从未真正生效过，本函数文档里"70K+ 行日志"的
    # 噪音问题实际仍然存在，只是错误地看起来已经被抑制。
    logger.disable("autoflowcfd.grid.mesh_gen.extraction.face_extractor")
    try:
        faces = FaceExtractor.extract_faces(cells.astype(np.int32), node_arr)
    finally:
        logger.enable("autoflowcfd.grid.mesh_gen.extraction.face_extractor")
    diag = validator.compute_face_diagnostics(nodes, cells, faces)
    if len(diag['angle_deg']) > 0:
        face_bad = (
            (diag['angle_deg'] > validator.thresholds['max_orthogonality_angle'])
            | (diag['volume_ratio'] > validator.thresholds['max_adjacent_volume_ratio'])
        )
        bad[diag['owner'][face_bad]] = True
        bad[diag['neighbor'][face_bad]] = True

    return int(np.sum(bad))
