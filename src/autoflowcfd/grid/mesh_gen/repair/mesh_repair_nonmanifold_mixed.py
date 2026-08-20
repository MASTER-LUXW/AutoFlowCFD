"""针对跨越棱柱(BL)+四面体(transition/core)混合网格的非流形面局部 cavity
修补——是 mesh_repair_cavity.patch_nonmanifold_cavity 的混合网格版本。

单独拆成一个模块（而不是加进已经超过本项目 450 行上限的
mesh_repair_cavity.py），纯粹为了控制文件行数；两个模块用的是同一套
底层技巧。
"""

from typing import List, Tuple

import numpy as np
from loguru import logger

from .mesh_repair_cavity_shared import (
    _CAVITY_FACE_TEMPLATES,
    _cavity_boundary_faces,
    _count_bad_cells,
    _weld_near_coincident_boundary_points,
)
from .mesh_repair_nonmanifold_mixed_demote import _split_prisms_to_tets, demote_invalid_prisms_to_tets  # noqa: F401  (demote_invalid_prisms_to_tets 是本模块的公开出口，保持原有导入路径可用)

# _weld_near_coincident_boundary_points 的容差——空腔边界面自身中位边长
# 的比例，不是固定长度（见该函数自己的文档字符串了解为什么必须用局部
# 尺度）。V2.0 专项攻关记录（cube_demo BL 质量campaign 第十九轮）：这个
# 函数是 BL 挤出前沿被撕裂（见 mesh_front_collision.py 模块文档字符串）
# 后唯一实际"缝合"出退化 sliver 的地方——之前对重铺结果完全没有事后
# 质量校验，也没有对输入边界点集本身的近重合焊接，两者都在本轮补上。
#
# 0.10 是真实 cube_demo A/B 扫描出的安全上界，不是随意选的：
# <=0.10（含本值）在整个真实生产管线上稳定安全——0.05 时焊接从未
# 实际触发（tolerance 太紧，0 次焊接，相邻单元体积比的改善完全来自
# 下面的事后质量门控），0.08/0.10 时焊接开始真正生效（5~13 次，
# 每次合并几个到十几个点），对主指标（相邻单元体积比 336.57->33.20）
# 零额外影响、对非正交度等其它指标零/可忽略的额外副作用；但
# >=0.12 时同一真实网格上会崩溃——真实报错"3 faces are shared by
# more than 2 cells"，即真正的非流形撕裂：更松的容差开始把同一空腔
# 边界里*彼此独立、各自与外部保留单元有真实缝合关系*的不同点也当成
# "近重合"合并掉（0.15 时单个空腔一次合并 192 个点、丢弃 520 个退化
# 面——远超"一对撕裂角点"的规模），把其中一个的索引整个丢弃相当于
# 撕开该点与外部网格的缝合缝。焊接函数自身的"只丢弃索引、不移动幸存
# 点坐标"设计本来就是为了不去扰动外部共享的真实缝合点，但无法从
# 距离本身分辨"确实是同一撕裂点"和"恰好彼此靠近的两个不同缝合点"——
# 容差越松，后一类假阳性合并的概率越高，真实数据显示 0.10->0.12 之间
# 就是这个假阳性开始产生真实拓扑损伤的转折点。选 0.10（安全区间的
# 上沿，不是更保守的 0.05/0.08）是为了让焊接机制在真实撕裂案例上尽量
# 有实际参与的机会，而不是退化成事实上从不触发的死代码。
CAVITY_WELD_TOLERANCE_FRACTION = 0.10


def patch_nonmanifold_cavity_mixed(
    nodes: np.ndarray,
    prism_cells: np.ndarray,
    tet_cells: np.ndarray,
    prism_keep: np.ndarray,
    tet_keep: np.ndarray,
    bl_cell_groups: np.ndarray,
    cell_groups: np.ndarray,
    n_buffer_rings: int = 1,
    max_cavity_cells: int = 5000,
    max_clusters_attempted: int = 20_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """在棱柱(BL)+四面体(transition/core)混合网格上局部重铺非流形/标记坏的空腔。

    是 mesh_repair_cavity.patch_nonmanifold_cavity 的混合网格版本。
    不调用 face_extractor，而是直接在混合单元集上操作。
    repair_nonmanifold_mixed 自身的"保留最大、丢弃其余"策略会导致
    超过共享面丢失一侧的真实孔洞——参见
    mesh_repair_cavity.patch_nonmanifold_cavity 的文档字符串了解原因。

    每个连通的种子单元簇（接触超过共享面的，或被调用方的 keep 掩码
    标记为坏的——例如 mesh_background.py 的 BL 棱柱纵横比检查）都成为
    自己独立的空腔，这与 mesh_repair_cavity.remesh_core_cavity 的模块
    文档字符串中的理由完全相同：一个跨越多个不相关坏区域的合并空腔
    会 (a) 无谓地超过 max_cavity_cells 限制，(b) 不必要地重铺两个
    不相关区域之间的好几何。具体实例：一个真实的 cube_demo 运行有
    约 21,000 个被标记为"坍缩角"的 BL 棱柱（见 mesh_background.py
    的纵横比检查），它们几乎完全是散布在整个物体表面的独立小簇，
    而不是一个连续区域——将它们作为单个空腔处理（本函数的早期版本
    就是这样做的）仅一个缓冲环就让合并种子超过了 max_cavity_cells，
    然后对所有这些簇完全无操作。按簇拆分让每个独立的小空腔（通常
    只有几个单元）可以独立修补，即使总的标记计数很大。

    被卷入空腔的任何棱柱（种子或缓冲环）首先被拆分为 3 个四面体
    (_split_prisms_to_tets)，这样空腔可以作为纯 tet PLC 交给单个
    tetgen 调用——tetgen 自身没有棱柱基本体。重铺实际替换的每个
    单元（无论来自棱柱还是四面体）都返回为普通内部四面体；没有
    东西被重新提升为棱柱——这与 remesh_core_cavity 的局部重铺
    结果做出的刻意、有界的权衡相同。

    Args:
        nodes: 完整节点数组（两种单元类型的共享坐标空间）。
        prism_cells, tet_cells: 当前单元数组（在应用 prism_keep/tet_keep
            之前——两者都是提议，尚未执行）。
        prism_keep, tet_keep: bool 数组——False 标记一个否则会被无条件
            丢弃/标记为坏的单元。
        bl_cell_groups: (n_prism,) 字符串数组，与 prism_cells 平行。
        cell_groups: (n_tet,) 字符串数组，与 tet_cells 平行——每个
            新重铺的单元获得 ''（与 patch_nonmanifold_cavity 相同的约定）。
        n_buffer_rings: 在提取每个簇的边界之前，围绕每个簇的面邻接
            环数。
        max_cavity_cells: 单簇安全上限（不是总预算）——单个簇这么大
            表明结构上不同的东西（参见 patch_nonmanifold_cavity 的
            文档字符串）；跳过而不是尝试，与 remesh_core_cavity 的
            单簇大小上限相同。
        max_clusters_attempted: 本次调用将尝试的独立簇总数上限，
            与 remesh_core_cavity 的上限出于相同原因（许多小簇，
            每个都便宜，但每次调用 tetgen 有实际开销，仍会累积）。

    Returns:
        (new_nodes, new_prism_cells, new_tet_cells, new_bl_cell_groups,
        new_cell_groups) —— 如果两个 keep 掩码已经全为 True 则返回
        未修改的（非副本）原始数组；否则反映成功修补的簇数（0 或更多
        ——部分结果，超大/失败的簇保持调用方 keep 掩码找到的原样，
        是预期且正常的，不是错误）。
    """
    if prism_keep.all() and tet_keep.all():
        return nodes, prism_cells, tet_cells, bl_cell_groups, cell_groups

    from ..tetgen.mesh_tetgen_core import fill_core_volume, CORE_TETGEN_MINRATIO, CORE_TETGEN_MINDIHEDRAL
    from ...validation.quality_validator import MeshQualityValidator
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    validator = MeshQualityValidator()

    n_prism = len(prism_cells)
    n_tet = len(tet_cells)
    n_total = n_prism + n_tet

    # 本函数专用的全局单元 ID 约定：[0, n_prism) 是棱柱，
    # [n_prism, n_prism+n_tet) 是四面体。
    keep = np.concatenate([prism_keep, tet_keep])

    # 从后面实际用于重铺空腔的同一组已验证的 3-tet 拆分推导
    # 每个棱柱的 8 个边界三角形，而不是直接手写四边形对角线
    # （手写对角线猜测已确认与 3-tet 拆分的真实暴露边界不匹配）。
    # 棱柱的 3 个拆分 tet 在 cell_of_face 中共享该棱柱的索引，
    # 所以棱柱总是作为整体生长/替换，从不会部分操作。
    if n_prism:
        prism_as_tets = _split_prisms_to_tets(prism_cells)  # (3*n_prism, 4)
        prism_faces = prism_as_tets[:, _CAVITY_FACE_TEMPLATES].reshape(-1, 3)
        # _split_prisms_to_tets block-concatenates (all T1's, then all
        # T2's, then all T3's) rather than interleaving per prism.
        prism_cell_of_face = np.repeat(np.tile(np.arange(n_prism), 3), 4)
    else:
        prism_faces = np.empty((0, 3), dtype=np.int64)
        prism_cell_of_face = np.empty((0,), dtype=np.int64)

    tet_faces = tet_cells[:, _CAVITY_FACE_TEMPLATES].reshape(-1, 3) if n_tet else np.empty((0, 3), dtype=np.int64)
    tet_cell_of_face = (n_prism + np.repeat(np.arange(n_tet), 4)) if n_tet else np.empty((0,), dtype=np.int64)

    all_faces = np.vstack([prism_faces, tet_faces])
    cell_of_face = np.concatenate([prism_cell_of_face, tet_cell_of_face])
    # A degenerate (repeated-vertex) face - from a "collapsed corner"
    # prism whose growth froze at exactly one base vertex, splitting into
    # one fully-degenerate sub-tet - is not a real geometric face and
    # must not participate in adjacency/grouping at all: left in, it
    # collides with itself and with genuinely-unrelated faces that happen
    # to share the same repeated node, corrupting both the non-manifold
    # detection and the cavity-growing graph (confirmed directly: this
    # was the actual cause of ~23,000 phantom cavity seeds in an earlier,
    # unfiltered version of this function).
    degenerate = (
        (all_faces[:, 0] == all_faces[:, 1])
        | (all_faces[:, 0] == all_faces[:, 2])
        | (all_faces[:, 1] == all_faces[:, 2])
    )
    all_faces = all_faces[~degenerate]
    cell_of_face = cell_of_face[~degenerate]

    sorted_faces = np.sort(all_faces, axis=1)
    face_dtype = np.dtype((np.void, sorted_faces.dtype.itemsize * 3))
    voids = np.ascontiguousarray(sorted_faces).view(face_dtype).reshape(-1)
    _, group_id, group_counts = np.unique(voids, return_inverse=True, return_counts=True)
    group_id = group_id.ravel()

    nonmanifold_group = group_counts[group_id] > 2
    dropped_group = np.zeros(len(group_counts), dtype=bool)
    np.logical_or.at(dropped_group, group_id, ~keep[cell_of_face])
    seed_occurrence = nonmanifold_group | dropped_group[group_id]
    seed = np.zeros(n_total, dtype=bool)
    seed[cell_of_face[seed_occurrence]] = True

    # 仅限内部（count==2）面的面邻接图，用于将种子单元聚类为
    # 连通分量并生长每个簇的缓冲环——count>2（非流形）面没有
    # 单个明确的"另一侧"可以穿过，count==1（边界）面则完全没有。
    interior_group = np.flatnonzero(group_counts == 2)
    interior_occ = np.isin(group_id, interior_group)
    occ_cell = cell_of_face[interior_occ]
    occ_group = group_id[interior_occ]
    order = np.argsort(occ_group, kind='stable')
    occ_cell_sorted = occ_cell[order]
    owner = occ_cell_sorted[0::2]
    neighbor = occ_cell_sorted[1::2]

    seed_idx = np.flatnonzero(seed)
    if len(seed_idx) == 0:
        logger.warning("Non-manifold mixed-cavity patch: seed set empty after degenerate-face filtering - falling back to plain cell removal")
        return nodes, prism_cells, tet_cells, bl_cell_groups, cell_groups

    seed_pos = -np.ones(n_total, dtype=np.int64)
    seed_pos[seed_idx] = np.arange(len(seed_idx))
    edge_mask = seed[owner] & seed[neighbor]
    rows = seed_pos[owner[edge_mask]]
    cols = seed_pos[neighbor[edge_mask]]
    graph = coo_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(len(seed_idx), len(seed_idx)))
    n_clusters, labels = connected_components(graph, directed=False)

    if n_clusters > max_clusters_attempted:
        logger.warning(
            f"Non-manifold mixed-cavity patch: {n_clusters} candidate cluster(s) found, "
            f"capping at {max_clusters_attempted} attempts"
        )

    claimed = np.zeros(n_total, dtype=bool)
    accepted: List[dict] = []
    n_skipped_size = 0
    n_failed = 0
    n_rejected = 0

    for cluster_id in range(min(n_clusters, max_clusters_attempted)):
        cluster_seed_mask = np.zeros(n_total, dtype=bool)
        cluster_seed_mask[seed_idx[labels == cluster_id]] = True

        cavity = cluster_seed_mask & ~claimed
        for _ in range(n_buffer_rings + 1):
            touches = cavity[owner] | cavity[neighbor]
            if not np.any(touches):
                break
            newly = np.zeros_like(cavity)
            newly[owner[touches]] = True
            newly[neighbor[touches]] = True
            newly &= ~claimed
            if np.array_equal(newly | cavity, cavity):
                break
            cavity |= newly

        cavity_idx = np.flatnonzero(cavity)
        if len(cavity_idx) == 0:
            continue
        if len(cavity_idx) > max_cavity_cells:
            n_skipped_size += 1
            continue

        cavity_prism_idx = cavity_idx[cavity_idx < n_prism]
        cavity_tet_idx = cavity_idx[cavity_idx >= n_prism] - n_prism

        cavity_as_tets = np.vstack([
            _split_prisms_to_tets(prism_cells[cavity_prism_idx]) if len(cavity_prism_idx) else np.empty((0, 4), dtype=prism_cells.dtype),
            tet_cells[cavity_tet_idx] if len(cavity_tet_idx) else np.empty((0, 4), dtype=tet_cells.dtype),
        ]).astype(np.int64)

        boundary_faces = _cavity_boundary_faces(cavity_as_tets, np.arange(len(cavity_as_tets)))
        global_pts = np.unique(boundary_faces)
        local_of_global = -np.ones(len(nodes), dtype=np.int64)
        local_of_global[global_pts] = np.arange(len(global_pts))
        local_faces = local_of_global[boundary_faces].astype(np.int32)
        local_points = nodes[global_pts]

        # 焊接空腔自身边界点集里的近重合点对（撕裂的 BL 前沿留下的典型
        # 产物），在把边界原样交给 tetgen 之前——见
        # _weld_near_coincident_boundary_points 自己的文档字符串了解
        # 为什么这必须在 tetgen 调用*之前*做，"重铺完再拒绝"本身不能
        # 修复由退化输入边界导致的退化输出。
        local_points, local_faces, global_pts = _weld_near_coincident_boundary_points(
            local_points, local_faces, global_pts,
            tolerance_fraction=CAVITY_WELD_TOLERANCE_FRACTION,
        )

        try:
            retiled_nodes, retiled_tets, _, _ = fill_core_volume(
                local_points, local_faces, verbose=False,
                minratio=CORE_TETGEN_MINRATIO, mindihedral=CORE_TETGEN_MINDIHEDRAL,
            )
        except Exception as e:
            # 只计数会让排查为什么某个 cavity retile 失败变得很困难，
            # 这里把具体异常记下来（debug 级别，不打断批量修复流程）。
            logger.debug(f"  Cavity retile failed for cluster with {len(global_pts)} boundary points: {e}")
            n_failed += 1
            continue

        n_boundary_pts = len(local_points)
        if not np.array_equal(retiled_nodes[:n_boundary_pts], local_points):
            n_failed += 1
            continue

        # 事后体积/形状质量门控——这个混合网格版本此前完全没有（对照
        # mesh_repair_cavity.remesh_core_cavity/_attempt_cavity_retile_
        # clusters 已有的 is_improvement/is_acceptable_fallback 门控），
        # 是本函数把撕裂"缝合"成 sliver 却从不拒绝的直接原因（见
        # mesh_front_collision.py 模块文档字符串了解撕裂如何产生）。
        # 与 Stage B' 相同的双重接受条件：严格改善，或者原始空腔本来就
        # 只有很少（<=2）坏单元时至少不变差——用于在困难几何特征上
        # tetgen 找不到完美解时打破死锁，而不是无限拒绝。
        old_bad = _count_bad_cells(validator, nodes, cavity_as_tets)
        bad_new = _count_bad_cells(validator, retiled_nodes, retiled_tets)
        is_improvement = bad_new < old_bad
        is_acceptable_fallback = (old_bad <= 2 and bad_new <= old_bad)
        if not is_improvement and not is_acceptable_fallback:
            logger.debug(
                f"  Cavity retile of {len(cavity_idx)} cell(s) ({old_bad} bad) -> "
                f"{len(retiled_tets)} cell(s) ({bad_new} bad) - not an improvement, "
                f"keeping original cells"
            )
            n_rejected += 1
            continue

        claimed[cavity_idx] = True
        accepted.append(dict(
            cavity_prism_idx=cavity_prism_idx, cavity_tet_idx=cavity_tet_idx,
            global_pts=global_pts, retiled_nodes=retiled_nodes, retiled_tets=retiled_tets,
            n_boundary_pts=n_boundary_pts,
        ))

    if not accepted:
        logger.warning(
            f"Non-manifold mixed-cavity patch: {n_clusters} cluster(s) found, none "
            f"accepted (skipped_size={n_skipped_size}, rejected={n_rejected}, "
            f"failed={n_failed}) - falling back to plain cell removal"
        )
        return nodes, prism_cells, tet_cells, bl_cell_groups, cell_groups

    keep_prism_outside = np.ones(n_prism, dtype=bool)
    keep_tet_outside = np.ones(n_tet, dtype=bool)
    for res in accepted:
        keep_prism_outside[res['cavity_prism_idx']] = False
        keep_tet_outside[res['cavity_tet_idx']] = False

    new_nodes_parts = [nodes]
    new_tet_parts = [tet_cells[keep_tet_outside]]
    new_group_parts = [cell_groups[keep_tet_outside]]
    interior_start = len(nodes)

    for res in accepted:
        global_pts = res['global_pts']
        retiled_nodes = res['retiled_nodes']
        retiled_tets = res['retiled_tets']
        n_boundary_pts = res['n_boundary_pts']

        is_boundary = retiled_tets < n_boundary_pts
        remapped = np.empty_like(retiled_tets)
        remapped[is_boundary] = global_pts[retiled_tets[is_boundary]]
        remapped[~is_boundary] = interior_start + (retiled_tets[~is_boundary] - n_boundary_pts)

        new_interior_nodes = retiled_nodes[n_boundary_pts:]
        new_nodes_parts.append(new_interior_nodes)
        new_tet_parts.append(remapped.astype(tet_cells.dtype))
        new_group_parts.append(np.full(len(remapped), '', dtype=object))
        interior_start += len(new_interior_nodes)

    new_nodes = np.vstack(new_nodes_parts)
    new_prism_cells = prism_cells[keep_prism_outside]
    new_bl_cell_groups = bl_cell_groups[keep_prism_outside]
    new_tet_cells = np.vstack(new_tet_parts)
    new_cell_groups = np.concatenate(new_group_parts)

    n_cavity_cells_replaced = sum(len(r['cavity_prism_idx']) + len(r['cavity_tet_idx']) for r in accepted)
    n_new_cells = sum(len(r['retiled_tets']) for r in accepted)
    logger.info(
        f"Non-manifold mixed-cavity patch: {len(accepted)}/{n_clusters} cluster(s) patched "
        f"({n_cavity_cells_replaced} cell(s) -> {n_new_cells} local retile cell(s); "
        f"skipped_size={n_skipped_size}, rejected={n_rejected}, failed={n_failed})"
    )
    return new_nodes, new_prism_cells, new_tet_cells, new_bl_cell_groups, new_cell_groups
