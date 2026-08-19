"""非流形 cavity 局部修补：用重新四面体化替代直接删除单元。

从 mesh_repair_cavity.py 拆分出来。patch_nonmanifold_cavity 是
repair_nonmanifold_cells（mesh_tetgen_postprocess.py）"保留最大体积、
丢弃其余"这个默认修复策略的替代方案——当被丢弃的单元其实是几何上真实存在
的一块区域（只是恰好和另一侧重复占用了同一空间）时，直接删除会在网格里
留下一个洞；这里改成局部重新铺一层四面体来填补它。
"""

from typing import Optional, Tuple

import numpy as np
from loguru import logger

from .mesh_repair_cavity_shared import _CAVITY_FACE_TEMPLATES, _cavity_boundary_faces


def patch_nonmanifold_cavity(
    nodes: np.ndarray,
    cells: np.ndarray,
    keep_mask: np.ndarray,
    cell_groups: np.ndarray,
    n_bl_cells: int,
    n_buffer_rings: int = 1,
    max_cavity_cells: int = 5000,
    bad_cell_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, Optional[np.ndarray]]:
    """对 mesh_tetgen_core.repair_nonmanifold_cells 标记的区域进行局部
    重新四面体化，代替仅删除它标记移除的单元（keep_mask False）并在
    原位留下孔洞。

    为何需要此函数：repair_nonmanifold_cells 自身对 3+ 单元共享面的
    修复是"保留最大、丢弃其余"——在额外单元是真正冗余重复时正确，
    但当它们来自网格的两个不同区域（例如过渡四面体阶段和 tetgen
    核心填充）都合法试图在锐角处占据相同空间时，丢弃"输方"侧的
    单元会移除真实几何而没有生成替换——最终网格中的真实间隙。
    已直接确认，非理论：真实 cube_demo 运行测得 0.189 m^3 缺失
    （合并网格体积 vs. 精确的 bbox 减 body 孔体积，由
    mesh_domain_classify._signed_volume 独立计算）在
    repair_nonmanifold_cells 报告移除单元的同一位置，在导出网格
    截图中可见为沿 body 一侧的虚空。

    方法：将接触过度共享（非流形）面的每个单元——在任一侧，不仅
    是 keep_mask 会丢弃的——作为空腔种子（仅丢弃"输方"单元并
    仅围绕它们重铺有风险使重铺自身的边界仍接触另一个非流形面），
    用 `n_buffer_rings` 圈普通（流形邻接）邻居填充，使空腔自身的新
    边界落在已经良好的区域上，并将其边界交给新的、无约束的 tetgen
    调用——与 remesh_core_cavity 已使用的相同局部空腔技巧，但无条件
    （无质量门控拒绝：消除真实孔洞是严格收益，无论替换单元自身
    的偏斜/正交得分，不像 remesh_core_cavity 对仅低质量单元的
    "仅接受可证明改进"门槛，而非物理缺失）。

    刻意自包含而非重用 FaceExtractor 获取邻接图：此处的输入按定义
    在部分地方是非流形的（这就是调用此函数的原因）——正好是
    FaceExtractor.extract_faces 自身严格模式验证存在以拒绝的条件，
    且即使其非严格模式也被确认（本项目自身历史）在单元完全不被
    任何面引用时仍会硬失败，这是此函数存在以修复的缺陷附近的
    真实风险。

    Args:
        nodes, cells: 任何移除之前的完整网格（repair_nonmanifold_
            cells 自身的 keep_mask 是提议，尚未应用）
        keep_mask: (n_cells,) bool，来自 repair_nonmanifold_cells
            ——False 标记否则会被无条件丢弃的单元
        cell_groups: (n_cells,) 与 cells 平行的字符串数组——每个新
            重铺单元得到 ''（与 remesh_core_cavity 自身对其创建单元
            的约定一致：从不会被重新分类为"过渡"或物理壁面组，因为
            跨越旧过渡/核心接缝的补丁比它替换的任何一侧都更接近
            普通内部几何）
        n_bl_cells: cells[:n_bl_cells] 在来源上为过渡阶段（参见
            generate_hybrid_mesh 自身的 n_bl_cells 约定）——被扫入
            空腔且未被逐字保留的那些单元数减少；每个新重铺单元
            追加到末尾，即始终计在此分割的核心/通用侧，从不计在
            过渡侧
        n_buffer_rings: 在提取非流形单元簇边界之前填充的普通邻居
            面邻接圈数
        max_cavity_cells: 安全上限——这么大的缺陷表明有结构性问题
            值得自行调查，不适合局部补丁；回退到简单删除
        bad_cell_mask: 可选 (n_cells,) bool 数组，与 cells 平行——
            与 cell_groups 完全同步保持（每个新重铺单元得到 False，
            即"不已知坏"——与 remesh_core_cavity 自身对其创建单元
            的约定一致），因此跟踪自身坏单元掩码的调用方
            （remesh_core_cavity 自身的重试循环）不必在此调用后
            单独重建它。None（默认）表示调用方没有此类数组要跟踪。

    Returns:
        (new_nodes, new_cells, new_cell_groups, new_n_bl_cells,
        new_bad_cell_mask) - 如果 keep_mask 已全真、空腔超过
        max_cavity_cells、或局部重铺失败/自身仍产生非流形，
        则节点/单元/单元组/bad_cell_mask 不变（非副本）且
        n_bl_cells 原样传递（任一情况都记录日志；调用方自身的
        repair_nonmanifold_cells 删除是对此无法修复情况的安全网）。
        new_bad_cell_mask 为 None 当且仅当 bad_cell_mask 为 None。
    """
    if keep_mask.all():
        return nodes, cells, cell_groups, n_bl_cells, bad_cell_mask

    from ..tetgen.mesh_tetgen_core import fill_core_volume, repair_nonmanifold_cells, CORE_TETGEN_MINRATIO, CORE_TETGEN_MINDIHEDRAL

    n_cells = len(cells)
    all_faces = cells[:, _CAVITY_FACE_TEMPLATES].reshape(-1, 3)
    cell_of_face = np.repeat(np.arange(n_cells), 4)
    sorted_faces = np.sort(all_faces, axis=1)
    face_dtype = np.dtype((np.void, sorted_faces.dtype.itemsize * 3))
    voids = np.ascontiguousarray(sorted_faces).view(face_dtype).reshape(-1)
    _, group_id, group_counts = np.unique(voids, return_inverse=True, return_counts=True)
    group_id = group_id.ravel()

    # Seed: every cell touching a face some OTHER cell also touches
    # (interior, count>=2) where either side is non-manifold (count>2) or
    # keep_mask already flagged one of the sharers for removal - i.e. the
    # whole locally-contested cluster, not just the "losing" cells.
    nonmanifold_group = group_counts[group_id] > 2
    dropped_group = np.zeros(len(group_counts), dtype=bool)
    np.logical_or.at(dropped_group, group_id, ~keep_mask[cell_of_face])
    seed_occurrence = nonmanifold_group | dropped_group[group_id]
    cavity = np.zeros(n_cells, dtype=bool)
    cavity[cell_of_face[seed_occurrence]] = True

    for _ in range(n_buffer_rings + 1):
        group_has_cavity = np.zeros(len(group_counts), dtype=bool)
        np.logical_or.at(group_has_cavity, group_id, cavity[cell_of_face])
        touches_cavity_group = group_has_cavity[group_id] & (group_counts[group_id] >= 2)
        grown = cavity.copy()
        grown[cell_of_face[touches_cavity_group]] = True
        if np.array_equal(grown, cavity):
            break
        cavity = grown

    cavity_idx = np.flatnonzero(cavity)
    if len(cavity_idx) == 0 or len(cavity_idx) > max_cavity_cells:
        logger.warning(
            f"Non-manifold cavity patch: {len(cavity_idx)} cell(s) implicated "
            f"(cap {max_cavity_cells}) - falling back to plain cell removal"
        )
        return nodes, cells, cell_groups, n_bl_cells, bad_cell_mask

    boundary_faces = _cavity_boundary_faces(cells, cavity_idx)
    global_pts = np.unique(boundary_faces)
    local_of_global = -np.ones(len(nodes), dtype=np.int64)
    local_of_global[global_pts] = np.arange(len(global_pts))
    local_faces = local_of_global[boundary_faces].astype(np.int32)
    local_points = nodes[global_pts]

    try:
        retiled_nodes, retiled_tets, _, _ = fill_core_volume(
            local_points, local_faces, verbose=False,
            minratio=CORE_TETGEN_MINRATIO, mindihedral=CORE_TETGEN_MINDIHEDRAL,
        )
    except Exception as e:
        logger.warning(f"Non-manifold cavity patch: local retile failed ({e}), falling back to plain cell removal")
        return nodes, cells, cell_groups, n_bl_cells, bad_cell_mask

    n_boundary_pts = len(local_points)
    if not np.array_equal(retiled_nodes[:n_boundary_pts], local_points):
        logger.warning(
            "Non-manifold cavity patch: boundary points weren't preserved "
            "verbatim by the local retile, falling back to plain cell removal"
        )
        return nodes, cells, cell_groups, n_bl_cells, bad_cell_mask

    keep_outside = np.ones(n_cells, dtype=bool)
    keep_outside[cavity_idx] = False
    interior_start = len(nodes)
    is_boundary = retiled_tets < n_boundary_pts
    remapped = np.empty_like(retiled_tets)
    remapped[is_boundary] = global_pts[retiled_tets[is_boundary]]
    remapped[~is_boundary] = interior_start + (retiled_tets[~is_boundary] - n_boundary_pts)

    new_interior_nodes = retiled_nodes[n_boundary_pts:]
    new_nodes = np.vstack([nodes, new_interior_nodes])
    new_cells = np.vstack([cells[keep_outside], remapped.astype(cells.dtype)])
    new_cell_groups = np.concatenate([
        cell_groups[keep_outside], np.full(len(remapped), '', dtype=object)
    ])
    new_n_bl_cells = int(np.sum(keep_outside[:n_bl_cells]))
    new_bad_cell_mask = (
        np.concatenate([bad_cell_mask[keep_outside], np.zeros(len(remapped), dtype=bool)])
        if bad_cell_mask is not None else None
    )

    #  整体 点 的 retiling 代替 的 deleting is 到 结束 向上 没有
    # a non-manifold defect - verify that actually happened before
    # accepting; if the same corner produces another non-manifold cluster
    # on retile (e.g. a genuinely self-intersecting input geometry, not
    # just an unlucky tetgen tiling choice), fall back rather than accept
    # a patch that didn't fix anything.
    patch_keep = repair_nonmanifold_cells(new_nodes, new_cells)
    if not patch_keep.all():
        logger.warning(
            "Non-manifold cavity patch: retile still produced non-manifold "
            "faces, falling back to plain cell removal"
        )
        return nodes, cells, cell_groups, n_bl_cells, bad_cell_mask

    logger.info(
        f"Patched a {len(cavity_idx)}-cell non-manifold cavity with a "
        f"{len(remapped)}-cell local retile ({len(new_interior_nodes)} new "
        f"interior point(s)) instead of deleting it"
    )
    return new_nodes, new_cells, new_cell_groups, new_n_bl_cells, new_bad_cell_mask
