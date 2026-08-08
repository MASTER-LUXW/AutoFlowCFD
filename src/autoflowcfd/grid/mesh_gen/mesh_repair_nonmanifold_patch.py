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
    """Locally re-tetrahedralize the region mesh_tetgen_core.
    repair_nonmanifold_cells flagged, instead of just deleting the cells it
    marked for removal (keep_mask False) and leaving a hole in their place.

    Why this exists: repair_nonmanifold_cells's own fix for a face shared by
    3+ cells is "keep the largest, drop the rest" - correct when the extra
    cells are genuinely redundant duplicates, but when they instead come
    from two DIFFERENT regions of the mesh (e.g. the transition-tet stage
    and the tetgen core fill) both legitimately trying to occupy the same
    space at a sharp corner, dropping the "losing" side's cells removes
    real geometry with nothing generated to replace it - a literal gap in
    the final mesh. Confirmed directly, not theoretical: a real cube_demo
    run measured 0.189 m^3 missing (merged-mesh volume vs. the exact
    bbox-minus-body-hole volume, computed independently via
    mesh_domain_classify._signed_volume) at the same location
    repair_nonmanifold_cells reported removing cells, visible in an
    exported-mesh screenshot as a void along one edge of the body.

    Approach: treat every cell touching an over-shared (non-manifold) face
    - on EITHER side, not just the ones keep_mask would drop - as the
    cavity seed (dropping only the "losing" cells and retiling just around
    them risks the retile's own boundary still touching another
    non-manifold face), pad it with `n_buffer_rings` of ordinary
    (manifold-adjacent) neighbours so the cavity's own new boundary lands
    on already-good territory, and hand its boundary to a fresh, unconstrained
    tetgen call - the same local-cavity technique remesh_core_cavity already
    uses, but unconditional (no quality-gate rejection: eliminating a real
    hole is a strict win regardless of the replacement cells' own skew/
    orthogonality score, unlike remesh_core_cavity's "only accept a
    provable improvement" bar for cells that were merely low-quality, not
    physically missing).

    Deliberately self-contained rather than reusing FaceExtractor for the
    adjacency graph: the input here is BY DEFINITION non-manifold in
    places (that's why this function is being called at all) - exactly
    the condition FaceExtractor.extract_faces's own strict-mode validation
    exists to reject, and even its non-strict mode was confirmed (this
    project's own history) to still hard-fail once a cell ends up
    referenced by no face at all, a real risk this close to the defect
    this function exists to fix.

    Args:
        nodes, cells: the full mesh BEFORE any removal (repair_nonmanifold_
            cells' own keep_mask is a proposal, not yet applied)
        keep_mask: (n_cells,) bool from repair_nonmanifold_cells - False
            marks a cell it would otherwise unconditionally drop
        cell_groups: (n_cells,) str array parallel to cells - every newly
            retiled cell gets '' (matching remesh_core_cavity's own
            convention for cells it creates: never re-classified as
            "transition" or a physical-wall group, since a patch spanning
            a former transition/core seam is closer to ordinary interior
            geometry than either side it replaced)
        n_bl_cells: cells[:n_bl_cells] are transition-stage in origin (see
            generate_hybrid_mesh's own n_bl_cells convention) - reduced by
            however many of THOSE specific cells got swept into the
            cavity and weren't preserved verbatim; every newly retiled
            cell is appended past the end, i.e. always counted on the
            core/generic side of this split, never the transition side
        n_buffer_rings: face-adjacency rings of ordinary neighbours padded
            around the non-manifold cell cluster before extracting its
            boundary
        max_cavity_cells: safety cap - a defect this large signals
            something structurally wrong worth its own investigation, not
            a good fit for a local patch; falls back to plain deletion
        bad_cell_mask: optional (n_cells,) bool array parallel to cells -
            kept in sync exactly like cell_groups (every newly retiled
            cell gets False, i.e. "not known bad" - matching remesh_core_
            cavity's own convention for cells it creates) so a caller that
            tracks its own bad-cell mask (remesh_core_cavity's own retry
            loop) doesn't have to separately reconstruct it after this
            call. None (default) if the caller has no such array to track.

    Returns:
        (new_nodes, new_cells, new_cell_groups, new_n_bl_cells,
        new_bad_cell_mask) - nodes/cells/cell_groups/bad_cell_mask
        unchanged (not copies) and n_bl_cells passed through as-is if
        keep_mask is already all-True, the cavity exceeds
        max_cavity_cells, or the local retile fails/still comes out
        non-manifold itself (logged either way; the caller's own
        repair_nonmanifold_cells deletion is the safety net for whatever
        this can't fix). new_bad_cell_mask is None iff bad_cell_mask was
        None.
    """
    if keep_mask.all():
        return nodes, cells, cell_groups, n_bl_cells, bad_cell_mask

    from .mesh_tetgen_core import fill_core_volume, repair_nonmanifold_cells, CORE_TETGEN_MINRATIO, CORE_TETGEN_MINDIHEDRAL

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

    # The whole point of retiling instead of deleting is to end up WITHOUT
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
