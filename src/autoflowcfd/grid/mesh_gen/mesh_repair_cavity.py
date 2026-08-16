"""Stage B': local cavity re-tiling for post-generation volume mesh repair.

remesh_core_cavity locally re-tetrahedralizes just the still-bad cells
(after Stage A smoothing and Stage B's BL-thickness cap, both in
mesh_repair.py / mesh_repair_bl_thickness.py) plus a buffer of good
neighbour cells, instead of nudging nodes (Stage A) or regenerating the
whole mesh. Extract the cavity's own boundary (padded by
`n_buffer_rings` of good cells so the cavity's new boundary sits in
already-good territory, not through an already-degenerate cell), hand JUST
that small closed shell to its own standalone tetgen call (nobisect=True,
no competing region/volume constraint - the same default,
already-proven-good tetgen usage this whole package's core fill uses
everywhere else), and splice the result back in place of the removed
cells. This structurally cannot leak refinement outward the way the old
core-side region approach did (see mesh_repair_bl_thickness.py's own
docstring for that history), because the cavity's own tetgen call never
sees the rest of the domain at all - there's nothing for it to leak into.
Gated on a strict quality improvement over the cells it replaces - if the
local retile doesn't actually help, the original cells are kept and the
case falls through to whatever repair stage runs next.

Scope (core-only cells vs. BL cells touching their own wall) is documented
on remesh_core_cavity's own docstring below, not repeated here.

Split out of mesh_repair.py purely to keep file size down - re-exported
from there (see the bottom of mesh_repair.py) so existing callers keep
working unchanged.

进一步拆分：cavity 扩张/边界提取/质量评分的共享工具在
mesh_repair_cavity_shared.py；patch_nonmanifold_cavity（非流形修补，与
remesh_core_cavity 共用同一套局部重新四面体化技巧，但用途不同）在
mesh_repair_nonmanifold_patch.py。两者都在本文件重新转出，外部代码一律
仍从 `mesh_repair_cavity` 导入即可。

remesh_core_cavity 自身体积过大（超过 400 行上限），其中"逐簇尝试局部
重新四面体化"的循环体（占了函数体的大半）进一步拆到了
mesh_repair_cavity_cluster_attempt.py 的 _attempt_cavity_retile_clusters，
本文件里的 remesh_core_cavity 只保留候选簇的连通分量计算和最终拼接。
"""

from typing import List, Tuple, TYPE_CHECKING

import numpy as np
from loguru import logger

from .mesh_repair_cavity_shared import (
    _CAVITY_FACE_TEMPLATES,
    _grow_cavity_rings,
    _cavity_boundary_faces,
    _count_bad_cells,
)
from .mesh_repair_cavity_cluster_attempt import _attempt_cavity_retile_clusters
from .mesh_repair_nonmanifold_patch import patch_nonmanifold_cavity

if TYPE_CHECKING:
    from ..schema.grid_faces import FaceData
    from ..validation.quality_validator import MeshQualityValidator

__all__ = [
    '_CAVITY_FACE_TEMPLATES',
    'remesh_core_cavity',
    'patch_nonmanifold_cavity',
]


def remesh_core_cavity(
    nodes: np.ndarray,
    cells: np.ndarray,
    cell_groups: np.ndarray,
    n_bl_cells: int,
    faces: 'FaceData',
    bad_cell_mask: np.ndarray,
    validator: 'MeshQualityValidator',
    n_buffer_rings: int = 1,
    max_cavity_cells: int = 20_000,
    max_clusters_attempted: int = 15_000,
    max_seconds: float = 400.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Stage B': locally re-tetrahedralize just the still-bad cells (plus a
    buffer of good neighbours) against their own fixed boundary, instead of
    nudging nodes (Stage A) or regenerating the whole mesh (Stage B, BL
    side). See this module's docstring for why this structurally avoids the
    old core-side region approach's failure mode.

    Originally scoped to core-only cells; extended to also cover BL cells
    (including ones touching the wall they were extruded from - see below)
    after real-case measurement showed the BL/corner-adjacent failure mode
    this was meant to complement (thickness capping, Stage B) can't reach a
    sharp-convex-edge cell at all: Stage A refuses to move a wall node
    (correctly - it's physical geometry), and Stage B only shortens BL
    columns, it can't re-tile the already-generated shape of one. A direct
    local retile can.

    Scope: a cell touching a physical boundary face is only eligible if
    that cell is itself a BL cell (index < n_bl_cells) - expected, since a
    BL cell is always adjacent to the wall it was extruded from, and that
    wall facet's own node indices are preserved verbatim through the local
    retile (nobisect=True), so identify_boundaries_from_surface's existing
    node-index-matching fallback recovers its boundary-group attribution
    without this function needing to track it. A CORE cell touching a
    physical boundary face (inlet/outlet/tunnel/farfield-type) remains out
    of scope - a genuinely different, unvalidated scenario (that facet may
    carry a tetgen facet-marker/region attribution this function doesn't
    handle). A cavity that would need to grow into an out-of-scope cell
    simply stops there instead - that cell (and any bad cell only
    reachable through it) is left for whichever repair stage runs next,
    not a reason to abort the whole operation.

    Args:
        nodes, cells: the full merged mesh (post Stage A)
        cell_groups: (n_cells,) str array, parallel to cells - boundary
            group per cell (see mesh_background._build_merged_mesh);
            replaced cells always get '' - correct for a replaced CORE
            cell (which, per scope above, could never have owned a real
            boundary face) and harmless for a replaced BL cell touching
            the wall (identify_boundaries_from_surface's node-index
            fallback re-derives that attribution independently of this
            array, per the scope note above)
        n_bl_cells: BL cells occupy cells[:n_bl_cells] - eligible (see
            scope above), just never subject to the physical-boundary
            exclusion core cells are
        faces: FaceData already extracted from this exact (nodes, cells)
            pair (the caller's own pre-repair or Stage-A-output extraction)
        bad_cell_mask: (n_cells,) bool, which cells still fail the quality
            gate after Stage A
        validator: reused for its per-cell skewness/face-diagnostic methods
            and thresholds, to gate acceptance (see below)
        n_buffer_rings: how many face-adjacency rings of good cells to pad
            each cavity with before extracting its boundary
        max_cavity_cells: safety cap - a cavity this large signals the bad
            region is too widespread for a "local" patch to make sense
            (and tetgen cost no longer resembles a cheap local operation);
            skipped rather than attempted
        max_clusters_attempted: safety cap on the total NUMBER of separate
            cavity clusters this call will attempt, regardless of how many
            candidate clusters exist. Each cluster is its own independent
            fill_core_volume (tetgen) call - individually cheap, but
            bad_cell_mask can legitimately contain tens of thousands of
            scattered, disconnected single/small clusters when it was fed
            from a widespread geometric defect rather than a handful of
            localized ones (observed directly: mesh_overlap_check.py
            flagging cells across a sharp-corner-heavy real mesh - see
            mesh_background.py's own comment where it folds overlap cells
            into bad_cell_mask). Attempting all of them sequentially, each
            with real (if small) per-call overhead, is what actually made
            this stage look hung on that case, not any single cluster
            being slow. Remaining bad cells beyond this cap are left as-is
            for whichever repair stage runs next (Stage B's BL thickness
            cap, or Stage C's global backoff) - consistent with every
            other cap in this function (size, quality-gate rejection):
            graceful fallthrough, never a hard failure. Raised from an
            earlier 2,000 to 15,000 after a real sharp-edge-heavy case
            (cube_demo) produced 9,013 candidate clusters and the old cap
            silently left 7,013 of them completely untried, well before
            the quality-gate-rejection question (see below) even entered
            the picture - offline mesh generation has minutes to spend
            here already, and each additional cluster attempt is cheap.
        max_seconds: safety cap on this call's own total wall-clock time,
            checked between clusters (not interrupting one already in
            progress). A pure cluster-count cap doesn't bound cost by
            itself if cluster size/tetgen difficulty varies widely; this
            is the second, independent net for that case. Raised from an
            earlier 90s alongside max_clusters_attempted, for the same
            reason.

    Returns:
        (new_nodes, new_cells, new_cell_groups, new_bad_cell_mask,
        action_log) - all unchanged (not copies) if no eligible cavity was
        found or every candidate cavity failed its acceptance gate.
        new_bad_cell_mask carries bad_cell_mask forward across the same
        cell removals/insertions applied to new_cells (every newly-inserted
        cell is marked good=False, since it already passed this function's
        own acceptance gate) - the caller's next repair stage can use it
        directly instead of re-validating the whole mesh from scratch.
    """
    actions: List[str] = []
    n_cells = len(cells)

    boundary_face_idx = faces.get_boundary_face_indices()
    touches_physical_boundary = np.zeros(n_cells, dtype=bool)
    touches_physical_boundary[faces.connectivity[boundary_face_idx, 0]] = True

    # A BL cell touching a physical boundary face is expected, not
    # disqualifying (it's always adjacent to the wall it was extruded
    # from) - only a CORE cell touching a physical boundary face (inlet/
    # outlet/tunnel/farfield-type) stays out of scope: that's a genuinely
    # different, unvalidated scenario (a core cell's boundary facet may
    # carry a facet marker/region attribution this function doesn't handle
    # - see the wall-facet handling note above). The wall facet itself
    # is preserved node-for-node by nobisect=True in the local retile
    # (same guarantee _cavity_boundary_faces already relies on for a kept
    # neighbour's shared face), so identify_boundaries_from_surface's
    # existing node-index-matching fallback recovers the wall group
    # attribution for whichever new cell ends up owning that facet,
    # without this function needing to track it itself.
    ineligible = touches_physical_boundary.copy()
    ineligible[:n_bl_cells] = False

    seed = bad_cell_mask & ~ineligible
    if not np.any(seed):
        actions.append("Stage B': no eligible bad cells (all touch an out-of-scope core boundary) - skipping")
        return nodes, cells, cell_groups, bad_cell_mask, actions

    interior_mask = faces.connectivity[:, 1] >= 0
    owner = faces.connectivity[interior_mask, 0]
    neighbor = faces.connectivity[interior_mask, 1]

    # Each connected cluster of eligible seed cells becomes its own
    # separate cavity - a single combined cavity spanning several unrelated
    # bad pockets would (a) risk exceeding max_cavity_cells for no reason
    # and (b) needlessly re-tetrahedralize the good cells geometrically
    # between two unrelated pockets.
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    seed_idx = np.flatnonzero(seed)
    seed_pos = -np.ones(n_cells, dtype=np.int64)
    seed_pos[seed_idx] = np.arange(len(seed_idx))
    edge_mask = seed[owner] & seed[neighbor]
    rows = seed_pos[owner[edge_mask]]
    cols = seed_pos[neighbor[edge_mask]]
    graph = coo_matrix(
        (np.ones(len(rows), dtype=bool), (rows, cols)), shape=(len(seed_idx), len(seed_idx))
    )
    n_clusters, labels = connected_components(graph, directed=False)

    # 逐簇尝试局部重新四面体化的循环体拆到了
    # mesh_repair_cavity_cluster_attempt._attempt_cavity_retile_clusters
    # （原文件超过 400 行上限）——两阶段设计（先只读原始数组决定每个簇的
    # 取舍，最后才一次性拼接进新网格）的原因见本文件模块文档字符串顶部，
    # 这里不重复。
    accepted, claimed, n_skipped_size, n_rejected, n_failed, n_skipped_budget = (
        _attempt_cavity_retile_clusters(
            nodes, cells, bad_cell_mask, validator,
            seed_idx, labels, n_clusters,
            owner, neighbor, ineligible,
            n_buffer_rings, max_cavity_cells, max_clusters_attempted, max_seconds,
        )
    )

    from .mesh_tetgen_core import repair_nonmanifold_cells

    if not accepted:
        actions.append(
            f"Stage B': {n_clusters} candidate cavity cluster(s) found, "
            f"none accepted (skipped_size={n_skipped_size}, rejected={n_rejected}, "
            f"failed={n_failed}, skipped_budget={n_skipped_budget})"
        )
        logger.info(
            f"Stage B': 0/{n_clusters} cavity cluster(s) remeshed "
            f"(skipped_size={n_skipped_size}, rejected={n_rejected}, "
            f"failed={n_failed}, skipped_budget={n_skipped_budget})"
        )
        return nodes, cells, cell_groups, bad_cell_mask, actions

    keep_mask = ~claimed
    new_nodes_parts = [nodes]
    new_cells_parts = [cells[keep_mask]]
    new_groups_parts = [cell_groups[keep_mask]]
    new_bad_parts = [bad_cell_mask[keep_mask]]
    interior_start = len(nodes)

    for res in accepted:
        n_boundary_pts = res['n_boundary_pts']
        global_pts = res['global_pts']
        retiled_nodes = res['retiled_nodes']
        retiled_tets = res['retiled_tets']

        def _remap(local_idx: np.ndarray, _global_pts=global_pts, _n_boundary=n_boundary_pts,
                   _offset=interior_start) -> np.ndarray:
            is_boundary = local_idx < _n_boundary
            out = np.empty_like(local_idx)
            out[is_boundary] = _global_pts[local_idx[is_boundary]]
            out[~is_boundary] = _offset + (local_idx[~is_boundary] - _n_boundary)
            return out

        new_tets_global = _remap(retiled_tets.ravel()).reshape(-1, 4).astype(cells.dtype)
        new_interior_nodes = retiled_nodes[n_boundary_pts:]

        new_nodes_parts.append(new_interior_nodes)
        new_cells_parts.append(new_tets_global)
        new_groups_parts.append(np.full(len(new_tets_global), '', dtype=object))
        new_bad_parts.append(np.zeros(len(new_tets_global), dtype=bool))
        interior_start += len(new_interior_nodes)

        actions.append(
            f"Stage B': cavity of {len(res['cavity_idx'])} cells "
            f"({res['old_bad']} bad) -> retiled into {len(new_tets_global)} cells "
            f"({res['bad_new']} bad), {len(new_interior_nodes)} new interior point(s)"
        )

    new_nodes = np.vstack(new_nodes_parts)
    new_cells = np.vstack(new_cells_parts)
    new_cell_groups = np.concatenate(new_groups_parts)
    new_bad_cell_mask = np.concatenate(new_bad_parts)

    # generate_hybrid_mesh runs this same check once, right after the
    # INITIAL _build_merged_mesh output, but never again - so a
    # non-manifold overlap this function's own splicing introduces (e.g.
    # two accepted cavities' retiles coincidentally producing overlapping
    # tets at their shared boundary, the same class of defect
    # repair_nonmanifold_cells's own docstring already documents as its
    # reason for existing) went uncaught until the CALLER's next
    # FaceExtractor.extract_faces call - which doesn't merely warn about
    # it (that call already tolerates >2-cell faces in its own
    # non-strict mode) but hard-crashes on a stricter, separate check:
    # some cell ends up referenced by NO face at all (every one of its 4
    # faces was the "extra" >2-cell occurrence at some other cell's
    # expense), which validate_face_data treats as fatal regardless of
    # strictness. Confirmed directly, not theoretical: a real run hit
    # exactly this ("Face connectivity references N-6 cells, expected N")
    # right after a Stage B' iteration logged "204 invalid (>2 cells)"
    # faces. Running the same cleanup this function's own caller already
    # trusts for the initial mesh keeps that invariant true after THIS
    # function's own mutation too, instead of leaving it to be discovered
    # (fatally) one call later.
    # Try a local retile first, same rationale as patch_nonmanifold_
    # cavity's own docstring: plain "keep largest, drop rest" deletion
    # here was itself found to leave a real hole - confirmed directly,
    # not theoretical, once the OTHER two repair_nonmanifold_cells call
    # sites (mesh_background.py, both already patched) turned out NOT to
    # be where a real cube_demo run's remaining 0.147 m^3 deficit (of an
    # original 0.189 m^3) was coming from - it traced back to exactly
    # this block. n_bl_cells isn't updated from the patch's own return
    # here (discarded below) - this function's cavity-growing already
    # tolerates n_bl_cells staying approximate after ordinary splicing
    # (see this function's own docstring: a BL cell not touching a
    # physical boundary can already be replaced without n_bl_cells
    # changing), so the patch path doesn't need to be any stricter.
    keep = repair_nonmanifold_cells(new_nodes, new_cells)
    if not keep.all():
        new_nodes, new_cells, new_cell_groups, _n_bl_cells_unused, new_bad_cell_mask = patch_nonmanifold_cavity(
            new_nodes, new_cells, keep, new_cell_groups, n_bl_cells,
            bad_cell_mask=new_bad_cell_mask,
        )
        keep = repair_nonmanifold_cells(new_nodes, new_cells)
        if not keep.all():
            n_removed = int(np.size(keep) - np.count_nonzero(keep))
            actions.append(f"Stage B': removed {n_removed} non-manifold cell(s) introduced by cavity splicing")
            new_cells = new_cells[keep]
            new_cell_groups = new_cell_groups[keep]
            new_bad_cell_mask = new_bad_cell_mask[keep]

    logger.info(
        f"Stage B': {len(accepted)}/{n_clusters} cavity cluster(s) remeshed "
        f"(skipped_size={n_skipped_size}, rejected={n_rejected}, "
        f"failed={n_failed}, skipped_budget={n_skipped_budget})"
    )

    return new_nodes, new_cells, new_cell_groups, new_bad_cell_mask, actions
