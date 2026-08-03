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

Originally scoped to core-only cells (never a BL cell, never touching a
physical boundary face) so the result would never need to carry any
boundary-group attribution of its own. Extended to also cover BL cells,
including ones touching the wall they were extruded from, after real-case
measurement showed the failure mode Stage B (BL thickness capping) can't
reach a sharp-convex-edge cell at all: Stage A correctly refuses to move a
wall node (it's physical geometry, not a mesh-generation artifact), and
Stage B only shortens BL columns early, it can't re-tile the
already-generated shape of one that already exists. A direct local retile
can do both without either limitation. Core cells touching a physical
(non-wall) boundary face remain out of scope - see remesh_core_cavity's
own docstring for why the wall case doesn't need special attribution
handling but a general core-boundary-facet case would.

Split out of mesh_repair.py purely to keep file size down - re-exported
from there (see the bottom of mesh_repair.py) so existing callers keep
working unchanged.
"""

import time
from typing import List, Tuple, TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from ..schema.grid_faces import FaceData
    from ..validation.quality_validator import MeshQualityValidator

# Outward-oriented triangular faces of a POSITIVELY-oriented tetrahedron
# (v0,v1,v2,v3), one row per omitted vertex - see mesh_prism_to_tet.orient_tetrahedra
# for the positive-orientation convention this assumes. Verified against a
# reference unit tet (0,0,0)-(1,0,0)-(0,1,0)-(0,0,1): each row's cross-product
# normal points away from the tet's own centroid, i.e. outward.
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
    """Expand a seed cell mask outward by `n_rings` face-adjacency hops,
    never crossing into a `blocked_mask` cell (BL cells / cells that touch a
    physical boundary face - see remesh_core_cavity). The buffer rings exist
    so the cavity's own new boundary lands on already-good cells, not
    through an already-degenerate one.

    Args:
        owner, neighbor: (n_interior_faces,) cell indices on either side of
            each INTERIOR face only (a boundary face has no far side to
            adjoin through, so it's not part of this adjacency graph at all)

    Returns:
        Boolean cell mask, same shape as seed_mask, with blocked cells
        guaranteed False even if reachable.
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
    """Outward-oriented boundary faces of a cell subset (global node
    indices) - a face shared by two cavity cells is purely interior (tetgen
    will retile it away) and is excluded; a face shared with a cell OUTSIDE
    the subset, or with nothing (a real physical boundary face), appears
    exactly once among the subset's own faces and becomes part of the
    cavity's fixed PLC.
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
    """How many of `cells` trip skewness, non-orthogonality, or adjacent-
    volume-ratio - the same three criteria mesh_repair.py's own
    `_bad_cell_mask` uses for the whole mesh, evaluated here on a small
    retiled cavity so remesh_core_cavity's acceptance gate (see its call
    site) compares like-for-like against `bad_cell_mask`'s definition of
    "bad", not skewness alone. A fresh local retile is a handful to a few
    thousand cells (bounded by max_cavity_cells) - cheap to fully
    re-extract faces for, unlike re-validating the whole mesh.
    """
    from .face_extractor import FaceExtractor
    from ..schema.grid_nodes import NodeArray

    bad = validator.compute_cell_skewness(nodes, cells) > validator.thresholds['max_skewness']

    node_arr = NodeArray(
        x=np.ascontiguousarray(nodes[:, 0]),
        y=np.ascontiguousarray(nodes[:, 1]),
        z=np.ascontiguousarray(nodes[:, 2]),
    )
    # face_extractor logs several INFO/SUCCESS lines per call unconditionally
    # (no verbose= switch there, unlike fill_core_volume) - fine for the
    # normal one-call-per-mesh case, but this runs once per cavity CANDIDATE
    # (up to max_clusters_attempted of them, mostly rejected), so left
    # enabled it multiplies into tens of thousands of lines of routine noise
    # per repair pass on a real case with many small cavities (confirmed
    # directly: a single Stage B' pass produced 70K+ log lines this way).
    # Only this module's own per-cavity/summary lines (logged separately by
    # remesh_core_cavity itself) are actually useful at this granularity.
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
    max_clusters_attempted: int = 2_000,
    max_seconds: float = 90.0,
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
            graceful fallthrough, never a hard failure.
        max_seconds: safety cap on this call's own total wall-clock time,
            checked between clusters (not interrupting one already in
            progress). A pure cluster-count cap doesn't bound cost by
            itself if cluster size/tetgen difficulty varies widely; this
            is the second, independent net for that case.

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

    # Two-phase design, deliberately never mutating nodes/cells/
    # bad_cell_mask while clusters are still being decided: every cluster's
    # cavity is extracted and retiled purely by reading the ORIGINAL,
    # never-touched arrays, and `claimed` (not array mutation) is what
    # prevents a later cluster's buffer rings from re-entering an already-
    # accepted cavity. All accepted results are combined into the new mesh
    # in a single splice at the very end. Mutating cells/bad_cell_mask
    # cluster-by-cluster (an earlier version of this function did) silently
    # invalidates every *subsequent* cluster's cavity_idx/cavity_mask - both
    # were computed as absolute indices into the pre-mutation array, and
    # cell removal shifts everything after the removed rows - so a later
    # cluster would read and "retile" the wrong cells entirely (this was
    # caught empirically: nearly every cavity after the first reported "0
    # bad cells", because it was reading cells that had shifted into that
    # position instead of its own intended bad pocket).
    claimed = np.zeros(n_cells, dtype=bool)
    accepted: List[dict] = []
    n_skipped_size = 0
    n_rejected = 0
    n_failed = 0
    n_skipped_budget = 0

    from .mesh_tetgen_core import fill_core_volume

    if n_clusters > max_clusters_attempted:
        logger.warning(
            f"Stage B': {n_clusters} candidate cavity clusters found, capping "
            f"at {max_clusters_attempted} attempts (max_clusters_attempted) - "
            f"this many separate/scattered bad pockets usually means a "
            f"widespread defect (e.g. many sharp-corner cells, or a large "
            f"fraction of the mesh flagged by the overlap check) that this "
            f"per-cluster local-patch strategy isn't a good fit for; the "
            f"remainder is left for whichever repair stage runs next"
        )

    start_time = time.perf_counter()
    for cluster_id in range(n_clusters):
        if cluster_id >= max_clusters_attempted:
            n_skipped_budget = n_clusters - cluster_id
            break
        elapsed = time.perf_counter() - start_time
        if elapsed > max_seconds:
            n_skipped_budget = n_clusters - cluster_id
            logger.warning(
                f"Stage B': stopping after {elapsed:.1f}s (max_seconds="
                f"{max_seconds:.0f}) with {cluster_id}/{n_clusters} clusters "
                f"attempted - remaining {n_skipped_budget} left for "
                f"whichever repair stage runs next"
            )
            break

        cluster_seed_mask = np.zeros(n_cells, dtype=bool)
        cluster_seed_mask[seed_idx[labels == cluster_id]] = True

        cavity_mask = _grow_cavity_rings(cluster_seed_mask, owner, neighbor, ineligible | claimed, n_buffer_rings)
        cavity_idx = np.flatnonzero(cavity_mask)
        if len(cavity_idx) > max_cavity_cells:
            n_skipped_size += 1
            continue

        boundary_faces = _cavity_boundary_faces(cells, cavity_idx)
        global_pts = np.unique(boundary_faces)
        local_of_global = -np.ones(len(nodes), dtype=np.int64)
        local_of_global[global_pts] = np.arange(len(global_pts))
        local_faces = local_of_global[boundary_faces].astype(np.int32)
        local_points = nodes[global_pts]

        try:
            # verbose=False: this runs once per cavity cluster, potentially
            # many times per repair pass - each individual call's own
            # boundary-point-count/Steiner-budget/completion lines aren't
            # interesting on their own, only this function's own per-cavity
            # and final summary lines (logged separately, below/at the end)
            # are.
            retiled_nodes, retiled_tets, _, _ = fill_core_volume(local_points, local_faces, verbose=False)
        except Exception as e:
            logger.warning(f"Stage B': cavity remesh failed ({e}), keeping original cells")
            n_failed += 1
            continue

        n_boundary_pts = len(local_points)
        if not np.array_equal(retiled_nodes[:n_boundary_pts], local_points):
            # fill_core_volume already logs+handles this internally (coincident-
            # point stitching fallback), but the cavity's own boundary points
            # are exactly the ones that must stay pinned for the splice below
            # to be valid - if even the fallback couldn't preserve them
            # verbatim, don't risk stitching a silently-shifted boundary into
            # the still-good rest of the mesh.
            logger.warning(
                "Stage B': cavity boundary points weren't preserved "
                "verbatim by the local retile, keeping original cells"
            )
            n_failed += 1
            continue

        # Quality gate: only accept if the retile is a strict improvement
        # over what it replaces - Stage A's own "never trust a move blindly"
        # philosophy (see this module's docstring), applied here to a whole
        # cavity swap instead of a per-node position.
        #
        # old_bad_in_cavity counts cells bad by ANY of bad_cell_mask's
        # criteria (skew, non-orthogonality, adjacent-volume-ratio, and -
        # since mesh_background.py folds it in - physical overlap). Scoring
        # bad_new by skewness alone (an earlier version of this check) was
        # an apples-to-oranges comparison: a cavity that was bad mainly for
        # non-orthogonality or adjacent-volume-ratio reasons could get a
        # retile that genuinely fixes those, yet still get rejected here
        # because its (unrelated) skewness count didn't happen to improve -
        # confirmed directly on a real sharp-corner case where the vast
        # majority of retiles were being rejected this way. Evaluating the
        # SAME three criteria on the retiled cavity makes this an actual
        # apples-to-apples comparison.
        old_bad_in_cavity = int(np.sum(bad_cell_mask[cavity_idx]))
        bad_new = _count_bad_cells(validator, retiled_nodes, retiled_tets)
        if bad_new >= old_bad_in_cavity:
            logger.info(
                f"Stage B': cavity of {len(cavity_idx)} cells "
                f"({old_bad_in_cavity} bad) retiled into {len(retiled_tets)} cells "
                f"({bad_new} bad) - not an improvement, keeping original cells"
            )
            n_rejected += 1
            continue

        claimed[cavity_idx] = True
        accepted.append(dict(
            cavity_idx=cavity_idx, global_pts=global_pts,
            retiled_nodes=retiled_nodes, retiled_tets=retiled_tets,
            n_boundary_pts=n_boundary_pts,
            old_bad=old_bad_in_cavity, bad_new=bad_new,
        ))

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

    logger.info(
        f"Stage B': {len(accepted)}/{n_clusters} cavity cluster(s) remeshed "
        f"(skipped_size={n_skipped_size}, rejected={n_rejected}, "
        f"failed={n_failed}, skipped_budget={n_skipped_budget})"
    )

    return new_nodes, new_cells, new_cell_groups, new_bad_cell_mask, actions
