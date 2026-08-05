"""Stage D: last-resort local edge collapse for post-generation volume mesh
repair.

Runs strictly after Stage A (smoothing, mesh_repair.py) and Stage B'
(cavity re-tiling, mesh_repair_cavity.py) have both already been tried on a
cell and it is STILL bad: Stage A can't move a physical-boundary node, and
Stage B' frequently can't find any alternative local tetrahedralization
that's actually better (observed directly, repeatedly, on real cases: "Stage
B': cavity of N cells (M bad) retiled into N' cells (M' bad) - not an
improvement, keeping original cells"). For the residual handful of cells
neither can fix - typically genuine sliver tets with one near-zero-length
edge, sitting right at a sharp convex edge/corner - the only remaining
degree of freedom that doesn't require re-triangulating the same fixed
boundary Stage B' already failed against is to remove the degeneracy
entirely: collapse the sliver's shortest edge, merging its two endpoint
nodes into one.

This is the standard mesh-simplification "edge collapse" operation, not
cell deletion: every tetrahedron incident to the collapsed edge vanishes
(its two collapsed vertices become the same point, so its volume goes to
exactly zero - it was never really occupying independent volume, a sliver's
whole defect IS that near-zero-volume degeneracy), and every OTHER
tetrahedron that only touched one of the two endpoints has that one vertex
index redirected to the surviving node - it keeps its other 3 vertices
untouched, so the mesh stays perfectly watertight (every remaining internal
face is still shared by exactly 2 cells) with no new boundary/hole
introduced anywhere. This is fundamentally different from - and safe in a
way that - "just delete the bad cell and leave a gap" is not: a gap would
force an artificial wall/vacuum face into the interior of the domain right
where the user's forces/wake are most sensitive (near the body surface);
an edge collapse instead makes the mesh very slightly coarser in a tiny
local pocket in exchange for removing the degeneracy, with no topology
change anywhere else.

Safety guarantees, all-or-nothing per candidate collapse (same "propose,
validate, accept-or-reject" pattern as Stage A's damped relaxation and
Stage B's strict-improvement gate):
    - Never merges two nodes that are both on physical boundary geometry
      (body/tunnel/inlet/outlet surface, or the BL/core interface) - reuses
      compute_movable_node_mask, the exact same criterion Stage A already
      uses to decide which nodes it's allowed to relocate at all. If
      exactly one endpoint is movable, that one is always the one removed
      (merged onto the fixed one); if both are movable, either may be
      removed.
    - Rejects the candidate outright if it would leave any surviving cell
      with a duplicate vertex (degenerate) or a non-positive volume.
    - Rejects the candidate if it would turn any currently-GOOD neighbour
      cell newly-bad by skewness or aspect ratio - Stage D is only allowed
      to spend a neighbour's quality margin to fix the sliver it's
      attached to, never to quietly break a cell that was fine before.

Not gated on strict global improvement the way Stage B' is (a collapse by
construction always eliminates the specific sliver it targets - there's
nothing to compare against), so callers should re-run the quality gate on
the result rather than assume every original bad cell is now fixed - some
may have no safe collapse candidate at all (e.g. every edge either touches
two boundary nodes, or every remap would create a new negative-volume or
newly-bad neighbour) and are left exactly as they were, same as Stage
A/B' already do when they can't help.
"""

from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np
from loguru import logger

from .mesh_repair import compute_movable_node_mask
from ..validation import quality_metrics as _qm

if TYPE_CHECKING:
    from ..schema.grid_faces import FaceData
    from ..validation.quality_validator import MeshQualityValidator

# Same (i, j) local-vertex-index pairing/order as
# quality_metrics.tetrahedron_edge_lengths, so "shortest edge index" maps
# straight back to a (i, j) pair without re-deriving it.
_EDGE_VERTEX_PAIRS: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
]


def _build_node_to_cells(cells: np.ndarray) -> Dict[int, Set[int]]:
    """Node index -> set of (currently alive) cell indices containing it."""
    node_to_cells: Dict[int, Set[int]] = {}
    for c in range(len(cells)):
        for v in cells[c]:
            node_to_cells.setdefault(int(v), set()).add(c)
    return node_to_cells


def _evaluate_collapse(
    cur_nodes: np.ndarray,
    cur_cells: np.ndarray,
    node_to_cells: Dict[int, Set[int]],
    remove: int,
    target: int,
    validator: 'MeshQualityValidator',
) -> Optional[Tuple[List[int], np.ndarray, List[int]]]:
    """Check whether merging node `remove` into `target` is safe. Returns
    None if unsafe/rejected, else (remap_cell_ids, their_new_vertex_rows,
    doomed_cell_ids) - purely descriptive, does not mutate anything."""
    cells_r = node_to_cells.get(remove, set())
    cells_t = node_to_cells.get(target, set())
    doomed_set = cells_r & cells_t
    doomed = sorted(doomed_set)
    remap_ids = sorted(cells_r - doomed_set)

    if not remap_ids and not doomed:
        return None  # `remove` isn't actually incident to anything - no-op

    if not remap_ids:
        return [], np.zeros((0, 4), dtype=cur_cells.dtype), doomed

    old_verts = cur_cells[remap_ids].copy()
    new_verts = old_verts.copy()
    new_verts[new_verts == remove] = target

    # Degeneracy: a valid tet needs 4 distinct vertex indices.
    if np.any(np.array([len(set(row.tolist())) for row in new_verts]) < 4):
        return None

    new_vol = _qm.compute_tetrahedron_volumes(cur_nodes, new_verts)
    if np.any(new_vol <= 0.0):
        return None

    sk_th = validator.thresholds['max_skewness']
    ar_th = validator.thresholds['max_aspect_ratio']
    old_skew = _qm.compute_tetrahedron_skewness_values(cur_nodes, old_verts)
    new_skew = _qm.compute_tetrahedron_skewness_values(cur_nodes, new_verts)
    old_ar = _qm.compute_tetrahedron_aspect_ratios(cur_nodes, old_verts)
    new_ar = _qm.compute_tetrahedron_aspect_ratios(cur_nodes, new_verts)
    newly_bad = (
        ((old_skew <= sk_th) & (new_skew > sk_th))
        | ((old_ar <= ar_th) & (new_ar > ar_th))
    )
    if np.any(newly_bad):
        return None

    return remap_ids, new_verts, doomed


def _apply_collapse(
    cur_cells: np.ndarray,
    alive: np.ndarray,
    node_to_cells: Dict[int, Set[int]],
    remove: int,
    target: int,
    remap_ids: List[int],
    new_verts: np.ndarray,
    doomed: List[int],
) -> None:
    """Mutate cur_cells/alive/node_to_cells in place to actually perform an
    already-validated collapse."""
    for c in doomed:
        alive[c] = False
        for v in cur_cells[c]:
            node_to_cells.get(int(v), set()).discard(c)

    for c, nv in zip(remap_ids, new_verts):
        cur_cells[c] = nv
        node_to_cells.get(remove, set()).discard(c)
        node_to_cells.setdefault(target, set()).add(c)

    # `remove` should now have zero remaining incident cells - drop it
    # explicitly rather than relying on the discards above leaving an empty
    # set (defensive: guarantees no future candidate can accidentally treat
    # `remove` as still-live).
    node_to_cells[remove] = set()


def _compact_mesh(
    nodes: np.ndarray, cells: np.ndarray, cell_groups: np.ndarray, n_bl_cells: int,
    extra_referenced_nodes: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """Drop nodes no longer referenced by any cell and renumber cells
    against the resulting dense node index range.

    Args:
        extra_referenced_nodes: Optional node indices this function must
            NEVER drop even though no cell in `cells` references them -
            e.g. a mixed prism+tet mesh's BL/core interface nodes, which
            PrismCells (entirely outside this function's `cells` array -
            Stage D only ever operates on the tet portion) still needs.
            Confirmed as a real, not just theoretical, gap: on a real case
            this function silently dropped a node still referenced by
            631,033 prism cells, and the caller's later PrismCells.
            compute_volumes crashed with an out-of-bounds index - a node
            that had zero surviving TET references (every tet touching it
            got collapsed away) is not "unused", it was still load-bearing
            for the BL region this function has no visibility into.

    Returns:
        (new_nodes, new_cells, cell_groups, n_bl_cells, remap) - remap has
        shape=(len(nodes),), old index -> new index (or -1 if genuinely
        dropped), for the caller to apply to any OTHER index array into
        the same original `nodes` (e.g. PrismCells.connectivity).
    """
    used = np.unique(cells.ravel())
    if extra_referenced_nodes is not None and len(extra_referenced_nodes):
        used = np.union1d(used, extra_referenced_nodes)
    remap = np.full(len(nodes), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    new_nodes = nodes[used]
    new_cells = remap[cells]
    return new_nodes, new_cells, cell_groups, n_bl_cells, remap


def collapse_bad_cells(
    nodes: np.ndarray,
    cells: np.ndarray,
    cell_groups: np.ndarray,
    n_bl_cells: int,
    faces: 'FaceData',
    bad_cell_mask: np.ndarray,
    validator: 'MeshQualityValidator',
    max_collapses: int = 5000,
    extra_referenced_nodes: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, List[str], np.ndarray]:
    """Stage D: collapse the shortest safe edge of each still-bad cell,
    worst (highest-skewness) cell first.

    Args:
        nodes, cells: the full merged mesh, post Stage A/B'
        cell_groups: (n_cells,) str array parallel to cells - boundary
            group per cell; doomed cells are simply dropped from it,
            remapped cells keep their existing entry unchanged (an edge
            collapse never changes which physical boundary facet, if any,
            a surviving cell owns - it only ever redirects one interior-side
            vertex)
        n_bl_cells: cells[:n_bl_cells] are BL-origin cells (see
            mesh_background._build_merged_mesh's convention)
        faces: FaceData already extracted from this exact (nodes, cells)
            pair - used only for compute_movable_node_mask
        bad_cell_mask: (n_cells,) bool, which cells still fail the quality
            gate after Stage A and Stage B'
        validator: reused for its thresholds and per-cell metric helpers
        max_collapses: safety cap on how many edges to collapse in one call
        extra_referenced_nodes: Optional node indices this function's final
            compaction must never drop even though `cells` itself no longer
            references them - see _compact_mesh's docstring for why (a
            mixed prism+tet mesh's PrismCells lives entirely outside this
            function's view but can share nodes at the BL/core interface).

    Returns:
        (new_nodes, new_cells, new_cell_groups, new_n_bl_cells, action_log,
        node_remap) - node_remap has shape=(len(nodes),), old index -> new
        index (identity if this call made no node-count change at all),
        for the caller to apply to any other index array into the same
        original `nodes` (e.g. PrismCells.connectivity).
    """
    actions: List[str] = []
    identity_remap = np.arange(len(nodes), dtype=np.int64)
    n_bad0 = int(np.sum(bad_cell_mask))
    if n_bad0 == 0:
        return nodes, cells, cell_groups, n_bl_cells, actions, identity_remap

    n_bad_bl = int(np.sum(bad_cell_mask[:n_bl_cells]))
    n_bad_core = n_bad0 - n_bad_bl
    logger.info(
        f"Stage D: {n_bad0} bad cells to process ({n_bad_bl} BL-region, "
        f"{n_bad_core} core-region, out of {n_bl_cells} BL / {len(cells) - n_bl_cells} core cells total)"
    )

    movable = compute_movable_node_mask(len(nodes), faces, n_bl_cells)

    cur_nodes = nodes.copy()
    cur_cells = cells.astype(np.int64).copy()
    alive = np.ones(len(cur_cells), dtype=bool)
    node_to_cells = _build_node_to_cells(cur_cells)

    # Worst (highest-skewness) cell first, so a shared edge between two bad
    # cells is spent on whichever is more severe.
    skew0 = validator.compute_cell_skewness(cur_nodes, cur_cells)
    bad_indices = np.flatnonzero(bad_cell_mask)
    bad_indices = bad_indices[np.argsort(-skew0[bad_indices])]

    n_collapsed = 0
    n_cells_removed_total = 0
    n_unresolved = 0

    for cell_idx in bad_indices:
        cell_idx = int(cell_idx)
        if n_collapsed >= max_collapses:
            actions.append(f"Stage D: reached max_collapses={max_collapses} limit, stopping")
            break
        if not alive[cell_idx]:
            continue  # already removed as part of an earlier collapse

        verts = cur_cells[cell_idx]
        if len(set(verts.tolist())) < 4:
            # Degenerated to zero volume by an earlier remap that touched
            # this cell without formally "doom"-ing it (shouldn't happen
            # given _evaluate_collapse's own degeneracy check, but treat
            # defensively as already-resolved rather than crashing on it).
            alive[cell_idx] = False
            continue

        pts = cur_nodes[verts]
        edge_lens = sorted(
            ((float(np.linalg.norm(pts[i] - pts[j])), i, j) for i, j in _EDGE_VERTEX_PAIRS),
            key=lambda t: t[0],
        )

        resolved = False
        for _, li, lj in edge_lens:
            n0, n1 = int(verts[li]), int(verts[lj])
            m0, m1 = bool(movable[n0]), bool(movable[n1])
            if not m0 and not m1:
                continue  # both endpoints are physical geometry - untouchable

            candidates = [(n0, n1)] if (m0 and not m1) else \
                         [(n1, n0)] if (m1 and not m0) else \
                         [(n0, n1), (n1, n0)]  # both movable - try either direction

            for remove, target in candidates:
                result = _evaluate_collapse(cur_nodes, cur_cells, node_to_cells, remove, target, validator)
                if result is None:
                    continue
                remap_ids, new_verts, doomed = result
                _apply_collapse(cur_cells, alive, node_to_cells, remove, target, remap_ids, new_verts, doomed)
                n_collapsed += 1
                n_cells_removed_total += len(doomed)
                resolved = True
                break
            if resolved:
                break

        if not resolved:
            n_unresolved += 1

    if n_collapsed == 0:
        actions.append(
            f"Stage D: {n_bad0} bad cells remained after Stage A/B', but none had a "
            f"safe collapse candidate (every edge either touched two physical-boundary "
            f"nodes, or every merge would create a negative-volume or newly-bad "
            f"neighbour) - leaving as-is"
        )
        return nodes, cells, cell_groups, n_bl_cells, actions, identity_remap

    new_cells = cur_cells[alive]
    new_cell_groups = cell_groups[alive]
    new_n_bl_cells = int(np.sum(alive[:n_bl_cells]))
    new_nodes, new_cells, new_cell_groups, new_n_bl_cells, node_remap = _compact_mesh(
        cur_nodes, new_cells, new_cell_groups, new_n_bl_cells,
        extra_referenced_nodes=extra_referenced_nodes,
    )
    actions.append(
        f"Stage D: {n_bad0} bad cells after Stage A/B' -> collapsed {n_collapsed} edge(s), "
        f"removing {n_cells_removed_total} cells ({n_unresolved} left unresolved, no safe "
        f"candidate found)"
    )
    logger.info(actions[-1])

    return new_nodes, new_cells.astype(np.int32), new_cell_groups, new_n_bl_cells, actions, node_remap
