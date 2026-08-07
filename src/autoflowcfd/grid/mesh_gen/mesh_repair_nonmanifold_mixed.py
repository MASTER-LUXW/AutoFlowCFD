"""Local cavity patch for non-manifold faces spanning the mixed
prism(BL)+tet(transition/core) mesh - the mixed-mesh counterpart of
mesh_repair_cavity.patch_nonmanifold_cavity.

Split into its own module (rather than added to mesh_repair_cavity.py,
already over this project's 450-line-per-file guideline) purely to keep
file size down; the two modules share the same underlying technique.
"""

from typing import List, Tuple

import numpy as np
from loguru import logger

from .mesh_repair_cavity import _CAVITY_FACE_TEMPLATES, _cavity_boundary_faces

# A prism (v0,v1,v2,w0,w1,w2) splits into exactly these 3 tets - same
# diagonal-consistency rule convert_layers_to_tetrahedra uses, so a
# prism's boundary faces here are bit-identical to what that function
# would have produced for the same slab, and therefore automatically
# conformal with whatever un-split neighbour (prism or tet) still borders
# this patch - PROVIDED v0<v1<v2 by global node index (the bottom
# triangle's own vertices, sorted, with the SAME row permutation carried
# over to the top triangle so w_i stays "above" v_i - convert_layers_to_
# prisms' own convention when it FIRST builds a prism).
#
# That precondition does NOT survive downstream node remapping: mesh_
# background.generate_hybrid_mesh calls _dedupe_coincident_points (seam
# merge, final defensive pass) multiple times after prisms are built,
# each of which can reassign a node's GLOBAL index to an arbitrary
# representative of its coincident-point group - nothing about that
# remap preserves "v0's NEW index < v1's NEW index < v2's NEW index" just
# because it held for the OLD indices. Confirmed directly, not
# theoretical: calling this function on real post-remap prisms without
# re-sorting produced ~23,000 phantom "non-manifold" face groups in an
# ad-hoc diagnostic script - face_extractor.repair_nonmanifold_mixed's
# own _build_prism_face_occurrences (which DOES re-sort every call, see
# its own docstring) found zero on the exact same mesh, proving the
# ~23,000 was entirely an artifact of this function's missing sort, not a
# real defect. Sorting here unconditionally (cheap, always correct
# whether or not the caller's input happens to already be sorted) is
# both the fix and the safe default going forward.
def _split_prisms_to_tets(prisms: np.ndarray) -> np.ndarray:
    bottom = prisms[:, 0:3]
    top = prisms[:, 3:6]
    order = np.argsort(bottom, axis=1)
    row_idx = np.arange(len(prisms))[:, None]
    sb = bottom[row_idx, order]
    st = top[row_idx, order]
    v0, v1, v2 = sb[:, 0], sb[:, 1], sb[:, 2]
    w0, w1, w2 = st[:, 0], st[:, 1], st[:, 2]
    return np.concatenate([
        np.stack([v0, v1, v2, w2], axis=1),
        np.stack([v0, v1, w1, w2], axis=1),
        np.stack([v0, w0, w1, w2], axis=1),
    ], axis=0)


def demote_invalid_prisms_to_tets(
    prism_cells: np.ndarray,
    bl_cell_groups: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Guarantee no exported CPENTA references the same node twice.

    A "collapsed-corner" prism (growth frozen at exactly one base vertex,
    v_i == w_i - see quality_metrics.compute_prism_aspect_ratios' own
    docstring) is a valid nonzero-volume cell by this project's own
    tolerance, but as a CPENTA record it repeats one GRID id in two of its
    6 slots - a malformed element by Nastran's own definition, not merely
    a quality issue. Confirmed directly against a real cube_demo export:
    ANSA 21.0.1 rejected ~21,000 such CPENTA records ("invalid node
    combination"), one per collapsed-corner prism, which is what actually
    produced the reported "empty" patches in the imported mesh - not the
    small tet-volume deficit this project chased earlier (see ProjectFiles
    Part10 P39).

    patch_nonmanifold_cavity_mixed (the aspect-ratio repair pass in
    mesh_background.py already routes every one of these through it, via
    prism_ar <= 500.0) is a best-effort tetgen retile and silently leaves
    the cavity untouched whenever no cluster is `accepted` - and tetgen
    reliably fails or is skipped on cavities built from near-zero-volume
    geometry, which is exactly what a collapsed-corner prism's boundary
    is. Confirmed directly: on that same real export, 100% of the
    ~21,000 flagged prisms were still present, unpatched, with the
    original duplicate node id, despite the AR-based patch call having
    run. This function is the deterministic fallback with no failure
    mode: a collapsed prism splits into exactly 3 tets via the same
    diagonal-consistent rule used everywhere else in this module
    (_split_prisms_to_tets), of which exactly the one referencing the
    repeated node twice is degenerate and dropped; the other 2 are
    ordinary, valid, non-degenerate tets covering the same volume the
    prism did - pure arithmetic, cannot fail the way a tetgen call can.

    Args:
        prism_cells: (n_prism, 6) prism connectivity
        bl_cell_groups: (n_prism,) str array parallel to prism_cells -
            each demoted prism's group name is carried onto its surviving
            tets directly (as `cell_groups`/`direct_cell_groups`), so the
            wall boundary group the prism used to belong to is not lost.

    Returns:
        (new_prism_cells, new_bl_cell_groups, extra_tets, extra_tet_groups)
        - extra_tets/extra_tet_groups are empty arrays (not None) when
        nothing needed demoting, so the caller can always np.vstack/
        np.concatenate them onto merged_cells/cell_groups unconditionally.
    """
    empty_tets = np.empty((0, 4), dtype=prism_cells.dtype)
    empty_groups = np.empty((0,), dtype=object)
    if len(prism_cells) == 0:
        return prism_cells, bl_cell_groups, empty_tets, empty_groups

    has_dup = np.zeros(len(prism_cells), dtype=bool)
    for i in range(6):
        for j in range(i + 1, 6):
            has_dup |= prism_cells[:, i] == prism_cells[:, j]

    if not has_dup.any():
        return prism_cells, bl_cell_groups, empty_tets, empty_groups

    bad_idx = np.flatnonzero(has_dup)
    split_tets = _split_prisms_to_tets(prism_cells[bad_idx])  # (3*n_bad, 4), block layout: all T1s, then T2s, then T3s
    degenerate = (
        (split_tets[:, 0] == split_tets[:, 1]) | (split_tets[:, 0] == split_tets[:, 2]) |
        (split_tets[:, 0] == split_tets[:, 3]) | (split_tets[:, 1] == split_tets[:, 2]) |
        (split_tets[:, 1] == split_tets[:, 3]) | (split_tets[:, 2] == split_tets[:, 3])
    )
    valid_tets = split_tets[~degenerate]
    # np.tile (not np.repeat) matches _split_prisms_to_tets' block layout -
    # row r of the (3*n_bad,4) output belongs to source prism bad_idx[r % n_bad].
    source_idx = np.tile(bad_idx, 3)[~degenerate]

    logger.warning(
        f"{len(bad_idx)} prism(s) with a duplicate node id among their own 6 "
        f"vertices (collapsed-corner, invalid as a CPENTA record) - demoting "
        f"to {len(valid_tets)} plain tet(s), the deterministic fallback for "
        f"whatever the tetgen-based aspect-ratio patch above did not resolve"
    )

    keep_mask = ~has_dup
    return (
        prism_cells[keep_mask],
        bl_cell_groups[keep_mask],
        valid_tets.astype(prism_cells.dtype),
        bl_cell_groups[source_idx],
    )


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
    """Locally re-tetrahedralize non-manifold/flagged-bad cavities spanning
    BOTH prism and tet cells, instead of face_extractor.
    repair_nonmanifold_mixed's own "keep the largest, drop the rest" - see
    mesh_repair_cavity.patch_nonmanifold_cavity's own docstring for why
    unconditional deletion of the "losing" side of an over-shared face
    leaves a real hole.

    Each connected cluster of seed cells (touching an over-shared face, or
    flagged bad by the caller's own keep-mask - e.g. mesh_background.py's
    BL-prism aspect-ratio pass) becomes its OWN separate cavity, exactly
    the same reasoning mesh_repair_cavity.remesh_core_cavity's own module
    docstring gives: a single combined cavity spanning many unrelated bad
    pockets would (a) risk exceeding max_cavity_cells for no reason and
    (b) needlessly re-tetrahedralize the good geometry between two
    unrelated pockets. This matters concretely here: a real cube_demo run
    had ~21,000 flagged "collapsed-corner" BL prisms (see mesh_background.
    py's own aspect-ratio pass) that are almost entirely SEPARATE small
    clusters scattered across the whole body surface, not one contiguous
    region - treating them as a single cavity (an earlier version of this
    function did) made the combined seed balloon past max_cavity_cells
    after just one buffer ring and fall back to a complete no-op for ALL
    of them. Splitting per-cluster lets each individually-small cavity
    (typically a handful of cells) be patched independently even when the
    total flagged count is huge.

    Any prism swept into a cavity (seed or buffer ring) is first split
    into 3 tets (_split_prisms_to_tets) so that cavity can be handed to
    ONE tetgen call as a pure-tet PLC - tetgen has no prism primitive of
    its own. Every cell a retile actually replaces (prism- or tet-origin)
    comes back as a plain interior tet; nothing is re-promoted to a
    prism - the same deliberate, bounded trade remesh_core_cavity's own
    local retile results already make.

    Args:
        nodes: full node array (shared coordinate space for both cell types)
        prism_cells, tet_cells: current cell arrays (BEFORE prism_keep/
            tet_keep are applied - both proposals, not yet acted on)
        prism_keep, tet_keep: bool arrays - False marks a cell that would
            otherwise be unconditionally dropped/flagged bad
        bl_cell_groups: (n_prism,) str array parallel to prism_cells
        cell_groups: (n_tet,) str array parallel to tet_cells - every
            newly retiled cell gets '' (same convention as
            patch_nonmanifold_cavity)
        n_buffer_rings: face-adjacency rings of ordinary neighbours padded
            around each cluster before extracting its own boundary
        max_cavity_cells: per-CLUSTER safety cap (not a total budget) - a
            single cluster this large signals something structurally
            different (see patch_nonmanifold_cavity's own docstring);
            skipped rather than attempted, same as remesh_core_cavity's
            own per-cluster size cap
        max_clusters_attempted: safety cap on the total NUMBER of separate
            clusters this call will attempt, mirroring remesh_core_cavity's
            own cap for the identical reason (many small clusters, each
            cheap but with real per-call tetgen overhead, can still add up)

    Returns:
        (new_nodes, new_prism_cells, new_tet_cells, new_bl_cell_groups,
        new_cell_groups) - unchanged (not copies) only if BOTH keep masks
        are already all-True; otherwise reflects however many clusters
        were successfully patched (0 or more - a partial result, with
        oversized/failed clusters left exactly as the caller's own
        keep-mask found them, is expected and normal, not an error).
    """
    if prism_keep.all() and tet_keep.all():
        return nodes, prism_cells, tet_cells, bl_cell_groups, cell_groups

    from .mesh_tetgen_core import fill_core_volume, CORE_TETGEN_MINRATIO, CORE_TETGEN_MINDIHEDRAL
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n_prism = len(prism_cells)
    n_tet = len(tet_cells)
    n_total = n_prism + n_tet

    # Global cell id convention for this function only: [0, n_prism) is
    # prisms, [n_prism, n_prism+n_tet) is tets.
    keep = np.concatenate([prism_keep, tet_keep])

    # Derive each prism's 8 boundary triangles from the SAME verified
    # 3-tet split used later to actually retile a cavity, rather than
    # hand-listing the quad diagonals directly (a hand-written diagonal
    # guess was confirmed NOT to match the true exposed boundary of the
    # 3-tet split). All 3 of a prism's split tets share that prism's own
    # index in cell_of_face, so a prism is always grown/replaced as one
    # whole unit, never partially.
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

    # Face-adjacency graph restricted to interior (count==2) faces, for
    # both clustering the seed cells into connected components and
    # growing each cluster's own buffer rings - a count>2 (non-manifold)
    # face has no single well-defined "other side" to grow through, and a
    # count==1 (boundary) face has none at all.
    interior_group = np.flatnonzero(group_counts == 2)
    interior_occ = np.isin(group_id, interior_group)
    occ_cell = cell_of_face[interior_occ]
    occ_group = group_id[interior_occ]
    order = np.argsort(occ_group, kind='stable')
    occ_cell_sorted = occ_cell[order]
    occ_group_sorted = occ_group[order]
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

        try:
            retiled_nodes, retiled_tets, _, _ = fill_core_volume(
                local_points, local_faces, verbose=False,
                minratio=CORE_TETGEN_MINRATIO, mindihedral=CORE_TETGEN_MINDIHEDRAL,
            )
        except Exception:
            n_failed += 1
            continue

        n_boundary_pts = len(local_points)
        if not np.array_equal(retiled_nodes[:n_boundary_pts], local_points):
            n_failed += 1
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
            f"accepted (skipped_size={n_skipped_size}, failed={n_failed}) - "
            f"falling back to plain cell removal"
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
        f"skipped_size={n_skipped_size}, failed={n_failed})"
    )
    return new_nodes, new_prism_cells, new_tet_cells, new_bl_cell_groups, new_cell_groups
