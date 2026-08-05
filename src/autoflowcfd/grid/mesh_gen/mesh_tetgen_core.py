"""Constrained tetrahedralization of the domain core using tetgen.

Fills the volume enclosed by a closed piecewise-linear complex (PLC) - the
boundary-layer (BL) outer surface plus the unmodified outer-shell faces
(inlet/outlet/tunnel/symmetry-like boundaries) - with tetgen, instead of the
old arbitrary padded-bounding-box + Cartesian background grid. The PLC is by
construction exactly the closed surface the input mesh already describes, so
the result can never extend outside the real domain.
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from loguru import logger

# Core-fill tetgen quality/grading knobs, shared by every caller of
# fill_core_volume that wants this project's own tightened standard rather
# than tetgen's out-of-the-box defaults (minratio~2.0, mindihedral~0
# effectively unconstrained). Originally lived only in
# mesh_background_merge.py (the main core fill's own caller) - moved here,
# the lowest-level module every one of fill_core_volume's callers already
# imports from, specifically so mesh_repair_cavity.py's Stage B' (local
# cavity re-tiling) can use the SAME standard for its own, much smaller
# fill_core_volume calls instead of silently falling back to tetgen's
# looser defaults. That inconsistency was a real, measured gap, not
# theoretical: Stage B' was rejecting ~72% of its own cavity retile
# attempts as "not an improvement" on a real case, and the retile itself
# had no reason to actually BE an improvement over the original (already
# badly-graded) cavity while using looser shape-quality bounds than what
# produced that cavity's own neighbours in the first place.
CORE_TETGEN_MINRATIO = 1.15  # was 1.4; tetgen default ~2.0 (lower = stricter)
CORE_TETGEN_MINDIHEDRAL = 15.0  # unchanged - dihedral wasn't the implicated metric
CORE_VOLUME_CAP_FRACTION = 0.08  # was 0.15, of max_cell_size**3


def build_seam_taper_scale(
    n_nodes: int,
    extrude_faces: np.ndarray,
    core_faces: np.ndarray,
    taper_rings: int = 100,
) -> np.ndarray:
    """Compute a per-node [0, 1] BL-extrusion scale that tapers smoothly to
    zero at the seam shared with core-only faces (e.g. where a ground plane
    meets the tunnel wall).

    Hard-pinning the seam to exactly zero displacement (an earlier version
    of this function) is not enough on its own: every triangle that touches
    a pinned node collapses toward zero area at the outer BL layer (1-2 of
    its 3 vertices frozen while the third moves by the full BL thickness),
    handing tetgen a boundary surface with degenerate/near-zero-area facets
    right along the whole seam perimeter - this reliably crashed tetgen's
    native tetrahedralization on real automotive geometry. Smoothly ramping
    the scale up over enough rings of mesh connectivity keeps every facet's
    vertices within a comparable displacement range near the seam, avoiding
    that degeneracy while still guaranteeing exact conformality (scale is
    exactly 0, not just small, right at the seam itself).

    The default of 100 rings is deliberately generous, not a tight local
    estimate: on real automotive geometry, the seam can pass through a small
    but geometrically tight feature (e.g. a body's underbody contact patch
    welded to the ground, with near-90 degree edges only a few mm long) -
    verified empirically that a narrow taper (~4 rings) still produced a
    self-intersecting BL surface there, while widening it resolved that
    without needing a separate local-feature-size analysis.

    Args:
        n_nodes: total number of nodes in the shared node array
        extrude_faces: faces that will be BL-extruded
        core_faces: faces used unmodified as part of the outer PLC boundary
        taper_rings: number of mesh-connectivity hops over which the scale
            ramps from 0 (at the seam) to 1 (unaffected interior)

    Returns:
        float array in [0, 1], shape=(n_nodes,)
    """
    scale = np.ones(n_nodes, dtype=np.float64)
    if len(extrude_faces) == 0 or len(core_faces) == 0:
        return scale

    extrude_node_idx = np.unique(extrude_faces)
    core_node_idx = np.unique(core_faces)

    in_extrude = np.zeros(n_nodes, dtype=bool)
    in_extrude[extrude_node_idx] = True
    in_core = np.zeros(n_nodes, dtype=bool)
    in_core[core_node_idx] = True
    seam_nodes = np.flatnonzero(in_extrude & in_core)

    logger.info(f"Seam nodes (shared between extruded and core-only faces): {len(seam_nodes)}")
    if len(seam_nodes) == 0:
        return scale

    # Multi-source unweighted shortest-path (hop count) from every seam node,
    # restricted to the extrude-eligible face graph (only that region's
    # nodes actually move, so only its connectivity matters for the taper).
    edges = np.vstack([extrude_faces[:, [0, 1]], extrude_faces[:, [1, 2]], extrude_faces[:, [2, 0]]])
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_nodes, n_nodes))

    from scipy.sparse.csgraph import dijkstra
    hop_dist = dijkstra(graph, indices=seam_nodes, unweighted=True, min_only=True)

    t = np.clip(hop_dist / taper_rings, 0.0, 1.0)
    smoothstep = t * t * (3.0 - 2.0 * t)
    # Nodes unreachable from any seam node (not connected through the
    # extrude-face graph, e.g. an unrelated embedded shell) keep scale=1.
    unreachable = ~np.isfinite(hop_dist)
    smoothstep[unreachable] = 1.0

    scale = smoothstep
    logger.info(
        f"BL taper applied over {taper_rings} connectivity rings from the seam "
        f"({int(np.sum(scale < 1.0))} nodes affected)"
    )
    return scale


def compute_local_thickness_limit(
    nodes: np.ndarray,
    extrude_faces: np.ndarray,
    extrude_node_idx: np.ndarray,
    domain_size: float,
    safety_factor: float = 0.45,
    angle_threshold_deg: float = 60.0,
    search_radius_fraction: float = 0.08,
) -> np.ndarray:
    """Cap each extrude-eligible node's *cumulative* BL thickness to a
    fraction of its local geometric gap to the nearest facing surface, so
    two BL fronts growing toward each other across a tight feature (e.g. a
    body's underbody a few cm above the ground) stop before they can cross
    - instead of growing at a uniform rate and relying on
    repair_nonmanifold_cells to clean up the resulting overlap afterward.

    The gap is measured on the undeformed (layer-0) surface: for each node,
    search nearby surface nodes within `domain_size * search_radius_fraction`
    and keep only those roughly "ahead" of the node along its own outward
    normal (within
    `angle_threshold_deg`) - this is what distinguishes a genuine facing
    gap from the node's own immediately-neighbouring mesh (which is always
    spatially close simply from local mesh resolution, not a real gap, and
    lies roughly in-plane rather than ahead of the normal). The nearest
    qualifying point's distance is the local gap; `safety_factor` (< 0.5)
    leaves margin for the facing surface's own BL growth toward this one.

    This is a geometric heuristic, not a formal proof of non-intersection:
    it's evaluated once on the undeformed surface, so a strongly curved
    front whose true closest-approach point shifts as both sides extrude
    could still, in principle, converge faster than estimated. It's a
    substantial reduction in how often crossings occur, not a guarantee -
    repair_nonmanifold_cells remains in place as a safety net for whatever
    it doesn't catch.

    Args:
        nodes: (n_nodes, 3) ALL surface node coordinates (whole surface,
            not just the extrude-eligible subset - the nearest feature
            limiting a wall's BL growth may be a different wall entirely)
        extrude_faces: (m, 3) faces eligible for BL extrusion, used only to
            compute each node's own outward (extrusion) normal
        extrude_node_idx: node indices that will actually be extruded
        domain_size: overall domain characteristic length (bounding-box
            diagonal), bounding both the search radius and the fallback cap
        safety_factor: fraction of the raw gap distance kept as the cap
        angle_threshold_deg: half-angle of the "ahead of the normal" cone
            used to separate a facing gap from same-sheet mesh neighbors
        search_radius_fraction: fraction of domain_size used as the KDTree
            ball-query radius. BL growth is only ever *targeted* to reach
            ~2% of domain_size (see extrude_layers' bl_target_thickness);
            extrude_layers separately and unconditionally hard-stops at 40%
            of domain_size regardless of what this function computes, so
            that extreme case is already covered without this search having
            to reach anywhere near it. The previous default (0.4, i.e. the
            *same* 40% used for that unrelated hard-stop) made the query
            ball routinely enclose a large fraction of the whole surface
            mesh for typical external-aero domains, turning what should be
            a local neighbor search into something close to brute-force
            over every extrude-eligible node - risking multi-minute (or
            worse) runtimes on real fine meshes. A modest multiple of the
            2% target (default 8%) keeps comfortable margin over the
            normal case while cutting the searched volume, and therefore
            the typical candidate count per query, by roughly (0.4/0.08)^3
            = 125x.

    Returns:
        (n_nodes,) float array: max cumulative BL thickness in meters for
        each node (np.inf where no nearby facing feature was found)
    """
    from scipy.spatial import cKDTree

    n_nodes = len(nodes)
    limit = np.full(n_nodes, np.inf, dtype=np.float64)
    if len(extrude_node_idx) == 0:
        return limit

    face_normals = _face_normals(nodes, extrude_faces)
    avg_normal = _average_node_normals(n_nodes, extrude_faces, face_normals)

    search_radius = domain_size * search_radius_fraction
    cos_threshold = np.cos(np.radians(angle_threshold_deg))

    tree = cKDTree(nodes)
    query_points = nodes[extrude_node_idx]

    # Query and process in chunks rather than all at once, and vectorize the
    # per-candidate angle/distance test across each chunk instead of a
    # Python-level per-node loop. On a fine surface mesh, a domain-scale
    # search_radius can enclose tens of thousands of same-sheet candidates
    # per query (the angle test below discards nearly all of them - they're
    # in-plane neighbors, not a genuine facing feature) - materializing
    # every query's full candidate list at once (the previous unchunked
    # behavior) scales memory with n_queries * avg_candidates, which reached
    # multi-GB transient usage on a ~25k-surface-node case, and the
    # per-node Python loop that followed (34k+ iterations, each doing
    # several small numpy calls) dominated runtime - 100+ seconds per call,
    # repeated every BL-extrusion attempt. This produces numerically
    # identical results (same radius, same angle test, same nearest-ahead-
    # point selection) - it only changes how the work is batched.
    chunk_size = 200
    n_capped = 0
    min_cap_seen = np.inf

    for start in range(0, len(query_points), chunk_size):
        end = min(start + chunk_size, len(query_points))
        chunk_node_idx = extrude_node_idx[start:end]
        neighbor_lists = tree.query_ball_point(
            query_points[start:end], r=search_radius, workers=-1
        )
        counts = np.fromiter(
            (len(lst) for lst in neighbor_lists), dtype=np.int64, count=len(neighbor_lists)
        )
        if counts.sum() == 0:
            continue

        row_idx = np.repeat(np.arange(len(chunk_node_idx)), counts)
        flat_candidates = np.concatenate(
            [np.asarray(lst, dtype=np.int64) for lst in neighbor_lists if len(lst) > 0]
        )

        node_idx_per_row = chunk_node_idx[row_idx]
        d = nodes[flat_candidates] - nodes[node_idx_per_row]
        dist = np.linalg.norm(d, axis=1)
        real = dist > 1e-9
        safe_dist = np.where(real, dist, 1.0)
        cosang = np.einsum('ij,ij->i', d, avg_normal[node_idx_per_row]) / safe_dist
        ahead = real & (cosang > cos_threshold)
        if not np.any(ahead):
            continue

        dist_masked = np.where(ahead, dist, np.inf)
        seg_min = np.full(len(chunk_node_idx), np.inf)
        np.minimum.at(seg_min, row_idx, dist_masked)

        has_match = np.isfinite(seg_min)
        if np.any(has_match):
            capped_vals = seg_min[has_match] * safety_factor
            limit[chunk_node_idx[has_match]] = capped_vals
            n_capped += int(has_match.sum())
            min_cap_seen = min(min_cap_seen, float(capped_vals.min()))

    if n_capped:
        logger.info(
            f"Local BL thickness limiting: {n_capped} nodes capped by a "
            f"nearby facing feature (min cap {min_cap_seen:.4e} m)"
        )
    return limit


def _face_normals(nodes: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0, v1, v2 = nodes[faces[:, 0]], nodes[faces[:, 1]], nodes[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-10)
    return normals / norms


def _average_node_normals(n_nodes: int, faces: np.ndarray, face_normals: np.ndarray) -> np.ndarray:
    sums = np.zeros((n_nodes, 3), dtype=np.float64)
    counts = np.zeros(n_nodes, dtype=np.int64)
    flat_nodes = faces.ravel()
    np.add.at(sums, flat_nodes, np.repeat(face_normals, 3, axis=0))
    np.add.at(counts, flat_nodes, 1)

    mask = counts > 0
    avg = np.zeros_like(sums)
    avg[mask] = sums[mask] / counts[mask, np.newaxis]
    norms = np.maximum(np.linalg.norm(avg, axis=1, keepdims=True), 1e-10)
    avg[mask] = avg[mask] / norms[mask]
    return avg


def _dedupe_coincident_points(
    points: np.ndarray,
    faces: np.ndarray,
    tolerance: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse coincident points (within tolerance) and remap faces (or
    tetrahedral cells - `faces` is just an (n, k) index array, remapped via
    plain fancy indexing, so this works unmodified for k=3 or k=4).

    Also returns `remap` (shape=(len(points),), old index -> new index) so
    a caller holding ANY OTHER index array into the same original `points`
    (e.g. fill_core_volume's separately-read `tgen.trifaces`) can apply the
    identical remapping and stay consistent with the returned
    `unique_points`/`new_faces` - passing `remap[some_other_array]` does
    that. `remap` is the identity when no coincident points were found.

    Fully transitive (uses scipy connected_components over the coincidence
    graph, not a one-hop union), unlike the older `merge_conforming_meshes`
    node-dedup logic elsewhere in this package.

    Two call sites: the original fallback here in fill_core_volume, for
    when tetgen doesn't return a fully conformal boundary; and
    mesh_background.generate_hybrid_mesh's final defensive pass over the
    WHOLE merged mesh, for a distinct failure mode found on a real case -
    when mesh_repair.compute_bl_thickness_limit_override's reactive BL
    thickness cap needs to cap a very large fraction of surface vertices
    (itself a symptom of something upstream producing widespread, not
    localized, bad cells), many nodes' `remaining_budget` hits exactly zero
    within the same few layers, freezing them at an identical coordinate
    for every subsequent layer - each still gets its own distinct global
    node index (one new index per layer, unconditionally), so the result
    is a large number of geometrically-coincident points under different
    indices. That doesn't trip repair_nonmanifold_cells (which matches by
    exact node index, not geometry) and doesn't reliably trip the
    degenerate-volume filter either (a tet mixing a frozen node with a
    still-growing neighbour can have a small but non-negligible volume) -
    it's a silent topological tear (two geometrically-identical faces under
    different index sets, each independently counted as a normal boundary
    face) rather than a crash, so nothing upstream of this catches it.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    pairs = tree.query_pairs(tolerance)

    n_points = len(points)
    if not pairs:
        return points, faces, np.arange(n_points, dtype=np.int64)

    rows = [p[0] for p in pairs]
    cols = [p[1] for p in pairs]
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_points, n_points))
    n_components, labels = connected_components(graph, directed=False)

    # Use the smallest original index in each component as the representative.
    representative = np.full(n_components, n_points, dtype=np.int64)
    np.minimum.at(representative, labels, np.arange(n_points))

    new_index_of_label = np.arange(n_components)
    unique_points = points[representative]
    remap = new_index_of_label[labels]

    new_faces = remap[faces]
    logger.warning(
        f"Coincident-point fallback stitch: {n_points} -> {len(unique_points)} points "
        f"({n_points - len(unique_points)} merged)"
    )
    return unique_points, new_faces, remap


def _tet_volumes(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Unsigned tetrahedron volumes (orientation-independent)."""
    p0 = nodes[cells[:, 0]]
    p1 = nodes[cells[:, 1]]
    p2 = nodes[cells[:, 2]]
    p3 = nodes[cells[:, 3]]
    return np.abs(np.einsum('ij,ij->i', p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0


def repair_nonmanifold_cells(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Detect tetrahedra that make some triangular face shared by more than
    2 cells (non-manifold) and mark the redundant ones for removal.

    The main known cause is fixed at the source now: an isolated embedded
    solid (e.g. a car body) needs a tetgen hole seed or its own interior
    gets filled with spurious tetrahedra that overlap the BL prisms already
    occupying that space (see mesh_domain_classify.find_point_inside_closed_shell
    and fill_core_volume's `holes` parameter). This function remains as a
    safety net for whatever that doesn't catch (e.g. a hole point that
    couldn't be found for a very non-convex solid, or a genuinely tight BL
    seam producing a near-degenerate boundary facet that tetgen's
    `nobisect=True` core fill can't resolve by inserting a boundary point
    there). Left unrepaired, this is a real conservation violation: a
    finite-volume face extraction can only ever attribute a shared face to
    2 of the 3+ cells touching it, silently dropping flux through it for
    the rest (see face_extractor.py's hard failure on exactly this
    condition).

    A triangular face has at most one legitimate neighbouring cell per
    side (the tet whose 4th vertex, the "apex", lies on that side of the
    face's plane). When more than one cell shares an apex-side, they are
    physically overlapping duplicates of each other - keep only the
    largest-volume one and drop the rest, independently per over-shared
    face.

    Args:
        nodes: (n_nodes, 3) node coordinates
        cells: (n_cells, 4) tetrahedral connectivity

    Returns:
        Boolean keep-mask, shape=(n_cells,); False marks a cell to remove
    """
    n_cells = len(cells)
    keep = np.ones(n_cells, dtype=bool)
    if n_cells == 0:
        return keep

    face_templates = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
    ], dtype=np.int64)
    apex_of_face = np.array([3, 2, 1, 0], dtype=np.int64)

    all_faces = cells[:, face_templates].reshape(-1, 3)
    apex_nodes = cells[:, apex_of_face].reshape(-1)
    cell_of_face = np.repeat(np.arange(n_cells), 4)

    sorted_faces = np.sort(all_faces, axis=1)
    face_dtype = np.dtype((np.void, sorted_faces.dtype.itemsize * 3))
    face_voids = np.ascontiguousarray(sorted_faces).view(face_dtype).reshape(-1)

    order = np.argsort(face_voids, kind='stable')
    sorted_voids = face_voids[order]
    sorted_cells = cell_of_face[order]
    sorted_apex = apex_nodes[order]
    sorted_face_nodes = sorted_faces[order]

    change = np.flatnonzero(sorted_voids[1:] != sorted_voids[:-1]) + 1
    group_starts = np.concatenate([[0], change])
    group_ends = np.concatenate([change, [len(sorted_voids)]])
    counts = group_ends - group_starts

    invalid_groups = np.flatnonzero(counts > 2)
    if len(invalid_groups) == 0:
        return keep

    volumes = _tet_volumes(nodes, cells)
    n_removed = 0

    for gi in invalid_groups:
        s, e = group_starts[gi], group_ends[gi]
        face_cells = sorted_cells[s:e]
        n0, n1, n2 = sorted_face_nodes[s]
        p0 = nodes[n0]
        normal = np.cross(nodes[n1] - p0, nodes[n2] - p0)

        apexes = sorted_apex[s:e]
        signed_dist = (nodes[apexes] - p0) @ normal

        for side_mask in (signed_dist > 0, signed_dist <= 0):
            side_cells = face_cells[side_mask]
            if len(side_cells) <= 1:
                continue
            best = side_cells[np.argmax(volumes[side_cells])]
            for c in side_cells:
                if c != best and keep[c]:
                    keep[c] = False
                    n_removed += 1

    if n_removed:
        logger.warning(
            f"Repaired {len(invalid_groups)} non-manifold faces by removing "
            f"{n_removed} redundant overlapping tetrahedra"
        )
    return keep


def attribute_cells_from_trifaces(
    cells: np.ndarray,
    trifaces: np.ndarray,
    triface_markers: np.ndarray,
    marker_to_name: dict,
) -> np.ndarray:
    """Recover each cell's source boundary group from fill_core_volume's
    facet markers, for cells that own a marked boundary face.

    Needed once fill_core_volume runs with nobisect=False (graded max-cell-
    size regions): tetgen may subdivide an input boundary facet into many
    sub-facets to satisfy the size cap, so those sub-facets' node indices
    no longer exist in the pre-fill surface mesh, and plain node-index
    matching (the pre-existing mesh_boundary.py fallback) can no longer
    find them. tetgen's own facet markers are inherited by every
    sub-facet of a marked input facet regardless of subdivision, so
    matching a cell's own boundary face against the marker set (by node
    SET, not index into some external array) works unconditionally.

    Args:
        cells: (n_cells, 4) tetrahedral connectivity, in the SAME index
            space as `trifaces` (i.e. call this before any node reindexing
            - reindexing only changes what a node index means, never which
            cell owns which row, so the returned per-row group assignment
            stays valid across a later remap)
        trifaces: (n_tri, 3) boundary triangles from fill_core_volume
        triface_markers: (n_tri,) int32, 0 = no marker (interior-only
            facet, e.g. the BL/core interface - never a real exterior
            boundary so it's fine to leave unattributed)
        marker_to_name: maps a nonzero marker value back to its boundary
            group name

    Returns:
        (n_cells,) str array, '' where the cell owns no marked boundary
        face
    """
    n_cells = len(cells)
    cell_groups = np.full(n_cells, '', dtype=object)

    nonzero = triface_markers != 0
    if not np.any(nonzero):
        return cell_groups

    marked_tri = np.sort(trifaces[nonzero], axis=1)
    marked_markers = triface_markers[nonzero]
    tri_dtype = np.dtype((np.void, marked_tri.dtype.itemsize * 3))
    marked_hash = np.ascontiguousarray(marked_tri).view(tri_dtype).reshape(-1)

    order = np.argsort(marked_hash, kind='stable')
    sorted_hash = marked_hash[order]
    sorted_marker = marked_markers[order]

    face_templates = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)
    all_faces = cells[:, face_templates].reshape(-1, 3)
    cell_of_face = np.repeat(np.arange(n_cells), 4)
    face_hash = np.ascontiguousarray(np.sort(all_faces, axis=1)).view(tri_dtype).reshape(-1)

    pos = np.clip(np.searchsorted(sorted_hash, face_hash), 0, len(sorted_hash) - 1)
    matched = sorted_hash[pos] == face_hash

    for cell_idx, marker in zip(cell_of_face[matched].tolist(), sorted_marker[pos[matched]].tolist()):
        cell_groups[cell_idx] = marker_to_name[marker]

    return cell_groups


# Rough conversion from a target edge length to a tetgen maxvolume cap
# (regular-tet volume/edge^3 is ~0.118; Delaunay-refined tets are less
# regular and tetgen's own region cap isn't strictly tight in practice -
# so this is deliberately generous, not exact).
_VOLUME_SHAPE_FACTOR = 0.15


# NOTE: an earlier version of this module graded the core fill's max
# cell size outward from the wall via nested icosphere regions
# (build_graded_regions/_generate_icosphere). It was abandoned - tetgen's
# per-region variable-volume refinement does not reliably converge
# multiple simultaneous regions to their own targets when they compete
# for one shared Steiner budget (see fill_core_volume's `regions` doc) -
# in favor of the single flat region mesh_background.py builds directly.
# Removed rather than left unreferenced to avoid it being wired back in
# without that context.

def estimate_steinerleft(
    points: np.ndarray,
    regions: Optional[List[Tuple[np.ndarray, int, float]]],
) -> int:
    """Estimate a Steiner-point budget (tetgen's `steinerleft`) generous
    enough for the requested region(s), scaled to the actual problem size
    rather than a fixed constant.

    tetgen's default steinerleft=100000 is a global cap on how many Steiner
    points it will ever insert, shared across the WHOLE mesh - with a
    region's own maxvolume target well below the PLC's natural
    (unconstrained) tet size, it can run out long before that target is
    reached everywhere, silently leaving a long tail of oversized cells in
    whatever pockets happened to refine last (measured directly: a 5.5x3x3
    m domain capped at 0.05 m with a fixed 300,000 budget left 6-10% of
    cells over 1.5x the target and a worst-case cell ~5-6x over).

    The domain-wide grading region (present whenever max_cell_size is set -
    see mesh_background._build_merged_mesh) always has the LARGEST maxvol
    of any region passed here, so bbox_volume / coarsest_maxvol estimates
    how many cells it alone needs to fill the core - that's the number this
    behaves identically to when there is exactly one region (unchanged from
    the original single-region formula, and - as of Stage B's core-side
    local repair regions being removed, see mesh_repair.py's module
    docstring - the only case `regions` now ever actually contains in
    practice: at most 1 entry).

    The `n_extra_regions` handling below is dead in current usage but kept
    rather than special-cased away, in case a future caller legitimately
    passes more than one region again: dividing the FULL bbox by the
    smallest maxvol among several regions (an earlier version of this
    function, using `min(maxvol for ...)`) badly overestimates whenever one
    of them is a small local patch rather than a domain-wide target -
    observed directly on a real case with Stage B's now-removed core
    regions, an estimate of ~17.8 BILLION target-sized tets for a domain
    whose single-region core fill converged around 1.2M tets. Note this
    estimate is advisory only, not a hard constraint: tetgen was confirmed
    to converge to the *identical* actual tet count regardless of whether
    steinerleft was the (buggy) inflated value or this function's corrected
    one - the real 5x core-fill blowup that estimate coincided with
    (1.2M -> 6.1M tets) turned out to be a separate, still-unresolved
    tetgen multi-region-refinement behavior (see mesh_repair.py), not
    something this budget number was ever actually causing.

    Args:
        points: PLC boundary points, shape=(n, 3) - only used for its
            bounding-box volume
        regions: (seed_point, region_id, maxvol) tuples, or None/empty for
            an unconstrained (nobisect=True) fill

    Returns:
        steinerleft, clamped to [300_000, 20_000_000] - or 100_000
        (tetgen's own default) when no regions are active at all.
    """
    if not regions:
        return 100_000

    bbox_volume = float(np.prod(np.max(points, axis=0) - np.min(points, axis=0)))
    coarsest_maxvol = max(maxvol for _, _, maxvol in regions)
    estimated_tets = bbox_volume / max(coarsest_maxvol, 1e-30)
    n_extra_regions = len(regions) - 1
    extra_tets = n_extra_regions * 200_000

    logger.info(
        f"Steiner-point budget estimate: ~{estimated_tets:,.0f} domain-wide target-sized tets"
        + (f" + {n_extra_regions} local repair region(s) x 200,000" if n_extra_regions else "")
    )
    return int(np.clip((estimated_tets + extra_tets) * 3.0, 300_000, 20_000_000))


def fill_core_volume(
    points: np.ndarray,
    faces: np.ndarray,
    minratio: float = 1.4,
    mindihedral: float = 15.0,
    holes: Optional[List[np.ndarray]] = None,
    regions: Optional[List[Tuple[np.ndarray, int, float]]] = None,
    face_markers: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Constrained-tetrahedralize the volume enclosed by a closed PLC.

    Args:
        points: (n_points, 3) float64 PLC vertices
        faces: (n_faces, 3) int32 PLC triangles (closed, watertight)
        minratio: max radius-edge ratio quality bound (tetgen convention;
            lower = higher quality, 1.0 is a perfect tet)
        mindihedral: min dihedral angle quality bound (degrees)
        holes: points, one strictly inside each isolated embedded solid in
            the PLC (mesh_domain_classify.find_point_inside_closed_shell).
            Without these, tetgen has no way to know an internal closed
            surface bounds a solid rather than just another constraint -
            it fills the fluid region around it AND that solid's own
            (BL-extruded) interior, producing spurious tetrahedra that
            overlap the BL prisms already occupying that cavity.
        regions: (seed_point, region_id, maxvolume) tuples (built by the
            caller, mesh_background.py) for capping max cell size per
            graded tier. Note: tetgen's own background-mesh sizing (`bgmesh`/
            `metric`, tetgen 0.8.4) is not used here - it segfaults
            unconditionally in this environment and package version
            regardless of settings (reproducible on a trivial cube,
            matching an unresolved upstream issue with no test coverage
            for that path) - region-based grading is used instead: proven
            stable, if less smoothly continuous.

            Passing `regions` switches off `nobisect`: enforcing a max
            cell size near a coarse far-field boundary facet (e.g. a
            sparsely-triangulated tunnel/inlet/outlet wall) requires
            tetgen to be allowed to subdivide that facet itself - with
            `nobisect` on (the default, no `regions`), any region touching
            the domain's own outer boundary is provably unaffected by its
            volume cap at all (verified: identical output with and
            without the cap on a boundary-adjacent region), because
            `nobisect` forbids inserting points on or near boundary
            facets and that blocks volume-based splitting of the
            boundary-adjacent cells too, not just the facets themselves.
        face_markers: (n_faces,) int32, one marker per input face, required
            together with `regions` - the boundary attribution mechanism
            (mesh_background.py) can no longer match subdivided boundary
            faces back to their source group by node index (nobisect=False
            means those indices no longer exist verbatim in the input), so
            it uses tetgen's own facet markers instead, which are inherited
            by every sub-facet a marked facet gets split into and are
            returned via this function's 3rd/4th outputs.
        verbose: log this call's own routine per-call progress (boundary
            point/face counts, Steiner budget, completion) at INFO level
            (default, matching prior behavior exactly). False drops those
            same lines entirely (not merely demoted to DEBUG - this
            project's default loguru sink shows DEBUG and above, so a
            demotion alone would not actually reduce visible output) - for
            a caller that makes many small calls in a loop
            (mesh_repair.remesh_core_cavity, one call per repaired cavity)
            where each individual call's own progress isn't interesting on
            its own, only the caller's own summary is. Warnings (non-
            conformal boundary, self-intersection) always stay at their
            normal level regardless of this flag - they indicate something
            a caller needs to see, not routine progress.

    Returns:
        (nodes, tets, trifaces, triface_markers): nodes shape=(n, 3)
        float64 (input points preserved verbatim as the first len(points)
        rows, even under subdivision - verified empirically, tetgen only
        appends new points, it never reorders/replaces existing ones),
        tets shape=(m, 4) int64. trifaces/triface_markers are None unless
        `face_markers` was given, else the tetrahedralized boundary
        triangles (shape=(p, 3) int64, indices into `nodes`) and their
        inherited markers (shape=(p,) int32).
    """
    import tetgen

    points = np.ascontiguousarray(points, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    nobisect = not bool(regions)
    log = logger.info if verbose else (lambda *_a, **_k: None)

    log(
        f"Tetrahedralizing core volume: {len(points)} boundary points, "
        f"{len(faces)} boundary faces (tetgen, nobisect={nobisect})..."
    )

    if face_markers is not None:
        tgen = tetgen.TetGen(points, faces, np.ascontiguousarray(face_markers, dtype=np.int32))
    else:
        tgen = tetgen.TetGen(points, faces)
    if holes:
        for hole_pt in holes:
            tgen.add_hole(hole_pt)
        log(f"Marked {len(holes)} tetgen hole seed(s) for isolated embedded solids")
    if regions:
        for seed_pt, region_id, maxvol in regions:
            tgen.add_region(region_id, seed_pt, maxvol)
        log(f"Marked {len(regions)} graded max-cell-size region(s)")

    steinerleft = estimate_steinerleft(points, regions)
    log(f"Steiner-point budget: {steinerleft:,}")

    try:
        nodes, elems, _attr, _markers = tgen.tetrahedralize(
            plc=True, nobisect=nobisect, quality=True,
            minratio=minratio, mindihedral=mindihedral,
            regionattrib=bool(regions), varvolume=bool(regions),
            steinerleft=steinerleft,
        )
    except RuntimeError as e:
        if "self-intersection" in str(e).lower():
            raise RuntimeError(
                f"{e}. The BL outer surface self-intersects at a tight local "
                f"feature (common at small welded contact patches with sharp "
                f"edges). Try fewer/thinner BL layers (--max-layers, "
                f"--min-cell-size) - naive normal-offset extrusion has no "
                f"per-feature thickness limiting yet, so cumulative BL "
                f"thickness must stay well under the tightest local gap in "
                f"the geometry."
            ) from e
        if "removevertexbyflips" in str(e).lower() or "internal tetgen error" in str(e).lower():
            # Observed on a real case when Stage B's reactive BL thickness
            # cap (mesh_repair.compute_bl_thickness_limit_override) needs to
            # cap a very large fraction of surface vertices - itself already
            # a symptom of Stage A leaving widespread, not localized, bad
            # cells - producing a boundary facet with enough near-coincident
            # points to exceed tetgen's own numerical robustness limits
            # internally (a tetgen implementation limitation, not a
            # meshing-strategy error on this codebase's side) rather than
            # failing with a clearer diagnostic like the self-intersection
            # case above.
            raise RuntimeError(
                f"{e}. tetgen hit an internal robustness limit - on a case "
                f"seen directly, this followed a very widespread Stage B "
                f"BL-thickness cap (a sign Stage A already found bad cells "
                f"across much of the surface, not just a few corners). Try "
                f"loosening --growth-rate/--min-cell-size/--max-layers so "
                f"Stage A has fewer bad cells to begin with."
            ) from e
        raise

    trifaces = None
    triface_markers = None
    if face_markers is not None:
        trifaces = tgen.trifaces.astype(np.int64)
        triface_markers = tgen.triface_markers.astype(np.int32)

    n_input = len(points)
    conformal = nodes.shape[0] >= n_input and np.array_equal(nodes[:n_input], points)

    if not conformal:
        logger.warning(
            "tetgen did not preserve all boundary points verbatim "
            "(likely near-duplicate/degenerate input facets); "
            "falling back to coincident-point stitching"
        )
        nodes, elems, remap = _dedupe_coincident_points(nodes, elems)
        if trifaces is not None:
            # trifaces was read from tgen.trifaces in the PRE-dedupe index
            # space (same node array `nodes` was in before the line above).
            # Left unremapped, it desynced from the now-renumbered
            # nodes/elems - mesh_background.attribute_cells_from_trifaces
            # matches trifaces against core_tets by sorted-node-triple, so
            # a stale index space made that matching silently miss or
            # misattribute boundary cells whenever this fallback and
            # face_markers (i.e. max_cell_size) were both active.
            trifaces = remap[trifaces]

    log(f"Core tetrahedralization complete: {len(nodes)} nodes, {len(elems)} tets")

    return nodes.astype(np.float64), elems.astype(np.int64), trifaces, triface_markers
