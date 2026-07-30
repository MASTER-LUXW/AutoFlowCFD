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
) -> np.ndarray:
    """Cap each extrude-eligible node's *cumulative* BL thickness to a
    fraction of its local geometric gap to the nearest facing surface, so
    two BL fronts growing toward each other across a tight feature (e.g. a
    body's underbody a few cm above the ground) stop before they can cross
    - instead of growing at a uniform rate and relying on
    repair_nonmanifold_cells to clean up the resulting overlap afterward.

    The gap is measured on the undeformed (layer-0) surface: for each node,
    search nearby surface nodes within `domain_size * 0.4` (the same
    cumulative-height cap extrude_layers already stops at, so nothing
    farther than that could ever be reached anyway) and keep only those
    roughly "ahead" of the node along its own outward normal (within
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

    search_radius = domain_size * 0.4
    cos_threshold = np.cos(np.radians(angle_threshold_deg))

    tree = cKDTree(nodes)
    query_points = nodes[extrude_node_idx]
    neighbor_lists = tree.query_ball_point(query_points, r=search_radius, workers=-1)

    n_capped = 0
    for local_i in range(len(extrude_node_idx)):
        node_idx = extrude_node_idx[local_i]
        candidates = np.asarray(neighbor_lists[local_i], dtype=np.int64)
        if len(candidates) <= 1:
            continue

        p = nodes[node_idx]
        n_p = avg_normal[node_idx]
        if not np.any(n_p):
            continue

        d = nodes[candidates] - p
        dist = np.linalg.norm(d, axis=1)
        real = dist > 1e-9
        if not np.any(real):
            continue

        cosang = (d[real] @ n_p) / dist[real]
        ahead = cosang > cos_threshold
        if not np.any(ahead):
            continue

        gap = float(dist[real][ahead].min())
        limit[node_idx] = gap * safety_factor
        n_capped += 1

    if n_capped:
        logger.info(
            f"Local BL thickness limiting: {n_capped} nodes capped by a "
            f"nearby facing feature (min cap {np.min(limit[np.isfinite(limit)]):.4e} m)"
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Collapse coincident points (within tolerance) and remap faces.

    Fully transitive (uses scipy connected_components over the coincidence
    graph, not a one-hop union), unlike the older `merge_conforming_meshes`
    node-dedup logic elsewhere in this package. Only invoked as a fallback
    when tetgen doesn't return a fully conformal boundary.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    pairs = tree.query_pairs(tolerance)

    n_points = len(points)
    if not pairs:
        return points, faces

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
    return unique_points, new_faces


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


def fill_core_volume(
    points: np.ndarray,
    faces: np.ndarray,
    minratio: float = 1.4,
    mindihedral: float = 15.0,
    holes: Optional[List[np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
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

    Returns:
        (nodes, tets): nodes shape=(n, 3) float64 (boundary points preserved
        verbatim as the first len(points) rows), tets shape=(m, 4) int64
    """
    import tetgen

    points = np.ascontiguousarray(points, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int32)

    logger.info(
        f"Tetrahedralizing core volume: {len(points)} boundary points, "
        f"{len(faces)} boundary faces (tetgen, nobisect)..."
    )

    tgen = tetgen.TetGen(points, faces)
    if holes:
        for hole_pt in holes:
            tgen.add_hole(hole_pt)
        logger.info(f"Marked {len(holes)} tetgen hole seed(s) for isolated embedded solids")
    try:
        nodes, elems, _attr, _markers = tgen.tetrahedralize(
            plc=True, nobisect=True, quality=True,
            minratio=minratio, mindihedral=mindihedral,
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
        raise

    n_input = len(points)
    conformal = nodes.shape[0] >= n_input and np.array_equal(nodes[:n_input], points)

    if not conformal:
        logger.warning(
            "tetgen did not preserve all boundary points verbatim despite "
            "nobisect=True (likely near-duplicate/degenerate input facets); "
            "falling back to coincident-point stitching"
        )
        nodes, elems = _dedupe_coincident_points(nodes, elems)

    logger.info(f"Core tetrahedralization complete: {len(nodes)} nodes, {len(elems)} tets")

    return nodes.astype(np.float64), elems.astype(np.int64)
