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
# see build_graded_regions - so this is deliberately generous, not exact).
_VOLUME_SHAPE_FACTOR = 0.15


def _generate_icosphere(
    center: np.ndarray, radius: float, subdivisions: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """A simple closed, outward-wound icosphere triangulation (a
    subdivided icosahedron, not a UV-sphere).

    Used as an internal grading-region boundary (build_graded_regions) - a
    sphere is always convex and simple regardless of the actual body's
    shape, so unlike offsetting the body's own (possibly non-convex)
    surface, it can never self-intersect. A UV-sphere was tried first and
    rejected: its pole vertices are shared by many (n_lon) triangles in a
    tight fan, and that pole singularity reproducibly segfaults tetgen's
    `add_hole` + `nobisect=True` + `quality=True` combination even on a
    trivial synthetic case (isolated, not related to the bgmesh crash
    elsewhere in this module) - a small hole box with ordinary box topology
    doesn't crash, isolating the pole fan as the trigger. An icosphere has
    uniform vertex valence (~5-6) everywhere with no singular vertex.

    Returns:
        (points, faces): points shape=(n,3), faces shape=(m,3) int64,
        node indices local to this sphere (caller must offset them when
        merging into a larger shared point array)
    """
    t = (1.0 + np.sqrt(5.0)) / 2.0
    base_verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=np.float64)
    base_verts /= np.linalg.norm(base_verts[0])

    base_faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)

    verts = base_verts
    faces = base_faces
    for _ in range(subdivisions):
        edge_cache: dict = {}
        new_faces = []

        def midpoint(i: int, j: int) -> int:
            key = (i, j) if i < j else (j, i)
            if key in edge_cache:
                return edge_cache[key]
            m = verts[i] + verts[j]
            m /= np.linalg.norm(m)
            idx = len(verts_list)
            verts_list.append(m)
            edge_cache[key] = idx
            return idx

        verts_list = list(verts)
        for a, b, c in faces:
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_faces.append([a, ab, ca])
            new_faces.append([b, bc, ab])
            new_faces.append([c, ca, bc])
            new_faces.append([ab, bc, ca])
        verts = np.array(verts_list, dtype=np.float64)
        faces = np.array(new_faces, dtype=np.int64)

    return verts * radius + center, faces


def build_graded_regions(
    center: np.ndarray,
    base_radius: float,
    domain_radius: float,
    near_wall_cell_size: float,
    max_cell_size: float,
    blocked_points: np.ndarray,
    n_tiers: int = 5,
    seed: int = 0,
    max_sphere_radius: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[np.ndarray, int, float]]]:
    """Build nested concentric sphere surfaces + tetgen region specs that
    grade the core fill's max cell size outward from the wall (continuing
    the BL's own near-wall size) up to `max_cell_size` far from the body,
    instead of leaving the whole core region uniformly unconstrained (see
    fill_core_volume) or uniformly capped everywhere (which would force
    the far-field just as fine as near the body, ballooning cell count).

    Each sphere is a genuinely simple, always-convex closed surface
    (_generate_icosphere) - unlike offsetting the body's own (possibly
    non-convex) surface outward, it can never self-intersect regardless of
    the body's actual shape.

    Args:
        center: (3,) grading center - typically the BL outer surface's own
            centroid
        base_radius: bounding radius of the BL outer surface from `center`
            - the innermost sphere is placed strictly outside this (with a
            safety margin) so it can't coincide/interfere with the BL
            surface itself
        domain_radius: distance from `center` to the farthest domain
            corner - radii and cell sizes are log-spaced from the BL
            surface out to this distance over exactly `n_tiers` bands (see
            below for why this is always true regardless of base_radius);
            the outermost band needs no bounding sphere of its own, the
            domain's real outer PLC shell already terminates it
        near_wall_cell_size: the BL's own final (outermost) layer
            thickness - the core fill's first tier continues growing from
            here, not from scratch
        max_cell_size: the hard cap the outermost tier converges to exactly
        blocked_points: (n, 3) points of real near-body geometry (the BL
            outer surface) - tier 0's seed point is verified to keep a
            minimum clearance from these via nearest-neighbor distance, so
            it can't accidentally land inside/on the real (possibly
            elongated/non-convex) BL block despite being outside its
            bounding sphere
        n_tiers: number of graded tiers to generate, ALWAYS exactly this
            many regardless of how base_radius compares to domain_radius
            (log-spaced, not geometric growth from base_radius - see the
            implementation note below for why that matters)
        seed: RNG seed for sampling candidate seed-point directions
        max_sphere_radius: Optional hard cap on how far any actual sphere
            surface may reach (e.g. the measured distance to the nearest
            OTHER extruded surface, such as a ground plane, that a sphere
            must not cross). When smaller than `domain_radius`, the
            n_tiers spheres are confined to [start_r, max_sphere_radius],
            and one additional sphere-less region is added covering
            [max_sphere_radius, domain_radius] at `max_cell_size` - so the
            true far-field (beyond the safe sphere zone) still gets capped,
            it just isn't graded within that outer region.

    Returns:
        extra_points: (p, 3) new points (the sphere surfaces' own
            vertices) - caller must append these to the shared PLC point
            array and offset extra_faces' indices accordingly
        extra_faces: (q, 3) int64, LOCAL indices into extra_points only
        region_specs: list of (seed_point_xyz, region_id, maxvolume) for
            tgen.add_region - includes the final (sphere-less) outer band
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(blocked_points)
    rng = np.random.default_rng(seed)

    def find_band_seed(r_inner: float, r_outer: float, min_clearance: float) -> Optional[np.ndarray]:
        r_mid = 0.5 * (r_inner + r_outer)
        for _ in range(30):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            candidate = center + direction * r_mid
            dist, _ = tree.query(candidate)
            if dist >= min_clearance:
                return candidate
        return None

    extra_point_rows: List[np.ndarray] = []
    extra_face_rows: List[np.ndarray] = []
    region_specs: List[Tuple[np.ndarray, int, float]] = []
    point_offset = 0
    region_id = 1

    # Radii and cell-size caps are log-spaced over exactly n_tiers bands,
    # from the BL surface out to the domain edge, REGARDLESS of how large
    # base_radius happens to be relative to domain_radius. A naive
    # "multiply by tier_growth until >= domain_radius" schedule (tried
    # first) can jump straight past domain_radius on its very first step
    # whenever the near-wall geometry itself already spans most of the
    # domain (e.g. a ground plane covering nearly the whole footprint,
    # common for a car underbody/ground pair) - collapsing to a single
    # region governing the entire remaining volume. That's a real
    # correctness problem, not just lost grading: tetgen's per-region
    # refinement queue does not reliably reach a tight cap when one region
    # spans that much of the domain from a single seed point (observed
    # ~1000x overshoot vs. a spatially compact region's ~1.2x), so a
    # requested cap can end up essentially unenforced. Always dividing
    # into n_tiers keeps every individual region small enough to refine
    # reliably.
    sphere_limit = domain_radius if max_sphere_radius is None else min(max_sphere_radius, domain_radius)

    start_r = base_radius * 1.15
    if start_r <= 0.0 or start_r >= sphere_limit:
        start_r = sphere_limit * 0.1
    # The sphere_limit*0.1 fallback is a blind fraction, not a measured
    # clearance - `center` itself can end up nowhere near the wall/body's
    # own centroid when the near-wall surfaces span wildly different
    # scales (e.g. a small isolated body's few hundred points averaged
    # together with a domain-spanning ground plane's own corner points),
    # so a sphere at that radius can slice straight through the real
    # geometry it was supposed to stay clear of. Grounding it in the
    # actual nearest measured distance from `center` to real near-wall
    # geometry closes that gap.
    min_dist_to_wall = float(tree.query(center)[0])
    start_r = max(start_r, min_dist_to_wall * 1.2)
    start_r = min(start_r, sphere_limit * 0.95)
    radii = np.geomspace(max(start_r, sphere_limit * 1e-6), sphere_limit, n_tiers + 1)

    # Cell-size caps are log-interpolated by RADIUS (not tier index) over
    # the FULL [start_r, domain_radius] span, even though the spheres
    # themselves stop at sphere_limit - so a tight sphere_limit (a nearby
    # wall) doesn't also truncate the size grading itself; the final
    # sphere-less region beyond sphere_limit still converges to exactly
    # max_cell_size at the true domain edge.
    start_size = min(near_wall_cell_size * 2.5, max_cell_size)

    def size_at_radius(r: float) -> float:
        if domain_radius <= start_r:
            return max_cell_size
        t = float(np.clip(
            (np.log(max(r, 1e-12)) - np.log(start_r)) / (np.log(domain_radius) - np.log(start_r)),
            0.0, 1.0,
        ))
        log_s0, log_s1 = np.log(max(start_size, 1e-12)), np.log(max_cell_size)
        return float(np.exp(log_s0 + t * (log_s1 - log_s0)))

    min_clearance = max(base_radius * 0.05, domain_radius * 0.01)
    prev_r = 0.0
    reaches_domain_edge = sphere_limit >= domain_radius

    for tier in range(n_tiers):
        inner_r = float(radii[tier + 1])
        cell_size = size_at_radius(inner_r)

        seed_r_inner = prev_r if tier > 0 else base_radius
        seed_pt = find_band_seed(seed_r_inner, inner_r, min_clearance)
        if seed_pt is not None:
            maxvol = cell_size ** 3 * _VOLUME_SHAPE_FACTOR
            region_specs.append((seed_pt, region_id, maxvol))
            region_id += 1

        if tier < n_tiers - 1 or not reaches_domain_edge:
            # A sphere is needed here unless this tier's outer edge already
            # IS the true domain edge (reaches_domain_edge and it's the
            # last tier) - the real domain PLC bounds that case already.
            sphere_pts, sphere_faces = _generate_icosphere(center, inner_r)
            extra_point_rows.append(sphere_pts)
            extra_face_rows.append(sphere_faces + point_offset)
            point_offset += len(sphere_pts)

        prev_r = inner_r

    if not reaches_domain_edge:
        # sphere_limit < domain_radius: there's real, uncovered domain
        # volume beyond the safe sphere zone (e.g. the rest of the way to
        # a far-off tunnel/inlet/outlet wall) - cap it too, just without
        # any further grading inside it (no sphere can safely subdivide
        # this band without risking the same wall-crossing this whole
        # `max_sphere_radius` mechanism exists to avoid).
        final_seed = find_band_seed(sphere_limit, domain_radius, min_clearance)
        if final_seed is not None:
            region_specs.append(
                (final_seed, region_id, max_cell_size ** 3 * _VOLUME_SHAPE_FACTOR)
            )
            region_id += 1

    extra_points = (
        np.vstack(extra_point_rows) if extra_point_rows else np.empty((0, 3), dtype=np.float64)
    )
    extra_faces = (
        np.vstack(extra_face_rows) if extra_face_rows else np.empty((0, 3), dtype=np.int64)
    )

    logger.info(
        f"Graded core sizing: {len(region_specs)} region tier(s), "
        f"sphere radii {radii[0]:.3f} -> {radii[-1]:.3f} m"
        + (f" (+1 flat-capped band to {domain_radius:.3f} m)" if not reaches_domain_edge else "")
        + f", cell size {near_wall_cell_size:.4f} -> {max_cell_size:.4f} m"
    )

    return extra_points, extra_faces, region_specs


def fill_core_volume(
    points: np.ndarray,
    faces: np.ndarray,
    minratio: float = 1.4,
    mindihedral: float = 15.0,
    holes: Optional[List[np.ndarray]] = None,
    regions: Optional[List[Tuple[np.ndarray, int, float]]] = None,
    face_markers: Optional[np.ndarray] = None,
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
        regions: (seed_point, region_id, maxvolume) tuples from
            build_graded_regions, for capping max cell size per graded
            tier. Note: tetgen's own background-mesh sizing (`bgmesh`/
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

    logger.info(
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
        logger.info(f"Marked {len(holes)} tetgen hole seed(s) for isolated embedded solids")
    if regions:
        for seed_pt, region_id, maxvol in regions:
            tgen.add_region(region_id, seed_pt, maxvol)
        logger.info(f"Marked {len(regions)} graded max-cell-size region(s)")

    # tetgen's default steinerleft=100000 is a global cap on how many
    # Steiner points it will ever insert, shared across the WHOLE mesh -
    # with a region's own maxvolume target well below the PLC's natural
    # (unconstrained) tet size, it can run out long before that target is
    # reached everywhere, silently leaving a long tail of oversized cells
    # in whatever pockets happened to refine last (measured directly: a
    # 5.5x3x3 m domain capped at 0.05 m with the previous fixed 300,000
    # budget left 6-10% of cells over 1.5x the target and a worst-case
    # cell ~5-6x over; the true fix is scaling the budget to the actual
    # problem size, not a fixed constant that only happens to be enough
    # for some domain/max_cell_size ratios).
    #
    # Estimate the number of target-sized tets the requested region(s)
    # imply from the PLC's own bounding-box volume (an upper bound on the
    # true fillable volume, so this errs toward *more* budget, never
    # less) divided by the smallest requested maxvol, then apply a
    # generous safety multiplier - empirically, tetgen's actual converged
    # tet count for a single flat max-cell-size region over a whole core
    # runs ~2-2.5x this naive packing estimate. Confirmed empirically that
    # tetgen stops on its own once its internal quality/size criteria are
    # satisfied well before exhausting a generous budget (identical tet
    # count and runtime whether the budget was 1e6, 2e6, or 8e6 for the
    # measured case above) - so a high ceiling costs nothing when it isn't
    # needed, it only matters for the cases that actually need the room.
    if regions:
        bbox_volume = float(np.prod(np.max(points, axis=0) - np.min(points, axis=0)))
        min_maxvol = min(maxvol for _, _, maxvol in regions)
        estimated_tets = bbox_volume / max(min_maxvol, 1e-30)
        steinerleft = int(np.clip(estimated_tets * 3.0, 300_000, 20_000_000))
        logger.info(
            f"Steiner-point budget: {steinerleft:,} (estimated ~{estimated_tets:,.0f} "
            f"target-sized tets needed, x3 safety margin)"
        )
    else:
        steinerleft = 100_000

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
        raise

    n_input = len(points)
    conformal = nodes.shape[0] >= n_input and np.array_equal(nodes[:n_input], points)

    if not conformal:
        logger.warning(
            "tetgen did not preserve all boundary points verbatim "
            "(likely near-duplicate/degenerate input facets); "
            "falling back to coincident-point stitching"
        )
        nodes, elems = _dedupe_coincident_points(nodes, elems)

    logger.info(f"Core tetrahedralization complete: {len(nodes)} nodes, {len(elems)} tets")

    trifaces = None
    triface_markers = None
    if face_markers is not None:
        trifaces = tgen.trifaces.astype(np.int64)
        triface_markers = tgen.triface_markers.astype(np.int32)

    return nodes.astype(np.float64), elems.astype(np.int64), trifaces, triface_markers
