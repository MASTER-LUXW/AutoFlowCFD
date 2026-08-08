"""tetgen 核心域填充：BL 缝合处（seam）过渡与局部厚度限制。

从 mesh_tetgen_core.py 拆分出来，专门负责两类问题：BL 挤出区域和
core-only 区域交界处（seam）的平滑过渡缩放，以及两侧 BL 前沿相向生长时
基于几何间隙的局部厚度上限（避免穿透）。供 mesh_background_merge.py 在
生成 core 填充所需的 PLC 边界之前调用。
"""

import numpy as np
from scipy.sparse import coo_matrix
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
    # Plain linear ramp, not the smoothstep (3t^2 - 2t^3) this used to be.
    # Smoothstep has ZERO slope at t=0 (the seam itself) by construction -
    # deliberately so at t=1 (blends smoothly into the untapered interior,
    # scale held constant at exactly 1.0 there), but the same flatness at
    # t=0 means the first several dozen rings out of taper_rings=100 stay
    # under ~10% scale (solving 3t^2-2t^3=0.1 gives t~=0.196, i.e. ~20
    # rings) - confirmed directly on cube_demo: ~12-14k BL prisms with a
    # near-zero-height vertical edge, concentrated in this flat near-seam
    # band, not a genuine defect but this taper's own by-design shape.
    # Linear has a CONSTANT slope of 1/taper_rings everywhere, which is
    # actually LOWER than smoothstep's own peak slope of 1.5/taper_rings
    # (reached at t=0.5) - so this is not "more aggressive than smoothstep
    # ever was" anywhere, it just removes the artificially flat start that
    # concentrated so many rings in the near-zero band. The one thing it
    # gives up is the zero-slope blend AT t=1 (linear meets the untapered
    # scale=1.0 plateau with a slope discontinuity of 1/taper_rings,
    # smoothstep met it with none) - a bounded, small (taper_rings=100)
    # discontinuity right at the taper zone's own edge, not concentrated
    # at the tight-feature seam this function's own docstring warns about.
    linear = t
    # Nodes unreachable from any seam node (not connected through the
    # extrude-face graph, e.g. an unrelated embedded shell) keep scale=1.
    unreachable = ~np.isfinite(hop_dist)
    linear[unreachable] = 1.0

    scale = linear
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
        limit = _smooth_thickness_limit(limit, extrude_faces)
    return limit


def _smooth_thickness_limit(
    limit: np.ndarray, extrude_faces: np.ndarray,
    max_ratio: float = 1.3, max_iterations: int = 50,
) -> np.ndarray:
    """Propagate each capped node's thickness_limit outward across the
    mesh so no two edge-adjacent extrude-eligible nodes differ by more
    than `max_ratio` - the SAME smooth-grading principle (~1.2-1.3 per
    step, general CFD meshing practice) this project already applies to
    the BL/transition growth rate itself, applied here to the CAP field
    instead.

    Without this, each node's cap is set independently from its own
    nearest-facing-feature query (see compute_local_thickness_limit
    above) with zero coordination against its mesh neighbours - a node
    capped tightly by a nearby facing feature (e.g. underbody-to-ground)
    can sit right next to an entirely uncapped neighbour a fraction of a
    millimetre away on the same surface. extrude_layers enforces that cap
    as a hard per-node stop, so the BL outer surface ends up with an
    abrupt local step between the frozen node and its still-growing
    neighbour - exactly the kind of sharp local jump this project has
    repeatedly found produces degenerate (near-zero-volume) cells at that
    seam. Those degenerate cells are then unconditionally dropped during
    post-processing with no repair attempt (nothing flags an now-EMPTY
    region as "bad" the way an existing-but-low-quality cell would),
    leaving a genuinely un-meshed gap in the final volume mesh. Confirmed
    directly as a real, not theoretical, effect on a real case: ~45-50%
    of all surface nodes ended up thickness-capped (a case with many
    tight facing features close together), and hundreds of thousands of
    degenerate cells were dropped per generation attempt.

    Implemented as vectorized Bellman-Ford-style relaxation: repeatedly
    propagate `own_limit * max_ratio` to every mesh-edge neighbour,
    keeping the minimum, until no value changes (or max_iterations is
    hit - going from an extreme (sub-millimetre) cap up to a typical
    far-field target size only takes on the order of
    log(target/cap)/log(max_ratio) hops, comfortably under 50 for any
    realistic combination this project's own min/max_cell_size ranges
    produce). A node with no facing-feature cap at all (np.inf) is only
    ever pulled down by this - it can never push a genuinely-capped
    neighbour's own tighter value back up.
    """
    if not np.any(np.isfinite(limit)):
        return limit

    edges = np.vstack([
        extrude_faces[:, [0, 1]], extrude_faces[:, [1, 2]], extrude_faces[:, [2, 0]],
    ])
    a, b = edges[:, 0], edges[:, 1]

    for _ in range(max_iterations):
        updated = limit.copy()
        np.minimum.at(updated, b, limit[a] * max_ratio)
        np.minimum.at(updated, a, limit[b] * max_ratio)
        if np.array_equal(updated, limit):
            break
        limit = updated

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
