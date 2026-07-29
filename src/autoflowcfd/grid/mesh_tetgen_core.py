"""Constrained tetrahedralization of the domain core using tetgen.

Fills the volume enclosed by a closed piecewise-linear complex (PLC) - the
boundary-layer (BL) outer surface plus the unmodified outer-shell faces
(inlet/outlet/tunnel/symmetry-like boundaries) - with tetgen, instead of the
old arbitrary padded-bounding-box + Cartesian background grid. The PLC is by
construction exactly the closed surface the input mesh already describes, so
the result can never extend outside the real domain.
"""

from typing import Tuple

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


def fill_core_volume(
    points: np.ndarray,
    faces: np.ndarray,
    minratio: float = 1.4,
    mindihedral: float = 15.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Constrained-tetrahedralize the volume enclosed by a closed PLC.

    Args:
        points: (n_points, 3) float64 PLC vertices
        faces: (n_faces, 3) int32 PLC triangles (closed, watertight)
        minratio: max radius-edge ratio quality bound (tetgen convention;
            lower = higher quality, 1.0 is a perfect tet)
        mindihedral: min dihedral angle quality bound (degrees)

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
