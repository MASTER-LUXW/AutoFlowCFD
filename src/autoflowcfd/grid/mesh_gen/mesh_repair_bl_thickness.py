"""Stage B (BL side): targeted BL-thickness-cap regeneration parameter.

compute_bl_thickness_limit_override is a pure function that turns a set of
still-bad BL-region cells (after Stage A smoothing, mesh_repair.py) into a
targeted regeneration parameter - a local BL thickness cap at specific
surface vertices - for the caller (mesh_background.generate_hybrid_mesh)
to feed into a second, targeted regeneration pass. Safe *because* it feeds
back into the exact same, already-correct generation path rather than
attempting a hand-rolled partial/local remesh.

A previous core-side counterpart to this (region-based local refinement)
was removed after being found net-harmful in practice: tetgen's per-region
refinement does not confine itself to the small local footprint of an
added region when a domain-wide grading region is also active in the same
connected volume, so a handful of small local repair regions could balloon
the whole core fill several-fold without actually improving quality there
(see mesh_background.py's own history). Stage B' (mesh_repair_cavity.py)
is the genuine local-remesh replacement for that core-side case; this
BL-side thickness cap has no equivalent failure mode (it only shortens BL
layers locally, never expands anything) and remains unchanged.

Split out of mesh_repair.py purely to keep file size down - re-exported
from there (see the bottom of mesh_repair.py) so existing callers keep
working unchanged.
"""

from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from loguru import logger


def compute_bl_thickness_limit_override(
    bad_cell_mask: np.ndarray,
    n_bl_cells: int,
    cells: np.ndarray,
    n_surface_nodes: int,
    cap_thickness: float,
    existing_thickness_limit: Optional[np.ndarray] = None,
    nodes_per_layer: Optional[int] = None,
    node_original_vertex: Optional[np.ndarray] = None,
    local_surface_faces: Optional[np.ndarray] = None,
    taper_rings: int = 3,
) -> Tuple[Optional[np.ndarray], List[int]]:
    """Stage B, BL side: for residual bad cells that live in the BL region
    (cell index < n_bl_cells), trace their nodes back to the original
    surface vertex that seeded their BL column (a BL node's global index is
    `layer_idx * nodes_per_layer + local_index`, so `node_idx %
    nodes_per_layer` recovers `local_index` regardless of layer) and cap
    the cumulative BL thickness there to `cap_thickness` (roughly 2-3
    layers' worth) - forcing the extrusion to stop growing early at
    exactly the vertices implicated in bad cells, everywhere else
    unaffected.

    Args:
        nodes_per_layer: The ACTUAL per-layer node stride - defaults to
            n_surface_nodes. Kept as an explicit parameter (rather than
            always assuming n_surface_nodes) so a future BL-generation path
            whose layer stride legitimately differs from n_surface_nodes
            doesn't silently recover the WRONG local index for almost every
            node the way an earlier version of this function did (that bug
            inflated a genuinely-local bad-cell cluster into a "21888 of
            25577 surface vertices" cap, which then fed into a tetgen
            internal-robustness crash - see mesh_tetgen_core.fill_core_
            volume's removevertexbyflips handling).
        node_original_vertex: Optional (nodes_per_layer,) array mapping a
            LOCAL index (post-modulo) back to its original
            (n_surface_nodes-space) vertex, for whenever nodes_per_layer
            differs from n_surface_nodes. None (default) is equivalent to
            the identity mapping, correct when nodes_per_layer ==
            n_surface_nodes (the normal case).
        local_surface_faces: Optional (m, 3) surface face connectivity
            (mesh_background._build_merged_mesh's own extrude_faces, in the
            same LOCAL/nodes_per_layer index space as node_original_vertex)
            - when given, the cap is TAPERED smoothly outward over
            taper_rings mesh-connectivity hops from the raw implicated
            LOCAL nodes (same hop-count + smoothstep technique as
            mesh_tetgen_core.build_seam_taper_scale), instead of applied as
            a hard cliff at exactly the implicated vertices. A hard cliff
            risks a severe degenerate/high-aspect-ratio cell right at the
            boundary between a capped node and its uncapped mesh neighbour
            (two directly-adjacent nodes at nearly the same (x, y), one
            frozen near its layer-0 height and the other grown at full
            rate) - tapering spreads that height mismatch across several
            hops instead of concentrating it at one face. None (default)
            keeps the previous hard-cliff behavior unchanged - safe
            whenever local_surface_faces isn't available (e.g. no BL
            region at all), just not tapered.
        taper_rings: hop-count width of the taper when local_surface_faces
            is given - deliberately small (unlike build_seam_taper_scale's
            default 100, which exists to survive a seam through a
            genuinely tight real-geometry feature): this taper only needs
            to smooth out one connector-width mismatch, not guarantee
            non-intersection across a large seam.

    Returns:
        (thickness_limit_array or None if nothing to do, affected surface
        vertex indices) - the array is sized (n_surface_nodes,), merge into
        an existing one via elementwise np.minimum (a caller already
        computing its own thickness_limit for tight-facing-feature capping
        should combine both, not replace one with the other).
    """
    bl_bad = np.flatnonzero(bad_cell_mask[:n_bl_cells])
    if len(bl_bad) == 0:
        return None, []

    stride = nodes_per_layer if nodes_per_layer is not None else n_surface_nodes
    bad_nodes = np.unique(cells[bl_bad].ravel())
    local_idx = np.unique(bad_nodes % stride)
    surface_verts = np.unique(node_original_vertex[local_idx]) if node_original_vertex is not None else local_idx

    limit = (
        existing_thickness_limit.copy()
        if existing_thickness_limit is not None
        else np.full(n_surface_nodes, np.inf)
    )

    if local_surface_faces is not None and len(local_surface_faces):
        # Taper outward from the raw implicated LOCAL nodes over the
        # post-split mesh's own connectivity (captures any new connector
        # adjacency directly - see this function's own doc) instead of a
        # hard cliff. Same hop-count + smoothstep technique as
        # build_seam_taper_scale; at hop 0 this reduces to exactly
        # cap_thickness, so it subsumes the old hard-cap behavior rather
        # than needing a separate step for the seed nodes themselves.
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra

        edges = np.vstack([
            local_surface_faces[:, [0, 1]],
            local_surface_faces[:, [1, 2]],
            local_surface_faces[:, [2, 0]],
        ])
        rows = np.concatenate([edges[:, 0], edges[:, 1]])
        cols = np.concatenate([edges[:, 1], edges[:, 0]])
        graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(stride, stride))
        hop_dist = dijkstra(graph, indices=local_idx, unweighted=True, min_only=True)

        t = np.clip(hop_dist / taper_rings, 0.0, 1.0)
        smoothstep = t * t * (3.0 - 2.0 * t)
        reachable = np.isfinite(hop_dist)

        tapered_local = np.full(stride, np.inf)
        tapered_local[reachable] = cap_thickness / np.maximum(1.0 - smoothstep[reachable], 1e-9)
        tapered_local[reachable & (t >= 1.0)] = np.inf

        orig_idx = node_original_vertex if node_original_vertex is not None else np.arange(stride)
        tapered_by_vertex = np.full(n_surface_nodes, np.inf)
        np.minimum.at(tapered_by_vertex, orig_idx, tapered_local)

        limit = np.minimum(limit, tapered_by_vertex)
        n_tapered = int(np.sum(np.isfinite(tapered_by_vertex)))
        logger.info(
            f"Stage B (BL side): capping cumulative BL thickness to {cap_thickness:.6f}m "
            f"at {len(surface_verts)} surface vertices implicated in {len(bl_bad)} residual bad "
            f"cells, tapered over {taper_rings} rings ({n_tapered} vertices affected in total)"
        )
    else:
        limit[surface_verts] = np.minimum(limit[surface_verts], cap_thickness)
        logger.info(
            f"Stage B (BL side): capping cumulative BL thickness to {cap_thickness:.6f}m "
            f"at {len(surface_verts)} surface vertices implicated in {len(bl_bad)} residual bad cells"
        )

    return limit, surface_verts.tolist()
