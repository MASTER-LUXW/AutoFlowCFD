"""Mesh extrusion module for boundary layer generation.

Implements surface extrusion along normals to create layered meshes
suitable for boundary layer resolution in CFD simulations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger

from .mesh_utils import compute_face_normals, check_reached_boundary


def extrude_layers(
    surface_nodes: np.ndarray,
    surface_faces: np.ndarray,
    normals: np.ndarray,
    bounding_box: Dict[str, np.ndarray],
    growth_rate: float = 1.2,
    max_layers: int = 30,
    min_cell_size: float = 0.001,
    taper_scale: 'Optional[np.ndarray]' = None,
    thickness_limit: 'Optional[np.ndarray]' = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Extrude surface along normals to create layered mesh with boundary layer resolution.

    Strategy (Two-stage extrusion):
    Stage 1 - Boundary Layer (Layers 1-8):
      - Fine resolution for y+ control
      - Growth rate: `growth_rate` (as passed in)
      - Target thickness: ~0.05-0.1m

    Stage 2 - Transition/Far-field (Layers 9-20):
      - Coarse resolution for domain filling
      - Growth rate: `growth_rate * 1.25`
      - Extend to far-field boundary

    Args:
        surface_nodes: Base surface nodes, shape=(n_nodes, 3)
        surface_faces: Surface connectivity, shape=(n_faces, 3)
        normals: Face normals for extrusion direction, shape=(n_faces, 3)
        bounding_box: Domain limits to prevent overshoot
        growth_rate: Geometric growth rate for stage-1 (BL) layer
            thickness; stage-2 (transition) layers grow at
            `growth_rate * 1.25`. Previously hardcoded to 1.2/1.5
            regardless of this argument - every caller's requested
            growth_rate (CLI --growth-rate, config growth_rate) had zero
            effect on the actual mesh.
        max_layers: Maximum number of layers to generate
        min_cell_size: Minimum allowable cell size in meters
        taper_scale: Optional float array in [0, 1], shape=(n_nodes,).
            Scales each node's per-layer displacement (1 = full extrusion,
            0 = stays exactly at its original position every layer). Used
            to taper the BL surface smoothly to zero right at a seam shared
            with a non-extruded boundary group, instead of either moving the
            seam (tearing the mesh open) or hard-pinning it (which collapses
            the seam's own triangles into zero-area slivers).
        thickness_limit: Optional float array in meters, shape=(n_nodes,)
            (np.inf where unconstrained), from
            mesh_tetgen_core.compute_local_thickness_limit. Caps each
            node's *cumulative* displacement across all layers so BL fronts
            converging on a tight local feature (e.g. a body's underbody
            close to the ground) freeze before they can cross, instead of
            growing at the uniform rate and overlapping.

    Returns:
        all_nodes: Concatenated nodes from all layers, shape=(total_nodes, 3)
        layer_connectivity: List of face indices per layer
    """
    # Calculate characteristic length and initial thickness
    domain_size = np.linalg.norm(
        bounding_box['max'] - bounding_box['min']
    )
    
    # For automotive CFD (Re ~ 1e6 - 1e7), first layer height should target y+ ~ 1-30
    # Using empirical formula: delta_y1 ≈ L * Re^(-0.5) / 100
    # Conservative estimate: 0.002 * L_char for first layer
    
    # CRITICAL FIX: min_cell_size should be the PRIMARY control for first layer thickness
    # Previous logic used domain_size * 0.002 which could be too large for tight geometries
    # Now we use min_cell_size as the base, with domain_size only as an upper bound
    base_thickness_from_domain = domain_size * 0.002  # 0.2% of domain size
    base_thickness = min(min_cell_size, base_thickness_from_domain)
    
    # Ensure minimum thickness is reasonable (but respect user's min_cell_size)
    # Only apply hard floor if min_cell_size is unreasonably small (< 0.1mm)
    if base_thickness < 0.0001:
        logger.warning(
            f"min_cell_size={min_cell_size}m is extremely small, using 0.0001m as safety floor"
        )
        base_thickness = 0.0001
    
    # Calculate optimal BL parameters for reduced layers
    bl_layers = min(8, max_layers)  # Use at most 8 layers for BL
    transition_layers = min(4, max_layers - bl_layers)  # Remaining layers for transition
    bl_target_thickness = domain_size * 0.02  # 2% of domain size for BL region

    # Stage 2 (transition) grows faster than stage 1 (BL) to reach the
    # far field in fewer layers - preserve the same relative step-up the
    # original hardcoded values had (1.2 -> 1.5 is a 1.25x jump) instead
    # of hardcoding both stages outright, which silently ignored the
    # caller's `growth_rate` (e.g. CLI --growth-rate / config growth_rate)
    # entirely regardless of what was requested.
    bl_growth_rate = growth_rate
    transition_growth_rate = growth_rate * 1.25

    logger.info(
        f"Two-stage extrusion strategy (optimized):\n"
        f"  Stage 1 (BL): {bl_layers} layers, growth_rate={bl_growth_rate}\n"
        f"  Stage 2 (Transition): {transition_layers} layers, growth_rate={transition_growth_rate}\n"
        f"  Initial thickness: {base_thickness:.6f}m\n"
        f"  Expected total cells: ~{len(surface_faces) * 3 * (bl_layers + transition_layers):,}"
    )
    
    all_nodes = [surface_nodes.copy()]
    layer_connectivity = [surface_faces.copy()]

    current_nodes = surface_nodes.copy()
    current_thickness = base_thickness
    current_growth_rate = bl_growth_rate  # Start with BL growth rate

    remaining_budget = thickness_limit.copy() if thickness_limit is not None else None
    if remaining_budget is not None:
        n_limited = int(np.sum(np.isfinite(remaining_budget)))
        logger.info(
            f"Local BL thickness limiting active for {n_limited} nodes "
            f"near tight facing features"
        )

    n_layers_generated = 0
    cumulative_height = 0.0

    for layer_idx in range(max_layers):
        # Check if we've reached domain boundary
        if check_reached_boundary(current_nodes, bounding_box):
            logger.info(
                f"Reached domain boundary at layer {layer_idx + 1}, "
                f"stopping extrusion (generated {n_layers_generated} layers)"
            )
            break

        # Switch to Stage 2 (Transition) after boundary layer
        if n_layers_generated == bl_layers and current_growth_rate < transition_growth_rate:
            current_growth_rate = transition_growth_rate
            logger.info(
                f"Switching to Stage 2 (transition) at layer {layer_idx + 1}, "
                f"growth_rate increased to {current_growth_rate}"
            )

        # Extrude nodes along averaged normals
        new_nodes = extrude_single_layer(
            current_nodes, surface_faces, normals, current_thickness,
            taper_scale=taper_scale, remaining_budget=remaining_budget
        )

        all_nodes.append(new_nodes)
        layer_connectivity.append(surface_faces.copy())
        n_layers_generated += 1
        cumulative_height += current_thickness
        
        # Update for next layer
        current_nodes = new_nodes
        current_thickness *= current_growth_rate
        
        # Log progress
        if (layer_idx + 1) % 5 == 0 or n_layers_generated <= 3:
            logger.info(
                f"  Layer {layer_idx + 1}: thickness={current_thickness:.6f}m, "
                f"cumulative_height={cumulative_height:.6f}m, "
                f"growth_rate={current_growth_rate}"
            )
        
        # Stop if we've filled enough of the domain (40% of domain size)
        if cumulative_height > domain_size * 0.4:
            logger.info(
                f"Cumulative height {cumulative_height:.4f}m exceeds 40% of domain, "
                f"stopping at layer {layer_idx + 1}"
            )
            break
    
    # Concatenate all layers
    all_nodes_array = np.vstack(all_nodes)
    
    logger.info(
        f"Extrusion completed: {n_layers_generated} layers generated, "
        f"total nodes: {len(all_nodes_array)}, "
        f"final cumulative height: {cumulative_height:.4f}m"
    )
    
    return all_nodes_array, layer_connectivity


def extrude_single_layer(
    nodes: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    thickness: float,
    taper_scale: 'Optional[np.ndarray]' = None,
    remaining_budget: 'Optional[np.ndarray]' = None,
) -> np.ndarray:
    """Extrude one layer of nodes.

    For each node, average the normals of adjacent faces and move
    along that direction. Vectorized implementation for performance.

    Args:
        nodes: Current layer nodes, shape=(n_nodes, 3)
        faces: Face connectivity, shape=(n_faces, 3)
        normals: Face normals, shape=(n_faces, 3)
        thickness: Extrusion distance in meters
        taper_scale: Optional float array in [0, 1], shape=(n_nodes,),
            scaling each node's displacement (see extrude_layers).
        remaining_budget: Optional float array in meters, shape=(n_nodes,),
            mutated in place - each node's displacement this layer is
            capped to what's left of its budget, which is then decremented
            by the same amount (see extrude_layers' thickness_limit).

    Returns:
        New node positions after extrusion, shape=(n_nodes, 3)
    """
    n_nodes = len(nodes)
    new_nodes = nodes.copy()

    # Build node-to-face mapping using vectorized operations
    node_normal_sum = np.zeros((n_nodes, 3))
    node_normal_count = np.zeros(n_nodes, dtype=np.int64)
    flat_nodes = faces.ravel()
    np.add.at(node_normal_sum, flat_nodes, np.repeat(normals, 3, axis=0))
    np.add.at(node_normal_count, flat_nodes, 1)

    # Compute average normals (avoid division by zero)
    mask = node_normal_count > 0
    avg_normals = np.zeros_like(node_normal_sum)
    avg_normals[mask] = node_normal_sum[mask] / node_normal_count[mask, np.newaxis]

    # Normalize
    norms = np.linalg.norm(avg_normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    avg_normals = avg_normals / norms

    if taper_scale is not None:
        node_thickness = thickness * taper_scale
    else:
        node_thickness = np.full(n_nodes, thickness, dtype=np.float64)

    if remaining_budget is not None:
        node_thickness = np.minimum(node_thickness, remaining_budget)
        remaining_budget -= node_thickness

    displacement = node_thickness[:, np.newaxis] * avg_normals

    # Extrude all nodes at once
    logger.info(f"Extruding layer with thickness={thickness:.6f}...")
    new_nodes[mask] += displacement[mask]

    return new_nodes


def convert_layers_to_tetrahedra(
    all_nodes: np.ndarray,
    layer_connectivity: List[np.ndarray],
    base_faces: np.ndarray
) -> np.ndarray:
    """Convert layered prism mesh to a *conformal* tetrahedral mesh.

    Each triangular prism between two consecutive layers is split into 3
    tetrahedra.  The split is chosen so that neighbouring prisms agree on the
    diagonal of every shared quadrilateral face, which is what makes the
    resulting mesh conformal (every interior face is shared by exactly two
    cells).  A fixed template applied blindly does *not* have this property and
    produces hanging faces that a finite-volume solver then mistakes for
    boundary faces.

    Rule: sort the three base vertices by global node index, v0 < v1 < v2, and
    let w_i be the corresponding vertices on the next layer.  Emit

        T1 = (v0, v1, v2, w2)
        T2 = (v0, v1, w1, w2)
        T3 = (v0, w0, w1, w2)

    The diagonals this induces on the three quad faces are v0-w1, v1-w2 and
    v0-w2, i.e. always "lower-indexed bottom vertex to higher-indexed top
    vertex".  That rule depends only on the two vertices of the shared edge, so
    two prisms sharing an edge necessarily pick the same diagonal.

    Tetrahedra are additionally oriented to have positive signed volume.

    Args:
        all_nodes: All nodes from all layers, shape=(total_nodes, 3)
        layer_connectivity: Face indices per layer
        base_faces: Original surface faces, shape=(n_faces, 3)

    Returns:
        Tetrahedral cell connectivity, shape=(n_tets, 4)
    """
    n_layers = len(layer_connectivity)
    n_base_faces = len(base_faces)

    if n_layers < 2:
        raise ValueError("Need at least 2 layers to create volume")

    n_total_nodes = len(all_nodes)
    nodes_per_layer = n_total_nodes // n_layers

    logger.info(f"Converting {n_layers-1} layer pairs to conformal tetrahedra...")

    # Sort each base triangle's vertices by global index once; the relative
    # order is identical on every layer (index = base + layer*nodes_per_layer),
    # so one sort is valid for the whole stack.
    sorted_base = np.sort(base_faces, axis=1)          # (n_faces, 3) -> v0<v1<v2

    n_tets = n_base_faces * (n_layers - 1) * 3
    tetrahedra = np.empty((n_tets, 4), dtype=np.int64)

    tet_idx = 0
    for layer_idx in range(n_layers - 1):
        off_lo = layer_idx * nodes_per_layer
        off_hi = (layer_idx + 1) * nodes_per_layer

        v0 = off_lo + sorted_base[:, 0]
        v1 = off_lo + sorted_base[:, 1]
        v2 = off_lo + sorted_base[:, 2]
        w0 = off_hi + sorted_base[:, 0]
        w1 = off_hi + sorted_base[:, 1]
        w2 = off_hi + sorted_base[:, 2]

        for quad in ((v0, v1, v2, w2),
                     (v0, v1, w1, w2),
                     (v0, w0, w1, w2)):
            sl = slice(tet_idx, tet_idx + n_base_faces)
            tetrahedra[sl, 0] = quad[0]
            tetrahedra[sl, 1] = quad[1]
            tetrahedra[sl, 2] = quad[2]
            tetrahedra[sl, 3] = quad[3]
            tet_idx += n_base_faces

    # Enforce positive signed volume (swap two vertices where inverted) so that
    # downstream code can rely on orientation instead of taking |det|.
    tetrahedra = orient_tetrahedra(all_nodes, tetrahedra)

    logger.info(f"Total tetrahedra generated: {len(tetrahedra)}")
    return tetrahedra


def orient_tetrahedra(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Flip inverted tetrahedra so every cell has positive signed volume.

    Signed volume = det(p1-p0, p2-p0, p3-p0) / 6.  Swapping two vertices flips
    the sign, so inverted cells are repaired in place.  Cells that are exactly
    degenerate (zero volume) cannot be repaired and are reported.

    Args:
        nodes: Node coordinates, shape=(n_nodes, 3)
        tets: Tetrahedral connectivity, shape=(n_tets, 4)

    Returns:
        Connectivity with all signed volumes >= 0.
    """
    p0 = nodes[tets[:, 0]]
    p1 = nodes[tets[:, 1]]
    p2 = nodes[tets[:, 2]]
    p3 = nodes[tets[:, 3]]
    det = np.einsum('ij,ij->i', p1 - p0, np.cross(p2 - p0, p3 - p0))

    inverted = det < 0.0
    n_inv = int(np.count_nonzero(inverted))
    if n_inv:
        # Swap last two vertices to restore positive orientation.
        tets[inverted, 2], tets[inverted, 3] = (
            tets[inverted, 3].copy(), tets[inverted, 2].copy()
        )
        logger.info(f"Re-oriented {n_inv} inverted tetrahedra")

    n_degen = int(np.count_nonzero(np.abs(det) < 1e-20))
    if n_degen:
        logger.warning(
            f"{n_degen} degenerate (zero-volume) tetrahedra detected; these "
            f"cannot be fixed by re-orientation and indicate collapsed layers"
        )

    return tets

