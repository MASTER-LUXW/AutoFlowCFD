"""Mesh extrusion module for boundary layer generation.

Implements surface extrusion along normals to create layered meshes
suitable for boundary layer resolution in CFD simulations. The per-layer
geometry step (normal averaging, sharp-corner miter compensation) lives in
mesh_layer_step.extrude_single_layer; converting the resulting layered
prism stack into tetrahedra lives in mesh_prism_to_tet - both split out of
this file to stay under this project's 450-line-per-file guideline.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger

from .mesh_utils import compute_face_normals, check_reached_boundary
from .mesh_layer_step import extrude_single_layer
from .mesh_front_collision import clamp_budget_for_convergence, freeze_self_colliding_nodes
from .mesh_bl_growth import _MAX_SAFETY_LAYERS, compute_layer_thickness

# Caps each layer's own thickness so its cell volume never jumps more than
# this multiple of the PREVIOUS layer's - a prism layer's volume scales
# with its thickness (base area is ~unchanged by a translational offset),
# so bounding the thickness ratio between consecutive layers is a direct,
# cheap proxy for keeping the actual adjacent-cell volume-ratio quality
# gate (quality_validator.py's own max_adjacent_volume_ratio=5.0,
# "STAR-CCM+-aligned Volume Change guidance") satisfied BY CONSTRUCTION,
# rather than relying on it merely showing up as a post-generation quality
# report failure to fix later. Near-wall growth rates are already
# conservative (~1.05-1.3 for y+ control) so this rarely actually binds in
# practice - kept as a safety backstop rather than removed.
MAX_ADJACENT_VOLUME_RATIO = 5.0


def _compute_sharp_angle_attenuation(
    nodes: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    normal_faces: Optional[np.ndarray] = None,
    sharp_angle_threshold: float = 45.0,
) -> np.ndarray:
    """Compute attenuation based on local dihedral angle (ANSA-style).
    
    Nodes at sharp corners (e.g., 90-degree edges) will have their 
    extrusion thickness attenuated to prevent self-intersection and distortion.
    
    Args:
        nodes: Surface nodes
        faces: Surface connectivity (topology)
        normals: Face normals
        normal_faces: The subset of faces corresponding to the normals
        sharp_angle_threshold: Angles below this (deviation from flat 180) are considered sharp
        
    Returns:
        attenuation: [0, 1] array. 0 = no extrusion (sharp corner), 1 = full extrusion.
    """
    n_nodes = len(nodes)
    detect_faces = normal_faces if normal_faces is not None else faces

    # 1. Calculate dihedral angle for each edge
    edge_map = {}  # (min_v, max_v) -> list of face indices
    for i, face in enumerate(detect_faces):
        for j in range(3):
            v1, v2 = int(face[j]), int(face[(j + 1) % 3])
            key = (min(v1, v2), max(v1, v2))
            if key not in edge_map:
                edge_map[key] = []
            edge_map[key].append(i)

    # Per node, track the SHARPEST touching edge as the MINIMUM dot(n1,n2)
    # (dot=1.0 <-> normals parallel <-> a flat continuation; dot values
    # further below 1.0 mean the two adjacent faces' normals diverge more,
    # i.e. a sharper fold). Initialize to 1.0 (flat) rather than a sentinel,
    # so a node touching no qualifying 2-face edge (isolated/patch-boundary
    # vertex) safely defaults to "smooth" instead of needing separate
    # handling.
    node_min_cos_angle = np.full(n_nodes, 1.0)

    for (v1, v2), face_indices in edge_map.items():
        if len(face_indices) >= 2:
            # Use the first two adjacent faces to estimate dihedral angle
            n1 = normals[face_indices[0]]
            n2 = normals[face_indices[1]]
            cos_angle = np.dot(n1, n2)  # 1.0 = flat (normals parallel)

            # MIN, not max: we want the node's SHARPEST touching edge (the
            # smallest dot product / most divergent pair of normals) to
            # govern its attenuation - a node touching even one sharp edge
            # should attenuate, regardless of how many other flat edges it
            # also touches. (Fixed from an earlier version that took max()
            # here despite the variable's own name and this comment saying
            # "min" - that bug meant a node attenuated only if EVERY edge
            # touching it was sharp, which on real geometry is close to
            # never, so this attenuation was silently inert.)
            node_min_cos_angle[v1] = min(node_min_cos_angle[v1], cos_angle)
            node_min_cos_angle[v2] = min(node_min_cos_angle[v2], cos_angle)

    # 2. Convert the normal-to-normal angle into the conventional surface
    # dihedral angle (180 deg = flat continuation, 90 deg = a right-angle
    # fold, decreasing further for a sharper fold) before applying the
    # smooth/sharp thresholds below, which are written in THAT convention.
    # arccos(dot(n1,n2)) alone is the angle BETWEEN THE NORMALS, which runs
    # the opposite way (~0 deg for flat, larger for sharper) - conflating
    # the two was a second, independent bug in an earlier version of this
    # function: flat regions (dot~1, normal-angle~0) satisfied
    # "angle < sharp_limit" and got attenuated to 0.1 almost everywhere,
    # not just at genuine sharp features.
    normal_angle_rad = np.arccos(np.clip(node_min_cos_angle, -1.0, 1.0))
    dihedral_rad = np.pi - normal_angle_rad

    # Define a "sharpness" range (in dihedral-angle terms)
    smooth_limit = np.radians(150)  # 150 degrees: treated as flat
    sharp_limit = np.radians(90)    # 90 degrees: treated as fully sharp

    attenuation = np.ones(n_nodes)

    # Mask for sharp regions
    sharp_mask = dihedral_rad < smooth_limit
    if np.any(sharp_mask):
        # Linearly interpolate between sharp_limit (0.2) and smooth_limit (1.0)
        # This creates a smooth taper from the edge
        t = (dihedral_rad[sharp_mask] - sharp_limit) / (smooth_limit - sharp_limit)
        attenuation[sharp_mask] = 0.2 + 0.8 * np.clip(t, 0, 1)

    # For very sharp corners (< 90 deg), keep a minimal thickness to avoid zero-volume cells
    # but prevent large extrusions
    very_sharp_mask = dihedral_rad < sharp_limit
    attenuation[very_sharp_mask] = 0.1

    return attenuation


def _compute_edge_distance_field(
    nodes: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    angle_threshold: float = 45.0,
    normal_faces: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute a distance field from each node to the nearest sharp edge.
    
    Args:
        nodes: Surface nodes, shape=(n_nodes, 3)
        faces: Surface connectivity, shape=(n_faces, 3) - used for topology
        normals: Face normals, shape=(n_normal_faces, 3) - MUST match `normal_faces`
        angle_threshold: Dihedral angle threshold (degrees) to consider an edge sharp.
        normal_faces: Optional subset of `faces` that the `normals` correspond to.
                      If None, assumes `normals` match `faces`.
        
    Returns:
        attenuation: Attenuation factor in [0, 1] for each node. 
                     1.0 means full thickness, < 1.0 means reduced thickness near edges.
    """
    n_nodes = len(nodes)
    
    # Use normal_faces for edge detection if provided, otherwise use faces
    detect_faces = normal_faces if normal_faces is not None else faces
    
    # 1. Identify sharp edges based on face normals
    # For each edge, check the angle between adjacent faces
    edge_map = {}  # (min_v, max_v) -> list of face indices (into detect_faces)
    
    for i, face in enumerate(detect_faces):
        for j in range(3):
            v1, v2 = int(face[j]), int(face[(j + 1) % 3])
            key = (min(v1, v2), max(v1, v2))
            if key not in edge_map:
                edge_map[key] = []
            edge_map[key].append(i)
            
    sharp_edges = set()
    for (v1, v2), face_indices in edge_map.items():
        if len(face_indices) >= 2:
            # Calculate dihedral angle
            n1 = normals[face_indices[0]]
            n2 = normals[face_indices[1]]
            cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            
            # Sharp if angle is significantly different from 180 (flat)
            # For a cube, we expect 90 degree angles
            if angle > angle_threshold and angle < (180 - angle_threshold):
                sharp_edges.add((v1, v2))
                
    if not sharp_edges:
        return np.ones(n_nodes)
        
    # 2. Compute distance from each node to the nearest sharp edge
    # For simplicity, we use distance to the nearest vertex participating in a sharp edge
    sharp_vertex_set = set()
    for v1, v2 in sharp_edges:
        sharp_vertex_set.add(v1)
        sharp_vertex_set.add(v2)
        
    sharp_vertices = nodes[list(sharp_vertex_set)]
    
    # Use KDTree for efficient nearest neighbor search
    from scipy.spatial import cKDTree
    tree = cKDTree(sharp_vertices)
    dists, _ = tree.query(nodes, k=1)
    
    # 3. Convert distance to attenuation using a smooth step function
    # Characteristic length scale: average edge length of the surface
    edge_lengths = []
    for v1, v2 in list(sharp_edges)[:100]: # Sample for performance
        edge_lengths.append(np.linalg.norm(nodes[v1] - nodes[v2]))
    char_length = np.mean(edge_lengths) if edge_lengths else 0.01
    
    # Smooth attenuation: 0 at distance 0, 1 at distance > 2*char_length
    attenuation = np.clip(dists / (2.0 * char_length), 0.0, 1.0)
    
    return attenuation


def extrude_layers(
    surface_nodes: np.ndarray,
    surface_faces: np.ndarray,
    normals: np.ndarray,
    bounding_box: Dict[str, np.ndarray],
    growth_rate: float = 1.2,
    min_cell_size: float = 0.001,
    taper_scale: 'Optional[np.ndarray]' = None,
    thickness_limit: 'Optional[np.ndarray]' = None,
    max_cell_size: 'Optional[float]' = None,
    bl_layers: 'Optional[int]' = None,
    normal_faces: 'Optional[np.ndarray]' = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Extrude surface along normals to create the boundary layer (BL) mesh.

    Fine, geometrically-graded extrusion for `bl_layers` layers (growth
    rate `growth_rate`, tuned for near-wall y+ control). Extrusion stops
    there - the remaining volume out to the domain boundary is filled
    directly from the BL's own real outer surface by
    mesh_background_merge._build_merged_mesh's single tetgen core-fill
    call, using TetGen's own unstructured grading instead of continuing
    structured layer extrusion (ProjectFiles Part13 P49; a separately-
    extruded "transition" stage bridging the BL to the core fill was tried
    and abandoned - a genuinely hard computational-geometry robustness
    problem on sharp-cornered bodies, not resolved after six different
    mitigation attempts).

    Args:
        surface_nodes: Base surface nodes, shape=(n_nodes, 3)
        surface_faces: Surface connectivity used for layer TOPOLOGY (what
            convert_layers_to_prisms/convert_layers_to_tetrahedra build
            cells from) - shape=(n_faces, 3). When mesh_corner_split.
            split_sharp_corners has been run upstream, this is its
            `topology_faces` (real triangles + bevel/cap triangles).
        normals: Face normals, shape=(len(normal_faces), 3) - NOTE: sized
            to `normal_faces`, not `surface_faces`, whenever the two
            differ (see `normal_faces` below).
        bounding_box: Domain limits to prevent overshoot
        growth_rate: Geometric growth rate for stage-1 (BL) layer
            thickness.
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
        max_cell_size: Optional target layer thickness (meters) used only
            as an ANSA-style upper bound (`0.5 * max_cell_size`) on how far
            this manual/structured extrusion is allowed to grow in total -
            a safety backstop that rarely actually binds given `bl_layers`
            already caps the layer count; the core tetgen fill handles the
            actual size range out to `max_cell_size` itself via its own
            unstructured grading.
        bl_layers: Optional override for how many layers to extrude before
            stopping. None (default) uses 8.
        normal_faces: Optional subset of `surface_faces` used for the
            per-NODE normal averaging that determines each node's offset
            direction (mesh_layer_step.extrude_single_layer's own
            averaging) - None (default, and every pre-existing caller)
            uses `surface_faces` itself, unchanged behaviour. Pass
            mesh_corner_split.split_sharp_corners' `real_face_mask`-
            filtered faces here (paired with `normals` computed from that
            SAME subset) to exclude its bevel/cap rows from normal
            averaging: a bevel triangle's own corner nodes already have a
            correct, single-patch normal from their real face - including
            the bevel face itself in the average would re-contaminate it
            with a third, arbitrary "connector" direction, defeating the
            whole point of having split the corner in the first place.
            Every node referenced anywhere in `surface_faces` is
            guaranteed to also appear in `normal_faces` (a bevel/cap
            triangle never introduces a node that isn't already used by
            some real face), so this never leaves a node undisplaced.

    Returns:
        all_nodes: Concatenated nodes from all layers, shape=(total_nodes, 3)
        layer_connectivity: List of face indices per layer
    """
    # Calculate characteristic length and initial thickness
    domain_size = np.linalg.norm(
        bounding_box['max'] - bounding_box['min']
    )
    normal_faces = surface_faces if normal_faces is None else normal_faces

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

    bl_layers = 8 if bl_layers is None else max(0, int(bl_layers))
    bl_growth_rate = growth_rate
    
    # Two independent attenuation heuristics, combined via the MINIMUM (the
    # more conservative of the two wins) rather than one silently
    # overwriting the other (the previous behaviour: this local computed
    # and logged, then immediately discarded when edge_attenuation was
    # reassigned below - real, wasted work that never affected the actual
    # extrusion). _compute_sharp_angle_attenuation reacts to a node's own
    # sharpest adjacent dihedral angle directly (floors at 0.1 for < 90
    # degrees); _compute_edge_distance_field reacts to Euclidean distance
    # from the nearest identified sharp vertex (floors at exactly 0 right
    # at that vertex, ramping up over 2x the mesh's own sharp-edge length
    # scale). Neither alone reliably attenuates enough on a coarsely
    # tessellated fillet (a physically-smooth curve that only spans 1-2
    # elements still reads as a sharp edge to both), so combining them
    # gives whichever one reacts first.
    logger.info("Computing sharp-angle attenuation for BL...")
    angle_attenuation = _compute_sharp_angle_attenuation(
        surface_nodes, surface_faces, normals,
        normal_faces=normal_faces
    )
    n_attenuated = int(np.sum(angle_attenuation < 0.9))
    if n_attenuated > 0:
        logger.info(f"Sharp-angle attenuation active for {n_attenuated} nodes")

    # Optimization: Compute edge distance field to attenuate BL thickness near sharp corners
    # This prevents severe geometric distortion and self-intersection at edges.
    logger.info("Computing sharp-edge distance field for BL attenuation...")
    distance_attenuation = _compute_edge_distance_field(
        surface_nodes, surface_faces, normals,
        normal_faces=normal_faces
    )
    n_attenuated = int(np.sum(distance_attenuation < 0.9))
    if n_attenuated > 0:
        logger.info(f"Sharp-edge attenuation active for {n_attenuated} nodes")

    edge_attenuation = np.minimum(angle_attenuation, distance_attenuation)

    # Manual (structured) extrusion also stops at ~0.5 * max_cell_size
    # (ANSA-style) as a safety backstop - see this cap's own Args doc above
    # for why it rarely actually binds given bl_layers already caps the
    # layer count.
    effective_max_thickness = max_cell_size * 0.5 if max_cell_size else np.inf

    logger.info(
        f"BL extrusion: {bl_layers} layers, growth_rate={bl_growth_rate}, "
        f"max adjacent volume ratio={MAX_ADJACENT_VOLUME_RATIO:.1f}x\n"
        f"  Max manual thickness: {effective_max_thickness:.4f}m\n"
        f"  Initial thickness: {base_thickness:.6f}m"
    )

    # Guard against degenerate inputs
    if growth_rate <= 1.0:
        raise ValueError(f"growth_rate must be > 1.0, got {growth_rate}")

    n_nodes = len(surface_nodes)
    current_nodes = surface_nodes.copy()
    all_layer_nodes: List[np.ndarray] = [current_nodes]

    # Track cumulative thickness
    current_thickness = 0.0
    n_layers_generated = 0

    # Previous layer's actual (post-cap) thickness, for
    # MAX_ADJACENT_VOLUME_RATIO below - None until the first layer commits.
    previous_layer_thickness: Optional[float] = None

    # Allocate remaining budget for thickness limiting
    remaining_budget = (
        thickness_limit.copy() if thickness_limit is not None
        else np.full(len(surface_nodes), np.inf, dtype=np.float64)
    )
    n_limited = int(np.sum(np.isfinite(remaining_budget)))
    if n_limited:
        logger.info(f"Local BL thickness limiting active for {n_limited} nodes")

    for layer_idx in range(_MAX_SAFETY_LAYERS):
        # Check if we've reached domain boundary
        if check_reached_boundary(current_nodes, bounding_box):
            logger.info(
                f"Reached domain boundary at layer {layer_idx + 1}, "
                f"stopping extrusion (generated {n_layers_generated} layers)"
            )
            break

        # Stop once the requested BL layer count is reached - the
        # remaining volume is filled directly from this real outer surface
        # by the core tetgen fill instead (see this function's own
        # docstring).
        if n_layers_generated == bl_layers:
            logger.info(
                f"Reached bl_layers={bl_layers}, stopping extrusion "
                f"(cumulative thickness={current_thickness:.6f}m) - remaining "
                f"volume filled directly by the core tetgen fill"
            )
            break

        # Compute target CUMULATIVE thickness for the end of this layer
        next_cumulative_thickness = compute_layer_thickness(
            current_thickness, growth_rate, base_thickness, n_layers_generated,
        )

        # Additional stop condition: ANSA-style max thickness cap
        if next_cumulative_thickness > effective_max_thickness:
            logger.info(f"Reached ANSA-style max thickness ({effective_max_thickness:.4f}m). Stopping manual extrusion.")
            break

        # The actual displacement for this layer
        layer_thickness = next_cumulative_thickness - current_thickness
        if layer_thickness <= 1e-12:
            logger.info(f"Layer thickness too small ({layer_thickness:.6e}m), stopping.")
            break

        # See MAX_ADJACENT_VOLUME_RATIO's own comment.
        if previous_layer_thickness is not None:
            max_layer_thickness = previous_layer_thickness * MAX_ADJACENT_VOLUME_RATIO
            if layer_thickness > max_layer_thickness:
                logger.info(
                    f"Layer {layer_idx + 1}: capped thickness {layer_thickness:.6f}m -> "
                    f"{max_layer_thickness:.6f}m to keep the adjacent-cell volume ratio "
                    f"at or below {MAX_ADJACENT_VOLUME_RATIO:.1f}x the previous layer"
                )
                layer_thickness = max_layer_thickness
                next_cumulative_thickness = current_thickness + layer_thickness

        # Reactive convergence clamp
        clamp_budget_for_convergence(current_nodes, surface_faces, remaining_budget)

        # Combine edge attenuation with taper_scale for smoother BL near sharp corners
        effective_taper = taper_scale * edge_attenuation if taper_scale is not None else edge_attenuation

        # Extrude nodes along averaged normals
        new_nodes = extrude_single_layer(
            current_nodes, normal_faces, normals, layer_thickness,
            taper_scale=effective_taper, remaining_budget=remaining_budget,
        )

        # Reactive local collision freeze
        frozen_now = freeze_self_colliding_nodes(
            new_nodes, current_nodes, surface_faces, remaining_budget,
        )
        if len(frozen_now):
            logger.warning(
                f"Layer {layer_idx + 1}: locally froze {len(frozen_now)} node(s) "
                f"where the advancing front would self-intersect; "
                f"extrusion continues elsewhere"
            )

        all_layer_nodes.append(new_nodes)
        current_nodes = new_nodes
        current_thickness = next_cumulative_thickness
        previous_layer_thickness = layer_thickness
        n_layers_generated += 1

    logger.info(
        f"Extrusion completed: {n_layers_generated} layers generated, "
        f"total nodes: {len(all_layer_nodes) * n_nodes}, "
        f"final cumulative height: {current_thickness:.4f}m"
    )

    # Return both the concatenated nodes and the layer connectivity.
    # layer_connectivity is a list of face arrays, one per layer.
    # Each layer uses the same topology (surface_faces), so we just replicate it.
    layer_connectivity = [surface_faces.copy() for _ in range(n_layers_generated)]
    return np.vstack(all_layer_nodes), layer_connectivity
