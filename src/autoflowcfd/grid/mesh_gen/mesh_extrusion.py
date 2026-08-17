"""边界层生成的网格挤出模块。

实现沿法向的表面挤出以创建层状网格，适用于 CFD 仿真中的边界层
分辨率。单层几何步骤（法向平均、尖角补偿）在 mesh_layer_step.extrude_single_layer
中实现；将层状棱柱堆栈转换为四面体在 mesh_prism_to_tet 中实现；
两个尖角特征衰减启发式方法在 mesh_extrusion_attenuation.py 中实现——
均从本文件拆分以满足项目 450 行/文件的规范。
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger

from .mesh_utils import compute_face_normals, check_reached_boundary
from .mesh_layer_step import extrude_single_layer
from .mesh_front_collision import clamp_budget_for_convergence, freeze_self_colliding_nodes
from .mesh_bl_growth import _MAX_SAFETY_LAYERS, compute_layer_thickness
from .mesh_extrusion_attenuation import (
    _compute_sharp_angle_attenuation,
    _compute_edge_distance_field,
)

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
