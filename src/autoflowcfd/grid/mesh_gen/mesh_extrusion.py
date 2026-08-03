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
    target_handoff_size: 'Optional[float]' = None,
    bl_layers: 'Optional[int]' = None,
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
        target_handoff_size: Optional target layer thickness (meters) for
            the LAST transition layer - normally the core fill's own
            max_cell_size (mesh_background._build_merged_mesh). The
            transition stage's growth rate is solved for (geometric
            interpolation over however many layers remain within
            max_layers) so cumulative thickness lands near this target by
            the final layer, instead of the previous fixed
            growth_rate*1.25 regardless of how large the gap to the core's
            own target cell size is. Without this, the BL/core interface
            facet size and the core's own target size can differ by a
            large, uncoordinated ratio - tetgen must then bridge that gap
            in the handful of tets immediately touching the interface,
            which is a common source of skewed/sliver cells right at the
            BL/core boundary. None (default, also used whenever
            max_cell_size itself is None) keeps the previous fixed-rate
            behavior unchanged.
        bl_layers: Optional override for how many of the available layers
            (out of max_layers) count as "Stage 1 (BL)" before switching to
            the transition growth rate. None (default) keeps the previous
            hardcoded `min(8, max_layers)` split. Clamped to
            [0, max_layers] regardless of what's passed - a caller-supplied
            value larger than max_layers would otherwise silently starve
            the transition stage of its own budget, the same way the old
            hardcoded 8 already could whenever max_layers <= 8 (see
            target_handoff_size above - a 0-layer transition stage never
            runs the size-matching logic at all).

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
    if bl_layers is None:
        bl_layers = min(8, max_layers)  # Use at most 8 layers for BL
    else:
        bl_layers = int(np.clip(bl_layers, 0, max_layers))
    bl_target_thickness = domain_size * 0.02  # 2% of domain size for BL region

    # Stage 2 (transition) grows faster than stage 1 (BL) to reach the
    # far field in fewer layers - preserve the same relative step-up the
    # original hardcoded values had (1.2 -> 1.5 is a 1.25x jump) instead
    # of hardcoding both stages outright, which silently ignored the
    # caller's `growth_rate` (e.g. CLI --growth-rate / config growth_rate)
    # entirely regardless of what was requested.
    bl_growth_rate = growth_rate
    transition_growth_rate = growth_rate * 1.25

    # transition_layers is only where Stage 2's faster growth rate kicks in
    # (n_layers_generated == bl_layers below) - it does NOT cap the loop,
    # which actually runs up to max_layers (or stops earlier on
    # check_reached_boundary/the 40%-of-domain-size rule). Sizing the
    # "expected total cells" estimate off bl_layers + transition_layers
    # alone used to understate it whenever max_layers exceeded that sum
    # (e.g. the CLI's own max_layers=12 default with the default bl_layers
    # split already reaches that sum, but any larger --max-layers would
    # silently under-report here).
    logger.info(
        f"Two-stage extrusion strategy (optimized):\n"
        f"  Stage 1 (BL): {bl_layers} layers, growth_rate={bl_growth_rate}\n"
        f"  Stage 2 (Transition): starts at layer {bl_layers + 1}, "
        f"growth_rate={transition_growth_rate} "
        f"(runs up to {max_layers - bl_layers} more layers, capped by "
        f"max_layers/domain-boundary/40%-height, whichever comes first)\n"
        f"  Initial thickness: {base_thickness:.6f}m\n"
        f"  Expected total cells (if max_layers is fully used): "
        f"~{len(surface_faces) * 3 * max_layers:,}"
    )

    all_nodes = [surface_nodes.copy()]
    layer_connectivity = [surface_faces.copy()]

    current_nodes = surface_nodes.copy()
    current_thickness = base_thickness
    current_growth_rate = bl_growth_rate  # Start with BL growth rate

    # Always allocated (not just when the caller passes an a-priori
    # thickness_limit) - the reactive self-collision freeze below needs a
    # budget array to zero out regardless of whether that static estimate
    # was supplied. All-inf is a no-op for any node the freeze never
    # touches: min(x, inf) == x, and inf minus any finite displacement is
    # still inf, so this changes nothing for a caller that omits
    # thickness_limit today.
    remaining_budget = (
        thickness_limit.copy() if thickness_limit is not None
        else np.full(len(surface_nodes), np.inf, dtype=np.float64)
    )
    n_limited = int(np.sum(np.isfinite(remaining_budget)))
    if n_limited:
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
            if target_handoff_size is not None and target_handoff_size > current_thickness > 0:
                # Solve for the constant per-layer growth rate that carries
                # current_thickness (the BL stage's own exit thickness) up
                # to target_handoff_size over exactly the layers still
                # available before max_layers - so whatever max_layers the
                # caller set, the transition stage uses however much growth
                # is actually needed to close the gap, rather than a fixed
                # 1.25x multiplier that may fall far short (leaving a large,
                # uncoordinated size jump right at the BL/core interface) or
                # overshoot it regardless of the actual gap size.
                n_remaining = max(1, max_layers - n_layers_generated)
                solved_rate = (target_handoff_size / current_thickness) ** (1.0 / n_remaining)
                # Never slower than the requested transition rate (that would
                # only undershoot the target further) and capped at 4x/layer
                # so a target far out of reach within the remaining layers
                # doesn't blow up into an equally abrupt jump the other way.
                current_growth_rate = float(np.clip(solved_rate, current_growth_rate, growth_rate * 4.0))
                logger.info(
                    f"Switching to Stage 2 (transition) at layer {layer_idx + 1}, "
                    f"growth_rate solved to {current_growth_rate:.4f} to reach "
                    f"target_handoff_size={target_handoff_size:.6f}m over "
                    f"{n_remaining} remaining layer(s) (BL exit thickness "
                    f"{current_thickness:.6f}m)"
                )
            else:
                logger.info(
                    f"Switching to Stage 2 (transition) at layer {layer_idx + 1}, "
                    f"growth_rate increased to {current_growth_rate}"
                )

        # Reactive convergence clamp: measure the CURRENT (this layer's
        # starting) distance between candidate near faces and cap each
        # involved node's remaining lifetime budget to at most half of
        # it, so this layer's step - however large growth_rate/the
        # solved transition rate makes it - can never move either side
        # more than halfway across whatever gap actually remains. Without
        # this, a single large step can "tunnel": two clean, non-
        # intersecting end-of-layer snapshots can still have their swept
        # prisms fully overlap in between (see mesh_front_collision.py's
        # module docstring - caught directly by this project's own test
        # suite). Must run BEFORE extrude_single_layer, on current_nodes.
        clamp_budget_for_convergence(current_nodes, surface_faces, remaining_budget)

        # Extrude nodes along averaged normals
        new_nodes = extrude_single_layer(
            current_nodes, surface_faces, normals, current_thickness,
            taper_scale=taper_scale, remaining_budget=remaining_budget,
        )

        # Reactive local collision freeze: a backstop for whatever the
        # clamp above doesn't happen to cover (its broad-phase search
        # radius is finite - see that function's own docstring). Check
        # the actual resulting geometry and, only where it genuinely
        # folds over, roll back and permanently freeze just those nodes -
        # independent of growth_rate/bl_layers/max_layers, every other
        # node keeps growing normally.
        frozen_now = freeze_self_colliding_nodes(
            new_nodes, current_nodes, surface_faces, remaining_budget,
        )
        if len(frozen_now):
            logger.warning(
                f"Layer {layer_idx + 1}: locally froze {len(frozen_now)} node(s) "
                f"where the advancing front would self-intersect; "
                f"extrusion continues elsewhere"
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
