"""Single-layer extrusion step: normal averaging and miter-join compensation.

Split out of mesh_extrusion.py (which retains the multi-layer orchestration
loop, extrude_layers) purely to keep both files under this project's
450-line-per-file guideline; extrude_layers is this module's only caller.
"""

import numpy as np
from typing import Optional
from loguru import logger

# Feature-angle handling for extrusion at sharp convex edges/corners (see
# extrude_single_layer). Below FEATURE_ANGLE_THRESHOLD_RAD, a vertex's
# adjacent faces are treated as locally smooth and get the plain averaged
# normal (unchanged behaviour). MITER_LIMIT caps the compensation factor
# the same way SVG/vector-graphics stroke "miter joins" do (default
# stroke-miterlimit there is 4) - past the limit a vertex falls back
# toward the uncompensated averaged-normal behaviour (a "bevel") instead of
# letting the offset blow up, which is what would happen at a near-reflex/
# needle feature.
#
# Was 3.0 ("a bit more conservative" than SVG's 4) until measured directly
# against the actual failure mode this compensation causes when left
# uncapped near a real feature: an axis-aligned box corner's natural
# (unclipped) compensation factor is ~1.73x, comfortably under the old 3.0
# cap - so 3.0 never actually constrained anything there. That 1.73x
# per-layer factor, compounded across every layer's cumulative height,
# produced tetrahedra up to ~50x larger in volume than typical near-wall
# cells at the SAME corner (still well-SHAPED - good skewness - just much
# bigger, a mesh-size-uniformity defect none of the shape-quality repair
# stages in mesh_repair.py even check for, let alone fix). Lowering the
# cap to 1.2 (tested directly on a real cube body, the worst realistic
# case: a genuine 3-face axis-aligned corner) cut that worst-case volume
# by ~41%, with no measured skewness regression (near-wall max skewness
# actually improved slightly, 0.899 -> 0.880) and no new degenerate/
# negative-volume cells. Not lowered all the way to 1.0 (no compensation
# at all, which measured even better on this specific case) to keep a
# margin against this mechanism's original purpose - preventing a
# disproportionately THIN layer at a sharper, more acute feature than a
# plain 90-degree box corner, which this specific test case doesn't
# exercise.
FEATURE_ANGLE_THRESHOLD_RAD = np.deg2rad(20.0)
MITER_LIMIT = 1.2


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

    At a sharp CONVEX edge/corner (e.g. a box's 90 degree edges), the
    plain averaged normal bisects the adjacent faces' normals, so moving
    along it by `thickness` only advances `thickness * cos(half the
    feature angle)` perpendicular to each actual face - the layer is
    disproportionately thin exactly at the edge, and that skew compounds
    every subsequent layer and every prism-to-tetrahedra split there,
    producing sliver/near-zero-volume tetrahedra concentrated at sharp
    features (empirically confirmed: on a plain cube body, the worst 0.1%
    of the whole volume mesh by shape quality were ~80% concentrated in
    the cube's own bounding box, at edge/corner-adjacent coordinates,
    despite the cube occupying under 1% of the domain). To compensate,
    each node's displacement is scaled up by the same "miter join" factor
    1/cos(half-angle) vector graphics stroking uses at sharp path corners
    - still along the same averaged direction (no topology change / vertex
    splitting), just far enough to restore the intended perpendicular
    offset. See FEATURE_ANGLE_THRESHOLD_RAD/MITER_LIMIT module constants.

    This compensation, and the optional remaining_budget cap, both only
    look at the UNDEFORMED surface / this node's own local neighbourhood -
    neither one can see whether the front they produce actually folds over
    somewhere else on the mesh. mesh_extrusion.extrude_layers checks the
    real result after calling this function and reactively freezes
    (remaining_budget = 0) any node that does, via
    mesh_front_collision.freeze_self_colliding_nodes - see that module's
    docstring for why a static, undeformed-surface estimate can't
    guarantee overlap-free geometry by itself.

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
    repeated_normals = np.repeat(normals, 3, axis=0)
    np.add.at(node_normal_sum, flat_nodes, repeated_normals)
    np.add.at(node_normal_count, flat_nodes, 1)

    # Compute average normals (avoid division by zero)
    mask = node_normal_count > 0
    avg_normals = np.zeros_like(node_normal_sum)
    avg_normals[mask] = node_normal_sum[mask] / node_normal_count[mask, np.newaxis]

    # Normalize
    norms = np.linalg.norm(avg_normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    avg_normals = avg_normals / norms

    # Per-node feature sharpness: how far the *worst* adjacent face normal
    # deviates from this node's own averaged normal. Using deviation-from-
    # mean (a single vectorized scatter-min) instead of the true max
    # pairwise angle between all adjacent-face-normal pairs (which would
    # need an O(valence^2) pass per node) - verified against the two cases
    # that matter here: a straight edge between two faces at angle theta
    # gives deviation=theta/2 exactly (recovers theta back out), and an
    # axis-aligned box corner (three mutually perpendicular faces) gives
    # ~54.7 degrees deviation vs. the true pairwise max of 90 degrees -
    # i.e. this slightly *over*-estimates sharpness at corners, which
    # applies more miter compensation exactly where the sliver problem was
    # empirically worst, not less.
    per_corner_dot = np.einsum('ij,ij->i', repeated_normals, avg_normals[flat_nodes])
    min_dot = np.ones(n_nodes, dtype=np.float64)
    np.minimum.at(min_dot, flat_nodes, per_corner_dot)
    deviation_angle = np.arccos(np.clip(min_dot, -1.0, 1.0))
    feature_angle = 2.0 * deviation_angle

    half_angle = np.clip(feature_angle / 2.0, 0.0, np.deg2rad(89.0))
    miter_scale = np.where(
        feature_angle > FEATURE_ANGLE_THRESHOLD_RAD,
        np.minimum(1.0 / np.cos(half_angle), MITER_LIMIT),
        1.0,
    )

    if taper_scale is not None:
        node_thickness = thickness * taper_scale * miter_scale
    else:
        node_thickness = thickness * miter_scale

    if remaining_budget is not None:
        node_thickness = np.minimum(node_thickness, remaining_budget)
        remaining_budget -= node_thickness

    displacement = node_thickness[:, np.newaxis] * avg_normals

    # Extrude all nodes at once
    logger.info(f"Extruding layer with thickness={thickness:.6f}...")
    new_nodes[mask] += displacement[mask]

    return new_nodes
