"""单层挤出步骤：法向平均与斜接（miter-join）补偿。

从 mesh_extrusion.py 拆出（该文件保留多层编排循环 extrude_layers），纯粹
为了让两个文件都控制在 450 行以内；extrude_layers 是本模块唯一调用方。
"""

import numpy as np
from typing import Optional
from loguru import logger

# MITER_LIMIT caps the offset-vector magnitude computed below the same way
# SVG/vector-graphics stroke "miter joins" do (default stroke-miterlimit
# there is 4) - past the limit a vertex falls back toward a shorter,
# still-correctly-directed offset instead of letting it blow up, which is
# what would happen at a near-reflex/needle feature (cos(half-angle) -> 0
# for a single 2-patch edge; the least-squares matrix in
# extrude_single_layer approaches singular for the general N-patch case).
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
MITER_LIMIT = 1.2

# When a node's remaining_budget (mesh_front_collision.clamp_budget_for_
# convergence's running cumulative cap - see extrude_layers) is already
# smaller than what this layer's nominal thickness would request, consume
# only this fraction of what's left THIS layer rather than all of it in
# one shot. Plain min(nominal, remaining_budget) (the previous behaviour)
# hits exactly 0 the very next time a layer's nominal request exceeds
# whatever was left - typically immediately, since layer thickness grows
# geometrically (each ~1.2-1.5x the last) while remaining_budget was
# already small BY DEFINITION of having triggered this path - producing a
# fully coincident (zero-volume) prism for every subsequent layer at that
# node. Confirmed directly on cube_demo: this was the entire explanation
# for its ~25,000 dropped BL prisms, with none of the OTHER collision
# mechanisms (freeze_self_colliding_nodes, compute_local_thickness_limit)
# showing any evidence of engaging at all. Geometric decay (consuming a
# fixed FRACTION of whatever remains, not a fixed amount) means a
# constrained node's height approaches its true limit asymptotically over
# several layers instead of hitting it in one - it only ever produces a
# vanishingly thin, never an EXACTLY zero-volume, layer, so nothing here
# needs to be dropped downstream. 0.5 (halve the remaining gap each time
# it's the binding constraint) mirrors this project's own
# CONVERGENCE_SAFETY_FRACTION-style reasoning elsewhere (mesh_front_
# collision.py) without literally reusing that specific constant, since it
# governs a different quantity (a spatial safety margin there, a temporal
# decay rate here).
BUDGET_TAPER_FRACTION = 0.5

# Pure geometric decay (BUDGET_TAPER_FRACTION alone) never hits exactly 0,
# but "never exactly 0" is not the same as "usable" - confirmed directly
# on cube_demo: eliminating every dropped prism this way produced a long
# tail of numerically vanishing survivors instead (minimum 3.3e-8 m,
# i.e. 33 nanometres, ~6% of all BL prisms under 0.1mm), which a real
# solver has no realistic use for (an aspect ratio in the millions,
# relative to that same cell's few-mm footprint) and is arguably a WORSE
# outcome than a clean drop - garbage-in for whatever mesh-quality gate
# or solver runs next, not just a smaller number in a log line. Below
# this fraction of the CURRENT layer's own nominal thickness, a tapered
# spend is treated as fully exhausted (snapped to 0, same as the old
# hard-stop behaviour) instead of continuing to shrink - a floor, not a
# guess: it keeps the smooth taper for every node that's still within a
# sane range of its neighbours' own cell size, and only reverts to a
# clean stop once continuing would produce a cell no downstream stage
# could do anything useful with anyway.
#
# 0.05 (the first value tried) still let survivors down to 0.073mm
# through - a compounding effect: a node can taper across SEVERAL
# consecutive layers before any single layer's floor check catches it
# (each layer's own floor is relative to that layer's own, larger,
# nominal thickness, but the survivor keeps shrinking by
# BUDGET_TAPER_FRACTION regardless), leaving a small but real tail below
# 0.1mm (~0.47% of all BL prisms on cube_demo). Measured directly across
# several values: 0.1 is the smallest that fully clears the 0.1mm tail
# (min survivor 0.130mm, 0 cells below 0.1mm), at a moderate cost in
# additional cleanly-dropped cells (20,828 vs 0.05's 13,807, both still
# well below the pre-taper 24,691 baseline). 0.3 clears a wider margin
# (min survivor 0.260mm) but pushes dropped cells to 32,213 - WORSE than
# not tapering at all - so higher is not simply safer; 0.1 was kept as
# the better-measured trade-off, not the most conservative available one.
MIN_TAPER_FRACTION_OF_NOMINAL = 0.1


def extrude_single_layer(
    nodes: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    thickness: float,
    taper_scale: 'Optional[np.ndarray]' = None,
    remaining_budget: 'Optional[np.ndarray]' = None,
    miter_decay: float = 1.0,
) -> np.ndarray:
    """Extrude one layer of nodes.

    For each node, compute the LEAST-SQUARES offset vector `d` (for a UNIT
    thickness) that best satisfies every adjacent face's own target
    distance simultaneously: `d = argmin sum_i (n_i . d - 1)^2` over that
    node's adjacent face normals `n_i`, solved per node via
    `(sum_i n_i n_i^T) d = sum_i n_i` (a 3x3 linear system, batched via
    np.linalg.pinv so a rank-deficient case - a flat node, a plain 2-patch
    edge - is handled by the SAME formula, not a special case).

    This is a direct generalisation of the "miter join" vector-graphics
    stroking uses at sharp path corners (1/cos(half-angle), still along
    the plain averaged-normal bisector) - PROVEN to reduce to exactly that
    formula for a 2-patch edge of any angle, and to exactly the plain
    normal for a flat/single-patch node (see this project's own validation
    script, not reproduced here - both are closed-form identities of the
    least-squares solution, not just empirically similar). The point of
    solving the full 3x3 system instead is the case neither the old plain-
    average-then-scale approach nor mesh_corner_split's own vertex
    splitting fully solved: a valence-3+ corner where 3+ patches meet at
    one point. A blended AVERAGE direction there is generically not even
    on the correct offset line for any single patch, let alone all of
    them - "no single blended direction simultaneously offsets three
    independent planes correctly" (mesh_corner_split's own module
    docstring). The least-squares solve is NOT a blend of directions - for
    3 independent normals in 3D it is the EXACT point satisfying all 3
    target distances at once (3 equations, 3 unknowns), and for more than
    3 it is the closest achievable compromise. Confirmed directly: for 3
    mutually perpendicular normals (an axis-aligned box corner) this
    solves to exactly (1,1,1) with magnitude sqrt(3) - the same "natural
    (unclipped) compensation factor ~1.73x" this module's own MITER_LIMIT
    comment already documented as the correct value from first principles,
    now reached by the general formula instead of only known as a special-
    cased reference number.

    Only the MAGNITUDE of the solved vector is capped (to MITER_LIMIT,
    same rationale as the old scalar cap - a near-reflex/needle feature
    can make the system near-singular) - never the direction, which is
    always exactly what the least-squares solve produced.

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
            mutated in place - a node whose nominal displacement this
            layer exceeds what's left of its budget only spends
            BUDGET_TAPER_FRACTION of the remainder (an asymptotic taper,
            never exactly exhausting it in one step - see that constant's
            own comment), decremented by the same amount it actually
            spent (see extrude_layers' thickness_limit).
        miter_decay: Blends the per-node offset vector toward the plain
            (unweighted, magnitude-1) averaged normal - 1.0 (default)
            keeps the full computed offset vector unchanged, 0.0 disables
            compensation entirely for this layer. extrude_layers ramps
            this down over the TRANSITION stage (see its own docstring for
            why): the offset vector is FIXED per-node (depends only on the
            undeformed surface's local feature angle, never recomputed
            against deformed geometry), so a sharp-corner node's
            cumulative height over many layers ends up in a constant
            ratio to its flat-region neighbours' cumulative height. That
            ratio is harmless at BL scale (a small absolute gap), but once
            the transition stage extrudes to a far-field target size, the
            SAME ratio applied to a much larger absolute height opens a
            large absolute gap between mesh-adjacent nodes - confirmed
            directly on a real case: a 90mm+ absolute height mismatch
            between a sharp-corner column and its flat-region neighbour,
            concentrated exactly at genuinely degenerate transition cells
            and a large fraction of the final mesh's boundary faces
            failing to match any real boundary group. Full compensation is
            still applied throughout the BL stage itself, where it matters
            most (near-wall cell shape quality) and cumulative heights
            stay small regardless.

    Returns:
        New node positions after extrusion, shape=(n_nodes, 3)
    """
    n_nodes = len(nodes)
    new_nodes = nodes.copy()

    # Build node-to-face mapping using vectorized operations
    node_normal_count = np.zeros(n_nodes, dtype=np.int64)
    flat_nodes = faces.ravel()
    repeated_normals = np.repeat(normals, 3, axis=0)
    np.add.at(node_normal_count, flat_nodes, 1)
    mask = node_normal_count > 0

    # Plain (unweighted) averaged normal, magnitude 1 - the miter_decay=0
    # "no compensation" fallback, and also the reference direction the
    # least-squares solve below is validated against for a flat node.
    node_normal_sum = np.zeros((n_nodes, 3))
    np.add.at(node_normal_sum, flat_nodes, repeated_normals)
    plain_avg_normal = np.zeros_like(node_normal_sum)
    plain_avg_normal[mask] = node_normal_sum[mask] / node_normal_count[mask, np.newaxis]
    plain_norms = np.maximum(np.linalg.norm(plain_avg_normal, axis=1, keepdims=True), 1e-10)
    plain_avg_normal = plain_avg_normal / plain_norms

    # Least-squares offset vector for unit thickness: per node, solve
    # (sum_i n_i n_i^T) d = sum_i n_i. Batched via np.linalg.pinv (handles
    # the rank-deficient flat/2-patch cases the same way as the general
    # one, via SVD, rather than needing a separate branch for them).
    outer = np.einsum('ki,kj->kij', repeated_normals, repeated_normals)
    A = np.zeros((n_nodes, 3, 3))
    np.add.at(A, flat_nodes, outer)
    A_pinv = np.linalg.pinv(A)
    offset_vec = np.einsum('kij,kj->ki', A_pinv, node_normal_sum)

    # Cap magnitude only (never redirect) - see MITER_LIMIT's own comment.
    offset_mag = np.linalg.norm(offset_vec, axis=1, keepdims=True)
    safe_mag = np.maximum(offset_mag, 1e-10)
    capped_mag = np.minimum(offset_mag, MITER_LIMIT)
    offset_vec = offset_vec / safe_mag * capped_mag

    if miter_decay != 1.0:
        offset_vec = plain_avg_normal + (offset_vec - plain_avg_normal) * miter_decay

    # avg_normals/miter_scale below keep their original names+meaning as a
    # unit direction and a separate scalar magnitude, so the rest of this
    # function (taper_scale/remaining_budget application) is unchanged.
    miter_scale = np.maximum(np.linalg.norm(offset_vec, axis=1), 1e-10)
    avg_normals = offset_vec / miter_scale[:, np.newaxis]

    if taper_scale is not None:
        node_thickness = thickness * taper_scale * miter_scale
    else:
        node_thickness = thickness * miter_scale

    if remaining_budget is not None:
        # See BUDGET_TAPER_FRACTION's own comment: a node whose remaining
        # budget is already tighter than this layer's nominal request only
        # spends a FRACTION of what's left, tapering asymptotically
        # instead of exhausting to exactly 0 in one step. A node with
        # plenty of budget left (the common case) is completely unaffected
        # - node_thickness already equals its own nominal request there.
        tight = remaining_budget < node_thickness
        tapered = remaining_budget * BUDGET_TAPER_FRACTION
        # See MIN_TAPER_FRACTION_OF_NOMINAL's own comment: stop cleanly
        # instead of continuing to taper toward a numerically meaningless
        # sliver once even the tapered spend would be negligible relative
        # to this layer's own scale.
        floor = thickness * MIN_TAPER_FRACTION_OF_NOMINAL
        exhausted = tight & (tapered < floor)
        node_thickness = np.where(tight & ~exhausted, tapered, node_thickness)
        node_thickness = np.where(exhausted, 0.0, node_thickness)
        remaining_budget -= node_thickness
        remaining_budget = np.where(exhausted, 0.0, remaining_budget)

    displacement = node_thickness[:, np.newaxis] * avg_normals

    # Extrude all nodes at once
    logger.info(f"Extruding layer with thickness={thickness:.6f}...")
    new_nodes[mask] += displacement[mask]

    return new_nodes
