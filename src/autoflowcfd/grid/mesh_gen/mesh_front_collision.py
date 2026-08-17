"""边界层前沿挤出的逐层自碰撞防护。

extrude_single_layer 的斜接补偿（mesh_layer_step.py 的 MITER_LIMIT）和
mesh_tetgen_core.compute_local_thickness_limit 的先验锥角预算都能减少挤出
前沿自折叠的频率，但都不能*保证*避免：斜接是按未变形表面一次性计算的
固定逐节点缩放，厚度限制预算是基于同一未变形几何的静态全程估计——
两者都不会查看即将生成的层的实际当前几何。尖锐凹曲线（当厚度超过局部
曲率半径时偏移线收敛）或价≥3的角点（三个或更多面片交汇，而非斜接补偿
建模的简单两面片边）无论参数如何都可能折叠——已直接确认：cube_demo 在
多种层数/增长率组合下仍显示数百个重叠单元。

两个互补机制弥补了这一缺陷，镜像了推进前沿方法（如 Pointwise 的 T-Rex）
的处理方式——两者均独立于 growth_rate、bl_layers 或任何其它挤出参数：

  clamp_budget_for_convergence - BEFORE each layer's step, measure the
      CURRENT distance between candidate non-adjacent face pairs and cap
      each involved node's remaining lifetime displacement to at most
      half of it. This is the primary defence, and matters even for a
      single, well-behaved pair of converging fronts: an AFTER-the-fact
      check alone can "tunnel" - two clean, non-intersecting end-of-layer
      snapshots can still have their SWEPT PRISMS fully overlap in
      between if a single step is large relative to the remaining gap
      (confirmed directly: this project's own test suite caught exactly
      this on two flat facing patches closing a tight gap over a handful
      of geometrically-growing layers - see
      test_mesh_front_collision.py). Recomputing from real geometry every
      layer (not a one-time undeformed-surface estimate) means the cap
      only ever tightens as fronts approach, converging them toward
      meeting near the midpoint of whatever gap remains, never past it.

  freeze_self_colliding_nodes - AFTER each layer's step, check the
      layer's actual resulting geometry for genuine self-intersection
      TWO ways and roll back + permanently freeze only the offending
      nodes:
        (a) find_self_colliding_faces - the new layer against itself
            (same snapshot, same as a single-pair tunneling check but
            for the whole mesh at once).
        (b) find_cross_state_colliding_faces - the new layer against the
            PREVIOUS layer - catches a fast-advancing face sweeping
            through space a different, slower/frozen neighbour was still
            occupying at the start of that same step, which (a) alone
            cannot see (neither snapshot it compares is self-
            intersecting on its own) and which clamp_budget_for_
            convergence's own first-order/instantaneous approximation is
            not guaranteed to predict ahead of time for a large enough
            step (confirmed directly: found real cases on cube_demo,
            spread across most of the BL stack's depth, that (a) alone
            missed entirely).
      Both are a backstop for whatever the pre-step clamp's pairwise,
      broad-phase-radius-bounded search doesn't happen to cover - defence
      in depth, not the primary mechanism.

See mesh_extrusion.py's extrude_layers for how both functions compose
with the pre-existing remaining_budget mechanism (tightening/freezing a
node is literally lowering/zeroing that node's remaining_budget).

底层的广相位候选对搜索 + 精确三角形相交/跨状态检测（find_self_colliding_
faces / find_cross_state_colliding_faces）拆到了同目录
mesh_front_collision_detect.py，本文件只保留事前裁剪
（clamp_budget_for_convergence）和事后冻结（freeze_self_colliding_nodes）
这两个真正对外使用的入口。
"""

import numpy as np
from loguru import logger

from ..validation.overlap_geometry import triangle_triangle_intersect, triangle_triangle_min_distance
from .mesh_front_collision_detect import (
    _face_geometry,
    _iter_candidate_pairs,
    find_self_colliding_faces,
    find_cross_state_colliding_faces,
)

# clamp_budget_for_convergence caps each side of a converging pair to this
# fraction of their CURRENT distance, not exactly 0.5. Mathematically any
# fraction <= 0.5 already guarantees the gap can never go negative (see
# that function's own docstring); strictly below 0.5 additionally
# guarantees it never fully closes to exactly 0 either, whenever a single
# layer's step is what exhausts a node's budget (the common case - see
# "Dropped tets" in mesh_prism_to_tet.py, and confirmed directly by this
# project's own test suite: with exactly 0.5, two perfectly symmetric
# facing fronts converge to exactly-coincident, duplicate geometry at the
# limit, which is a degenerate PLC input tetgen can reject (the same
# "vertices are coplanar" class of error already seen on a real case, see
# ProjectFiles Part5) even though the resulting zero-volume tets would
# themselves be handled fine by the dropping logic downstream). 0.45
# leaves a comfortable margin below the 0.5 boundary without materially
# changing how many layers convergence takes.
CONVERGENCE_SAFETY_FRACTION = 0.45

# clamp_budget_for_convergence must only restrict a candidate pair that is
# actually converging - being merely CLOSE is not enough. Proximity alone
# was tried first and confirmed directly (on cube_demo, a plain cube
# body) to be a serious bug, not just an over-conservative heuristic:
# every pair of small triangles straddling one of the cube's own CONVEX
# edges (ordinary, correctly-shaped mesh refinement near a feature, not a
# defect) sits well within a few face-widths of each other from the very
# first layer, exactly like a genuinely converging pair does - without a
# directional filter, clamp_budget_for_convergence could not tell that
# apart from real convergence and froze nodes along essentially every
# edge of the cube, producing a mesh with 131x MORE overlapping cells
# than the unfixed baseline (132,260 vs. 1,004) once the resulting mass
# of degenerate/near-duplicate frozen geometry reached tetgen.
#
# A plain dot(normal_a, normal_b) < 0 ("normals point back toward each
# other") test was tried next and is ALSO wrong, just for a narrower and
# sneakier class of case: a sharp CONVEX wedge (a thin fin/blade, e.g. an
# airfoil trailing edge) has near-OPPOSITE face normals purely because of
# how acute the wedge angle is (verified directly: a symmetric 10 degree
# wedge gives dot=-0.98), yet its two surfaces genuinely DIVERGE as they
# extrude outward, same as any other convex feature - the thinness of the
# material is irrelevant to which way the offset surfaces move. What
# actually matters is not the two normals' absolute directions but
# whether moving along them SHRINKS the distance between the two
# candidate faces: for centroid separation vector d = centroid_b -
# centroid_a and normal difference dn = normal_b - normal_a, the
# instantaneous rate of change of squared separation as both faces
# advance along their own normals is proportional to dot(d, dn) - and a
# pair is converging iff that is negative (verified directly against all
# four cases that matter: facing plates -0.1, convex 90deg edge +0.25,
# concave 90deg notch -0.25, sharp convex wedge +0.35 - each with the
# physically expected sign, including the wedge case the simpler normal-
# only test got backwards).
CONVERGING_CLOSING_RATE_THRESHOLD = 0.0


def clamp_budget_for_convergence(
    nodes: np.ndarray,
    faces: np.ndarray,
    remaining_budget: np.ndarray,
    search_multiplier: float = 3.0,
    chunk_size: int = 2000,
) -> None:
    """Tighten `remaining_budget` in place so no node is allowed to
    advance, over all remaining layers combined, further than
    CONVERGENCE_SAFETY_FRACTION (0.45) of its CURRENT distance to the
    nearest non-adjacent face it is actually converging with (relative
    closing rate below CONVERGING_CLOSING_RATE_THRESHOLD - see that
    constant's own comment for why this filter is required, not
    optional). Call this BEFORE extruding a layer, on that layer's
    starting (`current_nodes`) geometry - see module docstring for why a
    purely AFTER-the-fact check is not sufficient on its own.

    Every candidate pair found tightens BOTH sides' budgets to at most
    ~45% of their own current distance - symmetric, so if both sides
    spend their full allowance moving straight toward each other they
    stop just short of the midpoint, a strictly positive gap remains
    (never exactly 0, never crossing - see CONVERGENCE_SAFETY_FRACTION's
    own comment for why stopping strictly short of the midpoint matters).
    A node touching several converging pairs at once (e.g. a valence-3+
    concave corner where three fronts meet) ends up bounded by the
    MINIMUM over all of them, since each pairwise bound is enforced
    independently via a single vectorized scatter-min (np.minimum.at) -
    correct regardless of how many other pairs a node also participates
    in. Recomputed fresh every layer from the mesh's actual current
    geometry (not a one-time estimate), so as two fronts approach this
    only ever tightens further, converging them toward - never past -
    each other.

    A candidate pair already (exactly) intersecting on `current_nodes`
    should not occur in practice - `current_nodes` is always the
    previous layer's already-accepted, by-induction collision-free
    result - but is handled defensively by clamping straight to zero
    rather than calling triangle_triangle_min_distance, which (per its
    own docstring) is only meaningful for a non-intersecting pair.

    Args:
        nodes: (n_nodes, 3) CURRENT (pre-step) node positions
        faces: (n_faces, 3) triangle connectivity (int)
        remaining_budget: (n_nodes,) float, meters - mutated in place,
            only ever lowered, never raised
        search_multiplier: broad-phase KD-tree query radius as a multiple
            of each face's own sqrt(area) - larger than find_self_
            colliding_faces' default since this must also catch pairs
            that are merely close, not yet touching
        chunk_size: faces processed per KD-tree batch
    """
    n_faces = len(faces)
    if n_faces == 0:
        return

    tri, centroids, face_size, normal = _face_geometry(nodes, faces)
    budget_before = remaining_budget.copy()

    for row_idx, col_idx in _iter_candidate_pairs(
        faces, centroids, face_size, search_multiplier, chunk_size
    ):
        # Only a pair actually converging - moving along their own
        # normals shrinks the distance between them - gets restricted;
        # see CONVERGING_CLOSING_RATE_THRESHOLD's own comment for why
        # this must be the relative closing rate and not just whether the
        # two normals point at each other. Applied before the (more
        # expensive) exact geometric tests, both to skip work on excluded
        # pairs and because it is what makes this function safe to use at
        # all (see module docstring / this function's own name).
        d_vec = centroids[col_idx] - centroids[row_idx]
        n_diff = normal[col_idx] - normal[row_idx]
        closing_rate = np.einsum('ij,ij->i', d_vec, n_diff)
        converging = closing_rate < CONVERGING_CLOSING_RATE_THRESHOLD
        if not np.any(converging):
            continue
        row_idx, col_idx = row_idx[converging], col_idx[converging]

        a_nodes, b_nodes = tri[row_idx], tri[col_idx]
        intersects = triangle_triangle_intersect(
            a_nodes[:, 0], a_nodes[:, 1], a_nodes[:, 2],
            b_nodes[:, 0], b_nodes[:, 1], b_nodes[:, 2],
        )

        safe_budget = np.zeros(len(row_idx), dtype=np.float64)
        safe = ~intersects
        if np.any(safe):
            dists = triangle_triangle_min_distance(
                a_nodes[safe, 0], a_nodes[safe, 1], a_nodes[safe, 2],
                b_nodes[safe, 0], b_nodes[safe, 1], b_nodes[safe, 2],
            )
            safe_budget[safe] = CONVERGENCE_SAFETY_FRACTION * dists
        # intersecting pairs keep safe_budget == 0: clamp straight to zero.

        pair_nodes = np.concatenate([faces[row_idx], faces[col_idx]], axis=1)  # (M, 6)
        pair_budget = np.repeat(safe_budget, 6).reshape(-1, 6)
        np.minimum.at(remaining_budget, pair_nodes.ravel(), pair_budget.ravel())

    # Was previously silent - no logging at all despite being the primary
    # (proactive) defence the module docstring describes, which made it
    # invisible in a real run's own console output: a mesh that converged
    # entirely through this mechanism (remaining_budget tightened to ~0
    # BEFORE any actual collision could occur) produced zero self-
    # intersection warnings from freeze_self_colliding_nodes downstream,
    # reading as if nothing had constrained growth there at all - confirmed
    # directly on cube_demo, where this was the sole cause of ~25,000
    # dropped (fully budget-exhausted) BL prisms with no other mechanism
    # showing any evidence of why.
    tightened = remaining_budget < budget_before
    n_tightened = int(np.sum(tightened))
    if n_tightened:
        n_exhausted = int(np.sum(remaining_budget[tightened] <= 0.0))
        logger.info(
            f"Convergence budget clamp: {n_tightened} node(s) tightened this "
            f"layer ({n_exhausted} fully exhausted, remaining_budget=0 - that "
            f"column's front is done growing), min remaining "
            f"{float(np.min(remaining_budget[tightened])):.4e} m"
        )


def freeze_self_colliding_nodes(
    new_nodes: np.ndarray,
    current_nodes: np.ndarray,
    faces: np.ndarray,
    remaining_budget: np.ndarray,
    max_iterations: int = 5,
) -> np.ndarray:
    """Roll back and permanently freeze every node on a self-intersecting
    face, in place on `new_nodes` and `remaining_budget`.

    `current_nodes` is the PREVIOUS layer's already-accepted geometry -
    by induction it is itself collision-free, since this same check ran
    on it too when it was the "new" layer. Rolling a guilty node back to
    its `current_nodes` position can therefore only return that node's
    faces to a state already known to be collision-free with each other;
    it cannot make anything worse. Freezing is `remaining_budget = 0`:
    extrude_single_layer already clamps every node's per-layer
    displacement to what's left of its budget (see mesh_extrusion.py), so
    a frozen node simply stops moving for the remainder of the run -
    T-Rex's "terminate locally, continue elsewhere" semantics, with every
    other node unaffected.

    Iterates (bounded by `max_iterations`) because undoing one pair's
    collision can occasionally leave a still-moving neighbour
    intersecting something else that only became a problem once its
    neighbour rolled back (a cascade). Each iteration either freezes at
    least one additional node or finds nothing and stops, so this always
    terminates on its own; the cap only bounds worst-case per-layer cost
    - an unresolved cascade beyond the cap is still re-examined on the
    NEXT layer's own call, and caught regardless by the final mesh-wide
    overlap validation (mesh_overlap_check.py) that runs after
    tetrahedralization.

    Args:
        new_nodes: This layer's tentative node positions - mutated in
            place for any newly frozen node
        current_nodes: Previous (already-accepted) layer's node positions
        faces: (n_faces, 3) triangle connectivity, the same one used to
            extrude both layers
        remaining_budget: Per-node remaining extrusion budget (meters) -
            mutated in place, set to 0 for newly frozen nodes
        max_iterations: cascade-resolution cap (see above)

    Returns:
        int64 array of node indices frozen during this call (empty if
        none)
    """
    frozen = np.zeros(len(new_nodes), dtype=bool)
    total_frozen_count = 0

    for i in range(max_iterations):
        colliding_faces = find_self_colliding_faces(new_nodes, faces)
        # Also check for a fast-advancing face sweeping through a
        # different, slower/frozen neighbour's territory this same step -
        # see find_cross_state_colliding_faces' own docstring for why the
        # same-snapshot check above cannot see this on its own. Both
        # sides of a flagged pair are frozen (not just the "aggressor"),
        # matching the same-layer check's own all-nodes-of-both-faces
        # policy - simpler to reason about and never less safe.
        cross_faces = find_cross_state_colliding_faces(new_nodes, current_nodes, faces)
        colliding_faces = np.union1d(colliding_faces, cross_faces)
        if len(colliding_faces) == 0:
            break

        guilty = np.unique(faces[colliding_faces].ravel())
        guilty = guilty[remaining_budget[guilty] > 0]
        if len(guilty) == 0:
            break

        new_nodes[guilty] = current_nodes[guilty]
        remaining_budget[guilty] = 0.0
        frozen[guilty] = True
        total_frozen_count += len(guilty)
        
        if i == 0:
            logger.warning(f"Detected {len(colliding_faces)} self-intersecting faces in BL layer. "
                           f"Freezing {len(guilty)} nodes to prevent invalid geometry.")
        elif len(guilty) > 0:
            logger.debug(f"Cascade resolution: freezing {len(guilty)} additional nodes.")

    if total_frozen_count > 0:
        logger.info(f"Total nodes frozen in this BL layer: {total_frozen_count}")

    return np.flatnonzero(frozen)
