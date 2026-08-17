"""Unit tests for the reactive per-layer self-collision freeze mechanism
(mesh_front_collision.py) that mesh_extrusion.extrude_layers now calls
after every layer - see that module's own docstring for why a static,
undeformed-surface estimate (the miter join in mesh_layer_step, or an
a-priori thickness_limit) cannot by itself guarantee an extruded front
never folds over itself, regardless of growth_rate/bl_layers/transition_growth_rate.
"""

import numpy as np
import pytest

from autoflowcfd.grid.mesh_gen.extrusion.mesh_extrusion import extrude_layers
from autoflowcfd.grid.mesh_gen.utils.mesh_front_collision import (
    CONVERGENCE_SAFETY_FRACTION,
    clamp_budget_for_convergence,
    find_self_colliding_faces,
    freeze_self_colliding_nodes,
)
from autoflowcfd.grid.mesh_gen.utils.mesh_utils import compute_face_normals

# Two triangles that genuinely cross in 3D, sharing no vertices - verified
# directly against overlap_geometry.triangle_triangle_intersect. Triangle A
# lies flat in z=0 and contains the origin in its interior; edge B0-B1 of
# triangle B is the segment x=0,y=0,z in [-2,2], which pierces straight
# through that interior point.
A0, A1, A2 = np.array([-2., -2., 0.]), np.array([2., -2., 0.]), np.array([0., 2., 0.])
B0, B1, B2 = np.array([0., 0., -2.]), np.array([0., 0., 2.]), np.array([0., 3., 0.])


class TestFindSelfCollidingFaces:
    def test_crossing_triangles_are_detected(self):
        nodes = np.array([A0, A1, A2, B0, B1, B2])
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        colliding = find_self_colliding_faces(nodes, faces)
        assert sorted(colliding.tolist()) == [0, 1]

    def test_well_separated_triangles_are_not_flagged(self):
        far = np.array([B0, B1, B2]) + 100.0
        nodes = np.vstack([[A0, A1, A2], far])
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        assert len(find_self_colliding_faces(nodes, faces)) == 0

    def test_adjacent_faces_sharing_a_vertex_are_never_flagged(self):
        """A flat fan of triangles sharing a common centre vertex is
        ordinary, valid mesh topology, not a self-collision - even though
        every pair of fan blades literally touches at that shared vertex."""
        center = np.array([0., 0., 0.])
        n = 8
        rim = [np.array([np.cos(t), np.sin(t), 0.])
               for t in np.linspace(0, 2 * np.pi, n, endpoint=False)]
        nodes = np.array([center] + rim)
        faces = np.array([[0, 1 + i, 1 + (i + 1) % n] for i in range(n)])
        assert len(find_self_colliding_faces(nodes, faces)) == 0

    def test_empty_face_array_returns_empty(self):
        nodes = np.zeros((0, 3))
        faces = np.zeros((0, 3), dtype=np.int64)
        result = find_self_colliding_faces(nodes, faces)
        assert len(result) == 0


class TestFreezeSelfCollidingNodes:
    def _two_face_setup(self):
        """faces=[0,1,2] (A) and [3,4,5] (B); `current` is B held far away
        (the previous, already-accepted, collision-free layer) while
        `new` has B pulled back in to genuinely cross A."""
        new_nodes = np.array([A0, A1, A2, B0, B1, B2])
        current_nodes = new_nodes.copy()
        current_nodes[3:] = np.array([B0, B1, B2]) + 100.0
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        return new_nodes, current_nodes, faces

    def test_colliding_nodes_are_rolled_back_and_frozen(self):
        new_nodes, current_nodes, faces = self._two_face_setup()
        budget = np.full(6, np.inf)

        frozen = freeze_self_colliding_nodes(new_nodes, current_nodes, faces, budget)

        assert sorted(frozen.tolist()) == [0, 1, 2, 3, 4, 5]
        assert np.array_equal(new_nodes, current_nodes)
        assert np.all(budget == 0.0)
        # The rolled-back result must itself be collision-free - freezing
        # to the previous, already-accepted layer can never make things worse.
        assert len(find_self_colliding_faces(new_nodes, faces)) == 0

    def test_uninvolved_face_is_left_untouched(self):
        """A third, well-separated face must be unaffected by freezing the
        colliding pair - freezing is local to the offending nodes, not the
        whole layer."""
        new_nodes, current_nodes, faces = self._two_face_setup()
        # Far from both the origin (A/B's own coordinates) AND +100 (where
        # B gets rolled back to by this setup) so it can never accidentally
        # land near either.
        far_face = np.array([[1e5, 1e5, 1e5], [1e5 + 1, 1e5, 1e5], [1e5, 1e5 + 1, 1e5]])
        new_nodes = np.vstack([new_nodes, far_face])
        current_nodes = np.vstack([current_nodes, far_face + np.array([0., 0., 5.])])
        faces = np.vstack([faces, [[6, 7, 8]]])
        budget = np.full(9, np.inf)
        original_far = new_nodes[6:].copy()

        frozen = freeze_self_colliding_nodes(new_nodes, current_nodes, faces, budget)

        assert set(frozen.tolist()) == {0, 1, 2, 3, 4, 5}
        assert np.array_equal(new_nodes[6:], original_far), "uninvolved face must not move"
        assert np.all(budget[6:] == np.inf), "uninvolved nodes must keep their budget"

    def test_no_collision_freezes_nothing(self):
        nodes = np.array([A0, A1, A2])
        faces = np.array([[0, 1, 2]])
        budget = np.full(3, np.inf)

        frozen = freeze_self_colliding_nodes(nodes.copy(), nodes.copy(), faces, budget)

        assert len(frozen) == 0
        assert np.all(budget == np.inf)

    def test_two_independent_collisions_are_both_resolved(self):
        """Two unrelated colliding pairs elsewhere in the same layer (e.g.
        two different sharp corners of the same body) must both be caught
        and frozen in one call, not just the first one found."""
        new1, cur1, _ = self._two_face_setup()
        offset = np.array([500., 0., 0.])
        new2, cur2 = new1 + offset, cur1 + offset
        new_nodes = np.vstack([new1, new2])
        current_nodes = np.vstack([cur1, cur2])
        faces = np.vstack([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])
        budget = np.full(12, np.inf)

        frozen = freeze_self_colliding_nodes(new_nodes, current_nodes, faces, budget)

        assert set(frozen.tolist()) == set(range(12))
        assert len(find_self_colliding_faces(new_nodes, faces)) == 0

    def test_partially_frozen_pair_only_refreezes_the_still_moving_side(self):
        """If one side of a colliding pair was already frozen on an
        earlier layer (budget already 0, so extrude_single_layer already
        clamped its displacement to 0 - it never actually moved this
        layer either) and the other side is still advancing normally,
        only the still-moving side's nodes come back as newly frozen; the
        already-frozen side has nothing to roll back."""
        new_nodes, current_nodes, faces = self._two_face_setup()
        new_nodes[:3] = current_nodes[:3]  # triangle A already frozen: unmoved
        budget = np.array([0., 0., 0., np.inf, np.inf, np.inf])

        frozen = freeze_self_colliding_nodes(new_nodes, current_nodes, faces, budget)

        assert set(frozen.tolist()) == {3, 4, 5}
        assert np.array_equal(new_nodes, current_nodes)
        assert list(budget) == [0., 0., 0., 0., 0., 0.]
        assert len(find_self_colliding_faces(new_nodes, faces)) == 0


class TestClampBudgetForConvergence:
    def _facing_pair(self, gap=0.05):
        """Two triangles, each other's only close neighbour, `gap` apart,
        wound so their normals oppose (B's normal is -z, facing back down
        at A's +z) - a genuinely converging pair, not just two nearby
        triangles that happen to be parallel (see
        CONVERGING_DOT_THRESHOLD's own comment for why direction, not
        just proximity, is what this function must key off)."""
        a = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        b = np.array([[0., 0., gap], [1., 0., gap], [0., 1., gap]])
        nodes = np.vstack([a, b])
        faces = np.array([[0, 1, 2], [3, 5, 4]])  # note: 3,5,4 reverses B's winding
        return nodes, faces

    def test_tightens_to_safety_fraction_of_current_gap(self):
        gap = 0.05
        nodes, faces = self._facing_pair(gap)
        budget = np.full(6, np.inf)

        clamp_budget_for_convergence(nodes, faces, budget)

        expected = CONVERGENCE_SAFETY_FRACTION * gap
        assert budget == pytest.approx(np.full(6, expected))

    def test_fraction_is_strictly_below_one_half(self):
        """The whole point of the safety margin: two sides both fully
        spending a budget derived from strictly less than half the gap
        can never meet exactly (see CONVERGENCE_SAFETY_FRACTION's own
        comment) - regression guard against someone "simplifying" this
        back to an exact 0.5 and silently reintroducing exact-coincidence
        convergence."""
        assert CONVERGENCE_SAFETY_FRACTION < 0.5

    def test_never_loosens_an_already_tighter_budget(self):
        """A node frozen by an earlier layer (or by find_self_colliding_
        faces's own freeze) must stay frozen - this function may only
        tighten, never restore, a budget."""
        nodes, faces = self._facing_pair(gap=0.05)
        budget = np.array([0.001] * 3 + [np.inf] * 3)

        clamp_budget_for_convergence(nodes, faces, budget)

        assert budget[0] == pytest.approx(0.001)
        assert budget[3] == pytest.approx(CONVERGENCE_SAFETY_FRACTION * 0.05)

    def test_well_separated_faces_are_not_clamped(self):
        nodes, faces = self._facing_pair(gap=0.05)
        nodes[3:] += 1000.0  # push the second triangle far away
        budget = np.full(6, np.inf)

        clamp_budget_for_convergence(nodes, faces, budget)

        assert np.all(np.isinf(budget))

    def test_close_but_diverging_pair_across_a_convex_edge_is_not_clamped(self):
        """Regression test for a real, serious bug found on cube_demo:
        two small triangles straddling an ordinary CONVEX box edge
        (material at x<0 and y<0, shared edge at x=0,y=0) are close
        together near the shared edge from the very first layer - this is
        normal, correctly-shaped mesh refinement near a feature, not a
        defect - and moving each along its own normal (~(1,0,0) and
        ~(0,1,0)) INCREASES their separation (a convex edge's fronts
        diverge as they extrude, which is what the miter join in
        mesh_layer_step.py exists to handle). Clamping based on proximity
        alone (no directional filter at all) could not tell this apart
        from genuine convergence and froze nodes along essentially every
        edge of the cube - measured directly to produce 131x MORE
        overlapping cells than the unfixed baseline (132,260 vs. 1,004)
        once the resulting frozen, near-duplicate geometry reached
        tetgen. See CONVERGING_CLOSING_RATE_THRESHOLD's own comment -
        including for why a plain normal-vs-normal dot product test
        (tried second) is ALSO not sufficient, just for a narrower case
        this particular fixture doesn't happen to exercise (a sharp
        convex wedge/thin fin).
        """
        a0, a1, a2 = np.array([0., -0.2, 0.]), np.array([0., -0.05, 0.]), np.array([0., -0.125, 0.1])
        b0, b1, b2 = np.array([-0.2, 0., 0.]), np.array([-0.05, 0., 0.]), np.array([-0.125, 0., 0.1])
        nodes = np.array([a0, a1, a2, b0, b1, b2])
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        # Confirm the fixture is actually a close pair (a naive proximity-
        # only version of this function WOULD wrongly clamp it) - not a
        # vacuous test that passes only because nothing was ever nearby.
        centroid_a, centroid_b = nodes[:3].mean(axis=0), nodes[3:].mean(axis=0)
        assert np.linalg.norm(centroid_a - centroid_b) < 0.2
        budget = np.full(6, np.inf)

        clamp_budget_for_convergence(nodes, faces, budget)

        assert np.all(np.isinf(budget))

    def test_sharp_convex_wedge_is_not_clamped(self):
        """Regression test for a second, sneakier false-positive found
        while fixing the convex-edge bug above: a plain dot(normal_a,
        normal_b) < 0 filter (tried as the first fix) is ALSO wrong for a
        sharp CONVEX wedge (e.g. a thin fin or airfoil trailing edge) -
        its two faces have near-OPPOSITE normals purely because the wedge
        angle is acute (a symmetric 10 degree wedge gives dot=-0.98), yet
        the two surfaces genuinely DIVERGE as they extrude outward, same
        as any other convex feature; material thinness doesn't change
        which way the offset surfaces move. The closing-rate test this
        function actually uses gets this right where a plain normal-dot
        test would not (verified directly: +0.35 for this exact fixture,
        see CONVERGING_CLOSING_RATE_THRESHOLD's own comment)."""
        half_angle = np.deg2rad(5.0)
        top_dir = np.array([np.cos(half_angle), np.sin(half_angle), 0.])
        bot_dir = np.array([np.cos(half_angle), -np.sin(half_angle), 0.])
        a0, a1, a2 = 0.8 * top_dir, 1.0 * top_dir, 0.9 * top_dir + np.array([0., 0., 0.1])
        b0, b1, b2 = 0.8 * bot_dir, 1.0 * bot_dir, 0.9 * bot_dir + np.array([0., 0., 0.1])
        nodes = np.array([a0, a1, a2, b0, b1, b2])
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        centroid_a, centroid_b = nodes[:3].mean(axis=0), nodes[3:].mean(axis=0)
        assert np.linalg.norm(centroid_a - centroid_b) < 0.2  # confirm genuinely close
        budget = np.full(6, np.inf)

        clamp_budget_for_convergence(nodes, faces, budget)

        assert np.all(np.isinf(budget))

    def test_already_intersecting_pair_clamps_straight_to_zero(self):
        """A candidate pair that is already (exactly) intersecting isn't
        expected in practice (see the function's own docstring - by
        induction current_nodes is always already collision-free), but
        must be handled defensively (triangle_triangle_min_distance is
        only meaningful for a non-intersecting pair) rather than crash or
        silently skip clamping. Needs a pair that is BOTH overlapping AND
        converging (negative closing rate, see CONVERGING_CLOSING_RATE_
        THRESHOLD) - two tilted triangles crossing through each other,
        verified directly against triangle_triangle_intersect and the
        closing-rate formula before being fixed into this test."""
        a0, a1, a2 = np.array([0., 0., 0.4]), np.array([1., 0., 0.1]), np.array([0., 1., -0.1])
        b0, b1, b2 = np.array([0., 0., -0.1]), np.array([1., 0., -0.1]), np.array([0., 1., 0.1])
        nodes = np.array([a0, a1, a2, b0, b1, b2])
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        budget = np.full(6, np.inf)

        clamp_budget_for_convergence(nodes, faces, budget)

        assert np.all(budget == 0.0)

    def test_empty_face_array_does_not_crash(self):
        nodes = np.zeros((0, 3))
        faces = np.zeros((0, 3), dtype=np.int64)
        budget = np.zeros(0)
        clamp_budget_for_convergence(nodes, faces, budget)  # must not raise


class TestExtrudeLayersNeverProducesASelfIntersectingLayer:
    """End-to-end regression: two flat facing patches with a tight 0.05m
    gap between them (the same class of defect as a body's underbody
    approaching the ground - see mesh_tetgen_core.compute_local_thickness_
    limit's own docstring) extruded toward each other with NO a-priori
    thickness_limit supplied, so the only thing that can prevent the two
    fronts from crossing is the reactive freeze under test. Growth
    parameters are ordinary defaults, not tuned to make this pass - the
    point (per this project's own explicit requirement) is that no layer
    count or growth rate should be able to produce an overlap."""

    def _facing_patches(self, gap=0.05):
        # Patch A: unit square in z=0, wound so its normal is +z.
        a = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
        # Patch B: same footprint at z=gap, wound so its normal is -z
        # (facing directly back down at A).
        b = np.array([[0., 0., gap], [1., 0., gap], [1., 1., gap], [0., 1., gap]])
        nodes = np.vstack([a, b])
        faces = np.array([
            [0, 1, 2], [0, 2, 3],   # A, normal +z
            [4, 6, 5], [4, 7, 6],   # B, normal -z
        ])
        normals = compute_face_normals(nodes, faces)
        assert normals[0][2] == pytest.approx(1.0)
        assert normals[2][2] == pytest.approx(-1.0)
        return nodes, faces, normals

    def test_facing_fronts_never_cross_across_any_layer(self):
        surface_nodes, surface_faces, normals = self._facing_patches(gap=0.05)
        bounding_box = {
            'min': np.array([-10., -10., -10.]),
            'max': np.array([10., 10., 10.]),
        }

        all_nodes, layer_connectivity = extrude_layers(
            surface_nodes, surface_faces, normals, bounding_box,
            growth_rate=1.2, min_cell_size=0.005,
            bl_layers=20,
        )

        n_layers = len(layer_connectivity)
        npl = len(surface_nodes)
        for k in range(n_layers):
            layer_nodes = all_nodes[k * npl:(k + 1) * npl]
            colliding = find_self_colliding_faces(layer_nodes, surface_faces)
            assert len(colliding) == 0, (
                f"layer {k} self-intersects (faces {colliding.tolist()}) - "
                f"the reactive freeze failed to stop the fronts from crossing"
            )
            # A's four nodes (0-3, growing +z) must never overtake B's
            # corresponding four nodes (4-7, growing -z): the physical
            # meaning of "never crossed", checked directly on top of the
            # geometric self-intersection check above.
            assert np.all(layer_nodes[0:4, 2] <= layer_nodes[4:8, 2] + 1e-12)

        # The mechanism must actually have engaged (this gap is tight
        # enough that unconstrained growth at these settings would have
        # closed it well within the 20 available layers) - otherwise the
        # test above would pass vacuously just because nothing ever grew
        # far enough to matter.
        final_a_z = all_nodes[(n_layers - 1) * npl + 0, 2]
        final_b_z = all_nodes[(n_layers - 1) * npl + 4, 2]
        assert final_b_z - final_a_z < 0.05, "fronts should have advanced close to the gap"
        unconstrained_estimate = 0.005 * (1.2 ** (n_layers - 1))
        assert final_a_z < unconstrained_estimate, (
            "front A should have been frozen short of its unconstrained growth"
        )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
