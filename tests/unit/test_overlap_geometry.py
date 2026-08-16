"""Unit tests for validation/overlap_geometry.py's triangle-triangle
intersection test - no permanent test file existed for this module before
(the "15 hand-built edge cases + 3000-case stress test" mentioned in
ProjectFiles Part5 P2 were one-off validation scripts, never committed).

The thin-sliver-triangle regression cases here are not synthetic worst
cases - they were extracted directly from a real cube_demo BL extrusion
run (two triangles from different original surface faces, each correctly
shaped on its own, whose sharp-corner miter compensation left them thin
and similarly oriented) and confirmed as false positives via independent
brute-force point sampling before triangle_triangle_intersect was fixed.
"""

import numpy as np
import pytest

from autoflowcfd.grid.validation.overlap_geometry import (
    triangle_triangle_intersect,
    triangle_triangle_min_distance,
)

# Two triangles that genuinely cross in 3D, sharing no vertices (same
# fixture used throughout this project's mesh_front_collision tests).
A0, A1, A2 = np.array([-2., -2., 0.]), np.array([2., -2., 0.]), np.array([0., 2., 0.])
B0, B1, B2 = np.array([0., 0., -2.]), np.array([0., 0., 2.]), np.array([0., 3., 0.])

# The real cube_demo thin-sliver false-positive pair: two thin "fin"
# triangles from different original surface faces, offset by a genuine,
# unambiguous 0.01m gap along z (verified via independent brute-force
# sampling: true minimum distance ~0.01, nowhere near eps) - yet flagged
# as intersecting before the fix, because their near-degenerate shape
# makes each one's own plane only weakly sensitive to a real offset along
# the triangle's own long axis (see triangle_triangle_intersect's own
# "Thin-sliver-triangle correction" comment for the full mechanism).
SLIVER_A = np.array([
    [0.503, 0.241369, -0.055],
    [0.503, 0.24137, -0.045],
    [0.5025455844122716, 0.2525455844122716, -0.05],
])
SLIVER_B = np.array([
    [0.503, 0.241371, -0.035],
    [0.503, 0.24137200000000003, -0.025],
    [0.5025455844122716, 0.2525455844122716, -0.03],
])


def _intersects(p, q):
    return bool(triangle_triangle_intersect(
        p[0][None], p[1][None], p[2][None], q[0][None], q[1][None], q[2][None],
    )[0])


class TestTriangleTriangleIntersect:
    def test_crossing_triangles_are_detected(self):
        assert _intersects(np.array([A0, A1, A2]), np.array([B0, B1, B2]))

    def test_well_separated_triangles_are_not_flagged(self):
        far = np.array([B0, B1, B2]) + 100.0
        assert not _intersects(np.array([A0, A1, A2]), far)

    def test_coplanar_overlapping_triangles_are_detected(self):
        a = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
        b = np.array([[0.5, 0.5, 0.], [2.5, 0.5, 0.], [0.5, 2.5, 0.]])
        assert _intersects(a, b)

    def test_coplanar_non_overlapping_triangles_are_not_flagged(self):
        a = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        b = np.array([[10., 10., 0.], [11., 10., 0.], [10., 11., 0.]])
        assert not _intersects(a, b)

    def test_thin_sliver_triangles_with_a_real_gap_are_not_flagged(self):
        """Regression test for the real false positive found on
        cube_demo. The two triangles' z-extents are disjoint by a clear
        0.01m gap (10 million times larger than any float64 noise) -
        confirmed via triangle_triangle_min_distance and independent
        brute-force point sampling before the fix; the un-fixed function
        reported this pair as intersecting anyway."""
        assert not _intersects(SLIVER_A, SLIVER_B)

        dist = triangle_triangle_min_distance(
            SLIVER_A[0][None], SLIVER_A[1][None], SLIVER_A[2][None],
            SLIVER_B[0][None], SLIVER_B[1][None], SLIVER_B[2][None],
        )[0]
        assert dist == pytest.approx(0.01, abs=1e-4)

    def test_genuine_intersection_still_detected_regardless_of_correction(self):
        """The thin-sliver correction (triangle_triangle_min_distance as
        a second opinion) must only ever turn a false positive into a
        correct negative - a genuine, unambiguous intersection (min
        distance 0) must still be reported as True."""
        assert _intersects(np.array([A0, A1, A2]), np.array([B0, B1, B2]))
        dist = triangle_triangle_min_distance(
            A0[None], A1[None], A2[None], B0[None], B1[None], B2[None],
        )
        # Precondition-violating call (the pair DOES intersect) - only
        # used here to confirm it's not spuriously large; not a
        # meaningful "distance" for a genuine overlap (see that
        # function's own docstring).
        assert dist[0] < 1.0

    def test_shared_vertex_is_not_reported_as_intersecting(self):
        """Two triangles touching only at a single shared vertex have a
        zero-measure overlap interval - not reported as an intersection
        by this function's own construction (callers apply a separate
        node-sharing pre-filter for mesh purposes, but the geometric
        primitive itself must also behave this way at the exact
        boundary)."""
        shared = np.array([0., 0., 0.])
        a = np.array([shared, [1., 0., 0.], [0., 1., 0.]])
        b = np.array([shared, [-1., 0., 0.], [0., -1., 0.]])
        assert not _intersects(a, b)

    def test_vectorized_batch_matches_per_row_results(self):
        a_batch = np.stack([A0, SLIVER_A[0], np.array([0., 0., 0.])])
        a1_batch = np.stack([A1, SLIVER_A[1], np.array([1., 0., 0.])])
        a2_batch = np.stack([A2, SLIVER_A[2], np.array([0., 1., 0.])])
        b_batch = np.stack([B0, SLIVER_B[0], np.array([10., 10., 0.])])
        b1_batch = np.stack([B1, SLIVER_B[1], np.array([11., 10., 0.])])
        b2_batch = np.stack([B2, SLIVER_B[2], np.array([10., 11., 0.])])

        result = triangle_triangle_intersect(a_batch, a1_batch, a2_batch, b_batch, b1_batch, b2_batch)

        assert list(result) == [True, False, False]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
