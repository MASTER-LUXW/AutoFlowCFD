"""Unit tests for face_flux_points_exact_normal.py — the real fix for
core/fr_residual/inviscid_kernel.py's `true_normal`/`true_area_weight`
previously being one constant (flat, triangulated) value per face, reused
across every Flux Point on that face regardless of whether the face
(a prism quadrilateral side) is actually planar. See the module docstring
for the full mechanism.

These tests validate the core mathematical claims before the function is
wired into the real geometry pipeline:
1. For a PLANAR face (any tet face, or a prism quad whose 4 corners
   happen to be coplanar), the new per-FP computation must reduce to a
   SINGLE constant normal across all FPs, matching a hand-computed flat
   normal/area exactly (not approximately) — the batched exact-Jacobian
   route must not silently change results on the huge majority of faces
   where the old flat approximation was already exact.
2. For a genuinely WARPED (non-planar) prism quad, the normal must vary
   meaningfully across FPs (this is the actual bug being fixed — the old
   code used one constant value here too).
3. The batched Jacobian helpers must agree exactly with the existing,
   already-validated single-point `tet_exact_jacobian`/
   `prism_exact_jacobian` (curved_mapping_exact_jacobian.py) - the batched
   versions are a vectorization of the identical formulas, not a
   reimplementation, and must be bit-for-bit consistent on overlapping
   inputs (up to floating-point associativity).
"""

import numpy as np
import pytest

from autoflowcfd.fr.face_flux_points_exact_normal import (
    compute_exact_face_normals_and_weights,
    compute_exact_adj_rows,
    _tet_exact_jacobian_batched,
    _prism_exact_jacobian_batched,
)
from autoflowcfd.grid.curved_mapping.curved_mapping_exact_jacobian import (
    tet_exact_jacobian, prism_exact_jacobian,
)
from autoflowcfd.fr.face_flux_points import face_ref_grid
from autoflowcfd.fr.quadrature_points import gauss_legendre


class TestBatchedJacobianMatchesSinglePointVersion:
    def test_tet_batched_matches_loop_of_single_point(self):
        rng = np.random.default_rng(0)
        n_faces, n_fp = 5, 4
        ref_pts = rng.uniform(-1, 1, size=(n_fp, 3))
        cell_nodes = rng.uniform(-1, 1, size=(n_faces, 4, 3))

        batched = _tet_exact_jacobian_batched(ref_pts, cell_nodes)
        for f in range(n_faces):
            expected = tet_exact_jacobian(ref_pts, cell_nodes[f])
            np.testing.assert_allclose(batched[f], expected, rtol=1e-13, atol=1e-13)

    def test_prism_batched_matches_loop_of_single_point(self):
        rng = np.random.default_rng(1)
        n_faces, n_fp = 5, 4
        ref_pts = rng.uniform(-1, 1, size=(n_fp, 3))
        cell_nodes = rng.uniform(-1, 1, size=(n_faces, 6, 3))

        batched = _prism_exact_jacobian_batched(ref_pts, cell_nodes)
        for f in range(n_faces):
            expected = prism_exact_jacobian(ref_pts, cell_nodes[f])
            np.testing.assert_allclose(batched[f], expected, rtol=1e-13, atol=1e-13)


def _n1d_and_grids(order):
    n1d = order + 1
    sps_1d, weights_1d = gauss_legendre(n1d)
    return n1d, sps_1d, weights_1d


class TestPlanarFacesReduceToConstantNormal:
    def test_tet_face_gives_constant_normal_matching_flat_geometry(self):
        """A regular tet's 'a=-1' face (nodes 0,2,3, per TET_CUBE_FACES) is
        exactly planar by construction — every FP must get the identical
        normal, equal to the standard cross-product flat normal."""
        order = 2
        n1d, sps_1d, weights_1d = _n1d_and_grids(order)
        n_fp = n1d * n1d

        cell_nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        owner_cell = np.array([0], dtype=np.int64)
        owner_axis = np.array([0], dtype=np.int64)   # a
        owner_side = np.array([-1.0])                 # a=-1 -> TET_CUBE_FACES nodes (0,2,3)

        true_normal, true_area_weight = compute_exact_face_normals_and_weights(
            n_faces=1, n1d=n1d, sps_1d=sps_1d, weights_1d=weights_1d, n_prism=0,
            owner_cell=owner_cell, owner_axis=owner_axis, owner_side=owner_side,
            prism_conn=np.empty((0, 6), dtype=np.int64),
            tet_conn=np.array([[0, 1, 2, 3]], dtype=np.int64),
            node_coords=cell_nodes,
        )

        # every FP on this planar face must have the identical normal
        n = true_normal[0]
        np.testing.assert_allclose(n, np.tile(n[0], (n_fp, 1)), atol=1e-12)

        # face (0,2,3) in physical space: (0,0,0),(0,1,0),(0,0,1) - the x=0 plane,
        # outward normal (away from node 1 at x=1) must be -x direction.
        np.testing.assert_allclose(n[0], [-1.0, 0.0, 0.0], atol=1e-10)

        # total area must equal the true flat triangle area (0.5 for this right triangle)
        assert true_area_weight[0].sum() == pytest.approx(0.5, rel=1e-10)

    def test_planar_prism_quad_gives_constant_normal(self):
        """A rectangular-box prism's quad side is exactly planar - all FPs
        must agree on the same normal/direction, matching the flat
        cross-product normal of that rectangle."""
        order = 2
        n1d, sps_1d, weights_1d = _n1d_and_grids(order)
        n_fp = n1d * n1d

        # v0,v1,v2 bottom triangle; w0,w1,w2 top - an axis-aligned box,
        # side "a=-1" -> PRISM_CUBE_FACES nodes (0,2,5,3) = v0,v2,w2,w0.
        cell_nodes = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1],
        ], dtype=float)
        owner_cell = np.array([0], dtype=np.int64)
        owner_axis = np.array([0], dtype=np.int64)
        owner_side = np.array([-1.0])

        true_normal, true_area_weight = compute_exact_face_normals_and_weights(
            n_faces=1, n1d=n1d, sps_1d=sps_1d, weights_1d=weights_1d, n_prism=1,
            owner_cell=owner_cell, owner_axis=owner_axis, owner_side=owner_side,
            prism_conn=np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64),
            tet_conn=np.empty((0, 4), dtype=np.int64),
            node_coords=cell_nodes,
        )

        n = true_normal[0]
        np.testing.assert_allclose(n, np.tile(n[0], (n_fp, 1)), atol=1e-10)
        # quad v0(0,0,0)-v2(0,1,0)-w2(0,1,1)-w0(0,0,1) is the x=0 plane;
        # outward (away from v1 at x=1) is -x.
        np.testing.assert_allclose(n[0], [-1.0, 0.0, 0.0], atol=1e-10)
        # unit square face area = 1.0
        assert true_area_weight[0].sum() == pytest.approx(1.0, rel=1e-10)


class TestWarpedPrismQuadVariesAcrossFluxPoints:
    def test_non_planar_quad_normal_is_not_constant(self):
        """Displace one corner of the quad side out of plane - the old
        code's single constant normal is exactly the bug being fixed;
        the new per-FP computation must show real variation across FPs
        (that's the whole point of this change)."""
        order = 2
        n1d, sps_1d, weights_1d = _n1d_and_grids(order)

        cell_nodes = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1],
        ], dtype=float)
        # Warp the quad side "a=-1" = (v0,v2,w2,w0) by pushing w0 (node 3)
        # out of the x=0 plane - v0,v2,w2 stay at x=0, w0 moves to x=0.4.
        cell_nodes[3] = [0.4, 0.0, 1.0]

        owner_cell = np.array([0], dtype=np.int64)
        owner_axis = np.array([0], dtype=np.int64)
        owner_side = np.array([-1.0])

        true_normal, _ = compute_exact_face_normals_and_weights(
            n_faces=1, n1d=n1d, sps_1d=sps_1d, weights_1d=weights_1d, n_prism=1,
            owner_cell=owner_cell, owner_axis=owner_axis, owner_side=owner_side,
            prism_conn=np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64),
            tet_conn=np.empty((0, 4), dtype=np.int64),
            node_coords=cell_nodes,
        )

        n = true_normal[0]
        max_pairwise_diff = np.max(np.linalg.norm(n[:, None, :] - n[None, :, :], axis=-1))
        assert max_pairwise_diff > 1e-3, (
            "expected meaningfully different normals across FPs on a warped "
            f"quad, got max pairwise difference {max_pairwise_diff:.3e}"
        )
        # every direction must still be a unit vector
        np.testing.assert_allclose(np.linalg.norm(n, axis=-1), 1.0, atol=1e-10)


class TestComputeExactAdjRowsValidMask:
    """`compute_exact_adj_rows` is the shared primitive behind both
    `true_normal` (owner side, side-oriented+normalized) and the
    self-consistent-direction fix (owner AND neighbor side, raw/
    unnormalized) - this pins its `valid_mask` behaviour, which the
    neighbor-side callers rely on to skip boundary faces (whose
    neighbor_axis/neighbor_side are -1/0.0 sentinels, not real values)."""

    def test_invalid_entries_stay_zero(self):
        order = 1
        n1d, sps_1d, weights_1d = _n1d_and_grids(order)
        cell_nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)

        cell_arr = np.array([0, 0], dtype=np.int64)
        axis_arr = np.array([0, -1], dtype=np.int64)  # face 1: sentinel axis (boundary face, no neighbor)
        side_arr = np.array([-1.0, 0.0])
        valid_mask = np.array([True, False])

        adj_row = compute_exact_adj_rows(
            n_faces=2, n1d=n1d, sps_1d=sps_1d, n_prism=0,
            cell_arr=cell_arr, axis_arr=axis_arr, side_arr=side_arr,
            prism_conn=np.empty((0, 6), dtype=np.int64),
            tet_conn=np.array([[0, 1, 2, 3]], dtype=np.int64),
            node_coords=cell_nodes,
            valid_mask=valid_mask,
        )

        assert not np.allclose(adj_row[0], 0.0)  # valid entry: real (nonzero) adj(J) row
        np.testing.assert_array_equal(adj_row[1], 0.0)  # invalid entry: left untouched at zero

    def test_raw_row_is_unnormalized_and_not_side_oriented(self):
        """Unlike `compute_exact_face_normals_and_weights`'s `true_normal`,
        this is the *raw* adj(J) row - not a unit vector, not flipped by
        `side` - matching what inviscid_kernel.py's `a0,a1,a2` (before its
        own local `*oside` direction correction) currently gets from SP
        extrapolation."""
        order = 1
        n1d, sps_1d, weights_1d = _n1d_and_grids(order)
        cell_nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)

        adj_row = compute_exact_adj_rows(
            n_faces=1, n1d=n1d, sps_1d=sps_1d, n_prism=0,
            cell_arr=np.array([0], dtype=np.int64),
            axis_arr=np.array([0], dtype=np.int64),
            side_arr=np.array([-1.0]),
            prism_conn=np.empty((0, 6), dtype=np.int64),
            tet_conn=np.array([[0, 1, 2, 3]], dtype=np.int64),
            node_coords=cell_nodes,
        )
        # face (0,2,3) of this tet is the x=0 plane; raw adj(J) row for
        # axis=0 points in +x (into the cell, from node 1 at x=1) - the
        # *outward* direction (-x, matching true_normal's test above)
        # only appears after the caller's own `*side` correction.
        row = adj_row[0, 0]
        assert row[0] > 0  # NOT yet flipped to outward (-x) - raw value
        assert not np.isclose(np.linalg.norm(row), 1.0)  # NOT unit-normalized
