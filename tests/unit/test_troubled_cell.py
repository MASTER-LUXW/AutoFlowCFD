"""Unit tests for core/fr_operators/troubled_cell.py's numba median kernel.

`_median_abs_over_sps_kernel` replaced `np.median(np.abs(residual), axis=1)`
inside `suppress_residual_outliers` - real perf bottleneck on production
meshes (numpy's generic n-dimensional `partition`-based reduction pays a lot
of per-call dispatch overhead when reducing millions of tiny (n_sps,) groups
independently). These tests pin the numba kernel against `np.median` to
guarantee it is bit-exact, not an approximation, across both parities of
`np.median`'s definition (odd n_sps takes the middle value, even n_sps
averages the two middle values) - P0/P2/P3 have odd n_sps ((order+1)^3 for
even order+1... actually (order+1)^3 is odd only when order+1 is odd, i.e.
order even), P1/P3 have even/odd depending on order+1's parity, so all four
production orders are covered explicitly below.
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.troubled_cell import (
    _median_abs_over_sps_kernel,
    precompute_cell_face_misalignment,
    suppress_residual_outliers,
)


class TestMedianAbsOverSpsKernel:
    @pytest.mark.parametrize("n_sps", [1, 8, 27, 64])  # P0, P1, P2, P3
    def test_matches_numpy_median(self, n_sps):
        rng = np.random.default_rng(0)
        residual = rng.standard_normal((500, n_sps, 5)) * 10.0

        expected = np.median(np.abs(residual), axis=1)
        actual = _median_abs_over_sps_kernel(residual)

        np.testing.assert_array_equal(actual, expected)

    def test_matches_numpy_median_with_negative_and_zero_values(self):
        """Explicit small case covering the sign-flip and zero-value paths
        in the kernel's inline abs()."""
        residual = np.array([
            [[-3.0], [1.0], [0.0], [-2.0]],  # even n_sps=4 -> average of 2 middle
        ])
        expected = np.median(np.abs(residual), axis=1)
        actual = _median_abs_over_sps_kernel(residual)
        np.testing.assert_array_equal(actual, expected)


class TestSuppressResidualOutliers:
    def test_no_outliers_returns_input_unchanged(self):
        rng = np.random.default_rng(2)
        residual = rng.standard_normal((50, 8, 5)) * 1e-3
        reference = np.ones((50, 8, 5))
        out = suppress_residual_outliers(residual, reference)
        np.testing.assert_array_equal(out, residual)

    def test_single_outlier_sp_zeroed_siblings_untouched(self):
        n_cells, n_sps, n_vars = 4, 8, 5
        rng = np.random.default_rng(3)
        residual = rng.standard_normal((n_cells, n_sps, n_vars)) * 1e-3
        reference = np.ones((n_cells, n_sps, n_vars))
        # Inject one wildly-out-of-scale value in cell 0, SP 0, var 1.
        residual[0, 0, 1] = 1e5
        out = suppress_residual_outliers(residual, reference)
        assert out[0, 0, 1] == 0.0
        # Every other entry must be untouched.
        mask = np.ones_like(residual, dtype=bool)
        mask[0, 0, 1] = False
        np.testing.assert_array_equal(out[mask], residual[mask])


def _reference_cell_face_misalignment(mesh):
    """Original per-face Python-loop implementation, kept only as an
    independent oracle for the numba-based replacement (real perf bug: this
    version indexes `mesh.face_flux_points[f]` for every face, which since
    the flat-array refactor lazily *constructs* a full FaceFluxPointGeometry
    object per access - 1.88M such constructions measured to cost minutes on
    a production mesh)."""
    fc = mesh.face_connectivity
    ffp_list = mesh.face_flux_points
    n_prism = mesh.n_prism_cells
    ops = mesh.operators
    det_jacs = mesh.jacobians["det_jacs"].reshape(mesh.n_cells, -1)
    inv_jacs = mesh.jacobians["inv_jacs"].reshape(mesh.n_cells, -1, 3, 3)
    adj_j = det_jacs[..., None, None] * inv_jacs

    def extrap_to_face(cell, field, axis, side):
        E = ops.boundary_extrap_prism[(axis, side)] if cell < n_prism else ops.boundary_extrap_tet[(axis, side)]
        trailing = field.shape[1:]
        flat = E @ field.reshape(field.shape[0], -1)
        return flat.reshape((E.shape[0],) + trailing)

    def own_dir_outward(cell, axis, side):
        row = extrap_to_face(cell, adj_j[cell][:, axis, :], axis, side)
        mag = np.linalg.norm(row, axis=-1)
        return (row / np.maximum(mag[:, None], 1e-300)) * side

    cell_misalign = np.zeros(mesh.n_cells)
    for f in range(fc.n_faces):
        ffp = ffp_list[f]
        if ffp.owner_is_primary:
            owner_cell = int(fc.owner_cell[f])
            d = own_dir_outward(owner_cell, ffp.owner_axis, ffp.owner_side)
            misalign = 1.0 - np.sum(d * ffp.true_normal, axis=-1)
            cell_misalign[owner_cell] = max(cell_misalign[owner_cell], float(misalign.max()))
        if (not fc.is_boundary[f]) and ffp.neighbor_is_primary:
            neighbor_cell = int(fc.neighbor_cell[f])
            d = own_dir_outward(neighbor_cell, ffp.neighbor_axis, ffp.neighbor_side)
            misalign = 1.0 - np.sum(d * (-ffp.true_normal), axis=-1)
            cell_misalign[neighbor_cell] = max(cell_misalign[neighbor_cell], float(misalign.max()))
    return cell_misalign


class TestPrecomputeCellFaceMisalignment:
    @pytest.mark.parametrize("order", [0, 1, 2, 3])
    def test_matches_per_face_reference_loop(self, order):
        from .test_fr_residual_inviscid import _build_synthetic_mixed_mesh

        mesh = _build_synthetic_mixed_mesh(order)
        actual = precompute_cell_face_misalignment(mesh)
        expected = _reference_cell_face_misalignment(mesh)
        np.testing.assert_allclose(actual, expected, atol=1e-10)
