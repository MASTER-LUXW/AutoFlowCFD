"""Unit tests for core/utils/order_continuation.py's vectorized SPs
interpolation matrix.

`_build_linear_interp_matrix_3d` replaced a per-cell, per-variable Python
loop that constructed a fresh `scipy.interpolate.RegularGridInterpolator`
for every (cell, variable) pair - real performance bottleneck on
production-scale meshes (hundreds of thousands of cells) at every Order
Continuation phase transition. The replacement precomputes the linear
interpolation operator once (it only depends on the old/new SPs reference
coordinates, not on cell/variable data) and applies it to all cells/
variables in one vectorized `einsum` call. These tests pin the new
implementation against the original per-cell scipy loop to guarantee the
vectorization didn't change any numerical behavior (including the
`fill_value=None` linear extrapolation and the degenerate P0 single-point
grid case).
"""

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from autoflowcfd.core.utils.order_continuation import _build_linear_interp_matrix_3d
from autoflowcfd.fr.quadrature_points import gauss_legendre


def _reference_interp(U_old: np.ndarray, old_sps_1d: np.ndarray, new_sps_1d: np.ndarray) -> np.ndarray:
    """Original per-cell, per-variable RegularGridInterpolator loop, kept
    here only as an independent oracle for the vectorized replacement."""
    old_n1d = len(old_sps_1d)
    new_n1d = len(new_sps_1d)
    n_cells, old_n_sps, n_vars = U_old.shape
    new_xx, new_yy, new_zz = np.meshgrid(new_sps_1d, new_sps_1d, new_sps_1d, indexing='ij')
    new_pts = np.column_stack([new_xx.ravel(), new_yy.ravel(), new_zz.ravel()])

    new_U = np.zeros((n_cells, new_n1d ** 3, n_vars))
    for i in range(n_cells):
        for v in range(n_vars):
            u_old_3d = U_old[i, :, v].reshape((old_n1d, old_n1d, old_n1d))
            interp = RegularGridInterpolator(
                (old_sps_1d, old_sps_1d, old_sps_1d), u_old_3d,
                method='linear', bounds_error=False, fill_value=None
            )
            new_U[i, :, v] = interp(new_pts)
    return new_U


class TestBuildLinearInterpMatrix3D:
    @pytest.mark.parametrize("old_order,new_order", [(0, 1), (1, 2), (2, 3)])
    def test_matches_per_cell_scipy_loop(self, old_order, new_order):
        old_sps_1d, _ = gauss_legendre(old_order + 1)
        new_sps_1d, _ = gauss_legendre(new_order + 1)

        rng = np.random.default_rng(0)
        n_cells, n_vars = 12, 5
        old_n_sps = (old_order + 1) ** 3
        U_old = rng.standard_normal((n_cells, old_n_sps, n_vars))

        expected = _reference_interp(U_old, old_sps_1d, new_sps_1d)

        W = _build_linear_interp_matrix_3d(old_sps_1d, new_sps_1d)
        actual = np.einsum('ab,cbv->cav', W, U_old)

        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_p0_single_point_grid_broadcasts_constant(self):
        """P0's reference 'grid' is a single point per axis - the matrix
        must broadcast that one value to every new SP, not raise."""
        old_sps_1d, _ = gauss_legendre(1)
        new_sps_1d, _ = gauss_legendre(2)

        W = _build_linear_interp_matrix_3d(old_sps_1d, new_sps_1d)

        assert W.shape == (8, 1)
        np.testing.assert_allclose(W, np.ones((8, 1)))

    def test_identity_when_grids_match(self):
        """Interpolating onto the exact same SPs must reproduce the field
        exactly (the matrix should behave as an identity for this case)."""
        sps_1d, _ = gauss_legendre(2)
        W = _build_linear_interp_matrix_3d(sps_1d, sps_1d)
        np.testing.assert_allclose(W, np.eye(8), atol=1e-12)
