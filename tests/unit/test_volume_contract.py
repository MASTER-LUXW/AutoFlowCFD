"""Unit tests for core/fr_operators/volume_contract.py's numba adj(J) kernel.

`compute_adj_j` replaced `det_jacs[..., None, None] * inv_jacs` - a plain
numpy broadcast multiply that runs single-threaded regardless of thread
count settings. On the P2 over-integration fine grid (n_pts=64) for a
production-scale mesh this single line processes several GiB of data and
showed up as a real, measurable chunk of `compute_inviscid_residual_fr`'s
own (non-sub-function) time in a real cProfile run. The numba `prange`
kernel is mathematically the exact same elementwise product, just computed
in parallel across cells - these tests pin it against the original
broadcast expression for bit-exact equivalence.
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.volume_contract import compute_adj_j


class TestComputeAdjJ:
    @pytest.mark.parametrize("n_cells,n_pts", [(1, 1), (10, 8), (50, 27), (5, 64)])
    def test_matches_broadcast_multiply(self, n_cells, n_pts):
        rng = np.random.default_rng(0)
        det_jacs = rng.standard_normal((n_cells, n_pts))
        inv_jacs = rng.standard_normal((n_cells, n_pts, 3, 3))

        expected = det_jacs[..., None, None] * inv_jacs
        actual = compute_adj_j(det_jacs, inv_jacs)

        np.testing.assert_array_equal(actual, expected)

    def test_output_shape(self):
        det_jacs = np.ones((7, 3))
        inv_jacs = np.zeros((7, 3, 3, 3))
        out = compute_adj_j(det_jacs, inv_jacs)
        assert out.shape == (7, 3, 3, 3)
