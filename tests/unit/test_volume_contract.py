"""core/fr_operators/volume_contract.py 的 numba adj(J) 核的单元测试。

`compute_adj_j` 取代了 `det_jacs[..., None, None] * inv_jacs`——一个普通的
numpy 广播乘法，不论线程数设置如何都是单线程。在生产规模网格的 P2 过积分
细网格（n_pts=64）上，这一行要处理数 GiB 数据，在一次真实的 cProfile 运行
里占了 `compute_inviscid_residual_fr` 自身（非子函数）时间里可测的一大块。
numba `prange` 核在数学上是完全相同的逐元素乘积，只是跨单元并行计算——
这些测试把它与原来的广播表达式钉成逐位相等。
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
