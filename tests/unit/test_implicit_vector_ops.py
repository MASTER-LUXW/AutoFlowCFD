# -*- coding: utf-8 -*-
"""隐式求解的并行长向量运算（numpy 后端走 numba 并行路径）与 numpy 参照一致。"""

import numpy as np
import pytest

from autoflowcfd.core.time_integration.implicit import vector_ops


def test_parallel_dot_matches_numpy():
    rng = np.random.default_rng(0)
    n = 4 * vector_ops._PARALLEL_MIN + 7
    a, b = rng.standard_normal(n), rng.standard_normal(n)
    assert vector_ops.dot(np, a, b) == pytest.approx(float(np.dot(a, b)), rel=1e-12)


def test_parallel_axpy_is_in_place_and_matches_numpy():
    rng = np.random.default_rng(1)
    n = 3 * vector_ops._PARALLEL_MIN + 5
    y, x = rng.standard_normal((n // 5 + 1, 5)), rng.standard_normal((n // 5 + 1, 5))
    ref = y + 0.37 * x
    alias = y
    vector_ops.axpy_(np, y, 0.37, x)
    assert alias is y
    np.testing.assert_allclose(y, ref, rtol=0, atol=1e-15)


def test_axpy_refuses_non_contiguous_target():
    y = np.zeros((2 * vector_ops._PARALLEL_MIN, 2))[:, 0]
    with pytest.raises(ValueError):
        vector_ops.axpy_(np, y, 1.0, np.ones_like(y))
