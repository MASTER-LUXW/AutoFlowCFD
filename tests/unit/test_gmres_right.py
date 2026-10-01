# -*- coding: utf-8 -*-
"""后端无关的重启右预处理 GMRES（`core/time_integration/implicit/gmres.py`）。

判据：
1. 非对称系统上解到指定容差，且容差是**真实**线性残差 `||b - A x||`；
2. 重启（m 小于问题维数）仍收敛到同一解；
3. 右预处理：完美预处理（M = A）一步收敛；对角占优但对角量级差很大的
   系统上，对角预处理显著减少迭代数；
4. `max_iter` 用满返回 `info > 0` 与当时的最好解；非有限值返回 `info < 0`；
5. 归约对象可替换：一个把内积放大 1 倍再缩回的"假分布式"归约给出
   逐位相同的结果（算法只通过归约对象取标量）。
"""

import numpy as np
import pytest

from autoflowcfd.core.time_integration.implicit.gmres import gmres_right
from autoflowcfd.core.time_integration.implicit.reductions import LocalReductions


def _system(n, seed, spread=1.0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) / np.sqrt(n) + np.diag(np.logspace(0, spread, n))
    x_true = rng.standard_normal(n)
    return A, A @ x_true, x_true


@pytest.mark.parametrize("restart", [80, 15])
def test_solves_nonsymmetric_system_to_true_residual_tolerance(restart):
    A, b, x_true = _system(60, 0)
    x, it, info, _rel = gmres_right(lambda v: A @ v, b, lambda v: v, rtol=1e-10,
                              restart=restart, max_iter=2000)
    assert info == 0
    assert np.linalg.norm(b - A @ x) <= 1e-10 * np.linalg.norm(b) * 1.0001
    np.testing.assert_allclose(x, x_true, rtol=1e-7, atol=1e-8)


def test_perfect_preconditioner_converges_in_one_iteration():
    A, b, _ = _system(40, 1)
    Ainv = np.linalg.inv(A)
    x, it, info, _rel = gmres_right(lambda v: A @ v, b, lambda v: Ainv @ v, rtol=1e-10,
                              restart=30, max_iter=100)
    assert info == 0 and it == 1


def test_diagonal_preconditioning_helps_badly_scaled_system():
    A, b, _ = _system(80, 2, spread=5.0)
    d = np.diag(A).copy()
    _, it_plain, info_plain, _rel = gmres_right(lambda v: A @ v, b, lambda v: v, rtol=1e-8,
                                          restart=80, max_iter=400)
    _, it_prec, info_prec, _rel = gmres_right(lambda v: A @ v, b, lambda v: v / d, rtol=1e-8,
                                        restart=80, max_iter=400)
    assert info_prec == 0
    assert it_prec < it_plain


def test_max_iter_reports_positive_info_and_best_solution():
    A, b, _ = _system(60, 3, spread=4.0)
    x, it, info, _rel = gmres_right(lambda v: A @ v, b, lambda v: v, rtol=1e-14,
                              restart=10, max_iter=10)
    assert info > 0 and it == 10
    assert np.linalg.norm(b - A @ x) < np.linalg.norm(b)


def test_non_finite_is_reported():
    A, b, _ = _system(10, 4)
    x, it, info, _rel = gmres_right(lambda v: A @ v * np.nan, b, lambda v: v, rtol=1e-8,
                              restart=5, max_iter=20)
    assert info < 0


def test_reduction_object_is_the_only_source_of_scalars():
    class Doubling(LocalReductions):
        """sum 先乘 2 再除 2：数值等价，但走的是覆盖后的全局归约入口。"""

        def _allreduce_sum(self, value):
            return (2.0 * value) / 2.0

    A, b, _ = _system(50, 5)
    x1, it1, _, _rel = gmres_right(lambda v: A @ v, b, lambda v: v, rtol=1e-9, restart=20, max_iter=500)
    x2, it2, _, _rel = gmres_right(lambda v: A @ v, b, lambda v: v, rtol=1e-9, restart=20, max_iter=500,
                             red=Doubling())
    assert it1 == it2
    np.testing.assert_array_equal(x1, x2)


class _VaryingPreconditioner:
    """每次作用都不同的预处理（第 k 次用 `(1 + 0.3 sin k) D^{-1}` 的近似逆加一个随调用变化的
    低秩扰动）——K 循环那类非线性预处理的最小模型。"""

    def __init__(self, A, seed):
        self.d = np.diag(A).copy()
        self.rng = np.random.default_rng(seed)
        self.k = 0

    def __call__(self, v):
        self.k += 1
        u = self.rng.standard_normal(v.size)
        return (1.0 + 0.3 * np.sin(self.k)) * v / self.d + 0.2 * u * (u @ v) / v.size


@pytest.mark.parametrize("restart", [200, 12])
def test_flexible_mode_reaches_true_residual_with_varying_preconditioner(restart):
    """灵活模式下返回的 x 真实满足容差；同一预处理走非灵活模式时 `x = M^{-1}(V y)` 用的是
    又一次不同的作用，Givens 递推报告的残差与真实残差脱节。"""
    A, b, _ = _system(80, 7, spread=3.0)
    x, it, info, rel = gmres_right(lambda v: A @ v, b, _VaryingPreconditioner(A, 0), rtol=1e-9,
                                   restart=restart, max_iter=2000, flexible=True)
    assert info == 0
    assert np.linalg.norm(b - A @ x) <= 1.0001e-9 * np.linalg.norm(b)
    x2, _, info2, rel2 = gmres_right(lambda v: A @ v, b, _VaryingPreconditioner(A, 0), rtol=1e-9,
                                     restart=200, max_iter=200, flexible=False)
    assert np.linalg.norm(b - A @ x2) > 1e3 * max(rel2, 1e-9) * np.linalg.norm(b)


def test_flexible_mode_matches_standard_mode_for_linear_preconditioner():
    A, b, _ = _system(60, 8, spread=2.0)
    d = np.diag(A).copy()
    out_std = gmres_right(lambda v: A @ v, b, lambda v: v / d, rtol=1e-10, restart=15, max_iter=500)
    out_flex = gmres_right(lambda v: A @ v, b, lambda v: v / d, rtol=1e-10, restart=15, max_iter=500,
                           flexible=True)
    assert out_std[1] == out_flex[1]
    np.testing.assert_allclose(out_flex[0], out_std[0], rtol=1e-9, atol=1e-12)
