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

from autoflowcfd.core.time_integration.implicit.gmres import (
    SINGLE_BASIS_MIN_RTOL, basis_itemsize, gmres_right, projected_remaining,
)
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


def test_basis_precision_follows_the_tolerance_and_sets_the_restart_length():
    """容差不低于 `SINGLE_BASIS_MIN_RTOL` 时基向量按单精度存；同样的内存预算与下限给出两倍的重启长度。

    cube_demo SA P1 的尺寸（1109.8 万个未知量、灵活模式）：双精度被压在下限 30，单精度 60——同一个 CFL 1500
    的线性系统 GMRES(30) 198 次、GMRES(60) 46 次。"""
    from autoflowcfd.core.time_integration.implicit.jfnk import GMRES_RESTART, GMRES_RESTART_MAX, krylov_restart

    assert basis_itemsize(0.1) == 4 and basis_itemsize(SINGLE_BASIS_MIN_RTOL) == 4
    assert basis_itemsize(0.5 * SINGLE_BASIS_MIN_RTOL) == 8 and basis_itemsize(0.0) == 8
    red = LocalReductions()
    n = 11_097_528
    assert krylov_restart(n, red, flexible=True) == GMRES_RESTART == 30
    assert krylov_restart(n, red, flexible=True, itemsize=4) == 60
    assert krylov_restart(n // 4, red, flexible=False, itemsize=8) == 95
    assert krylov_restart(n // 4, red, flexible=False, itemsize=4) == 192
    assert krylov_restart(1000, red, itemsize=4) == krylov_restart(1000, red) == GMRES_RESTART_MAX


class _DtypeSpy:
    """记录作用时收到的数组精度（算子与预处理只应见到双精度）。"""

    def __init__(self, fn):
        self.fn, self.dtypes = fn, set()

    def __call__(self, v):
        self.dtypes.add(v.dtype)
        return self.fn(v)


@pytest.mark.parametrize("flexible", [True, False])
def test_single_precision_basis_reaches_the_true_residual_on_an_ill_conditioned_system(flexible):
    """条件数 1e6、解的分量比右端项大六个量级的系统，`rtol = 1e-4`：返回的 x 的**真实**残差满足容差，报告的
    相对残差就是真实值。

    单精度基的舍入让真实残差比 Givens 递推值多出 `eps32 * ||y||`：这个系统上递推判定收敛时真实残差是
    3.5e-3（2026-10-12 加核实之前返回的就是它，并报告为已收敛）。核实后接着迭代到真实达标，迭代数比
    双精度基多（正交性差到 1e-7 量级的代价）。"""
    rng = np.random.default_rng(11)
    n = 90
    A = np.diag(np.logspace(0, -6, n)) + 1e-8 * rng.standard_normal((n, n))
    b = rng.standard_normal(n)
    op = _DtypeSpy(lambda v: A @ v)
    prec = _DtypeSpy(lambda v: v.copy())
    x, it, info, rel = gmres_right(op, b, prec, rtol=1e-4, restart=200, max_iter=600, flexible=flexible)
    assert info == 0
    true_rel = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
    assert true_rel <= 1.0001e-4
    assert abs(true_rel - rel) <= 1e-9
    assert np.linalg.norm(x) > 1e3 * np.linalg.norm(b)                 # 解确实比右端项大得多
    assert op.dtypes == {np.dtype(np.float64)} and prec.dtypes == {np.dtype(np.float64)}
    assert x.dtype == np.float64
    # 用满迭代数时报告的也是真实残差
    x2, it2, info2, rel2 = gmres_right(lambda v: A @ v, b, lambda v: v.copy(), rtol=1e-4, restart=200,
                                       max_iter=it - 5, flexible=flexible)
    if info2 > 0:
        assert abs(np.linalg.norm(b - A @ x2) / np.linalg.norm(b) - rel2) <= 0.05 * rel2


def test_tight_tolerance_keeps_the_double_precision_basis():
    """容差低于单精度舍入能支撑的范围时仍用双精度基：解到 1e-12。"""
    A, b, x_true = _system(60, 21, spread=2.0)
    x, _, info, _ = gmres_right(lambda v: A @ v, b, lambda v: v, rtol=1e-12, restart=80, max_iter=200)
    assert info == 0
    assert np.linalg.norm(b - A @ x) <= 1.0001e-12 * np.linalg.norm(b)


def test_projected_remaining_iterations():
    """按最近 10 次迭代的平均下降率外推到容差还要多少次；不降为 inf，已到容差为 0。"""
    geometric = [0.9 ** (i + 1) for i in range(20)]
    expected = np.log(1e-3 / geometric[-1]) / np.log(0.9)
    assert abs(projected_remaining(geometric, 1e-3) - expected) < 1e-9
    assert projected_remaining([0.5] * 20, 0.1) == float("inf")
    assert projected_remaining([0.5, 0.6], 0.1) == float("inf")
    assert projected_remaining([0.5], 0.1) == float("inf")              # 一个点给不出下降率
    assert projected_remaining([0.5, 0.05], 0.1) == 0.0
    # 先平后陡：只看最近的下降率（cube_demo 的曲线，第 30 / 40 / 50 次 0.82 / 0.58 / 0.23）
    late = [0.95] * 30 + list(np.geomspace(0.82, 0.23, 21)[1:])
    assert projected_remaining(late, 0.1) < 15


def _plateau_then_fast_system(n=120, n_slow=40):
    """`n_slow` 个互不相同的小特征值加一簇 1 附近的特征值：GMRES 的残差先平（约 `n_slow` 次）后陡。"""
    rng = np.random.default_rng(4)
    eig = np.concatenate([np.geomspace(1e-3, 1e-1, n_slow), 1.0 + 0.01 * rng.standard_normal(n - n_slow)])
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = q @ np.diag(eig) @ q.T
    return A, q @ np.ones(n)


def test_review_point_keeps_a_solve_that_is_about_to_converge():
    """复核点处外推的剩余迭代数不超过"重来"的代价（`review_at`）时继续求解；超过时按未达到返回。

    2026-10-12 以前迭代到过时预算一律截断、重装配后从头再解：cube_demo SA P1（CFL 1650）在第 57 次迭代、
    线性残差 0.23 处被截断，重装配 60 s 后重新求解 64 次。"""
    A, b = _plateau_then_fast_system()

    def solve(**kw):
        return gmres_right(lambda v: A @ v, b, lambda v: v, rtol=1e-3, restart=200, max_iter=200, **kw)

    _, it_full, info_full, _ = solve()
    assert info_full == 0 and it_full > 40
    # 平台之后、离收敛只差几次的位置复核：继续，与不设复核点逐位相同
    x_keep, it_keep, info_keep, rel_keep = solve(review_at=it_full - 3)
    assert (it_keep, info_keep) == (it_full, 0)
    np.testing.assert_array_equal(x_keep, solve()[0])
    # 平台中段复核：按当前下降率还差得远，返回未达到，迭代数停在复核点
    _, it_cut, info_cut, rel_cut = solve(review_at=10)
    assert it_cut == 10 and info_cut > 0 and rel_cut > 1e-3
