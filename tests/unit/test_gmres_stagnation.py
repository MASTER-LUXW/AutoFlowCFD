"""GMRES 停滞提前终止（`implicit/krylov/gmres.py` 模块文档同名一节）。"""

import numpy as np

from autoflowcfd.core.time_integration.implicit.krylov.gmres import STAGNATION_WINDOW, _stagnated, gmres_right


def _identity(v):
    return v


def test_indefinite_rotation_stops_after_one_window_instead_of_the_full_budget():
    """循环移位算子：GMRES 在前 n-1 次迭代里残差完全不降（教科书里的最坏情形）。预算 400 次时停滞判据在
    窗口刚满时终止，返回 info > 0 与如实的线性残差。"""
    n = 600
    b = np.zeros(n)
    b[0] = 1.0
    calls = []

    def shift(v):
        calls.append(1)
        return np.roll(v, 1)

    _x, iters, info, rel = gmres_right(shift, b, _identity, rtol=0.1, restart=450, max_iter=400)
    assert info > 0 and rel > 0.99
    assert iters == STAGNATION_WINDOW + 1 and len(calls) == iters


def test_slow_but_sufficient_convergence_is_not_cut_short():
    """每 50 次迭代降到 0.7 倍：到 rtol=0.3 需要约 170 次。按当前速率到得了，就不终止。"""
    assert not _stagnated([0.7 ** (i / 50.0) for i in range(1, 120)], rtol=0.3, max_iter=400)
    # 同样的速率、预算只剩 30 次：窗口比值 0.7 < 0.98，仍不算停滞（判据只管"几乎不降"）
    assert not _stagnated([0.7 ** (i / 50.0) for i in range(1, 371)], rtol=1e-6, max_iter=400)
    # 几乎不降、但离容差只差一点、剩余预算够：不终止
    assert not _stagnated([0.101 * 0.999 ** (i / 50.0) for i in range(1, 60)], rtol=0.1, max_iter=4000)
    # 几乎不降、到不了：终止
    assert _stagnated([0.999 ** (i / 50.0) for i in range(1, 60)], rtol=0.1, max_iter=400)
    # 窗口没满不判
    assert not _stagnated([1.0] * STAGNATION_WINDOW, rtol=0.1, max_iter=400)


def test_well_conditioned_system_is_unchanged():
    rng = np.random.default_rng(0)
    a = np.eye(80) * 4.0 + 0.3 * rng.standard_normal((80, 80))
    b = rng.standard_normal(80)
    x, iters, info, rel = gmres_right(lambda v: a @ v, b, _identity, rtol=1e-10, restart=80, max_iter=200)
    assert info == 0 and iters < STAGNATION_WINDOW
    np.testing.assert_allclose(a @ x, b, atol=1e-8)
