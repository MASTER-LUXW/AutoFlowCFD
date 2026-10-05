# -*- coding: utf-8 -*-
"""块 ILU 预处理的分解跨步复用（`block_jacobi.py::BlockJacobiCache.preconditioner`，2026-10-05）。

此前每次 `preconditioner()` 都重新分解：湍流平板 P3（3072 单元）早期每个 Newton 步 GMRES 只 1 次，
分解却占步耗时 75%。现在 `J_cc` 不变时复用分解，直到 dtau 的几何平均比值超过 `DTAU_REBUILD_RATIO`
（伪时间项在两个时刻都已可忽略时不论怎么变都复用）；重装配 `J_cc` 后必然重新分解。
"""

import numpy as np
import pytest

from autoflowcfd.core.time_integration.implicit import block_jacobi as BJ
from tests.unit.test_block_ilu_precond import _setup


@pytest.fixture
def cache_and_counter(monkeypatch):
    from autoflowcfd.fr.native_padding import real_sps_per_cell

    mesh, res, U, r0, bp, bt, cp, n_real, colors, diag, off = _setup("tet", 1)
    npr, nte = real_sps_per_cell(1)
    cache = BJ.BlockJacobiCache(cell_is_prism=np.arange(mesh.n_cells) < mesh.n_prism_cells, colors=colors,
                                n_sps=mesh.n_sps_per_cell, n_real_prism=npr, n_real_tet=nte, n_var=5)
    assert cache.use_ilu, "本测试需要块 ILU 档"

    class _Assembler:
        want_coupling = True

        def __call__(self, u0, r0_):
            return bp, bt, cp

    cache.assembler = _Assembler()
    from autoflowcfd.core.time_integration.implicit.coarse.selection import CoarsePreconditionerFactory

    calls = {"n": 0}
    make = CoarsePreconditionerFactory.make

    def counting(self, *a, **k):
        calls["n"] += 1
        return make(self, *a, **k)

    monkeypatch.setattr(CoarsePreconditionerFactory, "make", counting)
    n_rows = mesh.n_cells * mesh.n_sps_per_cell
    u0 = U.reshape(-1, 5)
    cache.begin_step(res, u0, r0, np.ones(5), np.full(n_rows, 1e-6))
    return cache, calls, n_rows, (res, u0, r0)


def test_factorization_is_reused_until_dtau_drifts(cache_and_counter):
    cache, calls, n, _ = cache_and_counter
    p1 = cache.preconditioner(np.full(n, 1e-6), 5)
    assert cache.preconditioner(np.full(n, 1.5e-6), 5) is p1            # 1.5 倍：复用
    assert cache.preconditioner(np.full(n, 0.6e-6), 5) is p1            # 1/1.67 倍：复用
    assert calls["n"] == 1
    p2 = cache.preconditioner(np.full(n, 2.5e-6), 5)                    # 2.5 倍：重分解
    assert p2 is not p1 and calls["n"] == 2
    assert cache.preconditioner(np.full(n, 4.0e-6), 5) is p2            # 相对新基准 1.6 倍：复用


def test_negligible_pseudo_time_term_never_triggers_a_refactorization(cache_and_counter):
    """`1/dtau` 相对 `J_cc` 对角在两个时刻都低于 `PTC_NEGLIGIBLE` 时，dtau 翻倍也复用。"""
    cache, calls, n, _ = cache_and_counter
    big = 1e6 / cache._ptc_relative(np.ones(n))                          # 让 1/(dtau |J_rr|) ~ 1e-6
    assert cache._ptc_relative(np.full(n, big)) < BJ.PTC_NEGLIGIBLE
    p = cache.preconditioner(np.full(n, big), 5)
    for k in range(1, 6):
        assert cache.preconditioner(np.full(n, big * 2.0 ** k), 5) is p
    assert calls["n"] == 1


def test_reassembly_discards_the_factorization(cache_and_counter):
    cache, calls, n, (res, u0, r0) = cache_and_counter
    p = cache.preconditioner(np.full(n, 1e-6), 5)
    cache.refresh(res, u0, r0, np.ones(5), np.full(n, 1e-6))
    assert cache._factored is None
    assert cache.preconditioner(np.full(n, 1e-6), 5) is not p and calls["n"] == 2


def test_reused_factorization_still_solves_the_exact_system(cache_and_counter):
    """复用的（dtau 略过时的）分解仍是合法预处理子：右预处理 GMRES 收敛到同一个解。"""
    from autoflowcfd.core.time_integration.implicit.gmres import gmres_right

    cache, calls, n, _ = cache_and_counter
    dtau_old, dtau_new = np.full(n, 1e-6), np.full(n, 1.8e-6)
    cache.preconditioner(dtau_old, 5)
    stale = cache.preconditioner(dtau_new, 5)
    assert calls["n"] == 1
    rng = np.random.default_rng(2)
    A = rng.standard_normal((n * 5, n * 5)) * 1e-3 + np.diag(np.repeat(1.0 / dtau_new, 5))
    b = rng.standard_normal(n * 5)

    def matvec(v):
        return A @ v

    def prec(v):
        return stale.apply(v.reshape(-1, 5)).reshape(-1)

    x, _, info, _ = gmres_right(matvec, b, prec, rtol=1e-10, restart=60, max_iter=200,
                                flexible=cache.flexible)
    assert info == 0
    np.testing.assert_allclose(A @ x, b, atol=1e-8 * np.abs(b).max())
