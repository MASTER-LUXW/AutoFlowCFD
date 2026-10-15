# -*- coding: utf-8 -*-
"""预处理矩阵的对角位移（`core/time_integration/implicit/precond_shift.py`）与线性求解失败后的补救
（`BlockJacobiCache.recover`）。

判据：
1. `shifted_dtau`：逐行 `1/dtau_p = max(1/dtau, PRECONDITIONER_SHIFT * d)`，零填充槽位（`d == 0`）原样；
2. 位移默认不启用，预处理用的就是系统的 dtau；`recover` 启用它、复用中的块同时重装配，之后无可补救；
3. 启用后预处理与 dtau 无关（dtau 再大分解也不变），仍是合法的右预处理：GMRES 解的是原系统；
4. 装配矩阵与真实算子在近零模态上不一致时（预处理矩阵近奇异、真实算子不是）：不带位移的预处理让 GMRES
   用满迭代数，Newton 步里复核点截断、启用位移后同一步解到容差。
"""

import numpy as np

from autoflowcfd.core.time_integration.implicit.block_budget import plan_for_budget
from autoflowcfd.core.time_integration.implicit import precond_shift as ps
from autoflowcfd.core.time_integration.implicit.block_jacobi import BlockJacobiCache
from autoflowcfd.core.time_integration.implicit.gmres import gmres_right
from autoflowcfd.core.time_integration.implicit.jfnk import step_newton_krylov
from autoflowcfd.core.time_integration.implicit.physicality import ScaledFieldRowLimits

NV = 2


def test_shifted_dtau_floors_the_pseudo_time_term_relative_to_the_block_diagonal():
    diag = np.array([10.0, 10.0, 0.0, 2.0])
    dtau = np.array([1e-3, 1e6, 1e6, 1e6])
    out = ps.shifted_dtau(dtau, diag, np)
    floor = 1.0 / (ps.PRECONDITIONER_SHIFT * diag[[1, 3]])
    np.testing.assert_allclose(out, [1e-3, floor[0], 1e6, floor[1]], rtol=1e-15)
    assert out is not dtau and ps.shifted_dtau(dtau.reshape(2, 2), diag, np).shape == (2, 2)


def _cache(n):
    cip = np.ones(n, dtype=bool)
    return BlockJacobiCache(cell_is_prism=cip, colors=np.zeros(n, dtype=np.int64),
                            plan=plan_for_budget(cip, 1, 1, 5, NV, 2 ** 50),
                            n_sps=1, n_real_prism=1, n_real_tet=1, n_var=NV)


def _cellwise(n, seed, slow=0.0):
    """逐单元 2x2 线性残差 `A_c (u - t)`；`slow > 0` 时每个块有一个量级为 `slow` 的特征值。"""
    g = np.random.default_rng(seed)
    q = np.linalg.qr(g.standard_normal((n, NV, NV)))[0]
    eig = np.stack([np.full(n, slow if slow > 0.0 else 3.0), np.full(n, 5.0)], axis=1)
    A = np.einsum("nij,nj,nkj->nik", q, eig, q)
    t = 1.0 + 0.1 * g.standard_normal((n, NV))
    return A, t, (lambda u: np.einsum("nij,nj->ni", A, u - t))


def test_recover_enables_the_shift_once_and_rebuilds_reused_blocks():
    n = 40
    _, t, res = _cellwise(n, 0)
    cache = _cache(n)
    u0 = np.ones((n, NV))
    dtau = np.full(n, 1e8)
    cache.begin_step(res, u0, res(u0), np.ones(NV), dtau)
    assert not cache.shifted and cache._preconditioner_dtau(dtau) is dtau
    assert cache.recoverable

    assert cache.recover(res, u0, res(u0), np.ones(NV), dtau)          # 刚装配的块：只启用位移
    assert cache.shifted and cache.n_builds == 1
    capped = cache._preconditioner_dtau(dtau)
    np.testing.assert_allclose(capped, 1.0 / (ps.PRECONDITIONER_SHIFT * cache._diag_scale()), rtol=1e-14)
    assert np.all(capped < 1e3)
    # 启用后预处理与 dtau 无关
    np.testing.assert_array_equal(cache._preconditioner_dtau(1e3 * dtau), capped)
    assert not cache.recoverable
    assert not cache.recover(res, u0, res(u0), np.ones(NV), dtau)      # 刚装配且已启用：无可补救

    cache.record(5, accepted=True)                                     # 之后块进入复用
    assert cache.recoverable
    assert cache.recover(res, u0, res(u0), np.ones(NV), dtau) and cache.n_builds == 2
    assert cache.shifted                                               # 位移跨重装配保持


def test_shifted_preconditioner_still_solves_the_original_system():
    """位移只改预处理：Newton 步（线性问题、dtau 极大）解到 forcing 容差，残差按原系统度量下降。"""
    n = 50
    _, t, res = _cellwise(n, 1)
    cache = _cache(n)
    u0 = np.ones((n, NV))
    cache.begin_step(res, u0, res(u0), np.ones(NV), np.full(n, 1e8))
    cache.recover(res, u0, res(u0), np.ones(NV), np.full(n, 1e8))
    u, info = step_newton_krylov(res, u0, np.full(n, 1e8), np.ones(NV), block_precond=cache,
                                 physicality=ScaledFieldRowLimits([1e-12, 1e-12]))
    assert info["gmres_info"] == 0 and info["theta"] == 1.0 and info["gmres_iters"] <= 10
    assert np.linalg.norm(res(u)) <= 0.1 * np.linalg.norm(res(u0)) * (1.0 + 1e-6)


class _InaccurateAssembler:
    """解析装配器的替身：给出的块在慢模态上与真实算子不一致（真实特征值 `slow` 量级，装配出的值小 2~6 个
    量级、逐单元各不相同、符号随机）——冻结系数的近似 Jacobian 在近零模态上的最小模型。"""

    want_coupling = False

    def __init__(self, A, slow):
        g = np.random.default_rng(9)
        w, v = np.linalg.eigh(0.5 * (A + A.transpose(0, 2, 1)))
        w[:, 0] = slow * g.choice([-1.0, 1.0], w.shape[0]) * 10.0 ** g.uniform(-6.0, -2.0, w.shape[0])
        self.blocks = np.einsum("nij,nj,nkj->nik", v, w, v)

    def __call__(self, u0, r0):
        return self.blocks.copy(), np.zeros((0, NV, NV))


def test_newton_step_recovers_from_an_unreliable_assembled_matrix():
    """装配矩阵在慢模态上不可信、dtau 大到伪时间项可忽略：不带位移的预处理把各单元的慢模态放大成散布四个量级、正负都有的特征值，
    GMRES 在复核点外推出到不了容差；补救启用位移后同一个 Newton 步解到容差。

    cube_demo SA P1（CFL 3000）上的对应现象：预处理按系统的 dtau 分解时 200 次只到 0.43，按 0.03 倍的
    dtau 分解时 20 次到容差。"""
    n = 200
    slow = 1e-2
    A, t, res = _cellwise(n, 2, slow=slow)
    # 逐单元的慢特征值各不相同：Krylov 不能靠几次迭代把它们一并消掉
    A *= np.linspace(1.0, 3.0, n)[:, None, None]
    u0 = np.ones((n, NV))
    dtau = np.full(n, 1e10)
    r0 = res(u0)

    # 对照：不带位移的预处理，直接解同一个线性系统，用满迭代数到不了容差
    plain = _cache(n)
    plain.assembler = _InaccurateAssembler(A, slow)
    plain.use_ilu = False                      # 装配器替身只给对角块：块 Jacobi 档
    plain.begin_step(res, u0, r0, np.ones(NV), dtau)
    prec = plain.preconditioner(dtau, NV)
    _, it, ginfo, rel = gmres_right(
        lambda x: (np.einsum("nij,nj->ni", A, x.reshape(n, NV)) + x.reshape(n, NV) / dtau[:, None]).reshape(-1),
        -r0.reshape(-1), lambda x: prec.apply(x.reshape(n, NV)).reshape(-1), rtol=0.1, restart=30, max_iter=120)
    assert ginfo > 0 and rel > 0.1, (it, rel)

    cache = _cache(n)
    cache.assembler = _InaccurateAssembler(A, slow)
    cache.use_ilu = False                      # 装配器替身只给对角块：块 Jacobi 档
    u, info = step_newton_krylov(res, u0, dtau, np.ones(NV), block_precond=cache, gmres_max_iter=120,
                                 gmres_restart=30, physicality=ScaledFieldRowLimits([1e-12, 1e-12]))
    assert cache.shifted, "线性求解未达容差后没有启用对角位移"
    assert info["gmres_info"] == 0 and info["theta"] == 1.0
    assert info["gmres_iters"] < 120
    assert np.linalg.norm(res(u)) <= 0.1 * np.linalg.norm(r0) * (1.0 + 1e-6)


class _CountingPreconditioner:
    """`jfnk` 用到的块缓存接口的替身：对角预处理，记录 `recover` 被调用的次数。"""

    flexible = False
    recoverable = True
    restart_cap = None

    def __init__(self):
        self.n_recover = 0

    def begin_step(self, *args):
        pass

    def stale_budget(self):
        return None

    def preconditioner(self, dtau, n_var):
        from autoflowcfd.core.time_integration.implicit.preconditioner import PseudoTransientDiagonal

        return PseudoTransientDiagonal(dtau, n_var)

    def recover(self, *args):
        self.n_recover += 1
        return False

    def record(self, *args, **kwargs):
        pass


def test_recovery_is_only_attempted_when_the_direction_is_untrusted():
    """没到容差但线性残差已减半的方向直接用，不重装配重解；线性残差没减半才补救。

    2026-10-13 以前只要没到容差就补救：cube_demo SA P1 稳态 CFL 3e4~1e5 的十几步里第一次求解 200 次到
    0.14~0.3，仍重装配再解 200 次。"""
    from autoflowcfd.core.time_integration.implicit.globalization import LINEAR_SOLVE_MAX_REL_RESIDUAL

    rng = np.random.default_rng(0)
    n = 120
    A = np.diag(np.geomspace(1.0, 300.0, n)) + rng.normal(0.0, 1.0, (n, n)) / np.sqrt(n)
    c = rng.normal(0.0, 1.0, (n, 1))

    def res(u):
        return A @ u + c

    def step(max_iter):
        prec = _CountingPreconditioner()
        u, info = step_newton_krylov(res, np.zeros((n, 1)), np.full(n, 1e12), np.ones(1), block_precond=prec,
                                     gmres_max_iter=max_iter, gmres_restart=max_iter,
                                     physicality=lambda u, du, red: np.ones(u.shape[0]))
        return prec, info

    # 对角预处理在 dtau 极大时等于不预处理；逐个迭代数上限看截断的求解落在哪一档
    seen = set()
    for max_iter in range(1, 80):
        prec, info = step(max_iter)
        if info["gmres_info"] == 0:
            assert prec.n_recover == 0
            break
        if info["linear_rel_residual"] <= LINEAR_SOLVE_MAX_REL_RESIDUAL:
            seen.add("usable")
            assert prec.n_recover == 0 and info["theta"] == 1.0
        else:
            seen.add("untrusted")
            assert prec.n_recover >= 1 and info["theta"] == 0.0
    assert seen == {"usable", "untrusted"}, seen


def test_block_diagonal_reports_only_the_nonlinear_block():
    """平均流块线性、湍流块非线性时只把湍流列标为非线性，`apply_linear` 只给平均流列；两块同类时不分列。"""
    from autoflowcfd.core.time_integration.implicit.coupled_step import _BlockDiagonal
    from autoflowcfd.core.time_integration.implicit.preconditioner import PseudoTransientDiagonal

    class _Flex(PseudoTransientDiagonal):
        flexible = True

    n = 4
    dtau = np.full(n, 0.5)
    lin, flex = PseudoTransientDiagonal(dtau, 5), _Flex(dtau, 1)
    combo = _BlockDiagonal(lin, flex, dtau, 1)
    assert combo.flexible and list(combo.nonlinear_columns) == [False] * 5 + [True]
    v = np.arange(n * 6, dtype=float).reshape(n, 6) + 1.0
    out = combo.apply_linear(v)
    np.testing.assert_allclose(out[:, :5], 0.5 * v[:, :5])
    assert np.all(out[:, 5:] == 0.0)
    assert _BlockDiagonal(flex, _Flex(dtau, 1), dtau, 1).nonlinear_columns is None
    assert _BlockDiagonal(lin, PseudoTransientDiagonal(dtau, 1), dtau, 1).nonlinear_columns is None


class _CoupledLike:
    """块缓存接口的替身：预处理平均流 2 列用固定对角逆（线性）、第 3 列每次作用都乘不同的因子（非线性）。"""

    recoverable = False

    restart_cap = None
    flexible = True

    def __init__(self, split: bool):
        self.split = split
        self.flexible_fraction = 1 / 3 if split else 1.0

    def begin_step(self, *args):
        pass

    def stale_budget(self):
        return None

    def recover(self, *args):
        return False

    def record(self, *args, **kwargs):
        pass

    def preconditioner(self, dtau, n_var):
        split = self.split

        class _Prec:
            flexible = True
            nonlinear_columns = np.array([False, False, True]) if split else None
            calls = [0]

            def apply(self, v):
                self.calls[0] += 1
                out = v / 4.0
                out[:, 2] *= 1.0 + 0.3 * np.sin(self.calls[0])
                return out

            def apply_linear(self, v):
                out = v / 4.0
                out[:, 2] = 0.0
                return out

        return _Prec()


def test_newton_step_with_partial_flexible_matches_full_flexible():
    """经 `step_newton_krylov`（真实行压缩、逐列加权范数）：只为非线性列存预处理向量与整份存储迭代数相同、
    Newton 方向一致。"""
    rng = np.random.default_rng(12)
    n_rows, nv = 40, 3
    A = 4.0 * np.eye(n_rows * nv) + rng.normal(0.0, 1.0, (n_rows * nv,) * 2) / np.sqrt(n_rows * nv)
    c = rng.normal(0.0, 1.0, (n_rows, nv))
    real = np.ones(n_rows, dtype=bool)
    real[::7] = False                                # 零填充行
    c[~real] = 0.0
    mask = np.repeat(real, nv)
    A[~mask, :] = 0.0
    A[:, ~mask] = 0.0

    def res(u):
        return (A @ u.reshape(-1)).reshape(n_rows, nv) + c

    def step(split):
        return step_newton_krylov(res, np.zeros((n_rows, nv)), np.full(n_rows, 1e12), np.array([1.0, 2.0, 0.5]),
                                  block_precond=_CoupledLike(split), real_rows=real,
                                  norm_weights=np.where(real, 1.0, 0.0), tol_nonlinear=1e-14,
                                  physicality=lambda u, du, red: np.ones(u.shape[0]))

    (u_full, info_full), (u_part, info_part) = step(False), step(True)
    assert info_part["gmres_info"] == 0 and info_part["gmres_iters"] == info_full["gmres_iters"]
    np.testing.assert_allclose(u_part, u_full, rtol=1e-3, atol=1e-9)
    assert np.all(u_part[~real] == 0.0)
