# -*- coding: utf-8 -*-
"""单元块 Jacobi 预处理（`core/time_integration/implicit/block_jacobi.py`）。

判据：
1. 距离 1 着色合法（面相邻单元不同色）；
2. 着色有限差分装配出的块**逐元素等于**真实的单元对角块——用一个线性残差
   `R = A U` 构造：`A` 同时含单元对角块与面邻居耦合块、并带原生基的零填充
   槽位，邻居耦合不得混进对角块，填充槽位不进块；
3. 预处理作用等于 `blockdiag(J_cc + I/dtau)` 的精确逆，填充槽位保持 `dtau*v`；
4. 刷新判据：首步装配、之后复用；迭代数劣化 / 步未被接受 / 超龄时重装配；
   同一步内多次取预处理子（dtau 缩档重试）不重装配；
5. 超内存上限时退回对角预处理；
6. 接入 `step_newton_krylov` 后在同一线性问题上 GMRES 迭代数少于对角预处理。
"""

import numpy as np
import pytest

from autoflowcfd.core.time_integration.implicit import block_jacobi as bj
from autoflowcfd.core.time_integration.implicit.block_jacobi import (
    BlockJacobiCache,
    CellBlockJacobian,
    CellBlockJacobiPreconditioner,
    greedy_cell_coloring,
)
from autoflowcfd.core.time_integration.implicit.jfnk import step_newton_krylov
from autoflowcfd.core.time_integration.implicit.preconditioner import PseudoTransientDiagonal

N_SPS, NV = 8, 5
N_REAL_P, N_REAL_T = 6, 4


def _chain_mesh(n_cells, n_prism):
    """一维链：单元 i 与 i+1 面相邻，两端各一个边界面。"""
    owner = np.r_[np.arange(n_cells - 1), 0, n_cells - 1]
    nb = np.r_[np.arange(1, n_cells), -1, -1]
    is_prism = np.arange(n_cells) < n_prism
    return owner, nb, is_prism


def _linear_system(n_cells, is_prism, rng):
    """`R = A U`：真实行含对角块 + 链上邻居耦合，填充行/列全零。"""
    n = n_cells * N_SPS * NV
    A = np.zeros((n, n))
    real = np.zeros((n_cells, N_SPS), dtype=bool)
    for c in range(n_cells):
        real[c, :(N_REAL_P if is_prism[c] else N_REAL_T)] = True
    real_dof = np.repeat(real.ravel(), NV)

    def blk(c):
        return slice(c * N_SPS * NV, (c + 1) * N_SPS * NV)
    for c in range(n_cells):
        A[blk(c), blk(c)] = rng.normal(size=(N_SPS * NV, N_SPS * NV)) + 20.0 * np.eye(N_SPS * NV)
        for d in (c - 1, c + 1):
            if 0 <= d < n_cells:
                A[blk(c), blk(d)] = rng.normal(size=(N_SPS * NV, N_SPS * NV))
    A[~real_dof, :] = 0.0
    A[:, ~real_dof] = 0.0
    return A, real


def _residual(A):
    def R(u_flat):
        return (A @ u_flat.reshape(-1)).reshape(-1, NV)
    return R


def test_coloring_is_proper():
    rng = np.random.default_rng(0)
    n = 500
    owner = rng.integers(0, n, 2000)
    nb = rng.integers(-1, n, 2000)
    keep = owner != nb
    owner, nb = owner[keep], nb[keep]
    col = greedy_cell_coloring(owner, nb, n)
    inner = nb >= 0
    assert np.all(col[owner[inner]] != col[nb[inner]])
    assert col.min() == 0


def test_blocks_equal_true_cell_diagonal_blocks():
    rng = np.random.default_rng(1)
    n_cells, n_prism = 7, 4
    owner, nb, is_prism = _chain_mesh(n_cells, n_prism)
    A, real = _linear_system(n_cells, is_prism, rng)
    u0 = rng.normal(size=(n_cells * N_SPS, NV))
    R = _residual(A)
    jac = CellBlockJacobian(R, u0, R(u0), np.ones(NV), n_sps=N_SPS, cell_is_prism=is_prism,
                            n_real_prism=N_REAL_P, n_real_tet=N_REAL_T,
                            colors=greedy_cell_coloring(owner, nb, n_cells))
    for blocks, cells, n_real in ((jac.blocks_prism, jac.prism_cells, N_REAL_P),
                                  (jac.blocks_tet, jac.tet_cells, N_REAL_T)):
        for i, c in enumerate(cells):
            dof = (c * N_SPS + np.arange(n_real))[:, None] * NV + np.arange(NV)[None, :]
            want = A[np.ix_(dof.ravel(), dof.ravel())]
            np.testing.assert_allclose(blocks[i], want, rtol=1e-5, atol=1e-5 * np.abs(want).max())
    # 次数 = 色数 x 真实解点数上限 x 变量数（与单元数无关）
    n_col = int(greedy_cell_coloring(owner, nb, n_cells).max()) + 1
    assert jac.n_residual_evals == n_col * N_REAL_P * NV


def test_preconditioner_is_exact_block_inverse_and_diagonal_on_padding():
    rng = np.random.default_rng(2)
    n_cells, n_prism = 5, 2
    owner, nb, is_prism = _chain_mesh(n_cells, n_prism)
    A, real = _linear_system(n_cells, is_prism, rng)
    u0 = rng.normal(size=(n_cells * N_SPS, NV))
    R = _residual(A)
    jac = CellBlockJacobian(R, u0, R(u0), np.ones(NV), n_sps=N_SPS, cell_is_prism=is_prism,
                            n_real_prism=N_REAL_P, n_real_tet=N_REAL_T,
                            colors=greedy_cell_coloring(owner, nb, n_cells))
    dtau = rng.uniform(0.1, 2.0, n_cells * N_SPS)
    M = CellBlockJacobiPreconditioner(jac, dtau)
    v = rng.normal(size=(n_cells * N_SPS, NV))
    out = M.apply(v)
    for c in range(n_cells):
        n_real = N_REAL_P if is_prism[c] else N_REAL_T
        rows = c * N_SPS + np.arange(n_real)
        dof = (rows[:, None] * NV + np.arange(NV)[None, :]).ravel()
        blk = A[np.ix_(dof, dof)] + np.diag(np.repeat(1.0 / dtau[rows], NV))
        np.testing.assert_allclose(out[rows].ravel(), np.linalg.solve(blk, v[rows].ravel()),
                                   rtol=1e-4, atol=1e-6)
        pad = c * N_SPS + np.arange(n_real, N_SPS)
        np.testing.assert_allclose(out[pad], v[pad] * dtau[pad, None])


def _cache(n_cells=6, n_prism=3):
    owner, nb, is_prism = _chain_mesh(n_cells, n_prism)
    return BlockJacobiCache(owner_cell=owner, neighbor_cell=nb, cell_is_prism=is_prism,
                            n_sps=N_SPS, n_real_prism=N_REAL_P, n_real_tet=N_REAL_T, n_var=NV)


def _begin(cache, rng):
    n_cells = cache.cell_is_prism.size
    A, _ = _linear_system(n_cells, cache.cell_is_prism, rng)
    u0 = rng.normal(size=(n_cells * N_SPS, NV))
    R = _residual(A)
    cache.begin_step(R, u0, R(u0), np.ones(NV))


def test_refresh_policy():
    rng = np.random.default_rng(3)
    c = _cache()
    _begin(c, rng)
    assert c.n_builds == 1
    c.preconditioner(np.ones(c.cell_is_prism.size * N_SPS), NV)   # 同一步多次取不重装配
    c.preconditioner(0.5 * np.ones(c.cell_is_prism.size * N_SPS), NV)
    assert c.n_builds == 1
    c.record(20, accepted=True)
    _begin(c, rng)
    assert c.n_builds == 1                     # 正常复用
    c.record(20 * 2 + 10 + 1, accepted=True)   # 迭代数劣化超过 2x+10
    _begin(c, rng)
    assert c.n_builds == 2
    c.record(15, accepted=False)               # 步未被接受
    _begin(c, rng)
    assert c.n_builds == 3
    for _ in range(bj.MAX_AGE):                # 超龄
        c.record(15, accepted=True)
    _begin(c, rng)
    assert c.n_builds == 4


def test_over_memory_budget_falls_back_to_diagonal(monkeypatch):
    monkeypatch.setattr(bj, "MAX_BYTES", 1)
    c = _cache()
    assert c.disabled_reason is not None
    _begin(c, np.random.default_rng(4))
    p = c.preconditioner(np.ones(c.cell_is_prism.size * N_SPS), NV)
    assert type(p) is PseudoTransientDiagonal
    assert c.n_builds == 0


def test_jfnk_block_precond_needs_fewer_gmres_iterations():
    """单元内稠密强耦合、奇异值跨 4 个量级，单元间弱耦合：高阶 FR 单元内
    刚性正是这种结构，也正是对角预处理看不到、块 Jacobi 该赢的地方。"""
    rng = np.random.default_rng(5)
    n_cells, n_prism = 12, 6
    owner, nb, is_prism = _chain_mesh(n_cells, n_prism)
    nb_ = N_SPS * NV
    n = n_cells * nb_
    A = np.zeros((n, n))
    real = np.zeros((n_cells, N_SPS), dtype=bool)
    for c in range(n_cells):
        real[c, :(N_REAL_P if is_prism[c] else N_REAL_T)] = True
        q1, _ = np.linalg.qr(rng.normal(size=(nb_, nb_)))
        q2, _ = np.linalg.qr(rng.normal(size=(nb_, nb_)))
        A[c * nb_:(c + 1) * nb_, c * nb_:(c + 1) * nb_] = q1 @ np.diag(np.logspace(-2, 2, nb_)) @ q2
        for d in (c - 1, c + 1):
            if 0 <= d < n_cells:
                A[c * nb_:(c + 1) * nb_, d * nb_:(d + 1) * nb_] = 1e-3 * rng.normal(size=(nb_, nb_))
    real_dof = np.repeat(real.ravel(), NV)
    A[~real_dof, :] = 0.0
    A[:, ~real_dof] = 0.0
    # 基态取物理状态（jfnk 的物理性限幅按守恒变量解释 rho/p），右端项小，
    # 使 Newton 步在限幅之内
    u0 = np.tile([1.0, 0.0, 0.0, 0.0, 2.5], (n_cells * N_SPS, 1))
    b = A @ u0.reshape(-1) + 1e-4 * rng.normal(size=n) * real_dof

    def R(u_flat):
        return (A @ u_flat.reshape(-1) - b).reshape(-1, NV)
    dtau = np.full(n_cells * N_SPS, 1e6)
    _, info_d = step_newton_krylov(R, u0, dtau, np.ones(NV))
    cache = BlockJacobiCache(owner_cell=owner, neighbor_cell=nb, cell_is_prism=is_prism,
                             n_sps=N_SPS, n_real_prism=N_REAL_P, n_real_tet=N_REAL_T, n_var=NV)
    _, info_b = step_newton_krylov(R, u0, dtau, np.ones(NV), block_precond=cache)
    assert info_b["theta"] > 0.0 and info_d["theta"] > 0.0
    assert info_b["gmres_iters"] * 5 <= info_d["gmres_iters"], (info_b["gmres_iters"], info_d["gmres_iters"])
    assert cache.n_builds == 1 and cache.age == 1
