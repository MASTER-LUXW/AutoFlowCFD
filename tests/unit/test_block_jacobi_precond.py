# -*- coding: utf-8 -*-
"""单元块 Jacobi 预处理（`core/time_integration/implicit/blocks/block_jacobi.py`）。

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

from autoflowcfd.core.time_integration.implicit.blocks import block_budget as bb
from autoflowcfd.core.time_integration.implicit.blocks.block_jacobi import BlockJacobiCache
from autoflowcfd.core.time_integration.implicit.blocks.cell_block_inverse import CellBlockJacobiPreconditioner
from autoflowcfd.core.time_integration.implicit.blocks.cell_blocks import CellBlockJacobian
from autoflowcfd.core.time_integration.implicit.blocks.coloring import greedy_cell_coloring
from autoflowcfd.core.time_integration.implicit.newton.jfnk import step_newton_krylov
from autoflowcfd.core.time_integration.implicit.blocks.preconditioner import PseudoTransientDiagonal

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


def test_apply_accumulates_stored_inverse_in_float64_and_accepts_column_views():
    """作用核按行号直接读写（`cell_block_inverse.py::_apply_inverse_blocks`）：等于 float32 存储的逆在 float64
    里乘向量；输入是更宽数组的列切片（块对角组合传入的平均流列）时结果逐位相同、输入不被改动。"""
    rng = np.random.default_rng(12)
    n_cells, n_prism = 6, 3
    owner, nb, is_prism = _chain_mesh(n_cells, n_prism)
    A, _ = _linear_system(n_cells, is_prism, rng)
    u0 = rng.normal(size=(n_cells * N_SPS, NV))
    R = _residual(A)
    jac = CellBlockJacobian(R, u0, R(u0), np.ones(NV), n_sps=N_SPS, cell_is_prism=is_prism,
                            n_real_prism=N_REAL_P, n_real_tet=N_REAL_T,
                            colors=greedy_cell_coloring(owner, nb, n_cells))
    dtau = rng.uniform(0.1, 2.0, n_cells * N_SPS)
    M = CellBlockJacobiPreconditioner(jac, dtau)
    wide = rng.normal(size=(n_cells * N_SPS, NV + 2))
    before = wide.copy()
    v = wide[:, 1:NV + 1]
    out = M.apply(v)
    assert np.array_equal(wide, before)
    assert np.array_equal(out, M.apply(np.ascontiguousarray(v)))
    for inv, cells, n_real in ((M._inv_prism, jac.prism_cells, N_REAL_P), (M._inv_tet, jac.tet_cells, N_REAL_T)):
        assert inv.dtype == np.float32
        for i, c in enumerate(cells):
            rows = c * N_SPS + np.arange(n_real)
            want = inv[i].astype(np.float64) @ v[rows].ravel()
            np.testing.assert_allclose(out[rows].ravel(), want, rtol=1e-12, atol=1e-12 * np.abs(want).max())


def test_block_diagonal_combination_reads_column_views_without_modifying_input():
    """平均流 / 湍流两块的组合（`coupled_step.py::_BlockDiagonal`）不复制输入：各块拿到列切片、输入不变，
    结果等于各块分别作用在连续副本上。"""
    from autoflowcfd.core.time_integration.implicit.newton.coupled_step import _BlockDiagonal

    rng = np.random.default_rng(13)
    n = 40
    mean = PseudoTransientDiagonal(rng.uniform(0.5, 2.0, n), 5)
    turb = PseudoTransientDiagonal(rng.uniform(0.5, 2.0, n), 1)
    combo = _BlockDiagonal(mean, turb, None, 1)
    v = rng.normal(size=(n, 6))
    before = v.copy()
    out = combo.apply(v)
    assert np.array_equal(v, before)
    assert np.array_equal(out[:, :5], mean.apply(v[:, :5].copy()))
    assert np.array_equal(out[:, 5:], turb.apply(v[:, 5:].copy()))
    lin = combo.apply_linear(v)
    assert np.array_equal(lin, out)          # 两块都线性


def _cache(n_cells=6, n_prism=3, budget=2 ** 50):
    owner, nb, is_prism = _chain_mesh(n_cells, n_prism)
    return BlockJacobiCache(colors=greedy_cell_coloring(owner, nb, is_prism.size), cell_is_prism=is_prism,
                            plan=bb.plan_for_budget(is_prism, N_REAL_P, N_REAL_T, NV, 0, budget),
                            n_sps=N_SPS, n_real_prism=N_REAL_P, n_real_tet=N_REAL_T, n_var=NV)


def _begin(cache, rng):
    n_cells = cache.cell_is_prism.size
    A, _ = _linear_system(n_cells, cache.cell_is_prism, rng)
    u0 = rng.normal(size=(n_cells * N_SPS, NV))
    R = _residual(A)
    cache.begin_step(R, u0, R(u0), np.ones(NV), np.full(u0.shape[0], 1e-2))


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
    # 迭代数劣化超过"基线 + 一次装配折合的迭代数"（差分装配按残差求值次数折算，refresh_cost.py）
    assert c.refresh_slack == max(3, round(0.73 * c.jac.n_residual_evals))
    c.record(20 + c.refresh_slack, accepted=True)       # 恰在阈值上：不重装配
    _begin(c, rng)
    assert c.n_builds == 1
    c.record(20 + c.refresh_slack + 1, accepted=True)
    _begin(c, rng)
    assert c.n_builds == 2
    c.record(15, accepted=False)               # 步未被接受
    _begin(c, rng)
    assert c.n_builds == 3
    for _ in range(100):                       # 迭代数不劣化时不按步数重装配（硬上限已删除）
        c.record(15, accepted=True)
        _begin(c, rng)
    assert c.n_builds == 3


def test_over_memory_budget_falls_back_to_diagonal():
    c = _cache(budget=1)
    assert c.disabled_reason is not None
    _begin(c, np.random.default_rng(4))
    p = c.preconditioner(np.ones(c.cell_is_prism.size * N_SPS), NV)
    assert type(p) is PseudoTransientDiagonal
    assert c.n_builds == 0


def test_single_copy_mode_freezes_dtau_and_rebuilds_on_drift():
    """两份放不下、一份放得下：装配后按当时 dtau 原地求逆、不留 J_cc；预处理用这份逆
    （与传入的 dtau 无关）；dtau 几何平均漂移超过 DTAU_REBUILD_RATIO 才重装配。"""
    # 预算恰好放得下一份、放不下两份
    rng = np.random.default_rng(6)
    c = _cache(budget=bb.block_mode_bytes("single", 3, 3, N_REAL_P, N_REAL_T, NV))
    assert c.single_copy and c.disabled_reason is None and not c.use_ilu
    n_cells = c.cell_is_prism.size
    A, _ = _linear_system(n_cells, c.cell_is_prism, rng)
    u0 = rng.normal(size=(n_cells * N_SPS, NV))
    R = _residual(A)
    d0 = np.full(n_cells * N_SPS, 0.3)
    c.begin_step(R, u0, R(u0), np.ones(NV), d0)
    v = rng.normal(size=(n_cells * N_SPS, NV))

    def check(dtau_inverted, dtau_passed):
        out = c.preconditioner(dtau_passed, NV).apply(v)
        for cell in range(n_cells):
            n_real = N_REAL_P if c.cell_is_prism[cell] else N_REAL_T
            rows = cell * N_SPS + np.arange(n_real)
            dof = (rows[:, None] * NV + np.arange(NV)[None, :]).ravel()
            blk = A[np.ix_(dof, dof)] + np.diag(np.repeat(1.0 / dtau_inverted[rows], NV))
            np.testing.assert_allclose(out[rows].ravel(), np.linalg.solve(blk, v[rows].ravel()),
                                       rtol=1e-4, atol=1e-6)
            pad = cell * N_SPS + np.arange(n_real, N_SPS)
            np.testing.assert_allclose(out[pad], v[pad] * dtau_passed[pad, None])

    check(d0, 0.5 * d0)                        # 冻结：用的是求逆时的 d0
    c.record(10, accepted=True)
    c.begin_step(R, u0, R(u0), np.ones(NV), 1.5 * d0)
    assert c.n_builds == 1                     # 漂移 1.5 倍 < 2：复用
    c.record(10, accepted=True)
    c.begin_step(R, u0, R(u0), np.ones(NV), 3.0 * d0)
    assert c.n_builds == 2                     # 漂移 3 倍：重装配并按新 dtau 求逆
    check(3.0 * d0, 3.0 * d0)


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
    cache = BlockJacobiCache(colors=greedy_cell_coloring(owner, nb, is_prism.size), cell_is_prism=is_prism,
                             plan=bb.plan_for_budget(is_prism, N_REAL_P, N_REAL_T, NV, 0, 2 ** 50),
                             n_sps=N_SPS, n_real_prism=N_REAL_P, n_real_tet=N_REAL_T, n_var=NV)
    _, info_b = step_newton_krylov(R, u0, dtau, np.ones(NV), block_precond=cache)
    # 对角预处理在这个系统上用满迭代数也解不到容差：线性残差没减半的方向不走出去（theta=0，
    # `globalization.LINEAR_SOLVE_MAX_REL_RESIDUAL`），块预处理一次解到容差
    assert info_b["theta"] > 0.0 and info_b["gmres_info"] == 0
    assert info_b["gmres_iters"] * 5 <= info_d["gmres_iters"], (info_b["gmres_iters"], info_d["gmres_iters"])
    assert cache.n_builds == 1 and cache.age == 1


def test_stale_blocks_are_refreshed_within_the_step():
    """复用的 J_cc 在本步的过时预算（基线 + 一次装配折合的迭代数）内解不到容差时，
    当场重装配再解，而不是把整步的 GMRES 预算烧完、等下一步才刷新。"""
    from autoflowcfd.core.time_integration.implicit.blocks.block_jacobi import BlockJacobiCache
    from autoflowcfd.core.time_integration.implicit.newton.jfnk import step_newton_krylov
    from autoflowcfd.core.time_integration.implicit.newton.physicality import ScaledFieldRowLimits

    rng = np.random.default_rng(5)
    n = 60

    def make(seed):
        g = np.random.default_rng(seed)
        A = 5.0 * np.eye(2)[None] + 3.0 * g.standard_normal((n, 2, 2))
        t = 1.0 + 0.1 * g.standard_normal((n, 2))
        return lambda u: np.einsum("nij,nj->ni", A, u - t), t

    cip = np.ones(n, dtype=bool)
    cache = BlockJacobiCache(cell_is_prism=cip, colors=np.zeros(n, dtype=np.int64),
                             plan=bb.plan_for_budget(cip, 1, 1, 5, 2, 2 ** 50),
                             n_sps=1, n_real_prism=1, n_real_tet=1, n_var=2)
    dtau = np.full(n, 1e8)
    u = 1.0 + 0.1 * rng.standard_normal((n, 2))
    res1, _ = make(1)
    u, info1 = step_newton_krylov(res1, u, dtau, np.ones(2), block_precond=cache, gmres_max_iter=200,
                                  physicality=ScaledFieldRowLimits([1e-12, 1e-12]))
    assert cache.n_builds == 1 and info1["gmres_iters"] <= 2

    res2, t2 = make(2)                 # 块整体换掉：复用的预处理对它几乎无效
    budget = cache.stale_budget()
    assert budget is not None and budget < 200
    r_before = np.linalg.norm(res2(u))
    u, info2 = step_newton_krylov(res2, u, dtau, np.ones(2), block_precond=cache, gmres_max_iter=200,
                                  physicality=ScaledFieldRowLimits([1e-12, 1e-12]))
    assert cache.n_builds == 2, "本步没有当场重装配"
    # 补救同时启用预处理的对角位移（`precond_shift.py`）：重装配后的预处理不再是精确逆，按 forcing 容差
    # 0.1 解到为止，迭代数仍远在"把 200 次预算烧完"之下
    assert cache.shifted
    assert info2["gmres_iters"] <= budget + 10, info2["gmres_iters"]
    assert info2["gmres_info"] == 0 and info2["theta"] == 1.0
    assert np.linalg.norm(res2(u)) <= 0.1 * r_before * (1.0 + 1e-6)


def test_block_modes_share_one_budget_with_mean_flow_first():
    """平均流与湍流两个缓存共享一个预算，平均流优先（plate_demo P3 实测：各自按自己的上限选档时湍流 ILU
    8.38 GiB + 平均流单份 8.9 GiB 叠加 OOM）。"""
    sizes = (19872, 159365, 40, 20)                    # plate_demo P3
    single_mean = bb.block_mode_bytes("single", *sizes, 5)
    single_turb = bb.block_mode_bytes("single", *sizes, 2)
    assert bb.plan_block_modes(sizes, 5, 2, single_mean + single_turb) == ("single", "single")
    # 预算不够两者都放单份：平均流仍然拿到单份，湍流退回对角预处理
    assert bb.plan_block_modes(sizes, 5, 2, single_mean + single_turb - 1) == ("single", None)
    # 合计不超预算
    budget = 12 * 2 ** 30
    m, t = bb.plan_block_modes(sizes, 5, 2, budget)
    assert bb.block_mode_bytes(m, *sizes, 5) + bb.block_mode_bytes(t, *sizes, 2) <= budget
    # 层流：平均流独占预算
    assert bb.plan_block_modes(sizes, 5, 0, single_mean) == ("single", None)


def test_one_equation_turbulence_shares_the_budget():
    """SA（1 个湍流未知量）与 SST 一样在合计预算内：平均流按 1 个变量的湍流单份预留，两者合计不超预算。

    2026-10-15 以前按"湍流就是 2 个变量"写死：n_var=1 的湍流缓存走"不在合计预算内"的分支、独占全部预算，
    平均流给它预留的又是 2 个变量的单份——cube_demo 粗网格 SA P2 两个缓存合计 10.2 GiB（平均流两份 9.12 +
    湍流 ILU 1.07），只是恰好没超过写死的 11 GiB。"""
    sizes = (90649, 195744, 18, 10)                    # cube_demo 粗网格 P2（28.6 万单元）
    single_mean = bb.block_mode_bytes("single", *sizes, 5)
    single_sa = bb.block_mode_bytes("single", *sizes, 1)
    assert bb.plan_block_modes(sizes, 5, 1, single_mean + single_sa) == ("single", "single")
    for budget in (single_mean + single_sa, 6 * 2 ** 30, 9 * 2 ** 30, 11 * 2 ** 30, 40 * 2 ** 30):
        m, t = bb.plan_block_modes(sizes, 5, 1, budget)
        used = sum(bb.block_mode_bytes(mode, *sizes, n) for mode, n in ((m, 5), (t, 1)) if mode is not None)
        assert used <= budget, (budget, m, t)


def test_budget_follows_available_memory(monkeypatch):
    """预算 = 可用内存 − Krylov 基 − 残差求值工作数组 − 余量；可用内存少时降到单份而不是照样选最贵的档。

    cube_demo 粗网格 SA P2：写死的 11 GiB 选了平均流两份（9.12 GiB），第 2 步 GMRES 时进程 29.4 GB 内存不足。"""
    from autoflowcfd.core.time_integration.implicit.krylov.linear_solve import krylov_basis_bytes
    from autoflowcfd.core.time_integration.implicit.reductions import LocalReductions
    import autoflowcfd.core.utils.system_memory as sm

    red = LocalReductions()
    cip = np.arange(286393) < 90649                    # cube_demo 粗网格：9.06 万棱柱 + 19.6 万四面体
    sizes = (90649, 195744, 18, 10)
    real_entries = (90649 * 18 + 195744 * 10) * 6
    fixed = bb.RESIDUAL_WORK_FACTOR * 286393 * 27 * 6 * 8 + bb.MEMORY_HEADROOM
    # 平均流不是块 ILU 时它的预处理线性，只有 SA 那 1 列要存预处理向量（`gmres.py` 模块文档"部分灵活"）
    reserve = krylov_basis_bytes(real_entries, red, flexible=1 / 6) + fixed
    reserve_all = krylov_basis_bytes(real_entries, red, flexible=1.0) + fixed
    assert reserve < reserve_all

    def plan(available):
        monkeypatch.setattr(sm, "available_bytes", lambda xp: available)
        return bb.plan_block_memory(red, cip, 27, 18, 10, 5, 1)

    two_mean = bb.block_mode_bytes("two", *sizes, 5)
    ilu_sa = bb.block_mode_bytes("ilu", *sizes, 1)
    roomy = plan(reserve + two_mean + ilu_sa + 2 ** 30)
    assert (roomy.mean, roomy.turbulence) == ("two", "ilu") and roomy.reserve == reserve
    tight = plan(reserve + two_mean - 2 ** 30)
    assert tight.mean == "single" and tight.budget == two_mean - 2 ** 30
    assert plan(reserve // 2).budget == 0 and plan(reserve // 2).mean is None
    # 平均流放得下块 ILU 时它也变成非线性：按整份灵活预留
    ilu_mean = bb.block_mode_bytes("ilu", *sizes, 5)
    rich = plan(reserve_all + ilu_mean + ilu_sa + 2 ** 30)
    assert (rich.mean, rich.turbulence) == ("ilu", "ilu") and rich.reserve == reserve_all
    assert "可用" in roomy.describe() and "预留" in roomy.describe()


def test_available_memory_query_is_positive_and_below_physical():
    from autoflowcfd.core.utils.system_memory import available_bytes

    got = available_bytes(np)
    assert 0 < got < 2 ** 50


def test_restart_is_shortened_only_as_far_as_needed_to_fit_every_cache(monkeypatch):
    """湍流块差一点放不下单份时缩短 GMRES 重启长度换内存（只缩到刚好放得下）；缩到下限仍放不下则保持默认
    长度、退回对角预处理。Newton 步的重启长度按规划给出的上限截断。

    cube_demo 粗网格 SA P2：规划时可用 17.9 GiB，重启 60 时平均流单份之后湍流差 0.08 GiB 退回对角预处理。"""
    from autoflowcfd.core.time_integration.implicit.krylov.linear_solve import krylov_basis_bytes, krylov_restart
    from autoflowcfd.core.time_integration.implicit.reductions import LocalReductions
    import autoflowcfd.core.utils.system_memory as sm

    red = LocalReductions()
    cip = np.arange(286393) < 90649
    sizes = (90649, 195744, 18, 10)
    real_entries = (90649 * 18 + 195744 * 10) * 6
    fixed = bb.RESIDUAL_WORK_FACTOR * 286393 * 27 * 6 * 8 + bb.MEMORY_HEADROOM
    single_mean = bb.block_mode_bytes("single", *sizes, 5)
    single_sa = bb.block_mode_bytes("single", *sizes, 1)
    default_m = krylov_restart(real_entries, red, flexible=1.0, itemsize=4)

    def plan(available):
        monkeypatch.setattr(sm, "available_bytes", lambda xp: int(available))
        return bb.plan_block_memory(red, cip, 27, 18, 10, 5, 1)

    # 默认重启长度下差 1 MiB：缩一档就放得下
    short_by = krylov_basis_bytes(real_entries, red, flexible=1 / 6) + fixed + single_mean + single_sa - 2 ** 20
    p = plan(short_by)
    assert p.mean == "single" and p.turbulence is not None     # 缩一档腾出的内存够湍流取更好的档
    assert p.restart_cap == default_m - bb.RESTART_STEP and "重启长度缩到" in p.describe()
    assert krylov_restart(real_entries, red, flexible=1 / 6, itemsize=4, cap=p.restart_cap) == p.restart_cap
    # 缩到下限也放不下：保持默认长度
    floor = krylov_basis_bytes(real_entries, red, flexible=1 / 6, restart=bb.RESTART_FLOOR_FOR_BLOCKS) + fixed
    hopeless = plan(floor + single_mean - 2 ** 20)
    assert hopeless.restart_cap is None and (hopeless.mean is None or hopeless.turbulence is None)
    # 放得下时不缩
    assert plan(short_by + 2 ** 30).restart_cap is None
