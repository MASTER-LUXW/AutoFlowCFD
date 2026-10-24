"""单元级粗校正的分布式部分：装配结果按 rank 拆成 rank 内 / 跨 rank 粗块（`select_coarse_rows`），多 rank 的粗层
近似逆（以单元级粗矩阵为细层的全局粗校正，`coarse/cell_coarse.py` 模块文档"分布式"）。

本机没有 MPI：集合通信用线程模拟（`tests/unit/_thread_comm.py`），紧凑空间取全局单元编号。
"""

import numpy as np

from autoflowcfd.core.fr_residual.jacobian.coarse import CoarseBlocks, select_coarse_rows
from autoflowcfd.core.time_integration.implicit.coarse.cell_coarse import (
    CellCoarseMatrix, CellCoarseSolve, CellCoarseTopology,
)
from autoflowcfd.core.time_integration.implicit.krylov.gmres import gmres_right
from tests.unit._thread_comm import ThreadComm
from tests.unit.test_global_coarse_correction import _context

NV = 5
N_RANKS = 3


def _global_coarse(n, seed):
    """链 + 随机远邻的单元对、对流扩散型 5x5 块（对角占优不强，块 Jacobi 式分解收敛慢）。"""
    rng = np.random.default_rng(seed)
    pairs = {(c, c + 1) for c in range(n - 1)} | {(c + 1, c) for c in range(n - 1)}
    for _ in range(n):
        a, b = rng.integers(0, n, 2)
        if a != b:
            pairs |= {(int(a), int(b)), (int(b), int(a))}
    rows, cols = np.array(sorted(pairs)).T
    blocks = -0.45 * np.eye(NV)[None] + 0.02 * rng.normal(size=(rows.size, NV, NV))
    deg = np.bincount(rows, minlength=n)
    diag = (0.45 * deg + 0.01)[:, None, None] * np.eye(NV)[None] + 0.02 * rng.normal(size=(n, NV, NV))
    return CoarseBlocks(diag=diag, rows=rows, cols=cols, blocks=blocks)


def test_select_coarse_rows_splits_rank_local_and_cross_rank_pairs():
    g = _global_coarse(30, 1)
    ids = np.array([3, 4, 5, 6, 20, 21])                  # 本 rank 的单元（紧凑编号 = 全局编号）
    inner, cross = select_coarse_rows(g, ids)
    np.testing.assert_array_equal(inner.diag, g.diag[ids])
    pos = {int(c): k for k, c in enumerate(ids)}
    want_inner, want_cross = {}, {}
    for k, (r, y) in enumerate(zip(g.rows, g.cols)):
        if int(r) in pos and int(y) in pos:
            want_inner[pos[int(r)], pos[int(y)]] = g.blocks[k]
        elif int(r) in pos:
            want_cross[pos[int(r)], int(y)] = g.blocks[k]
    got_inner = {(int(r), int(y)): inner.blocks[k] for k, (r, y) in enumerate(zip(inner.rows, inner.cols))}
    got_cross = {(int(r), int(y)): cross.blocks[k] for k, (r, y) in enumerate(zip(cross.rows, cross.cols))}
    assert got_inner.keys() == want_inner.keys() and got_cross.keys() == want_cross.keys() and want_cross
    for k in want_inner:
        np.testing.assert_array_equal(got_inner[k], want_inner[k])
    for k in want_cross:
        np.testing.assert_array_equal(got_cross[k], want_cross[k])


def _dense(g, n, inv_dtau):
    A = np.zeros((n * NV, n * NV))
    for c in range(n):
        A[c * NV:(c + 1) * NV, c * NV:(c + 1) * NV] = g.diag[c] + inv_dtau[c] * np.eye(NV)
    for k, (r, y) in enumerate(zip(g.rows, g.cols)):
        A[r * NV:(r + 1) * NV, y * NV:(y + 1) * NV] += g.blocks[k]
    return A


def _partitioned(g, n, inv_dtau, with_global):
    parts = np.array_split(np.arange(n), N_RANKS)
    comm = ThreadComm(N_RANKS)

    def build(r):
        ids = parts[r]
        inner, cross = select_coarse_rows(g, ids)
        topo = CellCoarseTopology(ids.size, inner.rows, inner.cols, NV, cross.rows, cross.cols)
        mat = CellCoarseMatrix(topo, inner.diag, inner.blocks.astype(np.float32), cross.blocks)
        ctx = _context(comm, r, ids, n) if with_global else None
        return dict(ids=ids, solve=CellCoarseSolve(mat, inv_dtau[ids], ctx))

    return comm, comm.run(build)


def _apply_all(comm, ranks, v):
    outs = comm.run(lambda r: ranks[r]["solve"].solve(v.reshape(-1, NV)[ranks[r]["ids"]]))
    z = np.empty_like(v).reshape(-1, NV)
    for r, rk in enumerate(ranks):
        z[rk["ids"]] = outs[r]
    return z.reshape(-1)


def test_multi_rank_coarse_solve_matches_dense_definition():
    """多 rank：`z = z1 + M_loc(r - A_c z1)`，`z1 = P_g A_g^{-1} P_g^T r`，`M_loc` 为各 rank 自己那一块的精确逆
    （节点数小、局部走 LU），`A_c` 含跨 rank 粗块——验证跨 rank 粗块进了全局粗矩阵与 `A_c z1`。"""
    n = 90
    g = _global_coarse(n, 2)
    inv_dtau = np.full(n, 0.05)
    comm, ranks = _partitioned(g, n, inv_dtau, True)
    A = _dense(g, n, inv_dtau)
    gid = np.empty(n, dtype=np.int64)
    for rk in ranks:
        gid[rk["ids"]] = rk["solve"]._op._space.gid_local
    n_g = ranks[0]["solve"]._op._space.n_global
    P = np.zeros((n * NV, n_g * NV))
    for c in range(n):
        P[c * NV:(c + 1) * NV, gid[c] * NV:(gid[c] + 1) * NV] = np.eye(NV)
    r = np.random.default_rng(3).standard_normal(n * NV)
    z1 = P @ np.linalg.solve(P.T @ A @ P, P.T @ r)
    resid = r - A @ z1
    ref = z1.copy()
    for rk in ranks:
        idx = (rk["ids"][:, None] * NV + np.arange(NV)[None, :]).ravel()
        ref[idx] += np.linalg.solve(A[np.ix_(idx, idx)], resid[idx])
    # 粗矩阵的块按 float32 存（与块 ILU 档同一约定），稠密参考是 float64：差在 float32 舍入 x 条件数的量级
    np.testing.assert_allclose(_apply_all(comm, ranks, r), ref, rtol=0, atol=2e-6 * np.abs(ref).max())


def test_global_level_cuts_iterations_over_rank_local_coarse_solves():
    """跨 rank 的全局粗校正让以粗层近似逆为预处理的 GMRES 迭代数少于各 rank 只解自己那一块。"""
    n = 90
    g = _global_coarse(n, 4)
    inv_dtau = np.full(n, 0.01)
    A = _dense(g, n, inv_dtau)
    b = np.random.default_rng(6).standard_normal(n * NV)
    iters = {}
    for tag, with_global in (("local", False), ("global", True)):
        comm, ranks = _partitioned(g, n, inv_dtau, with_global)
        x, it, info, _ = gmres_right(lambda v: A @ v, b, lambda v: _apply_all(comm, ranks, v), rtol=1e-8,
                                     restart=300, max_iter=300)
        assert info == 0, (tag, it)
        iters[tag] = it
    assert iters["global"] < iters["local"], iters


def test_distributed_solver_with_coarse_tier_matches_single_machine(monkeypatch):
    """CPU 分布式（n_ranks=1）的隐式步在 `+cc` 档（把块 ILU 的内存估计抬高到放不下）同样装配单元级粗块：
    装配器分布式分支把限制权重放进紧凑空间、粗块按 rank 拆分，Newton 轨迹与单机一致（人工粘性开着，
    其标量耦合块也约化进粗块）。"""
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
    from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
    from autoflowcfd.core.time_integration import TimeIntegrationScheme
    from autoflowcfd.core.time_integration.implicit.blocks import block_budget as bb
    from autoflowcfd.fr.operators import generate_fr_operators
    from tests.unit.test_artificial_viscosity_nk_analytic import H, LX, LZ, P, RHO, U_INF, _solver
    from tests.validation._channel_mesh import build_face_exact_ghost_provider

    monkeypatch.setattr(bb, "estimate_ilu_bytes", lambda *a, **k: 2 ** 60)
    single = _solver()
    mesh = single.mesh
    bc = {n: {"type": "SYMMETRY"} for n in ("wall_bottom", "wall_top", "z_min", "z_max")}
    for n in ("x_min", "x_max"):
        bc[n] = {"type": "FARFIELD", "Q_free": [RHO, U_INF, 0.0, 0.0, P]}
    dist = DistributedFRSolver(
        mesh=mesh, ops=generate_fr_operators(1), face_connectivity=mesh.face_connectivity, n_ranks=1,
        order=1, turb_model_name="NONE",
        time_scheme=TimeIntegrationScheme.NEWTON_KRYLOV, mu_molecular=1.8e-5, rho_inf=RHO, vel_inf=U_INF,
        p_inf=P, bc_overrides=bc, artificial_viscosity_enabled=True)
    dist.local_solver.boundary_ghost_provider = build_face_exact_ghost_provider(mesh, LX, H, LZ, bc)
    n = mesh.n_cells
    dist.state.U[:n] = single.state.U
    dist.state.Q[:n] = conserved_to_primitive(np.asarray(single.state.U))
    for k in range(3):
        single.step(0.0)
        dist.step(0.0)
        a, b = dist._newton_last_info, single._newton_last_info
        assert a["gmres_iters"] == b["gmres_iters"], f"第 {k + 1} 步 GMRES 次数不同：{a} vs {b}"
    for s in (single, dist):
        cache = s._newton_block_precond
        assert cache.cell_coarse and cache._coarse_matrix is not None, "没有走单元级粗校正档"
    assert dist._newton_block_precond._coarse_topology.cross_rows.size == 0      # 单 rank：没有跨 rank 粗块
    rel = np.abs(dist.state.U[:n] - single.state.U).max() / np.abs(single.state.U).max()
    assert rel < 1e-9, rel
