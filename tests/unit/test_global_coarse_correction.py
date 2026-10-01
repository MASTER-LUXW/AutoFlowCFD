"""分布式块预处理的全局粗校正（`implicit/coarse/global_coarse.py`）的多 rank 判据。

本机没有 MPI：集合通信用线程模拟（`tests/unit/_thread_comm.py`，每个线程就是一个 rank）。
把通道 P1 四面体的真实解析块按单元切成 3 个"rank"，紧凑空间就取全局单元编号（`select_rows`
按它给出 rank 内与跨 rank 两组耦合块）。

1. 作用正确：每个 rank 的作用与按定义稠密构造的 `z1 + M_loc(r - A z1)`（`z1 = P_g A_g^{-1}
   P_g^T r`，`A` 是含跨 rank 耦合的完整矩阵）一致——同时验证了全局粗矩阵的三元组交换、halo
   粗编号与跨 rank 耦合块的矩阵乘；
2. 有效：全局粗校正让分区后的 GMRES 迭代数少于"各 rank 只有本地多层"（rank 间块 Jacobi）。
"""

import numpy as np

from autoflowcfd.core.fr_residual.jacobian.backend import select_rows
from autoflowcfd.core.time_integration.implicit.block_ilu import BlockCouplingStructure
from autoflowcfd.core.time_integration.implicit.cell_blocks import CellBlockJacobian
from autoflowcfd.core.time_integration.implicit.coarse import CoarseCommContext, CoarsePreconditionerFactory
from autoflowcfd.core.time_integration.implicit.coarse.galerkin import galerkin_coarse_matrix
from autoflowcfd.core.time_integration.implicit.gmres import gmres_right
from autoflowcfd.fr.native_padding import real_sps_per_cell
from tests.unit._thread_comm import ThreadComm
from tests.unit.test_block_ilu_precond import _setup
from tests.unit.test_multilevel_preconditioner import _dense_A, _dense_P

NV = 5
N_RANKS = 3


def _context(comm, rank, local_ids, n_compact):
    """全局粗校正的通信上下文：紧凑空间取全局单元编号，逐单元值经 allgather 铺满。"""
    def allreduce_sum(a):
        return comm.collective(rank, np.asarray(a, dtype=np.float64), lambda v: np.sum(v, axis=0))

    def compact_cell_values(values_local):
        ids = comm.allgather(rank, local_ids)
        vals = comm.allgather(rank, np.asarray(values_local, dtype=np.float64))
        full = np.full(n_compact, np.nan)
        full[ids] = vals
        return full

    return CoarseCommContext(rank=rank, n_ranks=comm.n, allgather=lambda a: comm.allgather(rank, a),
                             allreduce_sum=allreduce_sum, compact_cell_values=compact_cell_values)


def _partitioned(dtau_value, with_global: bool):
    mesh, res, U, r0, bp, bt, cp, n_real, colors, diag, off = _setup("tet", 1, smooth=True)
    nc, ns = mesh.n_cells, mesh.n_sps_per_cell
    assert mesh.n_prism_cells == 0          # 紧凑空间 = 全局单元编号（全是四面体）
    dtau = np.full(nc * ns, dtau_value)
    parts = np.array_split(np.arange(nc), N_RANKS)
    comm = ThreadComm(N_RANKS)
    npr, nte = real_sps_per_cell(1)

    def build(r):
        ids = parts[r]
        bp_l, bt_l, inner, cross = select_rows((bp, bt, cp), ids, 0, True)
        n_real_l = n_real[ids]
        struct = BlockCouplingStructure(inner, ids.size, n_real_l)
        jac = CellBlockJacobian.from_blocks(bp_l, bt_l, n_sps=ns, n_var=NV, cell_is_prism=np.zeros(ids.size, bool),
                                            n_real_prism=npr, n_real_tet=nte)
        ctx = _context(comm, r, ids, nc) if with_global else None
        fac = CoarsePreconditionerFactory(True, ctx)
        rows = (ids[:, None] * ns + np.arange(ns)[None, :]).ravel()
        prec = fac.make(jac, struct, fac.cross_coupling(cross, struct), dtau[rows])
        return dict(ids=ids, rows=rows, prec=prec, fac=fac, struct=struct, jac=jac)

    ranks = comm.run(build)
    return mesh, nc, ns, n_real, diag, off, dtau, r0, cp, ranks, comm


def _apply_all(ranks, comm, v_global, ns):
    """各 rank 同时作用（集合通信要求并发），拼回全局向量。"""
    def act(r):
        rk = ranks[r]
        return rk["prec"].apply(v_global.reshape(-1, ns, NV)[rk["ids"]].reshape(-1, NV)).reshape(-1)

    outs = comm.run(act)
    z = np.empty_like(v_global)
    for r, rk in enumerate(ranks):
        z.reshape(-1, ns * NV)[rk["ids"]] = outs[r].reshape(-1, ns * NV)
    return z


def test_action_matches_dense_definition():
    mesh, nc, ns, n_real, diag, off, dtau, r0, cp, ranks, comm = _partitioned(1e-1, True)
    struct_all = BlockCouplingStructure(cp, nc, n_real)
    A = _dense_A(nc, ns, n_real, diag, off, dtau, struct_all)
    # 全局粗节点（各 rank 的空间拼起来）
    gid = np.empty(nc, dtype=np.int64)
    for rk in ranks:
        sp = rk["prec"]._space
        gid[rk["ids"]] = sp.gid_local
    n_g = ranks[0]["prec"]._space.n_global
    Pg = _dense_P(nc, ns, n_real, gid, n_g)
    Ag = galerkin_coarse_matrix(diag, off, struct_all.row_dof, 1.0 / dtau, struct_all, gid, n_g, ns, NV).toarray()
    np.testing.assert_allclose(Ag, Pg.T @ A @ Pg, rtol=1e-9, atol=1e-9 * np.abs(Ag).max())
    r = np.random.default_rng(5).standard_normal(A.shape[0])
    z1 = Pg @ np.linalg.solve(Ag, Pg.T @ r)
    resid = r - A @ z1
    ref = z1.copy()
    for rk in ranks:
        loc = rk["prec"]._local
        sl = resid.reshape(-1, ns * NV)[rk["ids"]].reshape(-1, NV)
        ref.reshape(-1, ns * NV)[rk["ids"]] += loc.apply(sl).reshape(-1, ns * NV)
    z = _apply_all(ranks, comm, r, ns)
    np.testing.assert_allclose(z, ref, rtol=1e-6, atol=1e-8 * np.abs(ref).max())


def test_global_coarse_reduces_iterations_over_rank_local_preconditioning():
    iters = {}
    for tag, with_global in (("local", False), ("global", True)):
        mesh, nc, ns, n_real, diag, off, dtau, r0, cp, ranks, comm = _partitioned(1e-1, with_global)
        struct_all = BlockCouplingStructure(cp, nc, n_real)
        A = _dense_A(nc, ns, n_real, diag, off, dtau, struct_all)
        b = -np.asarray(r0, dtype=np.float64).reshape(-1)
        x, it, info, _ = gmres_right(lambda v: A @ v, b, lambda v: _apply_all(ranks, comm, v, ns), rtol=1e-6,
                                     restart=400, max_iter=400, flexible=True)
        assert info == 0, (tag, it)
        assert np.linalg.norm(b - A @ x) <= 1.0001e-6 * np.linalg.norm(b)
        iters[tag] = it
    assert iters["global"] < iters["local"], iters
