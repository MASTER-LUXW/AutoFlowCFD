"""多层预处理（`implicit/coarse/`）：

1. 组成部分（装配矩阵乘、转移算子、Galerkin 粗算子、粗层块稀疏存储、聚合层次）与按定义
   稠密构造的矩阵逐项对照（通道 P1 四面体的真实解析块）；
2. 两层作用 `z = P A_c^{-1} P^T r + M_ILU^{-1}(r - A z1)` 与稠密公式一致；
3. 有效：大 CFL、真实右端项 `-R` 下，灵活 GMRES 的迭代数少于块 ILU。"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from autoflowcfd.core.time_integration.implicit.block_ilu import BlockCouplingStructure, BlockILUPreconditioner
from autoflowcfd.core.time_integration.implicit.coarse import (
    MultilevelPreconditioner, build_hierarchy, preconditioner_hierarchy,
)
from autoflowcfd.core.time_integration.implicit.coarse.galerkin import (
    assembled_matvec, galerkin_coarse_matrix, prolong, restrict,
)
from autoflowcfd.core.time_integration.implicit.coarse.multilevel import _level_from_scipy
from autoflowcfd.core.time_integration.implicit.gmres import gmres_right
from tests.unit.test_block_ilu_precond import _setup

NV = 5


def _system():
    mesh, res, U, r0, bp, bt, cp, n_real, colors, diag, off = _setup("tet", 1)
    nc, ns = mesh.n_cells, mesh.n_sps_per_cell
    dtau = 2e-5 * (1.0 + np.random.default_rng(0).random(nc * ns))
    struct = BlockCouplingStructure(cp, nc, n_real)
    return nc, ns, n_real, diag, off, dtau, struct


def _dense_A(nc, ns, n_real, diag, off, dtau, struct):
    """全部槽位（含零填充）上的稠密 `A = blockdiag(J_cc + I/dtau) + {J_cy}`。"""
    N = nc * ns * NV
    A = np.diag(np.repeat(1.0 / dtau, NV))
    for c in range(nc):
        m = NV * n_real[c]
        b0 = c * ns * NV
        A[b0:b0 + m, b0:b0 + m] += diag[off[c]:off[c] + m * m].astype(np.float64).reshape(m, m)
        for k in range(struct.indptr[c], struct.indptr[c + 1]):
            y = struct.cols[k]
            my = NV * n_real[y]
            blk = struct.data[struct.offset[k]:struct.offset[k] + m * my].astype(np.float64).reshape(m, my)
            A[b0:b0 + m, y * ns * NV:y * ns * NV + my] += blk
    assert A.shape == (N, N)
    return A


def _dense_P(nc, ns, n_real, agg, n_agg):
    P = np.zeros((nc * ns * NV, n_agg * NV))
    for c in range(nc):
        for i in range(NV * n_real[c]):
            P[c * ns * NV + i, agg[c] * NV + i % NV] = 1.0
    return P


def test_assembled_matvec_matches_dense():
    nc, ns, n_real, diag, off, dtau, struct = _system()
    A = _dense_A(nc, ns, n_real, diag, off, dtau, struct)
    x = np.random.default_rng(1).standard_normal(A.shape[0])
    out = np.empty_like(x)
    assembled_matvec(x, diag, off, struct.row_dof, 1.0 / dtau, struct.indptr, struct.cols, struct.offset,
                     struct.data, ns, NV, out)
    np.testing.assert_allclose(out, A @ x, rtol=1e-10, atol=1e-10 * np.abs(A @ x).max())


def test_transfer_and_galerkin_match_dense():
    nc, ns, n_real, diag, off, dtau, struct = _system()
    (agg, n_agg), = build_hierarchy(struct.indptr, struct.cols, nc, nc - 1)
    A = _dense_A(nc, ns, n_real, diag, off, dtau, struct)
    P = _dense_P(nc, ns, n_real, agg, n_agg)
    rng = np.random.default_rng(2)
    r = rng.standard_normal(A.shape[0])
    np.testing.assert_allclose(restrict(r, struct.row_dof, agg, n_agg, ns, NV), P.T @ r, rtol=1e-12, atol=1e-12)
    zc = rng.standard_normal(n_agg * NV)
    z = np.empty(A.shape[0])
    prolong(zc, struct.row_dof, agg, ns, NV, z)
    np.testing.assert_array_equal(z, P @ zc)
    Ac = galerkin_coarse_matrix(diag, off, struct.row_dof, 1.0 / dtau, struct, agg, n_agg, ns, NV).toarray()
    ref = P.T @ A @ P
    np.testing.assert_allclose(Ac, ref, rtol=1e-9, atol=1e-9 * np.abs(ref).max())


def test_hierarchy_covers_every_node_with_connected_aggregates():
    nc, ns, n_real, diag, off, dtau, struct = _system()
    hier = build_hierarchy(struct.indptr, struct.cols, nc, 4)
    assert len(hier) >= 2
    ip, cs, n = struct.indptr, struct.cols, nc
    for agg, n_agg in hier:
        assert agg.shape == (n,) and agg.min() == 0 and agg.max() == n_agg - 1
        assert np.unique(agg).size == n_agg and n_agg < n
        rows = np.repeat(np.arange(n), np.diff(ip))
        same = agg[rows] == agg[cs]
        g = sp.csr_matrix((np.ones(int(same.sum())), (rows[same], cs[same])), shape=(n, n))
        n_comp, labels = connected_components(g, directed=False)
        # 每个聚合体在本层图上连通：连通分量与聚合体一一对应
        assert n_comp == n_agg
        assert all(np.unique(agg[labels == k]).size == 1 for k in range(n_comp))
        G = sp.csr_matrix((np.ones(cs.size), (agg[rows], agg[cs])), shape=(n_agg, n_agg))
        G.setdiag(0)
        G.eliminate_zeros()
        G.sort_indices()
        ip, cs, n = G.indptr.astype(np.int64), G.indices.astype(np.int64), n_agg


def test_coarse_level_storage_round_trips_galerkin_matrix():
    nc, ns, n_real, diag, off, dtau, struct = _system()
    (agg, n_agg), = build_hierarchy(struct.indptr, struct.cols, nc, nc - 1)
    Ac = galerkin_coarse_matrix(diag, off, struct.row_dof, 1.0 / dtau, struct, agg, n_agg, ns, NV)
    level = _level_from_scipy(Ac, NV)
    x = np.random.default_rng(3).standard_normal(n_agg * NV)
    ref = Ac @ x
    # 粗层块按 float32 存（与细层同一约定），对照容差按单精度
    np.testing.assert_allclose(level.matvec(x), ref, rtol=1e-5, atol=1e-6 * np.abs(ref).max())


def test_two_level_action_matches_dense_formula():
    """层次只有 p 粗化一层（每单元一个节点）时，作用就是精确两层公式。"""
    nc, ns, n_real, diag, off, dtau, struct = _system()
    hier = preconditioner_hierarchy(struct, high_order=True)
    assert len(hier) == 1 and hier[0][1] == nc
    agg, n_agg = hier[0]
    A = _dense_A(nc, ns, n_real, diag, off, dtau, struct)
    P = _dense_P(nc, ns, n_real, agg, n_agg)
    Ac = galerkin_coarse_matrix(diag, off, struct.row_dof, 1.0 / dtau, struct, agg, n_agg, ns, NV).toarray()
    ilu = BlockILUPreconditioner(diag, off, struct, dtau, ns)
    prec = MultilevelPreconditioner(diag, off, struct, hier, dtau, ns)
    assert prec.flexible
    r = np.random.default_rng(4).standard_normal(A.shape[0])
    z1 = P @ np.linalg.solve(Ac, P.T @ r)
    ref = z1 + ilu.apply((r - A @ z1).reshape(-1, NV)).reshape(-1)
    z = prec.apply(r.reshape(-1, NV)).reshape(-1)
    np.testing.assert_allclose(z, ref, rtol=1e-6, atol=1e-8 * np.abs(ref).max())


def test_fewer_iterations_than_block_ilu_on_true_residual():
    """多层（p 层 + 两层聚合，粗层走 K 循环）在大 CFL 的真实右端项上比块 ILU 少用迭代。"""
    from tests.unit.test_block_ilu_precond import _setup as setup
    mesh, res, U, r0, bp, bt, cp, n_real, colors, diag, off = setup("tet", 1, smooth=True)
    nc, ns = mesh.n_cells, mesh.n_sps_per_cell
    dtau = np.full(nc * ns, 1e-1)
    struct = BlockCouplingStructure(cp, nc, n_real)
    hier = [(np.arange(nc, dtype=np.int64), nc)] + build_hierarchy(struct.indptr, struct.cols, nc, 8)
    assert len(hier) >= 3            # p 层 + 至少一层中间聚合（走 K 循环）+ 最粗层

    def A(x):
        out = np.empty_like(x)
        assembled_matvec(x, diag, off, struct.row_dof, 1.0 / dtau, struct.indptr, struct.cols, struct.offset,
                         struct.data, ns, NV, out)
        return out

    b = -np.asarray(r0, dtype=np.float64).reshape(-1)
    iters = {}
    for tag, prec in (("ilu", BlockILUPreconditioner(diag, off, struct, dtau, ns)),
                      ("ml", MultilevelPreconditioner(diag, off, struct, hier, dtau, ns))):
        x, it, info, _ = gmres_right(A, b, lambda v: prec.apply(v.reshape(-1, NV)).reshape(-1), rtol=1e-6,
                                     restart=400, max_iter=400, flexible=prec.flexible)
        assert info == 0, (tag, it)
        assert np.linalg.norm(b - A(x)) <= 1.0001e-6 * np.linalg.norm(b)
        iters[tag] = it
    assert iters["ml"] < iters["ilu"], iters
