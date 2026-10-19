"""单元块 D-ILU 预处理（`implicit/blocks/block_ilu.py`）。

1. 作用正确：与按定义稠密构造的 `M = (D~ + L) D~^{-1} (D~ + U)` 的逆逐位对照
   （多色序、只修正对角块）；
2. 有效：CFL ~ 12 下 GMRES 迭代数显著少于块 Jacobi（同一组解析块）。
"""

import numpy as np

from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.fr_residual.jacobian import MeanFlowLinearization, assemble_mean_flow_blocks
from autoflowcfd.core.time_integration.implicit.blocks.block_ilu import BlockCouplingStructure, BlockILUPreconditioner
from autoflowcfd.core.time_integration.implicit.blocks.cell_block_inverse import CellBlockJacobiPreconditioner
from autoflowcfd.core.time_integration.implicit.blocks.cell_blocks import CellBlockJacobian
from autoflowcfd.core.time_integration.implicit.blocks.coloring import greedy_cell_coloring
from autoflowcfd.core.time_integration.implicit.krylov.gmres import gmres_right
from autoflowcfd.core.time_integration.implicit.krylov.jacobian_vector import MatrixFreeJacobian
from autoflowcfd.fr.native_padding import real_sps_per_cell
from autoflowcfd.fr.operators import generate_fr_operators
from tests.unit.test_mean_flow_analytic_jacobian import (
    H, LX, MACH_REF, MU, P_INF, RHO, SCALES, U_INF, _Residual, _channel, _state,
)


def _smooth_state(mesh):
    """光滑的通道状态（抛物线主流 + 正弦横向速度）：随机扰动状态的稳态 Jacobian
    在大伪时间步长下接近奇异，两种预处理都收敛不了，比较没有意义。"""
    X = np.asarray(mesh.sps_coords)
    y = X[..., 1] / H
    Q = np.zeros(X.shape[:2] + (5,))
    Q[..., 0] = RHO
    Q[..., 1] = U_INF * 4.0 * y * (1.0 - y)
    Q[..., 2] = 0.5 * np.sin(np.pi * X[..., 0] / LX)
    Q[..., 4] = P_INF
    U = Q.copy()
    U[..., 1:4] *= RHO
    U[..., 4] = P_INF / 0.4 + 0.5 * RHO * (Q[..., 1:4] ** 2).sum(-1)
    return U


def _setup(kind, order, smooth=False):
    mesh, prov = _channel(kind, order)
    ops = generate_fr_operators(order)
    U = _smooth_state(mesh) if smooth else _state(mesh, 31)
    res = _Residual(mesh, ops, prov, None, True, None)
    r0 = res(U.reshape(-1, 5))
    ctx = MeanFlowLinearization(mesh=mesh, ops=ops, ghost_provider=prov, mu=MU, mach_ref=MACH_REF,
                                low_mach=True)
    bp, bt, cp = assemble_mean_flow_blocks(ctx, U, residual=r0.reshape(U.shape), want_coupling=True)
    nc = mesh.n_cells
    npr, nte = real_sps_per_cell(order)
    cip = np.arange(nc) < mesh.n_prism_cells
    n_real = np.where(cip, npr, nte)
    flat = get_flat_face_geometry(mesh, ops)
    colors = greedy_cell_coloring(flat.owner_cell, flat.neighbor_cell, nc)
    diag = np.concatenate([bp.reshape(-1), bt.reshape(-1)]).astype(np.float32)
    off = np.concatenate([[0], np.cumsum((5 * n_real) ** 2)[:-1]]).astype(np.int64)
    return mesh, res, U, r0, bp, bt, cp, n_real, colors, diag, off


def test_wavefront_levels_respect_rank_dependencies():
    """波前层内单元互不依赖：同层任意两个单元不相邻，且每个单元名次更低的邻居都在更低的层。"""
    mesh, res, U, r0, bp, bt, cp, n_real, colors, diag, off = _setup("tet", 1)
    struct = BlockCouplingStructure(cp, mesh.n_cells, n_real)
    level = np.empty(mesh.n_cells, dtype=np.int64)
    for k, cells in enumerate(struct.levels):
        level[cells] = k
    assert sorted(np.concatenate(struct.levels).tolist()) == list(range(mesh.n_cells))
    for c in range(mesh.n_cells):
        for y in struct.cols[struct.indptr[c]:struct.indptr[c + 1]]:
            assert level[y] != level[c]
            assert (level[y] < level[c]) == (struct.rank[y] < struct.rank[c])


def _asymmetric_structure(rng, n=3000, n_var=2):
    """单元链 + 随机远邻的块稀疏结构，约 5% 的耦合只有单向（波前分层必须对任意结构成立，见
    `block_ilu.rcm_wavefronts` 文档）。"""
    pairs = set()
    for c in range(n - 1):
        pairs.add((c, c + 1))
        pairs.add((c + 1, c))
    for _ in range(3 * n):
        c, y = rng.integers(0, n, 2)
        if c != y:
            pairs.add((int(c), int(y)))
            if rng.random() > 0.05:
                pairs.add((int(y), int(c)))
    rows, cols = np.array(sorted(pairs)).T
    n_real = rng.integers(1, 4, n)
    sizes = (n_var * n_real[rows]) * (n_var * n_real[cols])
    offset = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    data = (0.1 * rng.normal(size=int(sizes.sum()))).astype(np.float32)
    indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=n))]).astype(np.int64)
    struct = BlockCouplingStructure.from_csr(indptr, cols, offset, data, n_real, n_var)
    m = n_var * n_real
    diag_off = np.concatenate([[0], np.cumsum(m * m)[:-1]]).astype(np.int64)
    diag = np.concatenate([(rng.normal(size=(k, k)) + 4.0 * np.eye(k)).ravel() for k in m]).astype(np.float32)
    return struct, diag, diag_off


def _serial_solve(inv, diag_off, s, x, n_sps):
    """D-ILU 前代 + 回代按名次逐个单元顺序做（与并行核同一个逐单元算式与累加顺序）。"""
    nv, y = s.n_var, np.zeros_like(x)
    order = np.argsort(s.rank)

    def blk(flat, start, r, c):
        return flat[start:start + r * c].astype(np.float64).reshape(r, c)

    for c in order:
        m, b0 = s.row_dof[c], c * n_sps * nv
        rhs = x[b0:b0 + m].copy()
        for k in range(s.indptr[c], s.indptr[c + 1]):
            yc = s.cols[k]
            if s.rank[yc] < s.rank[c]:
                B = blk(s.data, s.offset[k], m, s.row_dof[yc])
                for i in range(m):
                    acc = 0.0
                    for j in range(s.row_dof[yc]):
                        acc += B[i, j] * y[yc * n_sps * nv + j]
                    rhs[i] -= acc
        D = blk(inv, diag_off[c], m, m)
        for i in range(m):
            acc = 0.0
            for j in range(m):
                acc += D[i, j] * rhs[j]
            y[b0 + i] = acc
    for c in order[::-1]:
        m, b0 = s.row_dof[c], c * n_sps * nv
        t = np.zeros(m)
        for k in range(s.indptr[c], s.indptr[c + 1]):
            zc = s.cols[k]
            if s.rank[zc] > s.rank[c]:
                B = blk(s.data, s.offset[k], m, s.row_dof[zc])
                for i in range(m):
                    acc = 0.0
                    for j in range(s.row_dof[zc]):
                        acc += B[i, j] * y[zc * n_sps * nv + j]
                    t[i] += acc
        D = blk(inv, diag_off[c], m, m)
        for i in range(m):
            acc = 0.0
            for j in range(m):
                acc += D[i, j] * t[j]
            y[b0 + i] -= acc
    return y


def test_asymmetric_coupling_wavefronts_have_no_same_level_dependencies_and_solve_is_exact():
    """耦合模式不对称时：前代与回代读到的邻居都在更早的层（分层按对称化结构，`block_ilu.rcm_wavefronts`），
    并行解与按名次顺序的串行解逐位相同、重复作用逐位相同。修复前按单侧行分层，回代有同层读写冲突。"""
    from autoflowcfd.core.time_integration.implicit.blocks.block_ilu import factor_block_ilu, solve_block_ilu

    rng = np.random.default_rng(5)
    s, diag, diag_off = _asymmetric_structure(rng)
    n = s.n_real.size
    level = np.empty(n, dtype=np.int64)
    for k, cells in enumerate(s.levels):
        level[cells] = k
    asym = 0
    for c in range(n):
        row = s.cols[s.indptr[c]:s.indptr[c + 1]]
        for y in row:
            asym += c not in s.cols[s.indptr[y]:s.indptr[y + 1]]
            if s.rank[y] < s.rank[c]:
                assert level[y] < level[c], "前代读到同层或更晚层的值"
            else:
                assert level[y] > level[c], "回代读到同层或更早层的值"
    assert asym > 100            # 判据确实覆盖了单向耦合

    n_sps = 3
    inv = factor_block_ilu(diag, diag_off, s, np.full(n * n_sps, 0.5), n_sps)
    x = rng.normal(size=n * n_sps * s.n_var)
    got = [solve_block_ilu(inv, diag_off, s, x, np.zeros_like(x), n_sps) for _ in range(3)]
    ref = _serial_solve(inv, diag_off, s, x, n_sps)
    for g in got:
        assert np.array_equal(g, ref)


def test_apply_matches_dense_definition():
    mesh, res, U, r0, bp, bt, cp, n_real, colors, diag, off = _setup("tet", 1)
    nc, ns = mesh.n_cells, mesh.n_sps_per_cell
    dtau = np.full(nc * ns, 2e-5) * (1.0 + np.random.default_rng(0).random(nc * ns))
    struct = BlockCouplingStructure(cp, nc, n_real)
    prec = BlockILUPreconditioner(diag, off, struct, dtau, ns)
    rank = struct.rank
    # 稠密构造（只含真实自由度）
    dofs = [np.arange(5 * n_real[c]) + c * ns * 5 for c in range(nc)]
    idx = np.concatenate(dofs)
    pos = {c: np.arange(sum(5 * n_real[:c]), sum(5 * n_real[:c + 1])) for c in range(nc)}
    n = idx.size
    D = np.zeros((n, n))
    L = np.zeros((n, n))
    Uu = np.zeros((n, n))
    blocks = {}
    for g in cp.groups:
        for k in range(g.rows.size):
            blocks[(int(g.rows[k]), int(g.cols[k]))] = g.blocks[k].astype(np.float64)
    Dt = {}
    order = np.argsort(rank)
    for c in order:
        m = 5 * n_real[c]
        a = diag[off[c]:off[c] + m * m].astype(np.float64).reshape(m, m)
        a = a + np.diag(np.repeat(1.0 / dtau[c * ns:c * ns + n_real[c]], 5))
        for (r, y), B in blocks.items():
            if r == c and rank[y] < rank[c]:
                a = a - B @ np.linalg.inv(Dt[y]).astype(np.float32).astype(np.float64) @ blocks[(y, c)]
        Dt[c] = a
        D[np.ix_(pos[c], pos[c])] = a
    for (r, y), B in blocks.items():
        (L if rank[y] < rank[r] else Uu)[np.ix_(pos[r], pos[y])] = B
    Dinv = np.zeros_like(D)
    for c in range(nc):
        Dinv[np.ix_(pos[c], pos[c])] = np.linalg.inv(Dt[c])
    M = (D + L) @ Dinv @ (D + Uu)
    v = np.random.default_rng(1).standard_normal((nc * ns, 5))
    z = prec.apply(v.copy()).reshape(-1)
    z_ref = np.linalg.solve(M, v.reshape(-1)[idx])
    np.testing.assert_allclose(z[idx], z_ref, rtol=2e-4, atol=1e-6 * np.abs(z_ref).max())


def test_fewer_gmres_iterations_than_block_jacobi_at_large_cfl():
    mesh, res, U, r0, bp, bt, cp, n_real, colors, diag, off = _setup("tet", 1, smooth=True)
    nc, ns = mesh.n_cells, mesh.n_sps_per_cell
    npr, nte = real_sps_per_cell(1)
    cip = np.arange(nc) < mesh.n_prism_cells
    jac = CellBlockJacobian.from_blocks(bp, bt, n_sps=ns, n_var=5, cell_is_prism=cip,
                                        n_real_prism=npr, n_real_tet=nte)
    struct = BlockCouplingStructure(cp, nc, n_real)
    u0 = U.reshape(-1, 5)
    mf = MatrixFreeJacobian(res, u0, r0, SCALES)
    # 伪时间步长约为单元声学时间尺度的 12 倍（CFL ~ 12）
    dtau = np.full(nc * ns, 1e-3)

    def A(x):
        v = x.reshape(-1, 5)
        return (mf.matvec(v) + v / dtau[:, None]).reshape(-1)

    iters = {}
    for tag, prec in (("bj", CellBlockJacobiPreconditioner(jac, dtau)),
                      ("ilu", BlockILUPreconditioner(diag, off, struct, dtau, ns))):
        _, it, info, _ = gmres_right(A, -r0.reshape(-1), lambda x: prec.apply(x.reshape(-1, 5)).reshape(-1),
                                     rtol=1e-2, restart=30, max_iter=400)
        assert info == 0, (tag, it)
        iters[tag] = it
    assert iters["ilu"] * 1.6 <= iters["bj"], iters
