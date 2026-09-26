"""单元块 D-ILU 预处理（`implicit/block_ilu.py`）。

1. 作用正确：与按定义稠密构造的 `M = (D~ + L) D~^{-1} (D~ + U)` 的逆逐位对照
   （多色序、只修正对角块）；
2. 有效：CFL ~ 12 下 GMRES 迭代数显著少于块 Jacobi（同一组解析块）。
"""

import numpy as np

from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.fr_operators.kernels import resolve_ausm_precond_mode
from autoflowcfd.core.fr_residual.jacobian import MeanFlowLinearization, assemble_mean_flow_blocks
from autoflowcfd.core.time_integration.implicit.block_ilu import BlockCouplingStructure, BlockILUPreconditioner
from autoflowcfd.core.time_integration.implicit.block_jacobi import (
    CellBlockJacobian, CellBlockJacobiPreconditioner, greedy_cell_coloring,
)
from autoflowcfd.core.time_integration.implicit.gmres import gmres_right
from autoflowcfd.core.time_integration.implicit.jacobian_vector import MatrixFreeJacobian
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
                                precond_mode=resolve_ausm_precond_mode(), low_mach=True)
    bp, bt, cp = assemble_mean_flow_blocks(ctx, U, residual=r0.reshape(U.shape), want_coupling=True)
    nc, ns = mesh.n_cells, mesh.n_sps_per_cell
    npr, nte = real_sps_per_cell(order)
    cip = np.arange(nc) < mesh.n_prism_cells
    n_real = np.where(cip, npr, nte)
    flat = get_flat_face_geometry(mesh, ops)
    colors = greedy_cell_coloring(flat.owner_cell, flat.neighbor_cell, nc)
    diag = np.concatenate([bp.reshape(-1), bt.reshape(-1)]).astype(np.float32)
    off = np.concatenate([[0], np.cumsum((5 * n_real) ** 2)[:-1]]).astype(np.int64)
    return mesh, res, U, r0, bp, bt, cp, n_real, colors, diag, off


def test_apply_matches_dense_definition():
    mesh, res, U, r0, bp, bt, cp, n_real, colors, diag, off = _setup("tet", 1)
    nc, ns = mesh.n_cells, mesh.n_sps_per_cell
    dtau = np.full(nc * ns, 2e-5) * (1.0 + np.random.default_rng(0).random(nc * ns))
    struct = BlockCouplingStructure(cp, nc, n_real)
    prec = BlockILUPreconditioner(diag, off, struct, colors, dtau, ns)
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
    order = np.argsort(colors, kind="stable")
    for c in order:
        m = 5 * n_real[c]
        a = diag[off[c]:off[c] + m * m].astype(np.float64).reshape(m, m)
        a = a + np.diag(np.repeat(1.0 / dtau[c * ns:c * ns + n_real[c]], 5))
        for (r, y), B in blocks.items():
            if r == c and colors[y] < colors[c]:
                a = a - B @ np.linalg.inv(Dt[y]).astype(np.float32).astype(np.float64) @ blocks[(y, c)]
        Dt[c] = a
        D[np.ix_(pos[c], pos[c])] = a
    for (r, y), B in blocks.items():
        (L if colors[y] < colors[r] else Uu)[np.ix_(pos[r], pos[y])] = B
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
                      ("ilu", BlockILUPreconditioner(diag, off, struct, colors, dtau, ns))):
        _, it, info, _ = gmres_right(A, -r0.reshape(-1), lambda x: prec.apply(x.reshape(-1, 5)).reshape(-1),
                                     rtol=1e-2, restart=30, max_iter=400)
        assert info == 0, (tag, it)
        iters[tag] = it
    assert iters["ilu"] * 1.6 <= iters["bj"], iters
