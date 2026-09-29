"""P0 差分装配同时截取面邻居耦合块（`implicit/cell_blocks.py` 的 `coupling_graph`）。

判据：
1. 距离 2 着色的定义（同色单元既不相邻、也不共享面邻居）；
2. 截取的 `J_cc` 与 `J_cy` 对照稠密逐列差分 Jacobian（P0 残差模板是面邻居：
   单元内局部梯度对常数恒为零，粘性项只剩罚项），模板之外的块必须恰为零；
3. 块缓存在 P0 + 耦合图下改用块 ILU，大 dtau 下 GMRES 迭代数少于块 Jacobi。
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.time_integration.implicit.block_ilu import BlockILUPreconditioner
from autoflowcfd.core.time_integration.implicit.block_jacobi import BlockJacobiCache, CellBlockJacobiPreconditioner
from autoflowcfd.core.time_integration.implicit.cell_blocks import CellBlockJacobian
from autoflowcfd.core.time_integration.implicit.coloring import (
    coupling_graph_from_faces, distance2_cell_coloring, greedy_cell_coloring, stencil_pairs,
)
from autoflowcfd.core.time_integration.implicit.gmres import gmres_right
from autoflowcfd.core.time_integration.implicit.jacobian_vector import MatrixFreeJacobian
from autoflowcfd.fr.operators import generate_fr_operators
from tests.unit.test_mean_flow_analytic_jacobian import SCALES, _channel, _Residual, _state

MU = 1.8e-5


def test_distance2_coloring_definition():
    rng = np.random.default_rng(0)
    n = 400
    own = rng.integers(0, n, 1500)
    nb = rng.integers(-1, n, 1500)
    col = distance2_cell_coloring(own, nb, n)
    rows, cols = stencil_pairs(own, nb, n)
    nbrs = [set() for _ in range(n)]
    for r, c in zip(rows, cols):
        nbrs[r].add(c)
    for c in range(n):
        near = set(nbrs[c])
        for y in nbrs[c]:
            near |= nbrs[y]
        near.discard(c)
        assert all(col[y] != col[c] for y in near)
    # 距离 1 着色同样满足自己的定义
    col1 = greedy_cell_coloring(own, nb, n)
    assert all(col1[r] != col1[c] for r, c in zip(rows, cols))


def _p0_problem(kind):
    mesh, prov = _channel(kind, 0)
    ops = generate_fr_operators(0)
    U = _state(mesh, 1)
    mu_t = np.full((mesh.n_cells, 1), 5.0 * MU)
    res = _Residual(mesh, ops, prov, mu_t, True, None)
    u0 = U.reshape(-1, 5)
    fl = get_flat_face_geometry(mesh, ops)
    return mesh, res, u0, res(u0), fl


def _dense(res, u0, r0, n):
    h = 1.4901161193847656e-08 * (1.0 + np.sqrt(np.mean((u0 / SCALES) ** 2))) * SCALES
    J = np.zeros((5 * n, 5 * n))
    for c in range(n):
        for v in range(5):
            up = u0.copy()
            up[c, v] += h[v]
            J[:, 5 * c + v] = ((res(up) - r0) / h[v]).ravel()
    return J


@pytest.mark.parametrize("kind", ["tet", "prism"])
def test_p0_fd_coupling_matches_dense_jacobian(kind):
    mesh, res, u0, r0, fl = _p0_problem(kind)
    n = mesh.n_cells
    graph = coupling_graph_from_faces(fl.owner_cell, fl.neighbor_cell, n)
    cip = np.arange(n) < mesh.n_prism_cells
    fd = CellBlockJacobian(res, u0, r0, SCALES, n_sps=1, cell_is_prism=cip, n_real_prism=1, n_real_tet=1,
                           colors=None, coupling_graph=graph)
    assert fd.n_residual_evals == 5 * (int(graph.colors.max()) + 1)
    J = _dense(res, u0, r0, n)
    slot = np.empty(n, dtype=np.int64)
    slot[cip] = np.arange(cip.sum())
    slot[~cip] = np.arange((~cip).sum())
    for c in range(n):
        blk = (fd.blocks_prism if cip[c] else fd.blocks_tet)[slot[c]]
        ref = J[5 * c:5 * c + 5, 5 * c:5 * c + 5]
        assert np.abs(blk - ref).max() <= 1e-5 * np.abs(ref).max()
    covered = set()
    for g in fd.coupling.groups:
        for k in range(g.rows.size):
            r, c = int(g.rows[k]), int(g.cols[k])
            covered.add((r, c))
            ref = J[5 * r:5 * r + 5, 5 * c:5 * c + 5]
            assert np.abs(g.blocks[k] - ref).max() <= 1e-5 * np.abs(ref).max()
    for r in range(n):
        for c in range(n):
            if r != c and (r, c) not in covered:
                assert not np.any(J[5 * r:5 * r + 5, 5 * c:5 * c + 5])


def test_p0_cache_switches_to_block_ilu_and_beats_block_jacobi():
    mesh, res, u0, r0, fl = _p0_problem("tet")
    n = mesh.n_cells
    cip = np.arange(n) < mesh.n_prism_cells
    colors = greedy_cell_coloring(fl.owner_cell, fl.neighbor_cell, n)
    graph = coupling_graph_from_faces(fl.owner_cell, fl.neighbor_cell, n)
    cache = BlockJacobiCache(cell_is_prism=cip, colors=colors, n_sps=1, n_real_prism=1, n_real_tet=1, n_var=5,
                             coupling_graph=lambda: graph)
    cache.begin_step(res, u0, r0, SCALES, np.full(u0.shape[0], 1e-2))
    assert cache.coupling is not None
    mf = MatrixFreeJacobian(res, u0, r0, SCALES)
    dtau = np.full(n, 1e-2)          # 大 CFL：I/dtau 远小于 J

    def A(x):
        v = x.reshape(-1, 5)
        return (mf.matvec(v) + v / dtau[:, None]).reshape(-1)

    iters = {}
    for name, prec in (("ilu", cache.preconditioner(dtau, 5)),
                       ("bj", CellBlockJacobiPreconditioner(cache.jac, dtau))):
        assert isinstance(prec, BlockILUPreconditioner) == (name == "ilu")
        _, it, info, _ = gmres_right(A, -r0.reshape(-1), lambda x: prec.apply(x.reshape(-1, 5)).reshape(-1),
                                     rtol=1e-3, restart=60, max_iter=400)
        assert info == 0, (name, it)
        iters[name] = it
    assert iters["ilu"] * 1.5 <= iters["bj"], iters
