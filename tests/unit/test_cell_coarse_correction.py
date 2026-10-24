"""块 Jacobi 档的单元级叠加性粗校正（`coarse/cell_coarse.py`）：限制权重规则、粗矩阵 `R (J + I/dtau) P`、
粗层近似逆与加性组合。"""

import numpy as np
import pytest

from autoflowcfd.core.time_integration.implicit.blocks.preconditioner import PseudoTransientDiagonal
from autoflowcfd.core.time_integration.implicit.coarse.cell_coarse import (
    CellCoarseMatrix, CellCoarseSolve, CellCoarseTopology, CoarseCorrectedPreconditioner, restriction_weights,
)
from autoflowcfd.core.time_integration.implicit.krylov.gmres import gmres_right

NV = 5


def _uniform(W, real):
    """真实解点上合计为单元体积的均匀权重（正性限制器 `norm_weights` 的定义，测试里按定义构造）。"""
    w = np.where(real, W, 0.0)
    return np.where(real, (w.sum(axis=1) / real.sum(axis=1))[:, None], 0.0)


def test_restriction_weights_use_the_cell_integral_unless_it_has_negative_entries():
    """积分权重全部非负的单元用它；有负权重的单元改为真实解点上的均匀权重，合计同一体积；零填充槽位为 0。"""
    W = np.array([[0.0, 0.5, 0.5, 9.0],          # 棱柱式：顶点权重 0
                  [-0.05, 0.4, 0.4, 0.25],       # 四面体式：顶点权重为负
                  [0.2, 0.3, 7.0, 7.0]])         # 两个零填充槽位
    real = np.array([[True, True, True, False], [True, True, True, True], [True, True, False, False]])
    w = restriction_weights(W, _uniform(W, real))
    np.testing.assert_allclose(w[0], [0.0, 0.5, 0.5, 0.0])
    np.testing.assert_allclose(w[1], np.full(4, 1.0 / 4))
    np.testing.assert_allclose(w[2], [0.2, 0.3, 0.0, 0.0])
    with pytest.raises(ValueError):
        restriction_weights(np.array([[-1.0, 0.5]]), np.zeros((1, 2)))


def _fine_system(n_cells, n_sps, seed, pairs):
    """块结构的细层系统：每单元 2 或 `n_sps` 个真实解点（其余为零填充），对角块 `J_cc` 与面邻居耦合块 `J_cy`。
    返回稠密 `J`（零填充行列为 0）、真实掩码、`J_cc` 与按 `pairs` 顺序的 `J_cy`。"""
    rng = np.random.default_rng(seed)
    n_real = np.where(np.arange(n_cells) % 3 == 0, 2, n_sps)
    real = np.arange(n_sps)[None, :] < n_real[:, None]
    nd = n_cells * n_sps * NV
    J = np.zeros((nd, nd))
    Jcc = np.zeros((n_cells, n_sps, NV, n_sps, NV))
    for c in range(n_cells):
        m = n_real[c] * NV
        blk = rng.normal(0.0, 0.3, (m, m)) + 3.0 * np.eye(m)
        Jcc[c, :n_real[c], :, :n_real[c], :] = blk.reshape(n_real[c], NV, n_real[c], NV)
        o = c * n_sps * NV
        J[o:o + m, o:o + m] = blk
    Jcy = []
    for r, y in pairs:
        blk = rng.normal(0.0, 0.3, (n_real[r] * NV, n_real[y] * NV))
        full = np.zeros((n_sps, NV, n_sps, NV))
        full[:n_real[r], :, :n_real[y], :] = blk.reshape(n_real[r], NV, n_real[y], NV)
        Jcy.append(full)
        J[r * n_sps * NV:r * n_sps * NV + blk.shape[0], y * n_sps * NV:y * n_sps * NV + blk.shape[1]] = blk
    return J, real, Jcc, np.array(Jcy)


def _coarse_blocks(Jcc, Jcy, pairs, w):
    """单元对 5x5 约化（装配器给出的量）：行按 `w` 加权求和、列对真实解点求和。"""
    diag = np.einsum("cs,csatb->cab", w, Jcc)
    blocks = np.array([np.einsum("s,satb->ab", w[r], Jcy[k]) for k, (r, _) in enumerate(pairs)])
    return diag, blocks


def _prolongation(real):
    n_cells, n_sps = real.shape
    P = np.zeros((n_cells * n_sps * NV, n_cells * NV))
    for c in range(n_cells):
        for s in range(n_sps):
            if real[c, s]:
                for v in range(NV):
                    P[(c * n_sps + s) * NV + v, c * NV + v] = 1.0
    return P


def _chain_pairs(n_cells, extra=()):
    pairs = [(c, c + 1) for c in range(n_cells - 1)] + [(c + 1, c) for c in range(n_cells - 1)]
    return pairs + list(extra)


def test_coarse_matrix_is_r_j_plus_ptc_p():
    """粗矩阵（对角块 + 单元对耦合块 + 伪时间项 sum w/dtau）与稠密的 `R (J + I/dtau) P` 一致。"""
    n_cells, n_sps = 9, 3
    pairs = _chain_pairs(n_cells, extra=[(0, 5), (7, 2)])
    J, real, Jcc, Jcy = _fine_system(n_cells, n_sps, 1, pairs)
    rng = np.random.default_rng(2)
    w = np.where(real, rng.uniform(0.5, 1.5, real.shape), 0.0)
    dtau = rng.uniform(0.1, 1.0, n_cells * n_sps)
    diag, blocks = _coarse_blocks(Jcc, Jcy, pairs, w)
    topo = CellCoarseTopology(n_cells, np.array(pairs)[:, 0], np.array(pairs)[:, 1])
    mat = CellCoarseMatrix(topo, diag, blocks)
    inv_dtau = mat.ptc_diagonal(w, dtau, n_sps)
    P = _prolongation(real)
    R = (P * np.repeat(w.reshape(-1), NV)[:, None]).T
    A = J + np.diag(np.repeat(1.0 / dtau, NV))
    np.testing.assert_allclose(mat.scipy_matrix(inv_dtau).toarray(), R @ A @ P, rtol=2e-6, atol=1e-6)


def test_topology_rejects_diagonal_and_duplicate_pairs():
    with pytest.raises(ValueError):
        CellCoarseTopology(3, np.array([0, 1]), np.array([1, 1]))
    with pytest.raises(ValueError):
        CellCoarseTopology(3, np.array([0, 0]), np.array([1, 1]))


def _base(J, real, dtau, n_sps):
    """细层块 Jacobi（真实解点上逐单元的 `J_cc + I/dtau` 求逆，零填充行 `dtau v`）。"""
    n_cells = real.shape[0]
    A = J + np.diag(np.repeat(1.0 / dtau, NV))
    m = n_sps * NV
    inv = np.zeros_like(A)
    for c in range(n_cells):
        idx = np.arange(c * m, (c + 1) * m)
        keep = np.repeat(real[c], NV)
        sub = idx[keep]
        inv[np.ix_(sub, sub)] = np.linalg.inv(A[np.ix_(sub, sub)])
        pad = idx[~keep]
        inv[pad, pad] = np.repeat(dtau.reshape(n_cells, n_sps)[c], NV)[~keep]

    class _Base(PseudoTransientDiagonal):
        def apply(self, v):
            return (inv @ v.reshape(-1)).reshape(v.shape)

    return _Base(dtau, NV)


def test_small_problem_uses_an_exact_coarse_solve():
    """节点数不超过最粗层上限时粗层是精确解：作用等于 `M_bj^{-1} v + P A_c^{-1} R v`；拆分形式与整体作用一致，
    零填充行上粗校正为零。"""
    n_cells, n_sps = 9, 3
    pairs = _chain_pairs(n_cells)
    J, real, Jcc, Jcy = _fine_system(n_cells, n_sps, 3, pairs)
    W = np.where(real, 1.0, 0.0) * np.linspace(0.5, 1.5, n_sps)[None, :]
    w = restriction_weights(W, _uniform(W, real))
    dtau = np.full(n_cells * n_sps, 0.3)
    diag, blocks = _coarse_blocks(Jcc, Jcy, pairs, w)
    topo = CellCoarseTopology(n_cells, np.array(pairs)[:, 0], np.array(pairs)[:, 1])
    assert topo.hierarchy == []
    mat = CellCoarseMatrix(topo, diag, blocks)
    base = _base(J, real, dtau, n_sps)
    prec = CoarseCorrectedPreconditioner(base, CellCoarseSolve(mat, mat.ptc_diagonal(w, dtau, n_sps)), w, real,
                                         n_sps)
    v = np.random.default_rng(4).normal(size=(n_cells * n_sps, NV))
    v[~real.reshape(-1)] = 0.0
    P = _prolongation(real)
    R = (P * np.repeat(w.reshape(-1), NV)[:, None]).T
    A = J + np.diag(np.repeat(1.0 / dtau, NV))
    expect = base.apply(v).reshape(-1) + P @ np.linalg.solve(R @ A @ P, R @ v.reshape(-1))
    np.testing.assert_allclose(prec.apply(v).reshape(-1), expect, rtol=1e-5, atol=1e-8)
    z, c = prec.apply_split(v)
    np.testing.assert_array_equal(z, base.apply(v))
    prec.prolong_add(z, c)
    np.testing.assert_allclose(z, prec.apply(v), rtol=1e-12)
    assert np.all(prec.coarse_rows[~real.reshape(-1)] == -1)
    assert np.array_equal(prec.coarse_rows[real.reshape(-1)], np.nonzero(real)[0])
    assert prec.flexible and prec.n_coarse_cols == NV and prec.nonlinear_columns is None


def test_coarse_correction_cuts_gmres_iterations_on_a_long_chain():
    """长链（节点数超过最粗层上限，粗层走聚合 K 循环）上的对流扩散型块系统：加性粗校正的 FGMRES 迭代数
    明显少于块 Jacobi，且解满足容差。"""
    import scipy.sparse as sps

    n_cells, n_sps = 2400, 2
    rng = np.random.default_rng(7)
    real = np.ones((n_cells, n_sps), dtype=bool)
    m = n_sps * NV
    # 单元内：对角（与链耦合的行和只差 0.2%，接近奇异）；单元间：上风偏置的对流扩散（每个变量同样的链耦合）
    Jcc = np.zeros((n_cells, n_sps, NV, n_sps, NV))
    eye = np.eye(m).reshape(n_sps, NV, n_sps, NV)
    for c in range(n_cells):
        Jcc[c] = 1.002 * eye + 0.002 * rng.normal(size=(n_sps, NV, n_sps, NV))
    pairs = _chain_pairs(n_cells)
    Jcy = np.array([(-0.65 if r > y else -0.35) / n_sps * np.ones((n_sps, 1, n_sps, 1)) * np.eye(NV)[None, :, None, :]
                    for r, y in pairs])
    rows, cols, vals = [], [], []
    for c in range(n_cells):
        rr, cc = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
        rows.append(c * m + rr.ravel())
        cols.append(c * m + cc.ravel())
        vals.append(Jcc[c].reshape(m, m).ravel())
    for k, (r, y) in enumerate(pairs):
        rr, cc = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
        rows.append(r * m + rr.ravel())
        cols.append(y * m + cc.ravel())
        vals.append(Jcy[k].reshape(m, m).ravel())
    J = sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(n_cells * m, n_cells * m))
    dtau = np.full(n_cells * n_sps, 1e4)
    A = (J + sps.diags(np.repeat(1.0 / dtau, NV))).tocsr()
    inv_blocks = np.linalg.inv(np.array([A[c * m:(c + 1) * m, c * m:(c + 1) * m].toarray() for c in range(n_cells)]))

    class _BJ(PseudoTransientDiagonal):
        def apply(self, v):
            return np.einsum("cij,cj->ci", inv_blocks, v.reshape(n_cells, m)).reshape(v.shape)

    base = _BJ(dtau, NV)
    w = restriction_weights(np.full((n_cells, n_sps), 0.5), _uniform(np.full((n_cells, n_sps), 0.5), real))
    diag, blocks = _coarse_blocks(Jcc, Jcy, pairs, w)
    topo = CellCoarseTopology(n_cells, np.array(pairs)[:, 0], np.array(pairs)[:, 1])
    assert topo.hierarchy
    mat = CellCoarseMatrix(topo, diag, blocks)
    prec = CoarseCorrectedPreconditioner(base, CellCoarseSolve(mat, mat.ptc_diagonal(w, dtau, n_sps)), w, real,
                                         n_sps)
    b = rng.normal(size=n_cells * m)

    def run(p, flexible):
        return gmres_right(lambda x: A @ x, b, lambda v: p.apply(v.reshape(-1, NV)).reshape(-1), rtol=1e-6,
                           restart=60, max_iter=600, flexible=flexible)

    x_bj, it_bj, info_bj, _ = run(base, False)
    x_cc, it_cc, info_cc, _ = run(prec, True)
    assert info_cc == 0
    assert np.linalg.norm(b - A @ x_cc) <= 1.0001e-6 * np.linalg.norm(b)
    assert it_cc * 3 < it_bj, (it_cc, it_bj, info_bj)
