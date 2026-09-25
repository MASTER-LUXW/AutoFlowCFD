# -*- coding: utf-8 -*-
"""守恒的正性保持限制器（Zhang–Shu 型）判据。

它取代了逐点硬钳 `enforce_positivity`（不守恒、带 1 Pa 的量纲下限、且是
plate_demo 长程运行第 4134 步一步爆炸到 1e53 的放大器），见
`core/time_integration/positivity/__init__.py`。

判据：
1. **守恒**：每个单元的 `Σ_s w_s det(J)_s U_s` 限制前后相同（到舍入）；
2. **可容许**：限制后全部真实解点与全部面通量点上密度、压力 > 0；
3. **不触发时逐位不动**：可容许的单元一个比特都不改 —— 正常运行里它从不
   触发，结果必须与"没有限制器"逐位相同；
4. **两版实现一致**：CPU numba 核与 GPU 用的数组模块无关实现，在 numpy 上
   逐位（到舍入）对照；
5. **均值坏了就报错**：单元均值不可容许是真正的发散，不修；
6. 四面体解点求积权重对 <=P 次多项式精确（守恒权重的来源）。
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_solver.residual_diagnostics import SolverDivergedError
from autoflowcfd.core.fr_residual.inviscid import primitive_to_conserved
from autoflowcfd.core.time_integration.positivity import build_positivity_limiter
from autoflowcfd.core.time_integration.positivity.zhang_shu import zhang_shu_xp
from autoflowcfd.fr.operators import generate_fr_operators

from .test_fr_residual_inviscid import _build_synthetic_mixed_mesh

G = 1.4


def _pressure(U):
    return (G - 1.0) * (U[..., 4] - 0.5 * (U[..., 1] ** 2 + U[..., 2] ** 2 + U[..., 3] ** 2) / U[..., 0])


@pytest.fixture(scope="module", params=[1, 2])
def case(request):
    order = request.param
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    lim = build_positivity_limiter(mesh, ops, order=order)
    rng = np.random.default_rng(order)
    n, S = mesh.n_cells, mesh.n_sps_per_cell
    Q = np.empty((n, S, 5))
    Q[..., 0] = 1.2 * (1 + 0.05 * rng.uniform(-1, 1, (n, S)))
    Q[..., 1] = 30 + 5 * rng.uniform(-1, 1, (n, S))
    Q[..., 2] = 5 * rng.uniform(-1, 1, (n, S))
    Q[..., 3] = 5 * rng.uniform(-1, 1, (n, S))
    Q[..., 4] = 1e5 * (1 + 0.05 * rng.uniform(-1, 1, (n, S)))
    U = np.concatenate([primitive_to_conserved(Q), np.zeros((n, S, 2))], axis=-1)   # 7 列
    return dict(mesh=mesh, ops=ops, lim=lim, U=U, order=order)


def _conserved(lim, U3):
    return np.einsum("cs,csv->cv", lim.W, U3[..., :5])


def _all_point_states(lim, U3):
    """全部真实解点 + 全部通量点的守恒量 (n_pts_total, 5)。"""
    out = []
    for sel, nr, E in ((lim.cell_is_prism, lim.n_real_prism, lim.E_prism),
                       (~lim.cell_is_prism, lim.n_real_tet, lim.E_tet)):
        if sel.any():
            Uc = U3[sel, :nr, :5]
            out.append(Uc.reshape(-1, 5))
            out.append(np.einsum("qs,csv->cqv", E[:, :nr], Uc).reshape(-1, 5))
    return np.concatenate(out, axis=0)


def _violate(case):
    """在一个棱柱与一个四面体里各造一个负密度解点，在另一个单元里造负压力。"""
    U = case["U"].copy()
    n_prism = case["mesh"].n_prism_cells
    U[0, 0, 0] = -0.3 * U[0, 0, 0]                        # 棱柱：负密度
    U[n_prism, 1, 0] = -0.2 * U[n_prism, 1, 0]            # 四面体：负密度
    U[1, 2, 4] = 0.1 * 0.5 * (U[1, 2, 1] ** 2 + U[1, 2, 2] ** 2 + U[1, 2, 3] ** 2) / U[1, 2, 0]  # 负压力
    return U


def test_limited_state_is_admissible_and_conservative(case):
    lim = case["lim"]
    U = _violate(case)
    U3 = U.reshape(lim.n_cells, lim.n_sps, 7)
    before = _conserved(lim, U3).copy()
    lim(U.reshape(-1, 7))
    after = _conserved(lim, U3)
    np.testing.assert_allclose(after, before, rtol=1e-13, atol=0.0,
                               err_msg="限制器不守恒：单元的 Σ w J U 变了")
    P = _all_point_states(lim, U3)
    assert (P[:, 0] > 0).all(), "限制后仍有非正密度（解点或通量点）"
    assert (_pressure(P) > 0).all(), "限制后仍有非正压力（解点或通量点）"
    assert lim.n_limited_total >= 3


def test_admissible_cells_are_bit_identical(case):
    lim = case["lim"]
    U = _violate(case)
    ref = U.copy()
    lim(U.reshape(-1, 7))
    touched = np.zeros(lim.n_cells, dtype=bool)
    touched[[0, 1, case["mesh"].n_prism_cells]] = True
    assert np.array_equal(U[~touched], ref[~touched]), "限制器改动了本来可容许的单元"
    assert np.array_equal(U[..., 5:], ref[..., 5:]), "限制器不该碰湍流列"
    # 补零槽位不参与
    P = lim.cell_is_prism
    assert np.array_equal(U[P, lim.n_real_prism:], ref[P, lim.n_real_prism:])


def test_fully_admissible_state_is_untouched(case):
    lim = case["lim"]
    U = case["U"].copy()
    ref = U.copy()
    lim(U.reshape(-1, 7))
    assert np.array_equal(U, ref), "状态本来全部可容许，限制器却改了比特"


def test_numba_and_array_module_versions_agree(case):
    lim = case["lim"]
    U_a = _violate(case)
    U_b = U_a.copy()
    lim(U_a.reshape(-1, 7))
    zhang_shu_xp(np, U_b.reshape(lim.n_cells, lim.n_sps, 7), lim.W, lim.E_prism, lim.E_tet,
                 lim.cell_is_prism, lim.n_real_prism, lim.n_real_tet, lim.gamma)
    np.testing.assert_allclose(U_b, U_a, rtol=1e-12, atol=1e-12,
                               err_msg="numba 核与 GPU 用的向量化版给出不同结果")


def test_inadmissible_mean_is_a_hard_error(case):
    lim = case["lim"]
    U = case["U"].copy()
    U[0, :lim.n_real_prism, 0] = -1.0                      # 整个单元密度为负 -> 均值为负
    with pytest.raises(SolverDivergedError, match="单元均值"):
        lim(U.reshape(-1, 7))


@pytest.mark.parametrize("order", [1, 2, 3])
def test_tet_sp_weights_integrate_exactly(order):
    from autoflowcfd.fr.native_tet.basis import build_native_tet_operators
    from autoflowcfd.fr.native_tet.quadrature import build_native_tet_sp_weights
    from autoflowcfd.fr.quadrature_points import gauss_legendre

    w = build_native_tet_sp_weights(order)
    r, s, t = build_native_tet_operators(order)[0].T
    x, wq = gauss_legendre(12)
    A, B, C = (m.ravel() for m in np.meshgrid(x, x, x, indexing="ij"))
    WA, WB, WC = (m.ravel() for m in np.meshgrid(wq, wq, wq, indexing="ij"))
    Wd = WA * WB * WC * ((1 - B) / 2) * ((1 - C) / 2) ** 2
    R = (1 + A) * (1 - B) * (1 - C) / 4 - 1
    S = (1 + B) * (1 - C) / 2 - 1
    for a in range(order + 1):
        for b in range(order + 1 - a):
            for c in range(order + 1 - a - b):
                exact = Wd @ (R ** a * S ** b * C ** c)
                assert abs(w @ (r ** a * s ** b * t ** c) - exact) < 1e-13


def test_interleaved_cell_order_matches_prism_first(case):
    """分布式本地编号里棱柱/四面体交错：按逐单元掩码构建的限制器，对任意
    单元排列给出与"棱柱在前"排列逐位相同的结果（同一个核，不另写一份）。"""
    from autoflowcfd.core.time_integration.positivity import build_positivity_limiter_from_arrays

    mesh, lim = case["mesh"], case["lim"]
    perm = np.random.default_rng(7).permutation(lim.n_cells)
    det = np.asarray(mesh.jacobians["det_jacs"]).reshape(lim.n_cells, lim.n_sps)
    lim_p = build_positivity_limiter_from_arrays(det[perm], lim.cell_is_prism[perm],
                                                 case["ops"], case["order"])
    U_a = _violate(case)
    U_b = U_a[perm].copy()
    lim(U_a.reshape(-1, 7))
    lim_p(U_b.reshape(-1, 7))
    assert np.array_equal(U_b, U_a[perm]), "交错单元顺序下结果与棱柱在前不一致"
