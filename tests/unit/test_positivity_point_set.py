"""正性点集覆盖过积分细点（2026-09-26）。

回归对象：plate_demo 锐边贴壁单元的解点与面通量点都正常，一个过积分细点上
rho=7e-10、p<0——体积通量在那里按 1/rho 奇异，平均流残差对 1e-5 的扰动跳 13 个
数量级、Newton 步从此每步被拒。Zhang–Shu 限制器与隐式物理性限幅此前都只看解点
（前者加面通量点），这里钉住两者都覆盖格式求值通量的全部点。
"""

import numpy as np
import pytest

from autoflowcfd.core.time_integration.implicit.reductions import LocalReductions
from autoflowcfd.core.time_integration.positivity import build_positivity_limiter_from_arrays
from autoflowcfd.fr.operators import generate_fr_operators

GAMMA = 1.4


def _state_negative_only_at_fine_point(order):
    """一个棱柱单元：解点与面通量点上 rho>0，某个过积分细点上 rho<0。"""
    ops = generate_fr_operators(order)
    n_sps = ops.overint_interp_c2f_prism.shape[1]
    lim = build_positivity_limiter_from_arrays(np.ones((1, n_sps)), np.array([True]), ops, order)
    nr = lim.n_real_prism
    c2f = np.asarray(ops.overint_interp_c2f_prism)[:, :nr]
    n_face_rows = lim.E_prism.shape[0] - c2f.shape[0]
    E_face = lim.E_prism[:n_face_rows, :nr]
    rng = np.random.default_rng(0)
    for _ in range(20000):
        rho = rng.uniform(0.02, 1.5, nr)
        if (E_face @ rho).min() > 0.0 and (c2f @ rho).min() < 0.0:
            U = np.zeros((1, n_sps, 5))
            U[0, :nr, 0] = rho
            U[0, :nr, 4] = 2.5e5 / (GAMMA - 1.0) * rho / rho.mean() + 1e5
            return ops, lim, U, c2f, nr
    pytest.skip("随机搜索没有找到只在细点上为负的密度分布")


@pytest.mark.parametrize("order", [1, 2])
def test_limiter_covers_overintegration_points(order):
    ops, lim, U, c2f, nr = _state_negative_only_at_fine_point(order)
    U_flat = U.reshape(-1, 5).copy()
    lim(U_flat)
    rho = U_flat.reshape(U.shape)[0, :nr, 0]
    assert lim.min_theta < 1.0, "细点上 rho<0 但限制器没有触发"
    assert (c2f @ rho).min() > 0.0


@pytest.mark.parametrize("order", [1, 2, 3])
def test_newton_physicality_covers_overintegration_points(order):
    """逐单元 alpha 恰为"解点 + 面通量点 + 细点"上逐点限值的最小值，且点集含细点。"""
    from autoflowcfd.core.time_integration.implicit.jfnk import density_pressure_row_limits

    ops = generate_fr_operators(order)
    n_sps = ops.overint_interp_c2f_prism.shape[1]
    n_cells = 6
    is_prism = np.array([True, True, True, False, False, False])
    lim = build_positivity_limiter_from_arrays(np.ones((n_cells, n_sps)), is_prism, ops, order)
    n_fine_prism = ops.overint_interp_c2f_prism.shape[0]
    n_fine_tet = ops.overint_interp_c2f_tet.shape[0]
    np.testing.assert_allclose(lim.E_prism[-n_fine_prism:], np.asarray(ops.overint_interp_c2f_prism)[:, :n_sps])
    np.testing.assert_allclose(lim.E_tet[-n_fine_tet:], np.asarray(ops.overint_interp_c2f_tet)[:, :n_sps])

    rng = np.random.default_rng(order)
    U = np.zeros((n_cells, n_sps, 5))
    U[..., 0] = rng.uniform(0.8, 1.2, (n_cells, n_sps))
    U[..., 1:4] = rng.uniform(-30, 30, (n_cells, n_sps, 3)) * U[..., :1]
    U[..., 4] = 1e5 / (GAMMA - 1.0) + 0.5 * np.sum(U[..., 1:4] ** 2, axis=-1) / U[..., 0]
    dU = rng.normal(0.0, 1.0, U.shape) * np.array([0.3, 20.0, 20.0, 20.0, 1.2e5])
    red = LocalReductions(np)
    alpha = lim.density_pressure_limits(U.reshape(-1, 5), dU.reshape(-1, 5), red).reshape(n_cells, n_sps)
    for c in range(n_cells):
        nr = lim.n_real_prism if is_prism[c] else lim.n_real_tet
        E = (lim.E_prism if is_prism[c] else lim.E_tet)[:, :nr]
        P0 = np.vstack([U[c, :nr], E @ U[c, :nr]])
        PD = np.vstack([dU[c, :nr], E @ dU[c, :nr]])
        want = density_pressure_row_limits(P0, PD, red).min()
        np.testing.assert_allclose(alpha[c], want, rtol=1e-14)
