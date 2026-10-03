# -*- coding: utf-8 -*-
"""Newton 全局化的残差范数：线性求解与接受判据/SER 同一个范数，且是真正的范数（2026-10-03）。

两个缺陷（湍流平板层流 P2，`tests/validation/_flat_plate_case.py`，CFL 卡在 ~60 近 200 步）：

1. **线性求解与全局化用不同范数**：接受判据与 SER 用体积加权 RMS（`globalization.py::
   ResidualNorm`），GMRES 却在未加权 2 范数里最小化。被回溯的步上未加权线性残差 0.03~0.10、
   体积加权 1.1~2.5 倍（非线性误差只有 0.001~0.003）——线性模型本身就让全局度量恶化。
   现在 GMRES 对 `D A M^{-1} D^{-1}`（`D = sqrt(w)`）求解（`jfnk.py::_solve_direction`）。
2. **范数权重用了解点插值型求积权重**：原生基 P2/P3 有零与负权重，加权平方和只是半范数。
   现在取单元体积在真实解点上均分（`PositivityLimiter.norm_weights`）。
"""

import numpy as np
import pytest

from autoflowcfd.core.time_integration.implicit.globalization import ResidualNorm
from autoflowcfd.core.time_integration.implicit.jfnk import step_newton_krylov
from autoflowcfd.core.time_integration.implicit.reductions import LocalReductions
from autoflowcfd.core.time_integration.positivity.limiter import PositivityLimiter
from autoflowcfd.fr.native_padding import real_sps_per_cell
from autoflowcfd.fr.native_prism.quadrature import build_native_prism_sp_weights
from autoflowcfd.fr.native_tet.quadrature import build_native_tet_sp_weights


def test_truncated_linear_solve_does_not_increase_weighted_residual():
    """原始残差集中在小权重行（小单元）、GMRES 只走 2 次：方向必须在加权范数下不恶化。

    修复前（未加权最小化）同一构造下加权线性残差是 R0 的 18.7 倍，接受判据回溯；修复后
    由 GMRES 的最小残差性质保证不超过 1，步被完整接受。"""
    rng = np.random.default_rng(0)
    n = 120
    w = np.where(np.arange(n) < 20, 1e-6, 1.0)
    A = 4.0 * np.eye(n) + rng.normal(0.0, 1.0, (n, n)) / np.sqrt(n)
    c = rng.normal(0.0, 1.0, (n, 1))
    c[:20] *= 1e3

    def residual(u):
        return A @ u + c

    u0 = np.zeros((n, 1))
    u_new, info = step_newton_krylov(
        residual, u0, np.full(n, 1e12), np.ones(1), gmres_max_iter=2, gmres_restart=2,
        physicality=lambda u, du, red: np.ones(u.shape[0]), norm_weights=w)
    norm = ResidualNorm(w, LocalReductions())
    assert info["theta"] == 1.0
    assert norm(residual(u_new)) <= norm(residual(u0))
    assert info["linear_rel_residual"] <= 1.0 + 1e-12


@pytest.mark.parametrize("order", [2, 3])
def test_norm_weights_are_positive_cell_volume_shares(order):
    n_real_prism, n_real_tet = real_sps_per_cell(order)
    n_sps = max(n_real_prism, n_real_tet)
    is_prism = np.array([True, True, False, False])
    rng = np.random.default_rng(order)
    det = rng.uniform(0.5, 2.0, (4, 1)) * np.ones((1, n_sps))
    w = np.zeros((4, n_sps))
    w[is_prism, :n_real_prism] = build_native_prism_sp_weights(order)
    w[~is_prism, :n_real_tet] = build_native_tet_sp_weights(order)
    W = w * det
    # 插值型权重本身不是范数权重：P2/P3 在棱柱或四面体上有零或负值
    real = np.arange(n_sps)[None, :] < np.where(is_prism, n_real_prism, n_real_tet)[:, None]
    assert np.any(np.abs(W[real]) < 1e-12) or np.any(W[real] < 0.0)

    lim = PositivityLimiter(W, None, None, is_prism, n_real_prism, n_real_tet)
    nw = lim.norm_weights
    assert np.all(nw[real] > 0.0)
    assert np.all(nw[~real] == 0.0)
    np.testing.assert_allclose(nw.sum(axis=1), W.sum(axis=1), rtol=1e-13)
