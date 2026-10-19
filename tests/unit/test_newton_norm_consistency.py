# -*- coding: utf-8 -*-
"""Newton 全局化的残差范数：线性求解与接受判据/SER 同一个范数，且是真正的范数（2026-10-03）。

两个缺陷（湍流平板层流 P2，`tests/validation/_flat_plate_case.py`，CFL 卡在 ~60 近 200 步）：

1. **线性求解与全局化用不同范数**：接受判据与 SER 用体积加权 RMS（`globalization.py::
   ResidualNorm`），GMRES 却在未加权 2 范数里最小化。被回溯的步上未加权线性残差 0.03~0.10、
   体积加权 1.1~2.5 倍（非线性误差只有 0.001~0.003）——线性模型本身就让全局度量恶化。
   现在 GMRES 对 `D A M^{-1} D^{-1}`（`D = sqrt(w)`）求解（`linear_solve.py::_solve_direction`）。
2. **范数权重用了解点插值型求积权重**：原生基 P2/P3 有零与负权重，加权平方和只是半范数。
   现在取单元体积在真实解点上均分（`PositivityLimiter.norm_weights`）。
"""

import numpy as np
import pytest

from autoflowcfd.core.time_integration.implicit.newton.globalization import LINEAR_SOLVE_MAX_REL_RESIDUAL, ResidualNorm
from autoflowcfd.core.time_integration.implicit.newton.jfnk import step_newton_krylov
from autoflowcfd.core.time_integration.implicit.reductions import LocalReductions
from autoflowcfd.core.time_integration.positivity.limiter import PositivityLimiter
from autoflowcfd.fr.native_padding import real_sps_per_cell
from autoflowcfd.fr.native_prism.quadrature import build_native_prism_sp_weights
from autoflowcfd.fr.native_tet.quadrature import build_native_tet_sp_weights


def _small_cell_dominated_system():
    """原始残差集中在小权重行（小单元）的线性系统：`(residual, u0, weights)`。"""
    rng = np.random.default_rng(0)
    n = 120
    w = np.where(np.arange(n) < 20, 1e-6, 1.0)
    A = 4.0 * np.eye(n) + rng.normal(0.0, 1.0, (n, n)) / np.sqrt(n)
    c = rng.normal(0.0, 1.0, (n, 1))
    c[:20] *= 1e3
    return (lambda u: A @ u + c), np.zeros((n, 1)), w


def _step(residual, u0, w, max_iter, dtau=1e12):
    return step_newton_krylov(
        residual, u0, np.full(u0.shape[0], dtau), np.ones(1), gmres_max_iter=max_iter, gmres_restart=max_iter,
        physicality=lambda u, du, red: np.ones(u.shape[0]), norm_weights=w)


def test_truncated_linear_solve_does_not_increase_weighted_residual():
    """GMRES 在迭代数上限处截断（没到容差，但线性残差已降过一半）：方向必须在加权范数下不恶化。

    修复前（未加权最小化）同一构造下截断方向的加权线性残差大于 R0，接受判据回溯；修复后由 GMRES 的
    最小残差性质保证不超过 1，步被完整接受。"""
    residual, u0, w = _small_cell_dominated_system()
    u_new, info = _step(residual, u0, w, max_iter=5)
    norm = ResidualNorm(w, LocalReductions())
    assert info["gmres_info"] > 0                                    # 确实是截断的求解
    assert info["linear_rel_residual"] <= LINEAR_SOLVE_MAX_REL_RESIDUAL
    assert info["theta"] == 1.0
    assert norm(residual(u_new)) <= norm(residual(u0))


def test_direction_from_a_failed_linear_solve_is_not_taken():
    """线性求解没到容差、线性残差也没减半（2 次迭代只到 0.998）：这个方向不走出去。

    2026-10-11 以前它照常交给残差接受判据——判据只看范数增幅不超过 1.5 倍，拦不住一个与 Newton 方向
    无关的向量（cube_demo SA P1：线性残差 0.72~0.99 的步被接受，一步之内 Cd 0.736 -> 0.457，三步后
    最大速度 394 m/s）。现在按未接受处理：缩 dtau 重解；缩到线性系统接近对角主导时 2 次迭代就够，
    步被接受；本次调用的档数用完仍解不动则状态逐位不变、缩小后的档位带回给下一次调用。"""
    residual, u0, w = _small_cell_dominated_system()
    u_new, info = _step(residual, u0, w, max_iter=2)
    assert info["linear_rel_residual"] > LINEAR_SOLVE_MAX_REL_RESIDUAL and info["gmres_info"] > 0
    assert info["theta"] == 0.0 and info["n_dtau_cuts"] > 0 and info["dtau_scale"] < 1.0
    assert np.array_equal(u_new, u0)

    # 伪时间步小到 I/dtau 主导时同样 2 次迭代解到容差，步被完整接受
    u_small, info_small = _step(residual, u0, w, max_iter=2, dtau=1e-3)
    assert info_small["gmres_info"] == 0 and info_small["theta"] == 1.0 and info_small["n_dtau_cuts"] == 0
    assert not np.array_equal(u_small, u0)


def test_turbulence_residual_scale_is_the_field_magnitude():
    """湍流列的残差参考量级 = 场的体积加权 RMS（下限为模型尺度）；平均流列与对数列原样；零权重行不计。"""
    from autoflowcfd.core.time_integration.implicit.newton.coupled_step import residual_reference_scales

    n = 8
    x0 = np.zeros((n, 7))
    x0[:, 5] = [3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 1e9, 1e9]            # 后两行是补零槽位（权重 0）
    x0[:, 6] = 50.0                                                  # 对数列
    w = np.array([1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0])
    floor = np.array([1.2, 36.0, 36.0, 36.0, 1e5, 1e-5, 1.0])
    out = residual_reference_scales(x0, floor, w, LocalReductions(), log_columns=(1,))
    np.testing.assert_array_equal(out[:5], floor[:5])
    np.testing.assert_allclose(out[5], np.sqrt(12.5), rtol=1e-14)    # sqrt((9 + 16) / 2)
    assert out[6] == 1.0
    # 场还在来流量级以下：保持模型尺度
    x0[:, 5] = 1e-7
    assert residual_reference_scales(x0, floor, w, LocalReductions())[5] == 1e-5


def test_mean_flow_rows_are_solved_when_the_turbulence_residual_is_large():
    """两列系统：第 1 列（湍流）场量级是其尺度下限的 1e3 倍、残差随之大 1e3 倍，一次 Krylov 迭代就能解掉；
    第 0 列（平均流）要多次迭代。

    范数按尺度下限缩放（2026-10-11 上一提交的做法）时合成范数只是湍流列的范数：GMRES 一次迭代即到容差，
    平均流行的残差不降反升，步照样被完整接受——cube_demo SA P1 上"线性求解达标、整步接受"的一步里平均流
    残差 1.28 -> 8.42 的最小复现。按场的量级缩放后两列可比，同一步把平均流行的残差降下来。"""
    rng = np.random.default_rng(3)
    n = 60
    A = 4.0 * np.eye(n) + rng.normal(0.0, 1.0, (n, n)) / np.sqrt(n)
    c = rng.normal(0.0, 1.0, (n, 2))
    c[:, 1] *= 1e3

    def residual(u):
        return np.stack([A @ u[:, 0], u[:, 1]], axis=1) + c

    u0 = np.zeros((n, 2))
    mean = ResidualNorm(None, LocalReductions()).restricted(slice(0, 1))

    def mean_ratio(norm_scales):
        u_new, info = step_newton_krylov(
            residual, u0, np.full(n, 1e12), np.ones(2), norm_scales=norm_scales,
            physicality=lambda u, du, red: np.ones(u.shape[0]))
        assert info["gmres_info"] == 0 and info["theta"] == 1.0
        return mean(residual(u_new)[:, :1]) / mean(residual(u0)[:, :1])

    assert mean_ratio(None) > 1.0                       # 尺度下限：平均流残差变大而步被接受
    assert mean_ratio(np.array([1.0, 1e3])) < 0.2       # 场的量级：平均流行也解到容差量级

    with pytest.raises(ValueError, match="参考量级长度"):
        step_newton_krylov(residual, u0, np.full(n, 1e12), np.ones(2), norm_scales=np.ones(3))


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


def test_norm_is_taken_on_residuals_scaled_by_the_reference_magnitudes():
    """各方程先按参考量级无量纲化再求范数（2026-10-11）。此前直接对有量纲的残差求平方和，能量方程比质量方程
    大约 1e5 倍，范数实际只看能量方程（cube_demo SA P1：能量方程占 1.000）。"""
    red = LocalReductions()
    scales = np.array([1.2, 36.0, 36.0, 36.0, 1.0e5])
    r = np.tile(scales * np.array([3.0, 1.0, 1.0, 1.0, 1.0]), (7, 1))      # 无量纲残差：质量 3，其余 1
    raw = ResidualNorm(None, red)(r)
    scaled = ResidualNorm(None, red, scales)(r)
    assert scaled == pytest.approx(np.sqrt((9.0 + 4.0) / 5.0))
    assert raw == pytest.approx(np.sqrt((r[0] ** 2).mean()))               # 不给 scales 时不缩放
    # 只改质量方程的残差：缩放后的范数明显变化，有量纲的范数几乎不动
    r2 = r.copy()
    r2[:, 0] *= 10.0
    assert ResidualNorm(None, red, scales)(r2) > 5.0 * scaled
    assert ResidualNorm(None, red)(r2) == pytest.approx(raw, rel=1e-6)
    w = np.linspace(0.5, 2.0, 7)
    full = ResidualNorm(w, red, scales)
    assert full.restricted(slice(0, 4))(r[:, :4]) == pytest.approx(np.sqrt((9.0 + 3.0) / 4.0))
    with pytest.raises(ValueError, match="参考量级"):
        ResidualNorm(None, red, np.array([1.0, 0.0]))


def test_linear_solve_minimizes_the_scaled_residual():
    """GMRES 在同一个（按参考量级缩放的）范数里最小化：两个方程量级差 1e6、只给 1 次迭代时，不缩放的求解
    只顾大量级的那个方程，缩放后的范数里残差不降；现在接受判据所用的范数单调不增。"""
    n = 40
    rng = np.random.default_rng(3)
    scales = np.array([1.0, 1.0e6])
    A = np.zeros((2 * n, 2 * n))
    for k in range(n):
        A[2 * k:2 * k + 2, 2 * k:2 * k + 2] = np.array([[2.0, 0.3e-6], [0.4e6, 5.0]]) * rng.uniform(0.5, 1.5)
    c = (rng.standard_normal((n, 2)) * scales).reshape(-1)

    def residual(u):
        return (A @ u.reshape(-1) + c).reshape(n, 2)

    u0 = np.zeros((n, 2))
    u_new, info = step_newton_krylov(
        residual, u0, np.full(n, 1e12), scales, gmres_max_iter=1, gmres_restart=1,
        physicality=lambda u, du, red: np.ones(u.shape[0]))
    norm = ResidualNorm(None, LocalReductions(), scales)
    assert info["res_norm"] == pytest.approx(norm(residual(u0)))
    assert norm(residual(u_new)) <= norm(residual(u0)) * (1.0 + 1e-12)
    assert info["linear_rel_residual"] <= 1.0 + 1e-12
