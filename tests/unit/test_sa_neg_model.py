# -*- coding: utf-8 -*-
"""SA-neg 模型核心（`core/turbulence/sa`）：常数与来流、逐点函数的两支与连续性、模型接口。

逐点函数与一份按 TMR / Allmaras 2012 公式逐点写成的标量参考实现对照（测试内独立实现，不复用
被测代码），覆盖正支、`S_tilde` 修正分支、`r` 取上限、负支。
"""

import math

import numpy as np
import pytest

from autoflowcfd.core.turbulence.sa import SAModel
from autoflowcfd.core.turbulence.sa.constants import (
    C_B1, C_B2, C_N1, C_T3, C_V1, C_V2, C_V3, C_W1, C_W2, C_W3, CHI_FREESTREAM_TMR, KAPPA, R_LIM, SIGMA,
    chi_for_viscosity_ratio, eddy_viscosity_ratio,
)
from autoflowcfd.core.turbulence.sa.linearization import SAPointwise
from autoflowcfd.core.turbulence.sa.pointwise import (
    sa_diffusivity, sa_eddy_viscosity, sa_gradient_source, sa_source_terms, vorticity_magnitude,
)

NU = 1.5e-5


def _reference_source(nt, nu, d, Om, pf=1.0):
    """TMR SA-noft2 + Allmaras 2012 负支的逐点标量参考（`P - D`，单位质量）。"""
    chi = nt / nu
    if nt < 0.0:
        return pf * C_B1 * (1.0 - C_T3) * Om * nt + C_W1 * (nt / d) ** 2
    fv1 = chi ** 3 / (chi ** 3 + C_V1 ** 3)
    fv2 = 1.0 - chi / (1.0 + chi * fv1)
    sbar = nt * fv2 / (KAPPA ** 2 * d ** 2)
    if sbar >= -C_V2 * Om:
        st = Om + sbar
    else:
        st = Om + Om * (C_V2 ** 2 * Om + C_V3 * sbar) / ((C_V3 - 2.0 * C_V2) * Om - sbar)
    r = min(nt / (st * KAPPA ** 2 * d ** 2), R_LIM) if st > 0.0 else R_LIM
    g = r + C_W2 * (r ** 6 - r)
    fw = g * ((1.0 + C_W3 ** 6) / (g ** 6 + C_W3 ** 6)) ** (1.0 / 6.0)
    return pf * C_B1 * st * nt - C_W1 * fw * (nt / d) ** 2


def test_cw1_is_standard_combination():
    assert C_W1 == pytest.approx(C_B1 / KAPPA ** 2 + (1.0 + C_B2) / SIGMA, rel=1e-15)


@pytest.mark.parametrize("ratio", [1e-3, 0.21, 1.0, 10.0, 1e3])
def test_freestream_chi_inverts_viscosity_ratio(ratio):
    chi = chi_for_viscosity_ratio(ratio)
    assert eddy_viscosity_ratio(chi) == pytest.approx(ratio, rel=1e-12)


def test_tmr_freestream_value():
    assert eddy_viscosity_ratio(CHI_FREESTREAM_TMR) == pytest.approx(0.2104, abs=1e-4)
    with pytest.raises(ValueError):
        chi_for_viscosity_ratio(0.0)


def test_s_tilde_modification_is_c1_at_switch():
    """`S_bar = -c_v2 Omega` 两侧 `S_tilde` 的值与对 `S_bar` 的导数连续（Allmaras 2012）。"""
    Om = 2.0

    def s_tilde(sbar):
        if sbar >= -C_V2 * Om:
            return Om + sbar
        return Om + Om * (C_V2 ** 2 * Om + C_V3 * sbar) / ((C_V3 - 2.0 * C_V2) * Om - sbar)

    s0 = -C_V2 * Om
    h = 1e-7
    assert s_tilde(s0 - 1e-14) == pytest.approx(s_tilde(s0), abs=1e-12)
    left = (s_tilde(s0 - h) - s_tilde(s0 - 2 * h)) / h
    right = (s_tilde(s0 + 2 * h) - s_tilde(s0 + h)) / h
    assert left == pytest.approx(right, abs=1e-5)


def test_vectorized_source_matches_scalar_reference():
    rng = np.random.default_rng(3)
    n = 400
    nt = np.concatenate([rng.uniform(-30, 300, n - 40) * NU, np.zeros(10), -np.logspace(-9, -3, 30)])
    d = np.concatenate([rng.uniform(1e-6, 0.3, n - 40), rng.uniform(1e-6, 0.3, 40)])
    Om = rng.uniform(0.0, 5e3, n)
    Om[:20] = 0.0                       # Omega = 0：S_tilde 修正分支给 0，r 取上限
    src, damp = sa_source_terms(nt, NU, d, Om, 0.7, np)
    ref = np.array([_reference_source(a, NU, b, c, 0.7) for a, b, c in zip(nt, d, Om)])
    np.testing.assert_allclose(src, ref, rtol=1e-12, atol=1e-30)
    assert np.all(damp >= 0.0)


def test_branches_are_continuous_at_zero():
    d, Om = np.array([1e-3]), np.array([50.0])
    eps = 1e-14 * NU
    for f in (lambda x: sa_source_terms(x, NU, d, Om, 1.0, np)[0], lambda x: sa_eddy_viscosity(x, NU, np),
              lambda x: sa_diffusivity(x, np.ones(1), NU, np)):
        assert f(np.array([eps]))[0] == pytest.approx(f(np.array([-eps]))[0], abs=1e-12 * max(1.0, abs(f(np.zeros(1))[0])))


def test_negative_branch_diffusivity_stays_positive_and_nu_t_zero():
    chi = -np.linspace(0.0, 50.0, 5001)
    gamma = sa_diffusivity(chi * NU, np.ones_like(chi), NU, np)
    assert np.all(gamma > 0.0)
    assert np.all(sa_eddy_viscosity(chi[1:] * NU, NU, np) == 0.0)
    # 负支：f_n 公式本身
    chi3 = chi ** 3
    np.testing.assert_allclose(gamma * SIGMA, NU * (1.0 + chi * (C_N1 + chi3) / (C_N1 - chi3)), rtol=1e-12)


def test_gradient_source_formula():
    nt = np.array([2e-4, -1e-5])
    rho = np.array([1.2, 0.9])
    g = np.array([[1.0, 2.0, 0.5], [0.3, -0.2, 0.1]])
    gr = np.array([[0.01, 0.0, 0.02], [0.0, 0.05, 0.0]])
    mu = 1.8e-5
    out = sa_gradient_source(nt, rho, mu, g, gr, np)
    for i in range(2):
        nu = mu / rho[i]
        chi = nt[i] / nu
        fn = 1.0 if nt[i] >= 0 else (C_N1 + chi ** 3) / (C_N1 - chi ** 3)
        ref = (C_B2 * rho[i] * g[i] @ g[i] - (nu + nt[i] * fn) * (gr[i] @ g[i])) / SIGMA
        assert out[i] == pytest.approx(ref, rel=1e-13)


def test_vorticity_magnitude_is_curl_norm():
    rng = np.random.default_rng(0)
    G = rng.normal(size=(7, 3, 3))
    W = 0.5 * (G - np.swapaxes(G, -1, -2))
    ref = np.sqrt(2.0 * np.sum(W * W, axis=(-1, -2)))
    np.testing.assert_allclose(vorticity_magnitude(G, np), ref, rtol=1e-14)


def test_model_interface_roundtrip_and_limiter():
    m = SAModel(4, 3, nu_ref=NU, viscosity_ratio=1.0)
    assert m.n_transported == 1
    assert m.nu_tilde_inf == pytest.approx(chi_for_viscosity_ratio(1.0) * NU, rel=1e-15)
    x = m.newton_unknowns(np)
    assert x.shape == (12, 1)
    x[:, 0] = np.linspace(-1e-4, 1e-2, 12)
    m.set_newton_unknowns(x, np)
    np.testing.assert_array_equal(m.nu_tilde_field.ravel(), x[:, 0])
    m.nu_tilde_field[0, 0] = np.nan
    m.nu_tilde_field[0, 1] = 10.0 * m.nu_tilde_max
    m.nu_tilde_field[0, 2] = -3.0 * NU
    m.apply_positivity_limiter()
    assert m.nu_tilde_field[0, 0] == m.nu_tilde_inf
    assert m.nu_tilde_field[0, 1] == m.nu_tilde_max
    assert m.nu_tilde_field[0, 2] == -3.0 * NU         # 负值不裁剪（SA-neg）
    snap = m.field_snapshot()
    m.set_newton_unknowns(np.zeros((12, 1)), np)
    m.field_restore(snap)
    assert m.nu_tilde_field[0, 1] == m.nu_tilde_max
    other = m.like(2, 5)
    assert other.nu_tilde_field.shape == (2, 5) and other.nu_tilde_inf == m.nu_tilde_inf


def test_pointwise_evaluator_matches_model_source():
    rng = np.random.default_rng(1)
    n_cells, n_sps = 5, 4
    m = SAModel(n_cells, n_sps, nu_ref=NU, viscosity_ratio=3.0)
    m.production_factor = 0.8
    m.nu_tilde_field = rng.uniform(-5, 200, (n_cells, n_sps)) * NU
    Q = np.zeros((n_cells, n_sps, 5))
    Q[..., 0] = rng.uniform(1.0, 1.3, (n_cells, n_sps))
    grad_vel = rng.normal(scale=100.0, size=(n_cells, n_sps, 3, 3))
    d_wall = rng.uniform(1e-5, 0.1, (n_cells, n_sps))
    mu = NU * 1.225
    grad_nt = rng.normal(scale=1e-3, size=(n_cells, n_sps, 3))
    grad_rho = rng.normal(scale=1e-2, size=(n_cells, n_sps, 3))
    raw = m.compute_source_terms(Q, grad_vel, d_wall, mu)
    ev = SAPointwise(np, m, Q, grad_vel, d_wall, mu, grad_rho)
    S, G = ev(m.nu_tilde_field[..., None], grad_nt[..., None, :])
    rho = Q[..., 0]
    expected = raw + sa_gradient_source(m.nu_tilde_field, rho, mu, grad_nt, grad_rho, np)
    np.testing.assert_allclose(S[..., 0], expected, rtol=1e-13, atol=1e-20)
    np.testing.assert_allclose(G[..., 0], sa_diffusivity(m.nu_tilde_field, rho, mu, np), rtol=1e-14)
    assert math.isfinite(float(np.sum(S)))
