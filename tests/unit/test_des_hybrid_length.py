"""SST-DDES / SST-IDDES 混合长度尺度（Gritskevich et al. 2012）。

修复前的三处偏差：
1. IDDES 混合权重直接用 f_B，丢了 DDES 屏蔽分支 f~_d = max(1 - f_dt, f_B)：附着边界层里 d_w > ~0.5 h_max 处
   无论有无湍流内容都切到 LES（网格诱导分离）；
2. C_DES 未按 F1 混合（DDES 用了 SA 的 0.65，IDDES 固定 0.78）；
3. 高阶单元的网格尺度未计入阶数（P 阶单元分辨尺度 h/(P+1)）。
"""

import numpy as np
import pytest

from autoflowcfd.core.turbulence.des.ddes import DDESModel, des_constant
from autoflowcfd.core.turbulence.des.iddes import IDDESModel

BETA_STAR = 0.09
KAPPA = 0.41


def _shear(n_cells, n_sps, dudy):
    g = np.zeros((n_cells, n_sps, 3, 3))
    g[..., 0, 1] = dudy
    return g


def _reference_ddes(k, omega, nu_t, nu, d, g, F1, h, order):
    """独立转写（Gritskevich 2012 式 (9)-(11)）。"""
    s = np.sqrt(np.sum(g * g, axis=(-2, -1)))
    r = np.minimum((nu_t + nu) / (KAPPA ** 2 * d ** 2 * s), 10.0)
    f_d = 1.0 - np.tanh((20.0 * r) ** 3)
    l_rans = np.sqrt(k) / (BETA_STAR * omega)
    l_les = (0.78 * F1 + 0.61 * (1.0 - F1)) * (h / (order + 1))[:, None]
    return l_rans - f_d * np.maximum(l_rans - l_les, 0.0)


@pytest.mark.parametrize("order", [0, 1, 2])
def test_ddes_matches_reference(order):
    rng = np.random.default_rng(order)
    n, m = 7, 5
    k = rng.uniform(1e-4, 1.0, (n, m))
    omega = rng.uniform(1.0, 1e4, (n, m))
    nu_t = rng.uniform(1e-6, 1e-3, (n, m))
    d = rng.uniform(1e-4, 0.5, (n, m))
    g = rng.normal(scale=100.0, size=(n, m, 3, 3))
    F1 = rng.uniform(0.0, 1.0, (n, m))
    h = rng.uniform(1e-3, 0.1, n)
    got = DDESModel().length_scale(np, k, omega, nu_t, 1.5e-5, d, g, F1, h, order, BETA_STAR)
    np.testing.assert_allclose(got, _reference_ddes(k, omega, nu_t, 1.5e-5, d, g, F1, h, order), rtol=1e-12)


def test_des_constant_follows_f1():
    assert des_constant(1.0) == pytest.approx(0.78)
    assert des_constant(0.0) == pytest.approx(0.61)


def test_iddes_keeps_rans_in_attached_boundary_layer():
    """附着边界层（涡粘大、r_dt >> 1，f_dt -> 0）且 d_w 远大于 h_max（alpha < 0，f_B -> 0）：
    f~_d = max(1 - f_dt, f_B) = 1，取 RANS（乘 1 + f_e）；修复前用 f_B 作权重，在这里切到 LES。"""
    n, m = 1, 1
    k = np.full((n, m), 0.5)
    omega = np.full((n, m), 50.0)
    l_rans = np.sqrt(0.5) / (BETA_STAR * 50.0)
    d = np.full((n, m), 0.05)
    h_max = np.array([0.01])                      # alpha = 0.25 - 5 = -4.75，f_B ~ 0
    nu_t = np.full((n, m), 1e-2)                  # r_dt = 1e-2 / (0.168 * 2.5e-3 * 10) ~ 2.4 -> f_dt ~ 0
    g = _shear(n, m, 10.0)
    F1 = np.full((n, m), 0.5)
    l = IDDESModel().length_scale(np, k, omega, nu_t, 1.5e-5, d, g, F1, h_max, 0, BETA_STAR, h_wn=h_max)
    assert l[0, 0] == pytest.approx(l_rans, rel=1e-6)


def test_iddes_switches_to_les_away_from_walls_without_shielding():
    """远离壁面、涡粘小（r_dt -> 0，f_dt -> 1）：f~_d = f_B ~ 0，取 l_LES = C_DES * h_max/(P+1)。"""
    n, m = 1, 1
    order = 2
    k = np.full((n, m), 0.5)
    omega = np.full((n, m), 50.0)
    d = np.full((n, m), 1.0)
    h_max = np.array([0.03])
    nu_t = np.full((n, m), 1e-9)
    g = _shear(n, m, 10.0)
    F1 = np.zeros((n, m))
    l = IDDESModel().length_scale(np, k, omega, nu_t, 1.5e-5, d, g, F1, h_max, order, BETA_STAR, h_wn=h_max)
    assert l[0, 0] == pytest.approx(0.61 * 0.03 / (order + 1), rel=1e-6)


def test_wall_points_are_rans():
    """壁面解点（d_w = 0）：屏蔽比值饱和，DDES 与 IDDES 都取 RANS 长度。"""
    k = np.full((1, 2), 0.3)
    omega = np.full((1, 2), 80.0)
    d = np.array([[0.0, 0.0]])
    g = _shear(1, 2, 50.0)
    F1 = np.ones((1, 2))
    h = np.array([0.02])
    l_rans = np.sqrt(0.3) / (BETA_STAR * 80.0)
    l_d = DDESModel().length_scale(np, k, omega, np.zeros((1, 2)), 1.5e-5, d, g, F1, h, 1, BETA_STAR)
    l_i = IDDESModel().length_scale(np, k, omega, np.zeros((1, 2)), 1.5e-5, d, g, F1, h, 1, BETA_STAR, h_wn=h)
    np.testing.assert_allclose(l_d, l_rans, rtol=1e-12)
    np.testing.assert_allclose(l_i, l_rans, rtol=1e-12)


def test_apply_requires_current_f1():
    from types import SimpleNamespace

    model = SimpleNamespace(k_field=np.ones((1, 1)), omega_field=np.ones((1, 1)), nu_t=np.ones((1, 1)),
                            beta_star=BETA_STAR, _last_F1=None)
    with pytest.raises(RuntimeError, match="F1"):
        DDESModel().apply(model, d_w=np.ones((1, 1)), nu=1.5e-5, grad_u=_shear(1, 1, 1.0),
                          h_max=np.ones(1), h_wn=None, xp=np)
