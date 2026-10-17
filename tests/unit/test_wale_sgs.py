"""WALE 亚格子模型与高阶 LES 滤波宽度。

修复前：亚格子涡粘被硬截到 1e-3 m^2/s（真实尺度 U~50 m/s、Delta~1 cm 时 WALE 涡粘约 1e-2，被静默压掉
一个量级）；滤波宽度取 V^(1/3)，不计 FR 阶数（P 阶单元的分辨尺度约 V^(1/3)/(P+1)）。
"""

import numpy as np
import pytest

from autoflowcfd.core.turbulence.sgs import C_WALE, WALEModel, les_filter_width


def _reference_wale(g, delta):
    """Nicoud & Ducros (1999) 式 (12)-(13) 逐点转写。"""
    out = np.zeros(g.shape[:2])
    for c in range(g.shape[0]):
        for s in range(g.shape[1]):
            G = g[c, s]
            S = 0.5 * (G + G.T)
            G2 = G @ G
            Sd = 0.5 * (G2 + G2.T) - np.trace(G2) / 3.0 * np.eye(3)
            num = np.sum(Sd * Sd) ** 1.5
            den = np.sum(S * S) ** 2.5 + np.sum(Sd * Sd) ** 1.25
            out[c, s] = (C_WALE * delta[c]) ** 2 * num / den
    return out


def test_matches_reference_without_cap():
    rng = np.random.default_rng(4)
    g = rng.normal(scale=2e3, size=(4, 3, 3, 3))
    delta = rng.uniform(5e-3, 2e-2, 4)
    nu = WALEModel().compute_eddy_viscosity(g, delta)
    np.testing.assert_allclose(nu, _reference_wale(g, delta), rtol=1e-11)
    assert nu.max() > 1e-3                         # 修复前被截到 1e-3


def test_pure_shear_gives_zero():
    g = np.zeros((1, 1, 3, 3))
    g[0, 0, 0, 1] = 500.0
    assert WALEModel().compute_eddy_viscosity(g, np.array([0.01]))[0, 0] == pytest.approx(0.0, abs=1e-18)


def test_zero_gradient_gives_zero():
    assert WALEModel().compute_eddy_viscosity(np.zeros((2, 4, 3, 3)), np.array([0.01, 0.02])).max() == 0.0


@pytest.mark.parametrize("order, n_tet, n_prism", [(0, 1, 1), (1, 4, 6), (2, 10, 18), (3, 20, 40)])
def test_filter_width_is_volume_per_dof(order, n_tet, n_prism):
    V = np.array([8e-6, 8e-6])
    delta = les_filter_width(V, np.array([True, False]), order)
    np.testing.assert_allclose(delta, [(8e-6 / n_prism) ** (1 / 3), (8e-6 / n_tet) ** (1 / 3)], rtol=1e-14)
