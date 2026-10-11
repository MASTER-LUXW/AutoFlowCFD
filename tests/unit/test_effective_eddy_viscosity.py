"""非解点处涡粘的取用值（`flux_kernels/viscous.py::effective_eddy_viscosity`，2026-10-11）。

界面通量点上的涡粘是单元内涡粘多项式的外插。涡粘在贴壁第一层里陡升时外插到壁面为负，绝对值超过分子粘度，
总粘度 `mu + mu_t < 0`：粘性通量与壁面罚项变成反耗散（平板算例贴壁层内 78~90 m/s，来流 30；cube_demo 到
307 m/s）。
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.flux_kernels import (
    VBC_NOSLIP_WALL, effective_eddy_viscosity, effective_eddy_viscosity_array, viscous_physical_flux_point,
    viscous_physical_flux_point_effective,
)
from autoflowcfd.core.fr_residual.face_point_jumps import viscous_common_flux_point

MU = 1.8e-5


def test_piecewise_values_and_continuity_of_the_first_derivative():
    assert effective_eddy_viscosity(-50.0 * MU, MU) == 0.0
    assert effective_eddy_viscosity(0.0, MU) == 0.0                     # 层流逐位不变
    assert effective_eddy_viscosity(MU, MU) == MU
    assert effective_eddy_viscosity(37.5 * MU, MU) == 37.5 * MU         # 充分湍流逐位不变
    assert effective_eddy_viscosity(0.5 * MU, MU) == pytest.approx(0.375 * MU)
    h = 1e-7 * MU
    for x0, slope in ((0.0, 0.0), (MU, 1.0)):                           # 两个接点上左右导数一致
        left = (effective_eddy_viscosity(x0, MU) - effective_eddy_viscosity(x0 - h, MU)) / h
        right = (effective_eddy_viscosity(x0 + h, MU) - effective_eddy_viscosity(x0, MU)) / h
        assert left == pytest.approx(slope, abs=1e-6) and right == pytest.approx(slope, abs=1e-6)
    x = np.linspace(-3.0, 4.0, 701) * MU
    y = effective_eddy_viscosity_array(x, MU)
    np.testing.assert_allclose(y, [effective_eddy_viscosity(v, MU) for v in x], rtol=0, atol=1e-20)
    assert np.all(np.diff(y) >= 0.0) and np.all(y >= 0.0) and np.all(y <= np.maximum(x, 0.0) + 1e-20)
    assert (np.maximum(x, 0.0) - y).max() <= 0.149 * MU                 # 窄带内最多小 0.148 mu


def _state():
    Q = np.array([1.2, 30.0, 2.0, -1.0, 1.0e5])
    gv = np.array([[10.0, 4000.0, 30.0], [5.0, -8.0, 2.0], [1.0, 60.0, -2.0]])
    gT = np.array([3.0, -40.0, 5.0])
    return Q, gv, gT


def test_negative_extrapolated_eddy_viscosity_leaves_the_molecular_stress():
    """外插值为负时粘性通量就是分子粘性那一份；此前 `mu + mu_t = -20 mu`，应力整体反号。"""
    Q, gv, gT = _state()
    laminar = viscous_physical_flux_point(Q, gv, gT, MU, 0.72, 0.0, 0.9)
    np.testing.assert_array_equal(viscous_physical_flux_point(Q, gv, gT, MU, 0.72, -21.0 * MU, 0.9), laminar)
    np.testing.assert_array_equal(viscous_physical_flux_point_effective(Q, gv, gT, MU, 0.72, 0.0, 0.9), laminar)
    turbulent = viscous_physical_flux_point(Q, gv, gT, MU, 0.72, 40.0 * MU, 0.9)
    assert turbulent[1, 1] == pytest.approx(41.0 * laminar[1, 1])       # tau_xy 按 (mu + mu_t) 放大


def test_wall_penalty_stays_dissipative_when_the_extrapolated_eddy_viscosity_is_negative():
    """无滑移壁：罚项把本侧切向速度拉向壁面速度（与速度差反号）。外插涡粘为 -21 mu 时此前罚项系数为负，
    把速度差放大。"""
    Q_s, gv, gT = _state()
    Q_wall = Q_s.copy()
    Q_wall[1:4] = 0.0
    adj = np.array([0.0, -1.0, 0.0])                                     # 外法向 -y

    def jump(mut):
        return viscous_common_flux_point(Q_s, gv, gT, mut, Q_wall, gv, gT, mut, adj, 1e-3, VBC_NOSLIP_WALL,
                                         MU, 0.72, 0.9, 4.0)

    j_lam, j_neg, j_turb = jump(0.0), jump(-21.0 * MU), jump(40.0 * MU)
    np.testing.assert_array_equal(j_neg, j_lam)
    # 罚项主导（c_ip mu / h * u >> mu du/dy 的差）：x 动量的跳变量与本侧速度反号，湍流时更强
    assert j_lam[1] < 0.0 and j_turb[1] < 41.0 * j_lam[1] * 0.9
