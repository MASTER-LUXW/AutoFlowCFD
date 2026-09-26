"""SST 的环境维持项与 omega 下限。

历史：2026-09-07 为堵 P0 阶段 omega 塌陷（S 恒为零、无产生项，omega=1e-12 是
稳定不动点，cube_demo 真实网格升阶后 k 撞 k_max 的根因）加了 `0.1*omega_inf`
下限。2026-09-26 发现标准 SST 的来流衰减让外流算例的物理 omega 低于这条下限
（plate_demo P1 隐式稳态上 10% 以上的单元贴住它、Newton 与限制器拉锯），于是
加 Spalart–Rumsey 环境维持项（`core/turbulence/sst/ambient.py`）：来流是无剪切区
的精确不动点，下限回到"只在非物理暂态里起作用"的安全网角色。
"""

import numpy as np
import pytest

from autoflowcfd.core.turbulence.sst import SSTModelFR

K_INF, OMEGA_INF = 0.1666, 2268.1  # 与 cube_demo / plate_demo 真实来流同量级


def _build_source_term_inputs(n_cells, n_sps, grad_U_value=0.0):
    """构造 compute_source_terms 所需的最小合法输入集。"""
    Q = np.zeros((n_cells, n_sps, 5))
    Q[..., 0] = 1.225  # rho
    Q[..., 1] = 30.0   # u
    Q[..., 4] = 101325.0  # p
    grad_U = np.full((n_cells, n_sps, 3, 3), grad_U_value)
    d_wall = np.full((n_cells, n_sps), 0.01)
    grad_k = np.zeros((n_cells, n_sps, 3))
    grad_omega = np.zeros((n_cells, n_sps, 3))
    return Q, grad_U, d_wall, grad_k, grad_omega


def _sources(model, grad_U_value=0.0):
    n_cells, n_sps = model.k_field.shape
    Q, grad_U, d_wall, grad_k, grad_omega = _build_source_term_inputs(n_cells, n_sps, grad_U_value)
    return model.compute_source_terms(Q, grad_U, d_wall, mu=1.8e-5, grad_k=grad_k, grad_omega=grad_omega)


class TestAmbientSustainingTerms:
    def test_freestream_is_exact_equilibrium(self):
        """无剪切来流上 (k_inf, omega_inf) 的源项恰为零——标准 SST 在这里是
        `-beta* k omega` / `-beta omega^2` 的纯衰减。"""
        model = SSTModelFR(3, 2, k_inf=K_INF, omega_inf=OMEGA_INF)
        model.k_field[:] = K_INF
        model.omega_field[:] = OMEGA_INF
        Sk, Sw = _sources(model)
        rho = 1.225
        # 与单项量级（D_k ~ rho beta* k omega、D_omega ~ rho beta omega^2）相比为舍入级
        assert np.max(np.abs(Sk)) <= 1e-12 * rho * model.beta_star * K_INF * OMEGA_INF
        assert np.max(np.abs(Sw)) <= 1e-12 * rho * model.beta2 * OMEGA_INF ** 2

    def test_freestream_is_exact_equilibrium_under_des(self):
        """DES 分支（D_k = rho k^1.5 / l_eff）下同一个不动点同样精确成立。"""
        model = SSTModelFR(2, 1, k_inf=K_INF, omega_inf=OMEGA_INF)
        model.k_field[:] = K_INF
        model.omega_field[:] = OMEGA_INF
        model.des_length_scale = np.full((2, 1), 0.02)
        Sk, _ = _sources(model)
        assert np.max(np.abs(Sk)) <= 1e-12 * 1.225 * K_INF ** 1.5 / 0.02

    def test_decay_below_ambient_is_restored(self):
        """k、omega 低于环境值时源项为正（向环境值回复），高于时为负。"""
        model = SSTModelFR(2, 1, k_inf=K_INF, omega_inf=OMEGA_INF)
        model.k_field[:] = [[0.1 * K_INF], [10.0 * K_INF]]
        model.omega_field[:] = [[0.1 * OMEGA_INF], [10.0 * OMEGA_INF]]
        Sk, Sw = _sources(model)
        assert Sk[0, 0] > 0 and Sw[0, 0] > 0
        assert Sk[1, 0] < 0 and Sw[1, 0] < 0

    def test_collapsed_omega_is_not_a_fixed_point_at_p0(self):
        """P0（S=0、无产生项）下 omega=1e-12 不再是不动点：源项约为
        `rho beta omega_inf^2 > 0`，把它拉回来（2026-09-07 那次撞 k_max 的起点）。"""
        model = SSTModelFR(1, 1, k_inf=K_INF, omega_inf=OMEGA_INF)
        model.k_field[:] = 0.0514
        model.omega_field[:] = 1e-12
        _, Sw = _sources(model)
        assert Sw[0, 0] > 0.5 * 1.225 * model.beta2 * OMEGA_INF ** 2


class TestOmegaRealizabilityMin:
    def test_floor_is_pointwise_max_of_strain_and_ambient(self):
        """下限是逐点的 `max(0.1 S, 0.1 omega_inf)`，不被别处的应变尖峰抬高
        （2026-09-15 全域 max(S) 缺陷的回归）。"""
        n_cells, n_sps = 4, 1
        omega_inf = 100.0  # 故意设小，让强应变点的 0.1 S 更大
        model = SSTModelFR(n_cells, n_sps, k_inf=0.1, omega_inf=omega_inf)
        Q, grad_U, d_wall, grad_k, grad_omega = _build_source_term_inputs(n_cells, n_sps)
        grad_U[0, 0, 0, 0] = 1e5  # 只有这一点有强应变
        model.compute_source_terms(Q, grad_U, d_wall, mu=1.8e-5, grad_k=grad_k, grad_omega=grad_omega)
        rmin = np.asarray(model._omega_realizability_min)
        S = np.asarray(model.compute_strain_rate_magnitude(grad_U))
        np.testing.assert_allclose(rmin, np.maximum(0.1 * S, 0.1 * omega_inf), rtol=1e-14)
        assert rmin[0, 0] > 0.1 * omega_inf
        others = np.ones(rmin.shape, dtype=bool)
        others[0, 0] = False
        np.testing.assert_allclose(rmin[others], 0.1 * omega_inf, rtol=1e-14)

    def test_ambient_equilibrium_is_strictly_above_floor(self):
        """安全网的前提：来流不动点 (k_inf, omega_inf) 严格高于下限，限制器在
        那里不激活。"""
        model = SSTModelFR(2, 1, k_inf=K_INF, omega_inf=OMEGA_INF)
        model.k_field[:] = K_INF
        model.omega_field[:] = OMEGA_INF
        _sources(model)
        before = model.omega_field.copy()
        model.apply_positivity_limiter()
        np.testing.assert_array_equal(model.omega_field, before)
        assert np.all(before > 5.0 * np.asarray(model._omega_realizability_min))

    def test_collapsed_omega_uses_effective_value_and_is_restored(self):
        """P0（S=0）下被打到 1e-12 的 omega：限制器不再把它夹回去（不裁剪被输运的
        量，`bounds.py` 模块文档），模型项用 omega_eff = 0.1 omega_inf（涡粘有限），
        源项为正、把它推回来。"""
        model = SSTModelFR(2, 1, k_inf=K_INF, omega_inf=OMEGA_INF)
        model.k_field[:] = [[0.0514], [K_INF]]
        model.omega_field[:] = [[1e-12], [OMEGA_INF]]
        _, Sw = _sources(model)
        model.apply_positivity_limiter()
        assert model.omega_field[0, 0] == 1e-12
        assert Sw[0, 0] > 0.0
        nu_t_expected = 0.0514 / (0.1 * OMEGA_INF)
        assert model.nu_t[0, 0] == pytest.approx(nu_t_expected, rel=1e-12)


class TestTransportedFieldsAreNotClipped:
    """被输运的 k/omega 不裁剪下界，realizability 只作用于模型项求值
    （`core/turbulence/sst/bounds.py`，2026-09-26）。"""

    def test_negative_k_is_kept_and_restored(self):
        model = SSTModelFR(2, 1, k_inf=K_INF, omega_inf=OMEGA_INF)
        model.k_field[:] = [[-0.01], [K_INF]]
        model.omega_field[:] = OMEGA_INF
        Sk, _ = _sources(model)
        model.apply_positivity_limiter()
        assert model.k_field[0, 0] == -0.01
        assert model.nu_t[0, 0] == pytest.approx(1e-10 / OMEGA_INF, rel=1e-6), "涡粘用 k_bar = max(k, 0)"
        assert Sk[0, 0] > 0.0, "k 的原值为负时耗散变号、源项把它推回来"

    def test_limiter_only_recovers_non_finite_and_caps_upper(self):
        model = SSTModelFR(3, 1, k_inf=K_INF, omega_inf=OMEGA_INF)
        model.k_field[:] = [[np.nan], [2.0 * model.k_max], [-1.0]]
        model.omega_field[:] = [[np.inf], [2.0 * model.omega_max], [-5.0]]
        model.apply_positivity_limiter()
        np.testing.assert_array_equal(model.k_field[:, 0], [0.0, model.k_max, -1.0])
        np.testing.assert_array_equal(model.omega_field[:, 0], [OMEGA_INF, model.omega_max, -5.0])
