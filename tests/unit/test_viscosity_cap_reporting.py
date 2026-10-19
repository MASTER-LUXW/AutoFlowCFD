"""涡粘比上限激活的计数与告警（I-47，2026-10-19，`turbulence/limits.py::report_viscosity_cap`）。

此前 SST 的涡粘与 SA-neg 的 nu_tilde 撞到上限时静默截断，全尺度高雷诺数外流上若在物理解上激活没有任何报告。
"""

import numpy as np
from loguru import logger

from autoflowcfd.core.turbulence.limits import TURBULENT_VISCOSITY_RATIO_MAX
from autoflowcfd.core.turbulence.sa import SAModel
from autoflowcfd.core.turbulence.sst import SSTModelFR
from autoflowcfd.core.turbulence.sst.bounds import configure_sst_model

MU, RHO = 1.8e-5, 1.2


def _capture():
    messages = []
    return messages, logger.add(lambda msg: messages.append(str(msg)), level="WARNING")


def test_sst_cap_activation_is_counted_and_reported_with_doubling_throttle():
    m = SSTModelFR(1, 4, k_inf=1e-3, omega_inf=10.0)
    configure_sst_model(m, 30.0, "resolved")
    k = np.array([[1.0, 1.0, 1e-6, 1e-6]])
    omega = np.array([[1e-6, 1e-6, 100.0, 100.0]])                 # 前两点 nu_t = k/omega 远超上限
    rho, S, F2 = np.full_like(k, RHO), np.ones_like(k), np.zeros_like(k)
    messages, sink = _capture()
    try:
        nu_t = m.compute_eddy_viscosity(k, omega, rho, S, F2, MU)
        assert m.viscosity_cap_activations == 2 and len(messages) == 1
        assert np.all(nu_t[0, :2] == TURBULENT_VISCOSITY_RATIO_MAX * MU / RHO)
        m.compute_eddy_viscosity(k, omega, rho, S, F2, MU)          # 累计 4 = 2 x 上次告警时的 2：再告警
        m.compute_eddy_viscosity(k, omega, rho, S, F2, MU)          # 累计 6 < 8：不告警
    finally:
        logger.remove(sink)
    assert m.viscosity_cap_activations == 6 and len(messages) == 2


def test_sa_cap_activation_is_counted():
    m = SAModel(1, 3, nu_ref=MU / RHO, viscosity_ratio=3.0)
    m.nu_tilde_field = np.array([[1.0, 2.0 * m.nu_tilde_max, 3.0 * m.nu_tilde_max]])
    messages, sink = _capture()
    try:
        m.apply_positivity_limiter()
    finally:
        logger.remove(sink)
    assert m.viscosity_cap_activations == 2 and len(messages) == 1
    assert np.all(m.nu_tilde_field <= m.nu_tilde_max)


def test_values_below_the_cap_are_not_counted():
    m = SSTModelFR(1, 2, k_inf=1e-3, omega_inf=10.0)
    configure_sst_model(m, 30.0, "resolved")
    k, omega = np.full((1, 2), 1e-3), np.full((1, 2), 10.0)
    m.compute_eddy_viscosity(k, omega, np.full((1, 2), RHO), np.ones((1, 2)), np.zeros((1, 2)), MU)
    assert m.viscosity_cap_activations == 0
