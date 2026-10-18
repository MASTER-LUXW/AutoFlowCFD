"""omega 壁面目标值（`turbulence/sst/bounds.py::omega_wall_value`，CPU 与 GPU 共用），公式随近壁处理方式：

    resolved：Menter 放大式  omega_wall = 10 * 6 nu / (beta1 d1^2)
    auto    ：Menter 二项混合式（OpenFOAM `omegaWallFunction` 同式）
              omega_vis = 6 nu/(beta1 d1^2)，omega_log = sqrt(k)/(C_mu^0.25 kappa d1)，omega = sqrt(vis^2 + log^2)

常数：beta1 = 0.075、C_mu = beta* = 0.09、kappa = 0.41（OpenFOAM nutWallFunction 默认值）。2026-10-17 以前公式由
环境变量 `AFCFD_OMEGA_WALL_MODE` 选择，且 CPU 与 GPU 各写一份（I-137）。
"""

import types

import numpy as np
import pytest

from autoflowcfd.core.turbulence.sst.bounds import BETA_STAR, omega_wall_value
from autoflowcfd.core.turbulence.wall_model.law import KAPPA

BETA1 = 0.075


def test_constants_match_the_literature():
    assert BETA_STAR == pytest.approx(0.09) and KAPPA == pytest.approx(0.41)
    assert BETA_STAR ** 0.25 == pytest.approx(0.5477225575, rel=1e-9)


def test_resolved_is_the_amplified_form_and_independent_of_k():
    nu, d1 = 1.5e-5, 2.0e-4
    got = omega_wall_value(np, nu, d1, None, BETA1, "resolved")
    assert got == pytest.approx(10.0 * 6.0 * nu / (BETA1 * d1 ** 2), rel=1e-14)


def test_auto_is_the_blended_form():
    nu, d1, k = 1.5e-5, 2.0e-4, 0.1
    got = omega_wall_value(np, nu, d1, k, BETA1, "auto")
    o_vis = 6.0 * nu / (BETA1 * d1 ** 2)
    o_log = np.sqrt(k) / (0.09 ** 0.25 * 0.41 * d1)
    assert got == pytest.approx(np.hypot(o_vis, o_log), rel=1e-14)
    assert omega_wall_value(np, nu, d1, 1e2, BETA1, "auto") > 1.5 * omega_wall_value(np, nu, d1, 1e-8, BETA1, "auto")


def test_unknown_treatment_is_rejected():
    with pytest.raises(ValueError, match="近壁处理方式"):
        omega_wall_value(np, 1.5e-5, 2e-4, 0.1, BETA1, "blended")


@pytest.mark.parametrize("treatment", ["resolved", "auto"])
def test_cpu_wall_target_follows_the_model_treatment(treatment):
    """CPU 壁面目标值按湍流模型的 `wall_treatment` 取公式（auto 用贴壁单元 k 的真实自由度均值）。"""
    from autoflowcfd.core.turbulence.transport.omega_wall import _omega_wall_formula

    solver = types.SimpleNamespace(turb_model=types.SimpleNamespace(k_field=np.full((1, 8), 0.1),
                                                                    wall_treatment=treatment),
                                   mesh=types.SimpleNamespace(n_prism_cells=1))
    got = _omega_wall_formula(solver, np.zeros(1, dtype=np.int64), np.full(1, 1.5e-5), np.full(1, 2e-4), BETA1)[0]
    assert got == pytest.approx(omega_wall_value(np, 1.5e-5, 2e-4, 0.1, BETA1, treatment), rel=1e-14)
