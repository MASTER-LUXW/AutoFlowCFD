"""AutoFlowCFD V2.0 - TimeIntegrator 的格式分派。

DUAL_TIME 的每个物理步由带 BDF 物理时间项的 Newton 步求解（`time_integration/implicit/newton/physical_time.py`，
BDF1/BDF2 方程的正确性见 `test_dual_time_newton.py`），不经过显式积分器。
"""

import numpy as np
import pytest

from autoflowcfd.core.time_integration import TimeIntegrator, TimeIntegrationScheme


def test_generic_step_rejects_dual_time_scheme():
    """通用 step() 入口不能把 DUAL_TIME 悄悄当成显式格式推进，必须报错。"""
    ti = TimeIntegrator(scheme=TimeIntegrationScheme.DUAL_TIME)
    U_n = np.tile(np.array([1.2, 12.0, 0.0, 0.0, 2.5e5]), (4, 1))
    with pytest.raises(ValueError, match="DUAL_TIME"):
        ti.step(U_n, lambda u: 0.01 * u, np.full(U_n.shape[0], 1e-2))
    assert not hasattr(ti, "step_dual_time")


@pytest.mark.parametrize("scheme", [TimeIntegrationScheme.NEWTON_KRYLOV, TimeIntegrationScheme.DUAL_TIME])
def test_implicit_scheme_has_no_explicit_stage_table(scheme):
    """隐式格式没有 Shu-Osher 系数表：误走显式 stage 推进必须报错。此前取表时缺省回退到前向 Euler 表，静默推进。"""
    ti = TimeIntegrator(scheme=scheme)
    U = np.ones((2, 5))
    with pytest.raises(ValueError, match="不是显式格式"):
        ti._ssp_rk_stage_step(U, lambda u: np.zeros_like(u), np.ones(2))
