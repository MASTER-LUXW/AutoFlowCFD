"""隐式稳态步的"线性预测放大倍数"诊断（2026-10-10，`implicit/mean_flow_step.py::note_linear_growth`）。

伪瞬态 Newton 步解 `(I/dtau + J) dU = -R`，线性化残差 `R + J dU = -dU/dtau`。`J` 的特征值实部为正（定常解线性
稳定）时 `||dU/dtau|| / ||R|| <= 1` 的量级；有实部为负、量级约 `1/dtau` 的特征值时这个比值明显大于 1——
cube_demo P1 的停滞就是这种情形（分离剪切层的不稳定模态），此前只表现为"残差不降、CFL 上不去"，没有任何说明。
"""

from types import SimpleNamespace

import numpy as np
import pytest
from loguru import logger

from autoflowcfd.core.time_integration.implicit import step_newton_krylov
from autoflowcfd.core.time_integration.implicit.mean_flow_step import (
    UNSTABLE_LINEAR_GROWTH, UNSTABLE_STEPS_BEFORE_WARNING, note_linear_growth, reset_newton_state,
)
from tests.unit._implicit_jfnk_common import _N, _NV, _SCALES


def _diagonal_system(rate):
    """`R(U) = rate * (U - U*)`：`J = rate * I`，`rate < 0` 即增长率 `-rate` 的不稳定模态。"""
    u_star = _SCALES[None, :] * np.ones((_N, _NV))
    u0 = u_star * (1.0 + 1e-6)                      # 离平衡很近：更新量远小于物理性限幅

    def residual(u_flat):
        return rate * (u_flat - u_star)

    return residual, u0


@pytest.mark.parametrize("rate, dtau, expected", [
    (2.0, 0.5, 0.5),            # 稳定：1 / (1 + dtau*rate)
    (2.0, 1e9, 0.0),            # 纯 Newton：线性化残差为零
    (-1.0, 0.5, 2.0),           # 不稳定、dtau 为增长时间的一半：1 / (1 - 0.5)
    (-1.0, 0.8, 5.0),           # 越接近 1/增长率放大越厉害
])
def test_linear_growth_is_the_ptc_amplification(rate, dtau, expected):
    residual, u0 = _diagonal_system(rate)
    _u, info = step_newton_krylov(residual, u0, np.full(_N, dtau), _SCALES, gmres_max_iter=50, gmres_restart=20)
    assert info["linear_growth"] == pytest.approx(expected, rel=1e-3, abs=1e-6)


def _info(growth, ok):
    return dict(linear_growth=growth, theta=1.0 if ok else 0.5, n_dtau_cuts=0, gmres_info=0, linear_rel_residual=0.01)


def test_repeated_unstable_steps_are_reported_once_per_batch():
    solver = SimpleNamespace()
    reset_newton_state(solver)
    messages = []
    sink = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    try:
        # 完整接受的步、或放大倍数不超过阈值的步都不计
        note_linear_growth(solver, _info(10.0, ok=True))
        note_linear_growth(solver, _info(UNSTABLE_LINEAR_GROWTH, ok=False))
        note_linear_growth(solver, _info(float("nan"), ok=False))
        assert solver._newton_unstable_steps == 0 and not messages
        for _ in range(UNSTABLE_STEPS_BEFORE_WARNING - 1):
            note_linear_growth(solver, _info(4.75, ok=False))
        assert not messages
        note_linear_growth(solver, _info(4.75, ok=False))
        assert len(messages) == 1 and "solve transient" in messages[0] and "4.75" in messages[0]
        for _ in range(UNSTABLE_STEPS_BEFORE_WARNING):
            note_linear_growth(solver, _info(2.0, ok=False))
        assert len(messages) == 2
    finally:
        logger.remove(sink)
    reset_newton_state(solver)                       # 换阶后重新计数
    assert solver._newton_unstable_steps == 0


def test_unlimited_columns_are_left_out_of_the_relative_change_limit():
    """SA-neg 的 nu_tilde 对任意实数值都良态（`SAModel.NEWTON_UNLIMITED_COLUMNS`）：不做相对变化限幅。
    湍流/非湍流界面上它只有来流量级，限幅会把界面推进需要的几十倍增长压到百分之几（cube_demo P1 每步约
    3000 行）。SST 的 k / ln omega 仍然限幅。"""
    from autoflowcfd.core.time_integration.implicit.physicality import ScaledFieldRowLimits
    from autoflowcfd.core.time_integration.implicit.reductions import LocalReductions
    from autoflowcfd.core.turbulence.sa.model import SAModel
    from autoflowcfd.core.turbulence.sst.unknowns import _SSTTransportedMixin

    red = LocalReductions(np)
    u0 = np.array([[1e-5], [2e-3], [-1e-4]])
    du = np.array([[1e-3], [-5e-3], [4e-3]])                     # 涨 100 倍、越过零、由负转正
    limited = ScaledFieldRowLimits((1.5e-5,))(u0, du, red)
    assert limited.max() < 0.3                                    # 默认：每一行都被压住
    assert SAModel.NEWTON_UNLIMITED_COLUMNS == (0,)
    free = ScaledFieldRowLimits((1.5e-5,), unlimited_columns=SAModel.NEWTON_UNLIMITED_COLUMNS)(u0, du, red)
    np.testing.assert_array_equal(free, np.ones(3))

    # 两列里只放开一列：另一列的限值原样保留
    u2, d2 = np.array([[1.0, 1e-5]]), np.array([[-0.9, 1e-3]])
    both = ScaledFieldRowLimits((1e-3, 1.5e-5))(u2, d2, red)
    only_first = ScaledFieldRowLimits((1e-3, 1.5e-5), unlimited_columns=(1,))(u2, d2, red)
    assert both[0] < only_first[0] < 1.0
    assert _SSTTransportedMixin.NEWTON_UNLIMITED_COLUMNS == ()
