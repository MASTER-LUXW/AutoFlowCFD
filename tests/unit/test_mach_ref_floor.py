"""`mach_ref` 的物理下限（`core/fr_solver/mach_ref.py::MACH_REF_FLOOR`）。

`mach_ref` 直接决定 CFL 用的预处理波速 `c_pre ~ mach_ref*a`，也就直接决定 dt；下限太高白损步长，太低越出
显式稳定域。2026-09-17 用直接谱测量在 AUSM+up 物理声速档上标定（组装预处理后算子 `Gamma^-1 R` 的稠密
Jacobian 求特征值，与 SSP-RK3 稳定域 |Im| <= 1.732 比，**扫自洽的 mach_ref**——随 vel_inf 一起变）：

    vel_inf  M_true    用真实值 dt      用真实值裕度
       30    0.0882    1.04e-5           5.33x  稳定
       17    0.0500    1.78e-5           3.55x  稳定
      13.6   0.0400    2.22e-5           2.96x  稳定
      10.2   0.0300    2.96e-5           2.32x  稳定
       3     0.0088    1.01e-4           0.74x  越界
       1     0.0029    3.02e-4           0.25x  越界

下限的最坏情形恰好是 `M_true == FLOOR` 那一点，取 **0.05**（3.55 倍裕度）。2026-10-18 以前另有 AUSM+up
legacy 档的 0.1（随该档删除）。

覆盖：下限取值与裕度、真实求解器构造出的 `mach_ref`（高于下限原样保留、低于下限钳住）、钳制留日志。
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_solver.mach_ref import MACH_REF_FLOOR
from autoflowcfd.core.freestream_defaults import MU_MOLECULAR
from autoflowcfd.core.time_integration import TimeIntegrationScheme

#: 谱测量给出的各候选下限的裕度（见模块文档那张表）。改下限前必须重跑。
_MEASURED_MARGIN_AT_FLOOR = {0.05: 3.55, 0.04: 2.96, 0.03: 2.32}

#: 要求的最小裕度。为什么要这么多：越界一次之后 CFL 收缩救不回来，且失稳
#: 特征滞后 8~17 步（`core/time_integration/adaptive_cfl.py` 模块文档第 12
#: 条）——所以稳定性相关的默认值一律留 3 倍以上裕度。
_REQUIRED_MARGIN = 3.0


class TestFloorConstant:
    def test_floor_is_the_calibrated_value(self):
        assert MACH_REF_FLOOR == 0.05

    def test_floor_has_the_required_margin(self):
        """选定的下限必须在谱测量里留够裕度。"""
        m = _MEASURED_MARGIN_AT_FLOOR.get(MACH_REF_FLOOR)
        assert m is not None, (
            f"MACH_REF_FLOOR = {MACH_REF_FLOOR} 没有对应的谱测量数据（已测：{sorted(_MEASURED_MARGIN_AT_FLOOR)}）"
            f"——改这个下限必须先重跑那个测量，不能凭插值")
        assert m >= _REQUIRED_MARGIN, f"下限 {MACH_REF_FLOOR} 的实测裕度只有 {m}x，低于要求的 {_REQUIRED_MARGIN}x"


def _solver(vel):
    from tests.validation._channel_mesh import build_channel_mesh_prism

    from autoflowcfd.core.fr_solver import FRSolver

    mesh = build_channel_mesh_prism(1, 4, 3, 1, 1.0, 1.0, 0.25)
    return FRSolver(mesh=mesh, order=1, turb_model_name="NONE", rho_inf=1.225, vel_inf=vel, p_inf=101325.0,
                    time_scheme=TimeIntegrationScheme.SSP_RK3, mu_molecular=MU_MOLECULAR)


_A = float(np.sqrt(1.4 * 101325.0 / 1.225))


class TestRealSolverMachRef:
    def test_true_mach_is_used_at_plate_speed(self):
        """plate_demo 的工况（U=30, M=0.0882）高于下限：用真实值（dt 比旧的 0.1 下限 +14%）。"""
        s = _solver(30.0)
        assert s.freestream["mach_ref"] == pytest.approx(30.0 / _A, rel=1e-9)

    def test_clamps_far_below_the_floor(self):
        """Couette 量级（U=0.01, M=2.9e-5）：必须钳住（实测 max|dt*λ| = 684）。"""
        assert _solver(0.01).freestream["mach_ref"] == pytest.approx(MACH_REF_FLOOR)

    def test_floor_never_lowers_a_high_mach_number(self):
        """下限只能往上钳。M 高于下限时必须原样保留。"""
        assert _solver(0.5 * _A).freestream["mach_ref"] == pytest.approx(0.5, rel=1e-9)


class TestClampIsNotSilent:
    def test_clamping_emits_a_log_line(self):
        """本项目不接受静默兜底：钳制改变了数值方案，必须留痕。"""
        from loguru import logger

        seen = []
        sink = logger.add(lambda m: seen.append(str(m)), level="INFO")
        try:
            _solver(0.01)
        finally:
            logger.remove(sink)
        assert any("mach_ref" in m for m in seen), (
            "mach_ref 被钳制却没有任何日志痕迹")

    def test_no_log_when_no_clamping(self):
        """不钳制时不该刷这条信息。"""
        from loguru import logger

        seen = []
        sink = logger.add(lambda m: seen.append(str(m)), level="INFO")
        try:
            _solver(0.5 * _A)
        finally:
            logger.remove(sink)
        assert not any("低于当前 AUSM+up" in m for m in seen)
