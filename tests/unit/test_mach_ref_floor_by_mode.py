"""`mach_ref` 下限按 AUSM+up 预处理档分派，而不是一个与档位无关的 0.1。

## 缺口

`solver.py` 把按来流算出的 `mach_ref = vel_inf / a` 钳到 `_MACH_REF_FLOOR
= 0.1`。那条下限的依据是预处理通量的压差放大系数 `~1/mach_ref^2`（`Mp`
项分母 `fa*a_half_p^2`，`fa≈2*mach_ref`、`a_half_p^2 = beta2*a^2`、`beta2`
下限 `1.1*mach_ref^2`，两者同时随 mach_ref 塌缩），实证标定是 Couette
0.02 仍发散 / 0.05 稳定、TGV 0.0874 仍发散 / 0.1 稳定。

**但那次标定是在 `legacy` 档上做的**——2026-09-16 的 `PRECOND_PHYSICAL`
（现在的默认）把压力分裂改用**物理声速**，`1/mach_ref^2` 那条放大路径本身
就变了。于是一个已经不适用的下限在把每个算例的 `mach_ref` 往上抬，而
`mach_ref` 直接决定 CFL 用的预处理波速 `c_pre ~ mach_ref*a`，也就直接决定
dt。plate_demo 真实 M=0.0882 被抬到 0.1，白损 14% 的 dt。

## 重标定（直接谱测量，2026-09-17）

组装预处理后算子 `Gamma^-1 R` 的稠密 Jacobian 求特征值，与 SSP-RK3 稳定域
（|Im| <= 1.732）比。**扫的是自洽的 mach_ref**（随 vel_inf 一起变），不是
在固定来流上人为改 mach_ref——后者是"给 M=0.088 的流动配 M=0.0009 的
预处理"这种物理上不存在的组合（第一版就这么扫了，数据没有意义）：

    vel_inf  M_true    用 0.1 的裕度   用真实值 dt      用真实值裕度
       30    0.0882      6.50x        1.04e-5 (+14%)     5.33x  稳定
       17    0.0500      —            1.78e-5            3.55x  稳定
      13.6   0.0400      —            2.22e-5            2.96x  稳定
      10.2   0.0300      —            2.96e-5            2.32x  稳定
       3     0.0088      5.66x        1.01e-4            0.74x  越界
       1     0.0029      5.59x        3.02e-4            0.25x  越界

下限的最坏情形恰好是 `M_true == FLOOR` 那一点（那时 mach_ref 最小、dt
最大），所以第 2~4 行就是各候选下限的裕度。取 **0.05**（3.55 倍裕度）。

`legacy` 档保持 0.1：同一套测量下它在 mach_ref=0.0088 就已越界 4.5 倍
（max|dt*λ| = 7.81）、在 0.00088 是 768，正是那条 `1/mach_ref^2` 放大。

## 本文件覆盖

1. 两个下限常量的取值（改动它们必须先重跑谱测量）
2. 分派函数按档给出正确的下限
3. 真实求解器构造出的 `mach_ref`：physical 档用真实值、legacy 档被钳
4. 下限只能往上钳、不能往下改（M 高于下限时必须原样保留）
5. 钳制发生时必须留下日志痕迹（本项目不接受静默兜底）
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_operators.kernels import (
    PRECOND_LEGACY,
    PRECOND_PHYSICAL,
    PRECOND_PRESSURE_PHYSICAL,
)
from autoflowcfd.core.fr_solver.mach_ref import (
    _MACH_REF_FLOOR_LEGACY,
    _MACH_REF_FLOOR_PHYSICAL,
    _mach_ref_floor_for_mode,
)

#: 谱测量给出的各候选下限的裕度（见模块文档那张表）。改下限前必须重跑。
_MEASURED_MARGIN_AT_FLOOR = {0.05: 3.55, 0.04: 2.96, 0.03: 2.32}

#: 要求的最小裕度。为什么要这么多：越界一次之后 CFL 收缩救不回来，且失稳
#: 特征滞后 8~17 步（`core/time_integration/adaptive_cfl.py` 模块文档第 12
#: 条）——所以稳定性相关的默认值一律留 3 倍以上裕度。
_REQUIRED_MARGIN = 3.0


class TestFloorConstants:
    def test_physical_floor_is_the_calibrated_value(self):
        assert _MACH_REF_FLOOR_PHYSICAL == 0.05

    def test_legacy_floor_is_unchanged(self):
        """legacy 档的 0.1 标定仍然有效，不得跟着一起降。"""
        assert _MACH_REF_FLOOR_LEGACY == 0.1

    def test_physical_floor_has_the_required_margin(self):
        """选定的下限必须在谱测量里留够裕度。"""
        m = _MEASURED_MARGIN_AT_FLOOR.get(_MACH_REF_FLOOR_PHYSICAL)
        assert m is not None, (
            f"_MACH_REF_FLOOR_PHYSICAL = {_MACH_REF_FLOOR_PHYSICAL} 没有对应的"
            f"谱测量数据（已测：{sorted(_MEASURED_MARGIN_AT_FLOOR)}）——"
            f"改这个下限必须先重跑那个测量，不能凭插值")
        assert m >= _REQUIRED_MARGIN, (
            f"下限 {_MACH_REF_FLOOR_PHYSICAL} 的实测裕度只有 {m}x，"
            f"低于要求的 {_REQUIRED_MARGIN}x")

    def test_physical_floor_is_below_legacy_floor(self):
        """physical 档的放大路径更温和，下限必须更低（否则这次改动没意义）。"""
        assert _MACH_REF_FLOOR_PHYSICAL < _MACH_REF_FLOOR_LEGACY


class TestModeDispatch:
    @pytest.mark.parametrize("mode,expect", [
        (PRECOND_PHYSICAL, _MACH_REF_FLOOR_PHYSICAL),
        (PRECOND_PRESSURE_PHYSICAL, _MACH_REF_FLOOR_PHYSICAL),
        (PRECOND_LEGACY, _MACH_REF_FLOOR_LEGACY),
    ])
    def test_floor_for_mode(self, mode, expect):
        assert _mach_ref_floor_for_mode(mode) == expect

    def test_pressure_physical_shares_the_physical_floor(self):
        """`pressure_physical` 的压力分裂同样用物理声速，所以与 physical
        同档——这正是那条 `1/mach_ref^2` 放大被改掉的那一半。"""
        assert (_mach_ref_floor_for_mode(PRECOND_PRESSURE_PHYSICAL)
                == _mach_ref_floor_for_mode(PRECOND_PHYSICAL))


def _solver(vel, mode):
    import os
    import sys
    from pathlib import Path

    tests_dir = str(Path(__file__).resolve().parents[1])
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    from validation._channel_mesh import build_channel_mesh_prism

    from autoflowcfd.core.fr_solver import FRSolver

    old = os.environ.get("AFCFD_AUSM_PRECOND_MODE")
    os.environ["AFCFD_AUSM_PRECOND_MODE"] = mode
    try:
        mesh = build_channel_mesh_prism(1, 4, 3, 1, 1.0, 1.0, 0.25)
        return FRSolver(mesh=mesh, order=1, turb_model_name="NONE", n_vars=5,
                        rho_inf=1.225, vel_inf=vel, p_inf=101325.0)
    finally:
        if old is None:
            os.environ.pop("AFCFD_AUSM_PRECOND_MODE", None)
        else:
            os.environ["AFCFD_AUSM_PRECOND_MODE"] = old


_A = float(np.sqrt(1.4 * 101325.0 / 1.225))


class TestRealSolverMachRef:
    def test_physical_mode_uses_the_true_mach_at_plate_speed(self):
        """plate_demo 的工况（U=30, M=0.0882）：physical 档必须用真实值。

        这是本次改动的直接收益点——dt 因此 +14%（1.04e-5 vs 9.14e-6）。
        """
        s = _solver(30.0, "physical")
        assert s.freestream["mach_ref"] == pytest.approx(30.0 / _A, rel=1e-9)
        assert s.freestream["mach_ref"] < _MACH_REF_FLOOR_LEGACY

    def test_legacy_mode_still_clamps_at_plate_speed(self):
        s = _solver(30.0, "legacy")
        assert s.freestream["mach_ref"] == pytest.approx(0.1)

    def test_both_modes_clamp_far_below_the_floor(self):
        """Couette 量级（U=0.01, M=2.9e-5）：两档都必须钳住。

        physical 档在这个马赫数下同样越界（实测 max|dt*λ| = 684），所以
        这条下限**不是**纯 legacy 遗留——降它只能降到实测有裕度的那一档。
        """
        for mode, floor in (("physical", _MACH_REF_FLOOR_PHYSICAL),
                            ("legacy", _MACH_REF_FLOOR_LEGACY)):
            s = _solver(0.01, mode)
            assert s.freestream["mach_ref"] == pytest.approx(floor)

    def test_floor_never_lowers_a_high_mach_number(self):
        """下限只能往上钳。M 高于下限时必须原样保留，不能被改动。"""
        vel = 0.5 * _A          # M = 0.5
        for mode in ("physical", "legacy"):
            s = _solver(vel, mode)
            assert s.freestream["mach_ref"] == pytest.approx(0.5, rel=1e-9)


class TestClampIsNotSilent:
    def test_clamping_emits_a_log_line(self):
        """本项目不接受静默兜底：钳制改变了数值方案，必须留痕。"""
        from loguru import logger

        seen = []
        sink = logger.add(lambda m: seen.append(str(m)), level="INFO")
        try:
            _solver(0.01, "physical")
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
            _solver(0.5 * _A, "physical")
        finally:
            logger.remove(sink)
        assert not any("低于当前 AUSM+up" in m for m in seen)
