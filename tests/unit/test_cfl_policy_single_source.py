# -*- coding: utf-8 -*-
"""CFL 策略的唯一事实来源（`time_integration/adaptive_cfl/policy.py`）。

2026-09-25 之前六个后端构造点各写一份、已经分叉：控制器启用条件三种写法，
没有控制器时的 CFL 分别是 cfl_start(0.03) / 替身回退 0.1 / 构造参数 cfl(1.0)
—— 同一个双时间步算例的伪时间步长在不同后端上差 30 倍。

判据：
1. 规则本身（DUAL_TIME -> 固定 CFL；adaptive=False -> 固定 CFL；其余 -> 控制器；
   None 参数取控制器默认值，不另写一份）；
2. 结构：除 policy.py 外，`src/` 里不得再直接构造 `AdaptiveCFLController(`；
3. 行为：CPU 单机与 CPU 分布式在 IMEX / DUAL_TIME 下拿到同一个 CFL 来源。
"""

import inspect
import pathlib
import re

import pytest

from autoflowcfd.core.time_integration.adaptive_cfl import AdaptiveCFLController
from autoflowcfd.core.time_integration.adaptive_cfl.policy import (
    build_cfl_policy,
    current_cfl_number,
)
from autoflowcfd.core.time_integration.base import TimeIntegrationScheme as S

_DEFAULT_START = inspect.signature(AdaptiveCFLController.__init__).parameters["cfl_start"].default


@pytest.mark.parametrize("scheme", [S.SSP_RK3, S.SSP_RK2, S.FORWARD_EULER, S.IMEX_EULER,
                                    S.NEWTON_KRYLOV, "rk3", "imex"])
def test_adaptive_schemes_get_a_controller(scheme):
    ctrl, fixed = build_cfl_policy(scheme, cfl_start=0.02, cfl_max=0.05, cfl_min=0.01)
    assert fixed is None and ctrl is not None
    assert (ctrl.cfl_start, ctrl.cfl_max, ctrl.cfl_min) == (0.02, 0.05, 0.01)


@pytest.mark.parametrize("scheme", [S.DUAL_TIME, "dual-time", "dual_time"])
def test_dual_time_uses_fixed_cfl_start(scheme):
    ctrl, fixed = build_cfl_policy(scheme, cfl_start=0.037)
    assert ctrl is None and fixed == pytest.approx(0.037)


def test_non_adaptive_uses_fixed_cfl_start():
    ctrl, fixed = build_cfl_policy(S.SSP_RK3, adaptive=False, cfl_start=0.021)
    assert ctrl is None and fixed == pytest.approx(0.021)


def test_missing_values_come_from_the_controller_signature():
    """None 不得被某个构造点换成它自己的兜底值（多 GPU 此前是 `cfl`=1.0）。"""
    ctrl, _ = build_cfl_policy(S.SSP_RK3)
    ref = AdaptiveCFLController()
    assert (ctrl.cfl_start, ctrl.cfl_max, ctrl.cfl_min) == (ref.cfl_start, ref.cfl_max, ref.cfl_min)
    _, fixed = build_cfl_policy(S.DUAL_TIME)
    assert fixed == pytest.approx(_DEFAULT_START)


def test_current_cfl_number_priority():
    class Obj:
        pass

    o = Obj()
    o._cfl_controller, o.fixed_cfl_number = build_cfl_policy(S.DUAL_TIME, cfl_start=0.044)
    assert current_cfl_number(o) == pytest.approx(0.044)
    o._cfl_controller, o.fixed_cfl_number = build_cfl_policy(S.SSP_RK3, cfl_start=0.025)
    assert current_cfl_number(o) == pytest.approx(0.025)


def test_no_controller_constructed_outside_policy():
    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "autoflowcfd"
    pat = re.compile(r"AdaptiveCFLController\(")
    offenders = []
    for p in root.rglob("*.py"):
        rel = p.relative_to(root).as_posix()
        if rel.endswith("adaptive_cfl/policy.py"):
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if pat.search(code) and not code.lstrip().startswith("class "):
                offenders.append(f"{rel}:{i}")
    assert not offenders, (
        "CFL 控制器只能经 adaptive_cfl/policy.py::build_cfl_policy 构造，"
        f"发现直接构造：{offenders}")
