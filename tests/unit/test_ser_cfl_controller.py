# -*- coding: utf-8 -*-
"""隐式稳态（Newton-Krylov）的 SER CFL 律（`adaptive_cfl/ser.py`）。

判据：
1. 律本身：CFL 按 `R_{n-1}/R_n` 等比变化，每步限幅 `[shrink_limit, growth_limit]`，
   总体钳到 `[cfl_min, cfl_max]`；非有限残差按最大收缩倍率收缩且不污染基准；
2. `reset()` 回到起点且丢弃残差基准（换阶后残差量级不可比）；
3. 策略分派：`build_cfl_policy` 对 NEWTON_KRYLOV 给 SER，对显式格式给显式控制器；
   缺省值来自**所选控制器**的签名（两者差两个数量级，不能共用一份）。
"""

import inspect
import math

import pytest

from autoflowcfd.core.time_integration.adaptive_cfl import (
    AdaptiveCFLController,
    SERCFLController,
)
from autoflowcfd.core.time_integration.adaptive_cfl.policy import (
    build_cfl_policy,
    controller_default,
)
from autoflowcfd.core.time_integration.base import TimeIntegrationScheme as S


def test_first_update_only_records_baseline():
    c = SERCFLController(cfl_start=5.0, cfl_max=1e4, cfl_min=0.5)
    assert c.update(1e6) == 5.0


def test_cfl_scales_with_residual_ratio():
    c = SERCFLController(cfl_start=5.0, cfl_max=1e4, cfl_min=0.5)
    c.update(1e6)
    assert c.update(8e5) == pytest.approx(5.0 * 1.25)
    # 残差上升但 Newton 步被完整接受：物理暂态，保持
    assert c.update(1e6) == pytest.approx(5.0 * 1.25)
    # 残差上升且步没被完整接受：至少减半（比值 0.8 不够）
    assert c.update(1.25e6, step_ok=False) == pytest.approx(5.0 * 1.25 * 0.5)


def test_failed_step_never_grows_cfl():
    """残差略降但 Newton 步没被完整接受（例如线性求解失败）：不能因为比值
    > 1 就放大。真实数据：GMRES 用满 200 次只到 0.96 的那一步，旧规则把 CFL
    从 3351 放大到 3385。"""
    c = SERCFLController(cfl_start=100.0, cfl_max=1e4, cfl_min=0.5)
    c.update(4.03e5)
    assert c.update(3.89e5, step_ok=False) == pytest.approx(50.0)


def test_accepted_steps_grow_at_least_min_growth_on_a_plateau():
    """残差在平台上小幅起伏、每步都被完整接受：CFL 每步至少乘 MIN_GROWTH，不自锁在
    进展本身的量级上（plate_demo P1+SST 暂态里纯 SER 停在 25 附近 50 多步）。"""
    from autoflowcfd.core.time_integration.adaptive_cfl.ser import MIN_GROWTH, RISE_HOLD

    c = SERCFLController(cfl_start=25.0, cfl_max=1e4, cfl_min=0.5)
    c.update(2.0e6)
    for r in (1.99e6, 2.03e6, 2.01e6, 2.05e6):        # 下降 0.5% / 上升 2% / 下降 / 上升
        c.update(r)
    assert c.cfl_number == pytest.approx(25.0 * MIN_GROWTH ** 4)
    # 上升超过 RISE_HOLD：保持
    before = c.cfl_number
    assert c.update(2.05e6 * RISE_HOLD * 1.01) == pytest.approx(before)


def test_growth_while_holding_does_not_compound():
    """冲击启动：残差连续上升、每步都被完整接受 -> CFL 一直保持在起点。"""
    c = SERCFLController(cfl_start=5.0, cfl_max=1e4, cfl_min=0.5)
    r = 1.0
    for _ in range(50):
        r *= 1.3
        c.update(r)
    assert c.cfl_number == 5.0


def test_per_step_growth_and_shrink_are_limited():
    c = SERCFLController(cfl_start=5.0, cfl_max=1e4, cfl_min=0.01,
                         growth_limit=2.0, shrink_limit=0.1)
    c.update(1e6)
    assert c.update(1.0) == pytest.approx(10.0)       # 降 1e6 倍也只翻倍
    assert c.update(1e9, step_ok=False) == pytest.approx(1.0)   # 升 1e9 倍也只缩 10 倍


def test_clamped_to_bounds():
    c = SERCFLController(cfl_start=5.0, cfl_max=8.0, cfl_min=2.0)
    c.update(1.0)
    assert c.update(0.1) == 8.0
    for _ in range(5):
        c.update(1e3, step_ok=False)
    assert c.cfl_number == 2.0


@pytest.mark.parametrize("bad", [math.nan, math.inf, 0.0, -1.0])
def test_non_finite_residual_shrinks_and_keeps_baseline(bad):
    c = SERCFLController(cfl_start=5.0, cfl_max=1e4, cfl_min=0.01, shrink_limit=0.1)
    c.update(1e6)
    assert c.update(bad) == pytest.approx(0.5)
    # 基准仍是 1e6：下一步残差 5e5 应让 CFL 翻倍，而不是与坏值比较
    assert c.update(5e5) == pytest.approx(1.0)


def test_reset_restores_start_and_drops_baseline():
    c = SERCFLController(cfl_start=5.0, cfl_max=1e4, cfl_min=0.5)
    c.update(1e6)
    c.update(5e5)
    assert c.cfl_number == pytest.approx(10.0)
    c.reset()
    assert c.cfl_number == 5.0
    assert c.update(1.0) == 5.0      # 重置后的首步只记录基准


@pytest.mark.parametrize("kw", [dict(cfl_start=0.0), dict(cfl_max=math.inf), dict(cfl_min=-1.0),
                                dict(cfl_min=10.0, cfl_max=1.0), dict(growth_limit=0.9),
                                dict(shrink_limit=1.0)])
def test_invalid_parameters_rejected(kw):
    with pytest.raises(ValueError):
        SERCFLController(**kw)


def test_policy_dispatches_by_scheme():
    ctrl, fixed = build_cfl_policy(S.NEWTON_KRYLOV)
    assert fixed is None and isinstance(ctrl, SERCFLController)
    ref = SERCFLController()
    assert (ctrl.cfl_start, ctrl.cfl_max, ctrl.cfl_min) == (ref.cfl_start, ref.cfl_max, ref.cfl_min)
    for scheme in (S.SSP_RK3, S.SSP_RK2, S.FORWARD_EULER, S.IMEX_EULER):
        ctrl, _ = build_cfl_policy(scheme)
        assert type(ctrl) is AdaptiveCFLController


def test_defaults_come_from_the_selected_controller():
    for scheme, cls in ((S.NEWTON_KRYLOV, SERCFLController), (S.SSP_RK3, AdaptiveCFLController)):
        want = inspect.signature(cls.__init__).parameters["cfl_start"].default
        assert controller_default("cfl_start", scheme) == want
        _, fixed = build_cfl_policy(scheme, adaptive=False)
        assert fixed == pytest.approx(want)


def test_newton_krylov_default_cfl_max_lets_ptc_vanish():
    """NK 的默认上限 1e8（2026-10-04 由 1e4 调高，A/B 数据见 `ser.py` 模块文档）：1e4 时伪时间项远未
    可忽略，湍流平板 SA P3 尾段每步只降到 0.85 倍。CLI 不传 `--cfl-max` 时策略取的就是这个签名默认值。"""
    controller, _ = build_cfl_policy(S.NEWTON_KRYLOV)
    assert controller_default("cfl_max", S.NEWTON_KRYLOV) == 1.0e8
    assert controller.cfl_max == 1.0e8
    # 完整接受、残差每步减半：从 1e4 起每步翻倍，约 14 步到上限
    c = SERCFLController(cfl_start=1.0e4)
    c.update(1.0)
    r, n = 1.0, 0
    while c.cfl_number < c.cfl_max:
        r *= 0.5
        c.update(r)
        n += 1
    assert n == 14
