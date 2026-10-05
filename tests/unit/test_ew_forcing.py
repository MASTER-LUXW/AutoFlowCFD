# -*- coding: utf-8 -*-
"""inexact-Newton 的 forcing term：Eisenstat-Walker choice 1（`implicit/forcing.py`，2026-10-05 由 choice 2 改）。

choice 2 只看上一步残差掉了多少，湍流平板 SA P3 收敛末段把 eta 收紧到 1e-3、GMRES 每步 253 次，而非线性
残差每步只降 3~29 倍（远小于 1/eta）。choice 1 按线性模型与真实残差的吻合程度定 eta：吻合时收紧，
不吻合时保持宽松。
"""

import pytest

from autoflowcfd.core.time_integration.implicit.forcing import (
    _ETA_MAX, _ETA_MIN, _EW_PHI, EisenstatWalkerForcing,
)


def _two_steps(r0, lin_rel, r1, full=True):
    f = EisenstatWalkerForcing()
    f.next_eta(r0, 0.0)
    f.record_step(lin_rel, full_step=full)
    return f, f.next_eta(r1, 0.0)


def test_first_step_uses_the_upper_bound():
    assert EisenstatWalkerForcing().next_eta(1.0, 0.0) == _ETA_MAX


def test_tight_when_the_linear_model_predicts_the_nonlinear_residual():
    """真实残差 ~ 线性残差（Newton 模型可信）：eta = |r1 - lin|/r0 很小。"""
    _, eta = _two_steps(1.0, 1e-3, 1.05e-3)
    assert eta == pytest.approx(5e-5 if 5e-5 > _ETA_MIN else _ETA_MIN)


def test_loose_when_the_linear_model_overpredicts():
    """P3 末段的情形：线性求解到 1e-3，真实残差只降 5 倍 -> 不再收紧（choice 2 会给 0.9*0.2^2=0.036）。"""
    _, eta = _two_steps(1.0, 1e-3, 0.2)
    assert eta == _ETA_MAX


def test_modified_step_falls_back_to_the_upper_bound():
    """步被回溯或被物理性限幅松弛：线性模型不对应实际步，下一步取上界。"""
    _, eta = _two_steps(1.0, 1e-3, 1.05e-3, full=False)
    assert eta == _ETA_MAX
    _, eta = _two_steps(1.0, float("nan"), 1.05e-3)
    assert eta == _ETA_MAX


def test_over_contraction_guard():
    """`eta_{k-1}^phi > 0.1` 时不许一步收紧到更小（原文 (2.2)）。"""
    f = EisenstatWalkerForcing()
    f.next_eta(1.0, 0.0)                      # eta_0 = 0.1 -> guard 0.1^phi = 0.024 < 0.1：不触发
    f._eta = 0.5                              # 模拟上一步较宽松的 eta
    f.record_step(1e-3, full_step=True)
    eta = f.next_eta(1.05e-3, 0.0)
    assert eta == pytest.approx(min(_ETA_MAX, 0.5 ** _EW_PHI))


def test_nonlinear_tolerance_floor():
    """安全下限：别比"最终非线性容差的一半"还严。"""
    f, _ = _two_steps(1.0, 1e-3, 1.0001e-3)
    f.record_step(1e-6, full_step=True)
    eta = f.next_eta(1.0e-6, 2e-8)
    assert eta >= 0.5 * 2e-8 / 1.0e-6
