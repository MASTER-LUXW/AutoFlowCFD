# -*- coding: utf-8 -*-
"""`solve resume` 的时间格式：显式覆盖 > checkpoint 记录 > ssp_rk3（旧 checkpoint）。

checkpoint 记录什么格式就续算什么格式：隐式稳态不能被静默换回显式（CFL 默认值差两个数量级），瞬态
（dual_time）续算仍是 dual_time——2026-10-11 以前它被换成 ssp_rk3，"续算"一个瞬态算例跑的是显式稳态。
"""

from types import SimpleNamespace

import click
import pytest

from autoflowcfd.cli.solve.checkpoint_io.rebuild import resolve_resume_time_scheme
from autoflowcfd.core.time_integration.base import TimeIntegrationScheme as S
from autoflowcfd.core.utils.checkpoint_time import (
    INNER_ITERATIONS_KEY, restore_inner_iteration_limit, run_tolerance, time_metadata,
)


def test_recorded_scheme_is_used():
    assert resolve_resume_time_scheme(None, {"time_scheme": "newton_krylov"}) is S.NEWTON_KRYLOV
    assert resolve_resume_time_scheme(None, {"time_scheme": "ssp_rk3"}) is S.SSP_RK3


def test_old_checkpoint_without_record_is_rk3():
    assert resolve_resume_time_scheme(None, {}) is S.SSP_RK3


def test_override_wins():
    assert resolve_resume_time_scheme("newton-krylov", {"time_scheme": "ssp_rk3"}) is S.NEWTON_KRYLOV
    assert resolve_resume_time_scheme("rk3", {"time_scheme": "newton_krylov"}) is S.SSP_RK3


def test_transient_record_resumes_as_transient():
    assert resolve_resume_time_scheme(None, {"time_scheme": "dual_time"}) is S.DUAL_TIME


def test_transient_override_rejected():
    with pytest.raises(click.BadParameter):
        resolve_resume_time_scheme("dual-time", {})


def _solver(scheme, inner=20, dt=None):
    return SimpleNamespace(time_integrator=SimpleNamespace(scheme=scheme, dual_time_steps=inner), _physical_dt=dt)


def test_dual_time_checkpoint_carries_the_inner_iteration_limit_and_runs_all_steps():
    """内迭代次数上限随 checkpoint 写出与恢复；时间精确的续算不以残差下降为停止条件。"""
    meta = time_metadata(_solver(S.DUAL_TIME, inner=7, dt=1e-4))
    assert meta == {"dt": 1e-4, INNER_ITERATIONS_KEY: 7}
    assert INNER_ITERATIONS_KEY not in time_metadata(_solver(S.NEWTON_KRYLOV, dt=1e-3))

    resumed = _solver(S.DUAL_TIME, inner=20)
    restore_inner_iteration_limit(resumed, meta)
    assert resumed.time_integrator.dual_time_steps == 7
    restore_inner_iteration_limit(resumed, {})                     # 旧 checkpoint：保持当前值
    assert resumed.time_integrator.dual_time_steps == 7
    steady = _solver(S.NEWTON_KRYLOV, inner=20)
    restore_inner_iteration_limit(steady, meta)
    assert steady.time_integrator.dual_time_steps == 20

    assert run_tolerance(1e-6, resumed) == 0.0
    assert run_tolerance(1e-6, steady) == 1e-6
