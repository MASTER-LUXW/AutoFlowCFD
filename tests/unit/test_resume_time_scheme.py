# -*- coding: utf-8 -*-
"""`solve resume` 的时间格式：显式覆盖 > checkpoint 记录 > ssp_rk3（旧 checkpoint）。

隐式稳态运行的 checkpoint 续算时不能被静默换回显式格式（CFL 默认值差两个
数量级，换回显式后 CFL 5 直接越过显式稳定极限）；瞬态格式不属于 resume 的
稳态语义，退回 ssp_rk3 并警告。
"""

import click
import pytest

from autoflowcfd.cli.solve.checkpoint_io.rebuild import resolve_resume_time_scheme
from autoflowcfd.core.time_integration.base import TimeIntegrationScheme as S


def test_recorded_scheme_is_used():
    assert resolve_resume_time_scheme(None, {"time_scheme": "newton_krylov"}) is S.NEWTON_KRYLOV
    assert resolve_resume_time_scheme(None, {"time_scheme": "ssp_rk3"}) is S.SSP_RK3


def test_old_checkpoint_without_record_is_rk3():
    assert resolve_resume_time_scheme(None, {}) is S.SSP_RK3


def test_override_wins():
    assert resolve_resume_time_scheme("newton-krylov", {"time_scheme": "ssp_rk3"}) is S.NEWTON_KRYLOV
    assert resolve_resume_time_scheme("rk3", {"time_scheme": "newton_krylov"}) is S.SSP_RK3


def test_transient_record_falls_back_to_rk3():
    assert resolve_resume_time_scheme(None, {"time_scheme": "dual_time"}) is S.SSP_RK3


def test_transient_override_rejected():
    with pytest.raises(click.BadParameter):
        resolve_resume_time_scheme("dual-time", {})

