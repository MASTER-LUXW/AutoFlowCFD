"""物理时间随时间精确推进累计、随 checkpoint 持久化；是否时间精确推进随 checkpoint 恢复（I-170）。

此前 `time_integrator.current_time` 从不推进、checkpoint 也不记，`post transient-mean/rms/psd` 拿迭代数当物理时间
（--warmup-time 按迭代数比较）；`time_accurate` 不进 checkpoint，瞬态 rk3 的续算重建成稳态的局部时间步推进。
"""

from types import SimpleNamespace

import click
import numpy as np
import pytest

from autoflowcfd.cli.post.helpers import _physical_time
from autoflowcfd.cli.solve.checkpoint_io.rebuild import resolve_resume_time_accurate
from autoflowcfd.cli.solve.checkpoint_io.write import write_checkpoint
from autoflowcfd.core.freestream_defaults import MU_MOLECULAR, P_INF, RHO_INF
from autoflowcfd.core.time_integration import TimeIntegrationScheme
from autoflowcfd.core.time_integration.base import TimeIntegrator
from autoflowcfd.core.time_integration.time_accurate import advance
from autoflowcfd.core.utils.checkpoint import CheckpointManager
from autoflowcfd.core.utils.checkpoint_time import restore_time_state, time_metadata
from tests.validation._channel_mesh import build_channel_mesh_prism

_BC = {"x_min": "VELOCITY_INLET", "x_max": "PRESSURE_OUTLET", "wall_bottom": "WALL", "wall_top": "WALL",
       "z_min": "SYMMETRY", "z_max": "SYMMETRY"}


def _fake(scheme, time_accurate=False):
    s = SimpleNamespace(time_integrator=TimeIntegrator(scheme=scheme), time_accurate=time_accurate)

    def step(h):
        if time_accurate:
            s._ta_h = min(h, 0.3e-3)          # 显式子步：每步最多推进 0.3 ms
        return 1.0
    s.step = step
    return s


def test_advance_accumulates_physical_time_only_for_time_accurate_schemes():
    dual = _fake(TimeIntegrationScheme.DUAL_TIME)
    explicit = _fake(TimeIntegrationScheme.SSP_RK3, time_accurate=True)
    steady = _fake(TimeIntegrationScheme.SSP_RK3)
    for _ in range(3):
        advance(dual, 1e-3)
        advance(explicit, 1e-3)              # 每个物理步切成 4 个子步，最后一个恰好落在 t + dt
        advance(steady, 1e-3)
    assert dual.time_integrator.current_time == pytest.approx(3e-3, rel=1e-14)
    assert explicit.time_integrator.current_time == pytest.approx(3e-3, rel=1e-14)
    assert steady.time_integrator.current_time == 0.0


def test_time_metadata_round_trip_and_missing_physical_time_is_an_error():
    explicit = _fake(TimeIntegrationScheme.SSP_RK3, time_accurate=True)
    explicit.time_integrator.current_time = 0.125
    meta = time_metadata(explicit)
    assert meta["time_accurate"] is True and meta["physical_time"] == 0.125
    assert "physical_time" not in time_metadata(_fake(TimeIntegrationScheme.SSP_RK3))

    resumed = _fake(TimeIntegrationScheme.SSP_RK3, time_accurate=True)
    restore_time_state(resumed, meta)
    assert resumed.time_integrator.current_time == 0.125
    with pytest.raises(ValueError, match="物理时间"):
        restore_time_state(_fake(TimeIntegrationScheme.DUAL_TIME), {})
    restore_time_state(_fake(TimeIntegrationScheme.NEWTON_KRYLOV), {})     # 稳态：没有物理时间可恢复


def test_resume_time_accurate_rules():
    assert resolve_resume_time_accurate(None, {"time_accurate": True}) is True
    assert resolve_resume_time_accurate(None, {"time_accurate": False}) is False
    with pytest.raises(click.BadParameter):                 # 瞬态 rk3 不能换成稳态格式续算
        resolve_resume_time_accurate("newton-krylov", {"time_accurate": True})
    # 早于本字段的 checkpoint：能确定的确定，显式格式无法区分稳态与瞬态 rk3 时报错
    assert resolve_resume_time_accurate("rk3", {"time_scheme": "ssp_rk3"}) is False
    assert resolve_resume_time_accurate(None, {"time_scheme": "newton_krylov"}) is False
    assert resolve_resume_time_accurate(None, {"time_scheme": "dual_time"}) is False
    with pytest.raises(click.ClickException, match="time-scheme"):
        resolve_resume_time_accurate(None, {"time_scheme": "ssp_rk3"})


def test_post_reads_the_recorded_physical_time():
    assert _physical_time({"physical_time": 0.5}, "c.h5") == 0.5
    with pytest.raises(ValueError, match="物理时间"):
        _physical_time({}, "c.h5")                           # 此前回退成迭代数


def test_real_dual_time_run_writes_its_physical_time(tmp_path):
    from autoflowcfd.core.fr_solver import FRSolver

    mesh = build_channel_mesh_prism(1, 3, 3, 2, 0.4, 0.1, 0.08)
    mesh.boundary_bc_types = dict(_BC)
    s = FRSolver(mesh=mesh, order=1, turb_model_name="NONE", time_scheme=TimeIntegrationScheme.DUAL_TIME,
                 vel_inf=10.0, rho_inf=RHO_INF, p_inf=P_INF, mu_molecular=MU_MOLECULAR)
    s.order_continuation_enabled = False
    s.solve(max_iter=3, dt=2e-4, tol=0.0)
    assert s.time_integrator.current_time == pytest.approx(6e-4, rel=1e-14)
    path = write_checkpoint(s, str(tmp_path), 3, "mesh.nas", order=1, turbulence_model="none", backend="cpu",
                            quiet=True)
    meta = CheckpointManager().load(path)[3]
    assert meta["physical_time"] == pytest.approx(6e-4, rel=1e-14) and bool(meta["time_accurate"]) is False
    assert np.isfinite(meta["physical_time"])
