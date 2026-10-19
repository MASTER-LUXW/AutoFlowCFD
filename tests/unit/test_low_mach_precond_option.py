"""低马赫数伪时间预处理开关：构造参数 / CLI / 配置字段 / checkpoint 记录与续算（I-22，2026-10-18）。

此前开关的唯一外部入口是环境变量 `AFCFD_LOW_MACH_PRECOND`，且优先于构造参数：CLI 与配置文件表达不了，
checkpoint 也记不下——关掉预处理跑出的结果，续算时会静默换回开启。现在：
* `solve steady --low-mach-precond/--no-low-mach-precond`、`SteadyConfig.low_mach_precond` 给出请求值；
* 求解器记下请求值 `solver.low_mach_precond`（生效值 `low_mach_precond_enabled` 另按时间格式判定），
  checkpoint 写请求值（单机与分布式写入端共用 `solve_path_metadata`）；
* `solve resume` 默认沿用 checkpoint 记录，同名选项可覆盖；单机与四条分布式重建路径都接到。
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from autoflowcfd.cli.main import cli
from autoflowcfd.cli.solve.checkpoint_io import rebuild_solver_from_checkpoint, write_checkpoint
from autoflowcfd.cli.solve.checkpoint_io.rebuild import resolve_resume_low_mach_precond
from autoflowcfd.core.time_integration.base import TimeIntegrationScheme
from autoflowcfd.core.time_integration.time_accurate import configure_time_stepping
from autoflowcfd.core.utils.checkpoint_physics import solve_path_metadata
from tests.unit._host_state import with_host_state


# ---------------------------------------------------------------- 解析与记录
def test_resume_resolution_order():
    assert resolve_resume_low_mach_precond(None, {"low_mach_precond": False}) is False
    assert resolve_resume_low_mach_precond(None, {"low_mach_precond": True}) is True
    assert resolve_resume_low_mach_precond(True, {"low_mach_precond": False}) is True
    assert resolve_resume_low_mach_precond(False, {"low_mach_precond": True}) is False
    assert resolve_resume_low_mach_precond(None, {}) is True          # 旧 checkpoint：当时默认开启


@pytest.mark.parametrize("scheme,enabled", [
    (TimeIntegrationScheme.SSP_RK3, True),
    (TimeIntegrationScheme.NEWTON_KRYLOV, True),
    (TimeIntegrationScheme.DUAL_TIME, False),
])
def test_solver_records_requested_value_not_effective_value(scheme, enabled):
    """DUAL_TIME 的生效值恒为 False；checkpoint 记的是请求值，切到稳态格式续算时不会把预处理关掉。"""
    s = SimpleNamespace()
    configure_time_stepping(s, scheme, low_mach_precond=True)
    assert s.low_mach_precond is True and s.low_mach_precond_enabled is enabled
    s.time_integrator = SimpleNamespace(scheme=scheme)
    assert solve_path_metadata(s) == {"time_scheme": scheme.value, "low_mach_precond": True}

    configure_time_stepping(s, scheme, low_mach_precond=False)
    assert s.low_mach_precond is False and s.low_mach_precond_enabled is False


def test_environment_variable_is_ignored(monkeypatch):
    s = SimpleNamespace()
    monkeypatch.setenv("AFCFD_LOW_MACH_PRECOND", "1")
    configure_time_stepping(s, TimeIntegrationScheme.SSP_RK3, low_mach_precond=False)
    assert s.low_mach_precond_enabled is False


def test_steady_config_field():
    from autoflowcfd.config.solver_config import SteadyConfig

    assert SteadyConfig().low_mach_precond is True
    assert SteadyConfig(low_mach_precond=False).low_mach_precond is False


# ---------------------------------------------------------------- checkpoint 往返
def _fake_solver(low_mach_precond, order=0, n_cells=2, n_sps=1, n_vars=7):
    U = np.ones((n_cells, n_sps, n_vars))
    state = SimpleNamespace(U=U, Q=U.copy(), n_sps=n_sps, n_vars=n_vars)
    state._update_primitives = lambda: None
    return with_host_state(SimpleNamespace(
        state=state, order=order, current_order=order,
        mesh=SimpleNamespace(n_prism_cells=n_cells, jacobians={"det_jacs": np.ones(n_cells * n_sps)}),
        freestream={"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0},
        wall_treatment="resolved", low_mach_precond=low_mach_precond,
        mu_molecular=1.8e-5, _turbulence_intensity=0.01, _viscosity_ratio=5.0,
        artificial_viscosity_enabled=False, artificial_viscosity_alpha=1.0,
        time_integrator=SimpleNamespace(scheme=TimeIntegrationScheme.NEWTON_KRYLOV),
    ))


def _checkpoint(tmp_path, low_mach_precond):
    write_checkpoint(_fake_solver(low_mach_precond), str(tmp_path), 100, "volume.nas",
                     order=0, turbulence_model="none", backend="cpu", quiet=True)
    return str(tmp_path / "checkpoints" / "checkpoint_iter_000100.h5")


@pytest.mark.parametrize("override,expected", [(None, False), (True, True)])
def test_single_node_rebuild_restores_and_overrides(tmp_path, override, expected):
    ckpt = _checkpoint(tmp_path, low_mach_precond=False)
    built = {}

    def _fake_frsolver(**kwargs):
        built.update(kwargs)
        return _fake_solver(kwargs["low_mach_precond"], order=kwargs["order"])

    with patch("autoflowcfd.cli.solve.mesh_loader.load_mesh_for_solver", return_value=(MagicMock(), MagicMock())), \
         patch("autoflowcfd.core.FRSolver", side_effect=_fake_frsolver), \
         patch("autoflowcfd.cli.solve.wall_distance.compute_wall_distance_for_solver"):
        rebuild_solver_from_checkpoint(ckpt, low_mach_precond=override)
    assert built["low_mach_precond"] is expected


class _Built(Exception):
    pass


@pytest.mark.parametrize("multi_gpu,fully_distributed,target", [
    (False, False, "autoflowcfd.core.mpi.distributed_solver.DistributedFRSolver"),
    (True, False, "autoflowcfd.core.gpu.distributed.gpu_distributed.MultiGPUDistributedSolver"),
    (False, True, "autoflowcfd.core.mpi.distributed_mesh_loader.distributed_mesh_load_v2"),
    (True, True, "autoflowcfd.core.mpi.distributed_mesh_loader.distributed_mesh_load_v2"),
])
@pytest.mark.parametrize("override,expected", [(None, False), (True, True)])
def test_distributed_rebuild_paths_receive_the_switch(tmp_path, multi_gpu, fully_distributed, target,
                                                     override, expected):
    """四条分布式重建路径（CPU-MPI 传统/完全分布式、多 GPU 传统/完全分布式）都拿到同一个解析结果。"""
    from autoflowcfd.cli.solve.distributed_checkpoint_io import rebuild_distributed_solver_from_checkpoint

    ckpt = _checkpoint(tmp_path, low_mach_precond=False)
    seen = {}

    def _capture(*args, **kwargs):
        seen.update(kwargs)
        raise _Built

    with patch(target, side_effect=_capture), \
         patch("autoflowcfd.cli.solve.mesh_loader.load_mesh_for_solver", return_value=(MagicMock(), MagicMock())), \
         patch("autoflowcfd.fr.operators.generate_fr_operators", return_value=MagicMock()), \
         patch("autoflowcfd.cli.solve.distributed_checkpoint_io.wall_distance_source_if_needed", return_value=None), \
         pytest.raises(_Built):
        rebuild_distributed_solver_from_checkpoint(
            ckpt, n_ranks=2, multi_gpu=multi_gpu, fully_distributed=fully_distributed, low_mach_precond=override)
    assert seen["low_mach_precond"] is expected


# ---------------------------------------------------------------- CLI 接线
def test_steady_option_reaches_every_backend_branch(tmp_path):
    mesh = tmp_path / "m.pkl"
    mesh.write_bytes(b"")
    base = ["solve", "steady", str(mesh), "-o", str(tmp_path / "out")]
    for extra, fn in ((["--backend", "cpu"], "_run_single_node"),
                      (["--backend", "cpu", "--n-ranks", "2"], "_run_cpu_mpi"),
                      (["--backend", "gpu", "--multi-gpu", "--n-ranks", "2"], "_run_multi_gpu")):
        for flag, expected in (([], True), (["--no-low-mach-precond"], False)):
            with patch(f"autoflowcfd.cli.solve.steady.command.{fn}") as run:
                res = CliRunner().invoke(cli, base + extra + flag)
            assert res.exit_code == 0, res.output
            assert run.call_args.kwargs["low_mach_precond"] is expected, (fn, flag)


def _fake_resume_solver():
    s = MagicMock()
    s.solve.return_value = SimpleNamespace(converged=False, iterations=1, final_residual=1.0)
    s.order = s.current_order = 1
    return s


_META = {"input_file": "dummy.nas", "order": 1, "turbulence_model": "sst", "backend": "cpu", "surface_mesh": None}


@pytest.mark.parametrize("flag,expected", [([], None), (["--no-low-mach-precond"], False),
                                           (["--low-mach-precond"], True)])
def test_resume_option_is_forwarded(tmp_path, flag, expected):
    ck = tmp_path / "checkpoint_iter_000150.h5"
    ck.write_bytes(b"")
    with patch("autoflowcfd.cli.solve.commands.rebuild_solver_from_checkpoint") as reb, \
         patch("autoflowcfd.cli.solve.checkpoint_io.write.save_results"), \
         patch("autoflowcfd.cli.solve.checkpoint_io.write.write_checkpoint"), \
         patch("autoflowcfd.cli.solve.commands._report_aerodynamic_coefficients"):
        reb.return_value = (_fake_resume_solver(), 150, dict(_META))
        res = CliRunner().invoke(cli, ["solve", "resume", str(ck), "-n", "1"] + flag)
    assert res.exit_code == 0, res.output
    assert reb.call_args.kwargs["low_mach_precond"] is expected

    with patch("autoflowcfd.cli.solve.distributed_checkpoint_io.rebuild_distributed_solver_from_checkpoint") as reb, \
         patch("autoflowcfd.core.mpi.distributed_checkpoint.distributed_save_checkpoint"), \
         patch("autoflowcfd.core.mpi.distributed_checkpoint.distributed_save_results"), \
         patch("autoflowcfd.cli.solve.commands.report_distributed_aerodynamic_coefficients"):
        reb.return_value = (_fake_resume_solver(), 150, dict(_META))
        res = CliRunner().invoke(cli, ["solve", "resume", str(ck), "-n", "1", "--n-ranks", "2"] + flag)
    assert res.exit_code == 0, res.output
    assert reb.call_args.kwargs["low_mach_precond"] is expected
