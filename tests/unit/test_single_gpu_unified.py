# -*- coding: utf-8 -*-
"""单 GPU 统一到 `GPUFRSolver`（2026-10-04）。

此前单 GPU 有两套实现：`solve steady --backend gpu` 走 `GPUFRSolver`（自带一份只打印残差、不调 checkpoint
回调、返回字典的循环；CLI 不写中间 checkpoint、最终只存一份不含湍流场的 pickle），`solve transient
--backend gpu`、`solve resume`、Python API 走 `FRSolver(backend='gpu')`——后者的 GPU 分支用错的相对导入
（`..gpu` 解析成不存在的 `core.fr_solver.gpu`），装了 CuPy 的机器上第一次残差求值即 ImportError，CuPy
不可用时则静默退回 CPU。统一后：

* `GPUFRSolver` 与 `FRSolver` 共用 `SolveLoopMixin`（SolverResult、checkpoint 回调、残差历史与累计伪时间
  在单步里记录）；
* checkpoint 读写、`--init-from`、结果保存与气动力系数经主机视图（`host_view()` / `edit_host_state()`）；
* 全部单机入口经 `cli/solve/solver_factory.py::build_single_node_solver` 按后端构造。

GPU 部分用 numpy 替身完整构造真实 `GPUFRSolver`（替身与夹具见 `test_gpu_solver_order_continuation.py`）。
"""

import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from tests.unit._patch_pkg import patch_pkg_attr
from tests.unit.test_gpu_solver_order_continuation import _patch_gpu_modules  # noqa: F401（自动夹具）
from tests.validation._channel_mesh import build_channel_mesh_prism, channel_wall_source

_BC = {"x_min": "VELOCITY_INLET", "x_max": "PRESSURE_OUTLET", "y_min": "WALL", "y_max": "WALL",
       "z_min": "SYMMETRY", "z_max": "SYMMETRY"}


def _gpu_solver(turb="NONE", **kw):
    from autoflowcfd.core.gpu.solver.gpu_solver import GPUFRSolver

    mesh = build_channel_mesh_prism(1, 3, 3, 2, 0.4, 0.1, 0.08)
    mesh.boundary_bc_types = dict(_BC)
    solver = GPUFRSolver(mesh=mesh, order=1, turb_model_name=turb, vel_inf=10.0,
                         wall_distance_source=channel_wall_source(0.4, 0.1, 0.08), **kw)
    solver.order_continuation_enabled = False
    return solver


# ---------------------------------------------------------------------------
# 1. 共用求解循环
# ---------------------------------------------------------------------------

def test_gpu_solve_uses_the_shared_loop():
    from autoflowcfd.core.fr_solver.state import SolverResult

    solver = _gpu_solver()
    seen = []
    result = solver.solve(max_iter=4, dt=1e-3, tol=0.0, checkpoint_callback=lambda s, it: seen.append(it))
    assert isinstance(result, SolverResult) and result.iterations == 4
    assert seen == [1, 2, 3, 4]
    assert len(solver.residual_history) == 4              # 单步记录一次，不重复
    # 累计伪时间（与 CPU 同一个函数）：4 步之和严格大于最后一步
    assert solver.tau_accum.shape == solver.dt_cell_last.shape == (solver.mesh.n_cells,)
    assert np.all(solver.tau_accum > solver.dt_cell_last) and np.all(solver.dt_cell_last > 0)


def test_gpu_constructor_takes_the_cpu_parameter_names():
    import inspect

    from autoflowcfd.core.fr_solver.solver import FRSolver
    from autoflowcfd.core.gpu.solver.gpu_solver import GPUFRSolver

    gpu = inspect.signature(GPUFRSolver.__init__).parameters
    cpu = inspect.signature(FRSolver.__init__).parameters
    for name in ("dual_time_inner_iter", "n_threads", "sem_num_eddies", "cfl_start", "cfl_max", "cfl_min",
                 "turbulence_intensity", "viscosity_ratio", "mu_molecular", "aoa_deg", "aos_deg"):
        assert name in gpu and name in cpu, name
        assert gpu[name].default == cpu[name].default, name
    # 工厂总是显式传入的两个（默认值各自沿用：CPU 默认 SST、GPU 默认层流；时间格式枚举/字符串）
    assert "turb_model_name" in gpu and "time_scheme" in gpu
    assert gpu["ops"].default is None
    assert "backend" not in cpu


def test_gpu_dual_time_inner_iterations_reach_the_integrator():
    solver = _gpu_solver(time_scheme="dual_time", dual_time_inner_iter=7)
    assert solver.time_integrator.dual_time_steps == 7


# ---------------------------------------------------------------------------
# 2. 主机视图：checkpoint 往返
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("turb", ["SST", "SA"])
def test_checkpoint_round_trip_through_the_host_view(tmp_path, turb):
    from types import SimpleNamespace

    from autoflowcfd.cli.solve.checkpoint_io import restore_solver_state_from_fields, write_single_node_outputs
    from autoflowcfd.core.utils.checkpoint import CheckpointManager

    a = _gpu_solver(turb)
    a.solve(max_iter=3, dt=1e-3, tol=0.0)
    path = write_single_node_outputs(a, str(tmp_path), 3, "dummy.nas", turb.lower(), "gpu", quiet=True)
    _, _, _, meta = CheckpointManager(config=SimpleNamespace(), output_dir=".").load(path)
    fields = meta["fields"]
    names = a.turb_model_gpu.TRANSPORTED_FIELDS
    assert all(n in fields for n in names) and "nu_t" in fields and "tau_accum" in fields

    b = _gpu_solver(turb)
    with b.edit_host_state() as host:
        restore_solver_state_from_fields(host, fields, meta)

    np.testing.assert_array_equal(np.asarray(b.U_gpu), np.asarray(a.U_gpu))
    for n in names:
        np.testing.assert_array_equal(np.asarray(getattr(b.turb_model_gpu, n)), np.asarray(getattr(a.turb_model_gpu, n)))
    np.testing.assert_array_equal(np.asarray(b.turb_model_gpu.nu_t), np.asarray(a.turb_model_gpu.nu_t))
    np.testing.assert_array_equal(b.tau_accum, a.tau_accum)
    # 湍流场已恢复：产生项渐变直接到终点（与 CPU 同一处理）
    assert b.turb_model_gpu.production_factor == 1.0
    assert b._turb_ramp_step == b._turb_production_ramp_steps and b._resumed_from_checkpoint


def test_host_view_eddy_viscosity_matches_the_device_fields():
    solver = _gpu_solver("SST")
    solver.solve(max_iter=2, dt=1e-3, tol=0.0)
    host = solver.host_view()
    expected = np.asarray(solver.Q_gpu)[:, :, 0] * np.asarray(solver.turb_model_gpu.nu_t)
    np.testing.assert_array_equal(host._get_turbulent_viscosity_field(), expected)
    assert host.mesh is solver.mesh and host.freestream is solver.freestream


def test_host_view_pulls_only_what_is_read():
    """求解循环每步算 Cd/Cl 只读 `state`：湍流场不拷回；只读过 state 的视图 push 不碰湍流模型。"""
    solver = _gpu_solver("SST")
    host = solver.host_view()
    _ = host.state.Q
    assert "state" in host.__dict__ and "turb_model" not in host.__dict__ and "sgs_model" not in host.__dict__
    k_before = solver.turb_model_gpu.k_field
    host.push()
    assert solver.turb_model_gpu.k_field is k_before


# ---------------------------------------------------------------------------
# 3. 按后端分派
# ---------------------------------------------------------------------------

def _mesh_and_volume():
    mesh = build_channel_mesh_prism(1, 2, 2, 2, 0.4, 0.1, 0.08)
    mesh.boundary_bc_types = dict(_BC)
    return mesh, types.SimpleNamespace(boundaries={})


def test_factory_builds_a_gpu_solver_for_backend_gpu():
    from autoflowcfd.cli.solve.solver_factory import build_single_node_solver
    from autoflowcfd.core.gpu.solver.gpu_solver import GPUFRSolver

    mesh, volume = _mesh_and_volume()
    solver = build_single_node_solver("gpu", mesh, volume, order=1, turb_model_name="none", vel_inf=10.0)
    assert isinstance(solver, GPUFRSolver) and solver.turb_model_name == "NONE"


def test_factory_reports_missing_cupy_instead_of_falling_back(monkeypatch):
    import click

    import autoflowcfd.core.gpu as core_gpu
    from autoflowcfd.cli.solve.solver_factory import build_single_node_solver

    patch_pkg_attr(monkeypatch, core_gpu, "gpu_available", False)
    mesh, volume = _mesh_and_volume()
    with pytest.raises(click.ClickException, match="CuPy"):
        build_single_node_solver("gpu", mesh, volume, order=1, turb_model_name="none")


def test_factory_auto_picks_cpu_without_a_gpu(monkeypatch):
    import autoflowcfd.core.gpu as core_gpu
    from autoflowcfd.cli.solve.solver_factory import build_single_node_solver
    from autoflowcfd.core.fr_solver.solver import FRSolver

    patch_pkg_attr(monkeypatch, core_gpu, "gpu_available", False)
    mesh, volume = _mesh_and_volume()
    solver = build_single_node_solver("auto", mesh, volume, order=1, turb_model_name="none", vel_inf=10.0)
    assert type(solver) is FRSolver


def test_factory_rejects_the_cpu_only_entropy_stable_volume_on_gpu():
    import click

    from autoflowcfd.cli.solve.solver_factory import build_single_node_solver

    mesh, volume = _mesh_and_volume()
    with pytest.raises(click.BadParameter, match="CPU"):
        build_single_node_solver("gpu", mesh, volume, order=1, turb_model_name="none",
                                 entropy_stable_volume_enabled=True)


@pytest.mark.parametrize("command", ["steady", "transient"])
@pytest.mark.parametrize("args, hint", [
    (["--backend", "gpu", "--n-ranks", "2"], "--multi-gpu"),
    (["--backend", "gpu", "--multi-gpu"], "--n-ranks > 1"),
    (["--multi-gpu", "--n-ranks", "2"], "--backend gpu"),
])
def test_cli_rejects_inconsistent_gpu_options(tmp_path, command, args, hint):
    from autoflowcfd.cli.main import cli

    mesh_file = tmp_path / "mesh.pkl"
    mesh_file.write_bytes(b"")
    result = CliRunner().invoke(cli, ["solve", command, str(mesh_file), "--max-iter", "1", *args])
    assert result.exit_code != 0
    assert hint in result.output


# ---------------------------------------------------------------------------
# 4. solve transient 单机分支的中间 checkpoint 与参考面积
# ---------------------------------------------------------------------------

def test_transient_single_node_writes_periodic_checkpoints(tmp_path):
    from autoflowcfd.cli.main import cli

    mesh_file = tmp_path / "mesh.pkl"
    mesh_file.write_bytes(b"")
    solver = MagicMock()
    solver.current_order, solver.order = 1, 1

    def _solve(**kw):
        for i in range(1, 7):
            kw["checkpoint_callback"](solver, i)
        return types.SimpleNamespace(iterations=6, final_residual=1.0)

    solver.solve.side_effect = _solve
    with patch("autoflowcfd.cli.solve.transient.load_mesh_for_solver", return_value=(MagicMock(), MagicMock())), \
            patch("autoflowcfd.cli.solve.transient.build_single_node_solver", return_value=solver) as build, \
            patch("autoflowcfd.cli.solve.transient.resolve_reference_area", return_value=None) as ref, \
            patch("autoflowcfd.cli.solve.transient.write_single_node_outputs") as final_write, \
            patch("autoflowcfd.cli.solve.checkpoint_io.write.write_single_node_outputs") as periodic_write:
        result = CliRunner().invoke(cli, ["solve", "transient", str(mesh_file), "--max-iter", "6",
                                          "--checkpoint-interval", "3", "--backend", "gpu",
                                          "--output", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    assert build.call_args.args[0] == "gpu"
    ref.assert_called_once()
    assert [c.args[2] for c in periodic_write.call_args_list] == [3, 6]
    assert all(c.kwargs["transient"] for c in periodic_write.call_args_list)
    assert final_write.call_args.kwargs["transient"] is True
