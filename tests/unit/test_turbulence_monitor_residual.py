"""湍流方程的监控残差进每步监视行与收敛历史（I-77）。

此前显式路径的湍流方程没有任何监控残差：监视行与收敛历史只有平均流 5 个方程（隐式紧耦合步的湍流子系统残差
只在 Newton 一栏里打印、不进收敛历史）。现在显式更新记下逐未知量的完整时间导数除以尺度
（`turbulence/transported.py::advance_transported_fields`），四个后端在算平均流监控残差的同一处归约成全局 RMS
（`time_integration/residual_norm.py::explicit_turbulence_residual`）；收敛历史多一列 `turbulence`。
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from autoflowcfd.core.fr_solver.convergence_history import TURBULENCE_RESIDUAL, ConvergenceHistory
from autoflowcfd.core.time_integration.residual_norm import turbulence_monitor_residual
from tests.unit._module_source import module_source
from tests.unit.test_gpu_solver_order_continuation import _patch_gpu_modules  # noqa: F401（自动夹具）
from tests.validation._channel_mesh import build_channel_mesh_prism, channel_wall_source
from autoflowcfd.core.freestream_defaults import MU_MOLECULAR, P_INF, RHO_INF

_BC = {"x_min": "VELOCITY_INLET", "x_max": "PRESSURE_OUTLET", "wall_bottom": "WALL", "wall_top": "WALL",
       "z_min": "SYMMETRY", "z_max": "SYMMETRY"}
_N_STEPS = 3


def _mesh():
    mesh = build_channel_mesh_prism(1, 3, 3, 2, 0.4, 0.1, 0.08)
    mesh.boundary_bc_types = dict(_BC)
    return mesh


def _run(solver):
    solver.order_continuation_enabled = False
    solver.solve(max_iter=_N_STEPS, dt=1e-5, tol=0.0)
    return np.array(solver.convergence_history.turbulence)


def _cpu(turb):
    from autoflowcfd.core.fr_solver import FRSolver
    from autoflowcfd.core.time_integration import TimeIntegrationScheme

    from autoflowcfd.core.fr_solver.turbulence.wall_distance import apply_wall_distance_source

    s = FRSolver(mesh=_mesh(), order=1, turb_model_name=turb, time_scheme=TimeIntegrationScheme.SSP_RK3, vel_inf=10.0, rho_inf=RHO_INF, p_inf=P_INF, mu_molecular=MU_MOLECULAR)
    if turb != "NONE":
        apply_wall_distance_source(s, channel_wall_source(0.4, 0.1, 0.08))
    return s


@pytest.mark.parametrize("turb", ["SST", "SA"])
def test_explicit_turbulence_residual_is_recorded_every_step(turb):
    hist = _run(_cpu(turb))
    assert hist.shape == (_N_STEPS,) and np.all(np.isfinite(hist)) and np.all(hist > 0.0)


def test_laminar_has_no_turbulence_column():
    s = _cpu("NONE")
    hist = _run(s)
    assert np.all(np.isnan(hist))
    assert TURBULENCE_RESIDUAL not in s.convergence_history.to_checkpoint()["residuals"]


@pytest.mark.parametrize("turb", ["SST", "SA"])
def test_single_gpu_matches_cpu(turb):
    from autoflowcfd.core.gpu.solver.gpu_solver import GPUFRSolver
    from autoflowcfd.core.time_integration import TimeIntegrationScheme

    gpu = GPUFRSolver(mesh=_mesh(), order=1, turb_model_name=turb, vel_inf=10.0,
                      time_scheme=TimeIntegrationScheme.SSP_RK3,
                      wall_distance_source=channel_wall_source(0.4, 0.1, 0.08), rho_inf=RHO_INF, p_inf=P_INF, mu_molecular=MU_MOLECULAR)
    np.testing.assert_allclose(_run(gpu), _run(_cpu(turb)), rtol=1e-8)


@pytest.mark.parametrize("turb", ["SST", "SA"])
def test_cpu_distributed_matches_cpu(turb):
    from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
    from autoflowcfd.core.time_integration import TimeIntegrationScheme

    mesh = _mesh()
    dist = DistributedFRSolver(mesh=mesh, ops=mesh.operators, face_connectivity=mesh.face_connectivity, n_ranks=1,
                               order=1, turb_model_name=turb, time_scheme=TimeIntegrationScheme.SSP_RK3,
                               vel_inf=10.0, wall_distance_source=channel_wall_source(0.4, 0.1, 0.08), rho_inf=RHO_INF, p_inf=P_INF, mu_molecular=MU_MOLECULAR)
    np.testing.assert_allclose(_run(dist), _run(_cpu(turb)), rtol=1e-8)


def test_multi_gpu_reduces_the_written_back_rates():
    """多 GPU：视图写回模型时一并写回本分区的速率，监控残差在同一处归约（本机没有多 GPU 端到端替身）。"""
    write_back = module_source("autoflowcfd.core.gpu.distributed.gpu_distributed_init.turb_source")
    assert "advance_transported_fields(ctx.view" in write_back and "model.last_update_rates" in write_back
    monitor = module_source("autoflowcfd.core.gpu.distributed.gpu_distributed.residual")
    assert "explicit_turbulence_residual(red, self.turb_model_gpu" in monitor


def test_implicit_subsystem_residual_takes_precedence():
    s = SimpleNamespace(_newton_last_info={"res_norm_turbulence": 2.0}, _residual_turbulence=None)
    assert turbulence_monitor_residual(s) == 2.0
    assert turbulence_monitor_residual(SimpleNamespace(_newton_last_info=None, _residual_turbulence=5.0)) == 5.0


def test_history_column_round_trips_through_checkpoint():
    h = ConvergenceHistory()
    h.append(1.0, turbulence=0.5)
    h.append(0.5, turbulence=0.25)
    out = ConvergenceHistory.from_checkpoint(h.to_checkpoint(), iteration=2)
    assert out.turbulence == [0.5, 0.25]
    # 旧 checkpoint 没有这一列：恢复成 nan
    old = h.to_checkpoint()
    del old["residuals"][TURBULENCE_RESIDUAL]
    assert all(math.isnan(v) for v in ConvergenceHistory.from_checkpoint(old, iteration=2).turbulence)
