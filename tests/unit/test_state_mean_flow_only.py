# -*- coding: utf-8 -*-
"""求解器状态只含平均流 5 个守恒变量（2026-10-05 删除 SST 族的两个历史槽位 `U[..., 5:7]`）。

此前 CPU 单机 SST 族的状态是 7 列：两列 k/omega 槽位全仓库无人读取、残差恒为零，但
1. CPU SST 的残差 RMS 分母多算两列，比 GPU / 分布式（5 列）小 sqrt(7/5) 倍；
2. CPU 写出的 SST checkpoint 是 7 列，恢复到 5 列的单 GPU 时被形状校验拒绝；
3. 隐式步每次残差求值都要拷一份整状态再切片。
旧 checkpoint 的 7 列状态恢复时只取前 5 列。
"""

import numpy as np
import pytest

from tests.unit.test_gpu_solver_order_continuation import _patch_gpu_modules  # noqa: F401（自动夹具）
from tests.validation._channel_mesh import build_channel_mesh_prism, channel_wall_source

_BC = {"x_min": "VELOCITY_INLET", "x_max": "PRESSURE_OUTLET", "y_min": "WALL", "y_max": "WALL",
       "z_min": "SYMMETRY", "z_max": "SYMMETRY"}


def _mesh():
    mesh = build_channel_mesh_prism(1, 3, 3, 2, 0.4, 0.1, 0.08)
    mesh.boundary_bc_types = dict(_BC)
    return mesh


def _cpu(turb):
    from autoflowcfd.core.fr_solver.solver import FRSolver
    from autoflowcfd.core.fr_solver.turbulence import apply_wall_distance_source

    s = FRSolver(mesh=_mesh(), order=1, turb_model_name=turb, vel_inf=10.0)
    if turb != "NONE":
        apply_wall_distance_source(s, channel_wall_source(0.4, 0.1, 0.08))
    return s


@pytest.mark.parametrize("turb", ["NONE", "SA", "SST", "DDES"])
def test_every_turbulence_model_has_a_five_column_state(turb):
    from autoflowcfd.core.fr_solver.state import N_MEAN_FLOW_VARS

    s = _cpu(turb)
    assert N_MEAN_FLOW_VARS == 5
    assert s.state.U.shape[-1] == s.state.Q.shape[-1] == s.state.n_vars == 5


def test_cpu_and_gpu_report_the_same_residual_norm_for_sst():
    """同一网格、同一初场的第一步：CPU 与 GPU 的残差 RMS 口径相同（此前 CPU SST 小 sqrt(7/5) 倍）。"""
    from autoflowcfd.core.gpu.solver.gpu_solver import GPUFRSolver

    cpu = _cpu("SST")
    gpu = GPUFRSolver(mesh=_mesh(), order=1, turb_model_name="SST", vel_inf=10.0,
                      wall_distance_source=channel_wall_source(0.4, 0.1, 0.08))
    assert gpu.n_vars == cpu.state.n_vars
    # 非均匀初场（均匀来流的残差只是舍入噪声，相对比较没有意义）
    rng = np.random.default_rng(5)
    U = cpu.state.U * (1.0 + 0.01 * rng.standard_normal(cpu.state.U.shape))
    cpu.state.U = U.copy()
    cpu.state._update_primitives()
    gpu.U_gpu = U.copy()
    gpu._update_primitives_gpu()
    r_cpu, r_gpu = cpu.step(1e-3), gpu.step(1e-3)
    assert r_cpu > 1.0 and r_cpu == pytest.approx(r_gpu, rel=1e-9)


def test_old_seven_column_checkpoint_restores_the_mean_flow(tmp_path):
    from types import SimpleNamespace

    from autoflowcfd.cli.solve.checkpoint_io import restore_solver_state_from_fields

    s = _cpu("SST")
    rng = np.random.default_rng(3)
    U5 = s.state.U * (1.0 + 0.01 * rng.random(s.state.U.shape))
    old = np.concatenate([U5, rng.random(U5.shape[:2] + (2,))], axis=-1)     # 旧格式：多两列死槽位
    restore_solver_state_from_fields(s, {"U_sps": old}, {})
    np.testing.assert_array_equal(s.state.U, U5)
    assert s.state.U.flags["C_CONTIGUOUS"]


def test_checkpoint_with_wrong_mesh_is_still_rejected():
    import click

    from autoflowcfd.cli.solve.checkpoint_io import restore_solver_state_from_fields

    s = _cpu("NONE")
    bad = np.ones((s.state.n_cells + 1, s.state.n_sps, 5))
    with pytest.raises(click.ClickException, match="不匹配"):
        restore_solver_state_from_fields(s, {"U_sps": bad}, {})
    with pytest.raises(click.ClickException, match="不匹配"):
        restore_solver_state_from_fields(s, {"U_sps": np.ones(s.state.U.shape[:2] + (4,))}, {})


def test_newton_step_rejects_a_state_with_extra_columns():
    from autoflowcfd.core.time_integration.implicit.mean_flow_step import step_mean_flow_newton

    with pytest.raises(ValueError, match="5 个守恒变量"):
        step_mean_flow_newton(None, None, np.zeros((8, 7)), None, None, red=None, cell_is_prism=None,
                              cell_colors=None, order=1, filter_active=False, positivity=None)
