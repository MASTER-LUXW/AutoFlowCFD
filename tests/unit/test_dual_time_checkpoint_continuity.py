"""dual-time 瞬态续算的时间连续性（2026-10-09，`core/utils/checkpoint_time.py`）。

此前 checkpoint 既不记物理时间步长（`solve resume` 一律 dt=1e-3），也不存上一时间层（BDF2 续算第一步退化成
BDF1）。现在两者都随 checkpoint 持久化：跑 2 步 + 写出 + 恢复 + 再跑 1 步，与连续跑 3 步逐位相同。
"""

from types import SimpleNamespace

import numpy as np
import pytest

from autoflowcfd.cli.solve.checkpoint_io.restore import restore_solver_state_from_fields
from autoflowcfd.cli.solve.checkpoint_io.write import write_checkpoint
from autoflowcfd.core.time_integration import TimeIntegrationScheme
from autoflowcfd.core.time_integration.base import STEADY_DT
from autoflowcfd.core.utils.checkpoint import CheckpointManager
from autoflowcfd.core.utils.checkpoint_time import PREVIOUS_LEVEL_FIELD, resume_time_step
from tests.validation._channel_mesh import build_channel_mesh_prism

DT = 2e-4
_BC = {"x_min": "VELOCITY_INLET", "x_max": "PRESSURE_OUTLET", "wall_bottom": "WALL", "wall_top": "WALL",
       "z_min": "SYMMETRY", "z_max": "SYMMETRY"}


def _solver():
    from autoflowcfd.core.fr_solver import FRSolver

    mesh = build_channel_mesh_prism(1, 3, 3, 2, 0.4, 0.1, 0.08)
    mesh.boundary_bc_types = dict(_BC)
    s = FRSolver(mesh=mesh, order=1, turb_model_name="NONE", time_scheme=TimeIntegrationScheme.DUAL_TIME,
                 vel_inf=10.0, dual_time_inner_iter=3)
    s.order_continuation_enabled = False
    rng = np.random.default_rng(7)          # 非均匀初场：均匀来流几步之内不变，比不出差别
    s.state.U[..., 1:4] *= 1.0 + 0.05 * rng.standard_normal(s.state.U[..., 1:4].shape)
    s.state._update_primitives()
    return s


def _checkpoint(solver, tmp_path, iteration):
    path = write_checkpoint(solver, str(tmp_path), iteration, "mesh.nas", order=1, turbulence_model="none",
                            backend="cpu", quiet=True)
    _sol, _hist, _it, meta = CheckpointManager(SimpleNamespace(), output_dir=str(tmp_path)).load(path)
    return meta


def test_resumed_dual_time_run_continues_bitwise(tmp_path):
    reference = _solver()
    reference.solve(max_iter=3, dt=DT, tol=0.0)

    first = _solver()
    first.solve(max_iter=2, dt=DT, tol=0.0)
    meta = _checkpoint(first, tmp_path, 2)
    assert meta["dt"] == pytest.approx(DT)
    assert PREVIOUS_LEVEL_FIELD in meta["fields"]

    resumed = _solver()
    restore_solver_state_from_fields(resumed, meta["fields"], meta)
    resumed.solve(max_iter=1, dt=resume_time_step(None, meta, resumed), tol=0.0)
    np.testing.assert_array_equal(resumed.state.U, reference.state.U)
    # 定阶循环的残差下降基准也随 checkpoint 续接（此前不写，续算被报成"没有阶段起始残差记录"）
    assert meta["phase_initial_residual"] == first._phase_initial_residual
    assert resumed._phase_initial_residual == reference._phase_initial_residual

    # 对照：不恢复上一时间层（续算第一步退化成 BDF1）就接不上
    bdf1 = _solver()
    fields = {k: v for k, v in meta["fields"].items() if k != PREVIOUS_LEVEL_FIELD}
    restore_solver_state_from_fields(bdf1, fields, meta)
    bdf1.solve(max_iter=1, dt=DT, tol=0.0)
    assert not np.array_equal(bdf1.state.U, reference.state.U)


def test_resume_time_step_rules():
    dual = SimpleNamespace(time_integrator=SimpleNamespace(scheme=TimeIntegrationScheme.DUAL_TIME))
    rk3 = SimpleNamespace(time_integrator=SimpleNamespace(scheme=TimeIntegrationScheme.SSP_RK3))
    assert resume_time_step(5e-5, {"dt": 1e-4}, dual) == 5e-5          # 命令行优先
    assert resume_time_step(None, {"dt": 1e-4}, dual) == 1e-4          # 沿用 checkpoint
    assert resume_time_step(None, {}, rk3) == STEADY_DT                 # 旧 checkpoint、伪时间格式
    with pytest.raises(ValueError, match="--dt"):
        resume_time_step(None, {}, dual)                                # 物理时间步长不能猜


def test_distributed_checkpoint_carries_dt_and_previous_level(tmp_path):
    from autoflowcfd.core.mpi.distributed_checkpoint import distributed_load_checkpoint, distributed_save_checkpoint
    from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
    from autoflowcfd.fr.operators import generate_fr_operators

    mesh = build_channel_mesh_prism(1, 3, 3, 2, 0.4, 0.1, 0.08)
    mesh.boundary_bc_types = dict(_BC)

    def _dist():
        return DistributedFRSolver(mesh=mesh, ops=generate_fr_operators(1), face_connectivity=mesh.face_connectivity,
                                   n_ranks=1, order=1, turb_model_name="none", vel_inf=10.0,
                                   time_scheme=TimeIntegrationScheme.DUAL_TIME)
    a = _dist()
    a.order_continuation_enabled = False
    a.solve(max_iter=2, dt=DT, tol=0.0)
    path = distributed_save_checkpoint(a, str(tmp_path), 2, "mesh.nas", 1, "none", "cpu")
    b = _dist()
    _U, meta, _it = distributed_load_checkpoint(path, b)
    assert meta["dt"] == pytest.approx(DT)
    np.testing.assert_array_equal(b._dual_time_U_prev, a._dual_time_U_prev)
