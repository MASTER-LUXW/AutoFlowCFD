"""时间精确的显式推进（瞬态命令的 rk3 档，`core/time_integration/time_accurate.py`）。

修复前：瞬态命令默认 rk3，走的是稳态的逐单元局部时间步——忽略 `--dt`，各单元推进的物理时间不同，结果不是
时间精确解；全场同一步长只在验证算例里靠替换求解器方法实现。现在每个物理步切成全场同步长的子步
（子步长 = min(剩余时间, 全场显式稳定步长)），最后一个子步恰好落在 t + dt；固定 CFL、关低马赫预处理。
"""

import numpy as np
import pytest

from autoflowcfd.core.time_integration import TimeIntegrationScheme
from autoflowcfd.core.time_integration.time_accurate import advance, configure_time_stepping
from tests.unit.test_gpu_solver_order_continuation import _patch_gpu_modules  # noqa: F401（自动夹具）
from tests.validation._channel_mesh import build_channel_mesh_prism

LX, H, LZ = 0.4, 0.1, 0.08
_BC = {"x_min": "VELOCITY_INLET", "x_max": "PRESSURE_OUTLET", "wall_bottom": "WALL", "wall_top": "WALL",
       "z_min": "SYMMETRY", "z_max": "SYMMETRY"}


def _mesh(order):
    m = build_channel_mesh_prism(order, 3, 3, 2, LX, H, LZ)
    m.boundary_bc_types = dict(_BC)
    return m


def _perturbed(solver, seed=5):
    rng = np.random.default_rng(seed)
    U = solver.state.U.copy()
    U[..., 1:4] += 2.0 * rng.standard_normal(U[..., 1:4].shape) * U[..., :1]
    return U


def _cpu(order=1, **kw):
    from autoflowcfd.core.fr_solver import FRSolver

    s = FRSolver(mesh=_mesh(order), order=order, turb_model_name="NONE", time_scheme=TimeIntegrationScheme.SSP_RK3,
                 vel_inf=30.0, **kw)
    s.order_continuation_enabled = False
    return s


def test_substeps_are_uniform_and_land_exactly_on_t_plus_dt():
    """子步各自全场同一步长、总和恰为 dt；结果与"逐子步把局部步长换成全场同一值"的独立实现逐位相同。"""
    a = _cpu(time_accurate=True)
    b = _cpu(adaptive_cfl=False, low_mach_precond=False)      # 独立参照：同一固定 CFL 的普通显式步
    U0 = _perturbed(a)
    for s in (a, b):
        s.state.U = U0.copy()
        s.state._update_primitives()
    h0 = float(a._compute_local_time_step().min())
    dt = 2.6 * h0

    taken = []
    orig_step = a.step

    def recording_step(d):
        out = orig_step(d)
        taken.append(a._ta_h)
        return out
    a.step = recording_step
    advance(a, dt)
    assert len(taken) == a.last_substeps >= 3
    assert sum(taken) == pytest.approx(dt, rel=1e-13)
    assert all(h <= dt for h in taken)

    # 参照：每个子步用同一个 h，局部步长函数换成返回常数场
    remaining = dt
    orig_dt = b._compute_local_time_step
    for h_ref in taken:
        stable = float(orig_dt().min())
        assert min(remaining, stable) == pytest.approx(h_ref, rel=1e-14)
        b._compute_local_time_step = (lambda h: (lambda return_physical_too=False: (
            (np.full_like(orig_dt(), h),) * 2 if return_physical_too else np.full_like(orig_dt(), h))))(h_ref)
        b.step(h_ref)
        b._compute_local_time_step = orig_dt
        remaining -= h_ref
    np.testing.assert_array_equal(a.state.U, b.state.U)


def test_time_accurate_mode_fixes_cfl_and_disables_preconditioning():
    s = _cpu(time_accurate=True, cfl_start=0.02)
    assert s._cfl_controller is None and s.fixed_cfl_number == pytest.approx(0.02)
    assert not s.low_mach_precond_enabled


def test_time_accurate_needs_an_explicit_scheme():
    class Stub:
        pass
    with pytest.raises(ValueError, match="显式"):
        configure_time_stepping(Stub(), TimeIntegrationScheme.DUAL_TIME, time_accurate=True)


def test_single_gpu_matches_cpu():
    """单 GPU（numpy 替身）的时间精确推进与 CPU 子步数相同、状态一致到舍入。"""
    from autoflowcfd.core.gpu.solver.gpu_solver import GPUFRSolver

    cpu = _cpu(time_accurate=True)
    gpu = GPUFRSolver(mesh=_mesh(1), order=1, turb_model_name="NONE", time_scheme=TimeIntegrationScheme.SSP_RK3,
                      vel_inf=30.0, time_accurate=True)
    gpu.order_continuation_enabled = False
    U0 = _perturbed(cpu)
    cpu.state.U = U0.copy()
    cpu.state._update_primitives()
    gpu.U_gpu = U0.copy()
    gpu._update_primitives_gpu()
    dt = 2.6 * float(cpu._compute_local_time_step().min())
    advance(cpu, dt)
    advance(gpu, dt)
    assert gpu.last_substeps == cpu.last_substeps
    scale = np.abs(cpu.state.U).max(axis=(0, 1))
    assert (np.abs(np.asarray(gpu.U_gpu) - cpu.state.U) / scale).max() < 1e-11
