# -*- coding: utf-8 -*-
"""多 GPU 分布式 `step()` 的隐式分支接线：隐式稳态（NEWTON_KRYLOV）与双时间步（DUAL_TIME）。

隐式步的算法（`implicit/newton/mean_flow_step.py`）已由单机与 CPU 分布式的对照测试
覆盖；这里只验证多 GPU 分支自己的接线：Newton 步真的被调用、试探态经
`self.U_gpu` 临时写入再恢复、跨步状态落在求解器上、SER 律拿到 Newton 诊断。

判据用一个线性残差 `dU/dt = -5 (U - target)`：块 Jacobi 对这个逐单元解耦的
算子是精确的，一个 PTC-Newton 步的结果可以解析写出（与残差是不是真实 FR
通量无关）。
"""

import types
import numpy as np
import pytest

import autoflowcfd.core.gpu.distributed.gpu_distributed as gd_mod

from tests.unit._gpu_cupy_shim import patch_module_get_cupy
from tests.unit._numpy_as_cupy import NumpyAsCupy


@pytest.fixture(autouse=True)
def _patch_get_cupy(monkeypatch):
    patch_module_get_cupy(monkeypatch, gd_mod, NumpyAsCupy())
    # 残差是合成的线性算子、不是 FR 离散：解析单元块（对 FR 残差求导）不适用，
    # 块 Jacobi 用着色差分装配（对任意残差精确）
    import autoflowcfd.core.mpi.distributed_implicit as di
    monkeypatch.setattr(di, "distributed_mean_flow_assembler", lambda *a, **k: None)


def _conservative(n, n_sps, rng):
    rho = rng.uniform(1.15, 1.3, size=(n, n_sps))
    vel = [rng.uniform(20.0, 40.0, size=(n, n_sps)), rng.uniform(-5, 5, size=(n, n_sps)),
           rng.uniform(-5, 5, size=(n, n_sps))]
    p = rng.uniform(9.8e4, 1.03e5, size=(n, n_sps))
    U = np.zeros((n, n_sps, 5))
    U[..., 0] = rho
    for i in range(3):
        U[..., 1 + i] = rho * vel[i]
    U[..., 4] = p / 0.4 + 0.5 * rho * sum(v ** 2 for v in vel)
    return U


def _stub(n_local, n_sps, target):
    from autoflowcfd.core.gpu.distributed.gpu_distributed.timestep import _MultiGPUTimeStepMixin
    from autoflowcfd.core.time_integration.adaptive_cfl.ser import SERCFLController
    from autoflowcfd.core.time_integration.base import TimeIntegrationScheme
    from autoflowcfd.core.time_integration.implicit.newton.mean_flow_step import reset_newton_state

    s = types.SimpleNamespace()
    s.partition = types.SimpleNamespace(n_local_cells=n_local, rank=0, n_ranks=1)
    s.mesh = types.SimpleNamespace(n_sps_per_cell=n_sps)
    s.time_integrator = types.SimpleNamespace(scheme=TimeIntegrationScheme.NEWTON_KRYLOV)
    s.order = s.current_order = 1
    s.freestream = {"rho_inf": 1.225, "vel_inf": 30.0, "p_inf": 101325.0}
    s.boundary_ghost_provider = types.SimpleNamespace(code_to_config={})   # 无 SEM 入口
    s.turb_model_gpu = None
    s.turb_model_name = "NONE"
    s.low_mach_precond_enabled = False
    s.iteration = 0
    s.dist_flat_face = types.SimpleNamespace(perm=np.arange(n_local),
                                             compact_cell_type=np.zeros(n_local, dtype=int))
    s._inv_perm_gpu = s._perm_gpu = np.arange(n_local)
    s.gpu_halo = types.SimpleNamespace(exchange=None, n_vars=5)   # 单 rank、无 halo（解析装配器已被 fixture 关掉）
    s._block_jacobi_colors_local = np.arange(n_local) % 2
    # 合成状态没有面/细点几何：点集只有解点（额外求值点为空），物理性限幅仍按解点生效
    from autoflowcfd.core.time_integration.positivity import PositivityLimiter
    _lim = PositivityLimiter(np.ones((n_local, n_sps)), np.zeros((0, n_sps)), np.zeros((0, n_sps)),
                             np.ones(n_local, dtype=bool), n_sps, n_sps)
    s._get_positivity_limiter_gpu = lambda: _lim
    s._compute_turbulence_source_distributed = lambda dt: None
    s._monitor_residuals = lambda r: (float(np.linalg.norm(r)), np.zeros(5))
    big = np.full(n_local, 1e3)          # dtau 远大于 1/5：接近纯 Newton
    s._compute_local_time_step_gpu = lambda return_physical_too=False, nu_av_compact=None: (
        (big, big) if return_physical_too else big)
    s.compute_artificial_diffusivity_compact_gpu = lambda: None   # 人工粘性未启用
    s.artificial_viscosity_enabled = False
    s._cfl_controller = SERCFLController()
    s._update_cfl_controller = types.MethodType(_MultiGPUTimeStepMixin._update_cfl_controller, s)
    reset_newton_state(s)
    s.U_gpu = _conservative(n_local, n_sps, np.random.default_rng(3))

    def _compute_total_residual_gpu(mu_t_field=None, inviscid=True, viscous=True, nu_av_compact=None):
        return -5.0 * (s.U_gpu - target)          # dU/dt（生产约定）
    s._compute_total_residual_gpu = _compute_total_residual_gpu
    return s


def test_newton_branch_lands_on_the_fixed_point_and_feeds_ser():
    n_local, n_sps = 4, 6
    target = _conservative(n_local, n_sps, np.random.default_rng(7))
    s = _stub(n_local, n_sps, target)
    U_before = s.U_gpu.copy()
    cfl0 = s._cfl_controller.cfl_number

    res = gd_mod.MultiGPUDistributedSolver.step(s, dt=0.0)

    info = s._newton_last_info
    assert info is not None and info["theta"] == 1.0 and info["gmres_info"] == 0
    assert s._newton_block_precond is not None, "块 Jacobi 缓存应落在求解器上"
    # 块 Jacobi 对逐单元解耦的线性算子是精确的：一个 PTC-Newton 步正好是
    # (I/dtau + 5I) dU = -5 (U - target) 的解
    dtau = 1e3
    expected = U_before + (target - U_before) * 5.0 / (5.0 + 1.0 / dtau)
    # 线性求解停在 forcing 容差上、块 Jacobi 按 float32 存储：误差相对**步长**
    # 在 1e-5 以内（相对状态本身是 1e-6 量级）
    step = np.abs(expected - U_before).max(axis=(0, 1))
    err = np.abs(s.U_gpu - expected).max(axis=(0, 1))
    assert np.all(err <= 1e-5 * step), (err / step)
    # 报告的残差是步前的物理残差
    assert res == pytest.approx(np.linalg.norm(5.0 * (U_before - target)))
    # SER 拿到的是 Newton 诊断：首步只记录基准，第二步残差大降、CFL 放大
    gd_mod.MultiGPUDistributedSolver.step(s, dt=0.0)
    assert s._cfl_controller.cfl_number > cfl0


def test_dual_time_branch_solves_bdf1_then_bdf2():
    """DUAL_TIME：每个物理步解 BDF 方程。线性残差 `dU/dt = -5 (U - target)` 下可手算：
    BDF1 `U1 = (U0 + 5 dt T) / (1 + 5 dt)`，BDF2 `U2 = (2 U1 - U0/2 + 5 dt T) / (3/2 + 5 dt)`。
    上一时间层落在求解器上，伪时间 CFL 不被 step() 末尾的外层残差更新。"""
    from autoflowcfd.core.time_integration.base import TimeIntegrationScheme
    from autoflowcfd.core.turbulence.dual_time import reset_dual_time_history

    n_local, n_sps, dt = 4, 6, 0.05
    target = _conservative(n_local, n_sps, np.random.default_rng(7))
    s = _stub(n_local, n_sps, target)
    s.time_integrator = types.SimpleNamespace(scheme=TimeIntegrationScheme.DUAL_TIME, dual_time_steps=30,
                                              dual_time_tolerance=1e-9)
    reset_dual_time_history(s)
    outer_updates = []
    s._update_cfl_controller = lambda *a, **k: outer_updates.append(a)
    U0 = s.U_gpu.copy()

    gd_mod.MultiGPUDistributedSolver.step(s, dt=dt)
    U1 = s.U_gpu.copy()
    expected1 = (U0 + 5.0 * dt * target) / (1.0 + 5.0 * dt)
    step = np.abs(expected1 - U0).max(axis=(0, 1))
    assert np.all(np.abs(U1 - expected1).max(axis=(0, 1)) <= 1e-6 * step)
    assert s._dual_time_inner["converged"] and s._dual_time_inner["a0"] == 1.0
    np.testing.assert_array_equal(s._dual_time_U_prev.reshape(U0.shape), U0)

    gd_mod.MultiGPUDistributedSolver.step(s, dt=dt)
    expected2 = (2.0 * U1 - 0.5 * U0 + 5.0 * dt * target) / (1.5 + 5.0 * dt)
    step = np.abs(expected2 - U1).max(axis=(0, 1))
    assert np.all(np.abs(s.U_gpu - expected2).max(axis=(0, 1)) <= 1e-6 * step)
    assert s._dual_time_inner["a0"] == 1.5
    np.testing.assert_array_equal(s._dual_time_U_prev.reshape(U0.shape), U1)
    assert not outer_updates and s.iteration == 2
