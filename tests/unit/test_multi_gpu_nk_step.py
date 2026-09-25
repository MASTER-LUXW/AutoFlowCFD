# -*- coding: utf-8 -*-
"""多 GPU 分布式 `step()` 的隐式稳态（NEWTON_KRYLOV）分支接线。

隐式步的算法（`implicit/mean_flow_step.py`）已由单机与 CPU 分布式的对照测试
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


class _NumpyAsCupy:
    def __getattr__(self, name):
        return getattr(np, name)

    def asnumpy(self, x):
        return np.asarray(x)


@pytest.fixture(autouse=True)
def _patch_get_cupy(monkeypatch):
    patch_module_get_cupy(monkeypatch, gd_mod, _NumpyAsCupy())


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
    from autoflowcfd.core.time_integration.implicit.mean_flow_step import reset_newton_state

    s = types.SimpleNamespace()
    s.partition = types.SimpleNamespace(n_local_cells=n_local)
    s.mesh = types.SimpleNamespace(n_sps_per_cell=n_sps)
    s.time_integrator = types.SimpleNamespace(scheme=TimeIntegrationScheme.NEWTON_KRYLOV)
    s.order = s.current_order = 1
    s.freestream = {"rho_inf": 1.225, "vel_inf": 30.0, "p_inf": 101325.0}
    s.turb_model_gpu = None
    s.turb_model_name = "NONE"
    s.filter_func_gpu = None
    s.low_mach_precond_enabled = False
    s.residual_history, s.iteration = [], 0
    s.dist_flat_face = types.SimpleNamespace(perm=np.arange(n_local),
                                             compact_cell_type=np.zeros(n_local, dtype=int))
    s._inv_perm_gpu = s._perm_gpu = np.arange(n_local)
    s._block_jacobi_colors_local = np.arange(n_local) % 2
    s._get_positivity_limiter_gpu = lambda: None
    s._compute_turbulence_source_distributed = lambda dt: None
    s._global_residual_norm = lambda r: float(np.linalg.norm(r))
    big = np.full(n_local, 1e3)          # dtau 远大于 1/5：接近纯 Newton
    s._compute_local_time_step_gpu = lambda return_physical_too=False: (big, big)
    s._cfl_controller = SERCFLController()
    s._update_cfl_controller = types.MethodType(_MultiGPUTimeStepMixin._update_cfl_controller, s)
    reset_newton_state(s)
    s.U_gpu = _conservative(n_local, n_sps, np.random.default_rng(3))

    def _compute_total_residual_gpu(mu_t_field=None, inviscid=True, viscous=True):
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
