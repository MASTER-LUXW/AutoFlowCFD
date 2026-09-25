"""AutoFlowCFD V2.0 - 多GPU分布式 DUAL_TIME 支持验证（2026-09-02）。

真实 bug 修复（与 CPU 分布式 `DistributedFRSolver.step` 同一处修复、
同一个理由——用户明确要求"不允许出现完成度不是100%的功能点"后排查
发现）：`MultiGPUDistributedSolver.step()` 此前无论 `self.time_
integrator.scheme` 是什么都无条件走手动展开的 RK stage 逻辑，
`scheme` 只被用来查系数表——DUAL_TIME 请求了也完全无路可走。

本文件不重新验证 `GPUTurbulenceSST`/`compute_viscous_residual_gpu`
这类既有 GPU 数值 kernel（已在 `test_gpu_distributed_turbulence.py`
决定性验证过），也不重新验证积分器 `step_dual_time`（CPU/GPU 同一份）本身的 BDF 构造/
收敛迭代（单机 GPU 路径早已使用、是既有数值逻辑，不是本次改动的
对象）——只验证本次新增分支的真实风险点：`_spatial_residual` 闭包
是否正确地把 trial U 临时写入 `self.U_gpu`（再正确恢复）、
`_dual_time_U_prev` 是否正确持久化。用一个简单的、可手算验证的解析
残差函数（而不是真实的 FR 通量 kernel）隔离出这个风险点本身。
"""

import types

import numpy as np
import pytest

import autoflowcfd.core.gpu.distributed.gpu_distributed as gd_mod
import autoflowcfd.core.gpu.gpu_time_integration as gti_mod
from tests.unit._gpu_cupy_shim import patch_module_get_cupy


class _NumpyAsCupy:
    def __getattr__(self, name):
        return getattr(np, name)

    def asnumpy(self, x):
        return np.asarray(x)


@pytest.fixture(autouse=True)
def _patch_get_cupy(monkeypatch):
    shim = _NumpyAsCupy()
    patch_module_get_cupy(monkeypatch, gd_mod, shim)
    patch_module_get_cupy(monkeypatch, gti_mod, shim)


def _make_target(n_local, n_sps, rng):
    """构造一个真实物理量级的守恒变量场（rho~1.2/动量~30-150/能量~2.5e5）
    ——积分器每个 stage 都检查可容许性（密度、压力 > 0），测试数据须是
    物理上可容许的守恒量。"""
    rho = rng.uniform(1.15, 1.3, size=(n_local, n_sps))
    u = rng.uniform(20.0, 40.0, size=(n_local, n_sps))
    v = rng.uniform(-5.0, 5.0, size=(n_local, n_sps))
    w = rng.uniform(-5.0, 5.0, size=(n_local, n_sps))
    p = rng.uniform(9.5e4, 1.05e5, size=(n_local, n_sps))
    gamma = 1.4
    target = np.zeros((n_local, n_sps, 5))
    target[..., 0] = rho
    target[..., 1] = rho * u
    target[..., 2] = rho * v
    target[..., 3] = rho * w
    target[..., 4] = p / (gamma - 1.0) + 0.5 * rho * (u ** 2 + v ** 2 + w ** 2)
    return target


def _make_stub(n_local, n_sps, n_vars, target):
    """构造一个只暴露 `MultiGPUDistributedSolver.step()` DUAL_TIME
    分支真正读取的属性/方法的最小 stub——`_compute_total_residual_gpu`
    换成一个简单的线性残差 `R(U) = 5*(U - target)`（唯一不动点在
    U=target），不是真实的 FR 通量 kernel，但足以决定性验证"trial U
    是否正确临时写入 self.U_gpu 再恢复""dual_time_U_prev 是否正确
    持久化"这两个真实风险点：只要 DUAL_TIME 收敛，`self.U_gpu` 必须
    收敛到 `target`（与残差函数是不是真实 FR kernel 无关，这是纯粹的
    不动点性质）。"""
    from autoflowcfd.core.gpu.gpu_time_integration import GPUTimeIntegrator

    stub = types.SimpleNamespace()
    stub.partition = types.SimpleNamespace(n_local_cells=n_local)
    stub.mesh = types.SimpleNamespace(n_sps_per_cell=n_sps)
    stub.time_integrator = GPUTimeIntegrator(scheme="dual_time")
    # GPUTimeIntegrator 本身没有 dual_time_steps 属性（step() 的
    # DUAL_TIME 分支用 getattr(...,5) 兜底，与单机 GPU gpu_solver.py
    # 同一个既有约定），默认 5 次内迭代对这里刻意选的、需要真正收敛
    # 才能验证的线性残差问题远远不够，手动设置一个足够大的值。
    stub.time_integrator.dual_time_steps = 500
    stub._dual_time_U_prev = None
    # 正性限制器需要真实网格几何；本 stub 刻意不构造网格，给 None ——积分器
    # 退回"只检查、不修改"（`base._finish_stage`），对这里物理可容许的数据
    # 不改变任何值。
    stub._get_positivity_limiter_gpu = lambda: None
    # 初始猜测同样必须是真实物理量级的守恒变量（与 target 用不同随机种子，
    # 确保确实需要迭代收敛，不是从 target 本身出发的平凡情形）。
    stub.U_gpu = _make_target(n_local, n_sps, np.random.default_rng(999))
    stub.filter_func_gpu = None
    stub.residual_history = []
    stub.iteration = 0
    # 逐单元局部 CFL 步长（2026-09-14）：`step()` 的 DUAL_TIME 分支现在
    # 用 `_compute_local_time_step_gpu()` 算内层伪时间步长，而不是把调用
    # 方传入的全局 `dt` 铺满（那条"已接受的简化"已补齐，见
    # gpu_distributed.py::_compute_local_time_step_gpu 的重写说明）。
    # 本 stub 刻意不构造真实网格几何/紧凑面数据（那会把这个专注于
    # "trial U 是否正确临时写入再恢复""dual_time_U_prev 是否持久化"的
    # 用例变成一个完整求解器集成测试），所以这里给一个常量步长替身——
    # 与原先 `dt` 铺满的数值行为完全一致，这两个风险点的判据不受影响
    # （只要 DUAL_TIME 收敛，U_gpu 必须收敛到 target，是纯不动点性质）。
    _PSEUDO_DT = 1.0e-3

    def _compute_local_time_step_gpu(return_physical_too: bool = False):
        arr = np.full(n_local, _PSEUDO_DT)
        return (arr, arr) if return_physical_too else arr
    stub._compute_local_time_step_gpu = _compute_local_time_step_gpu
    # 紧凑<->原生排列置换：本 stub 只有 local（无 halo）、且不做"棱柱在前"
    # 重排，所以是恒等置换。
    stub._inv_perm_gpu = np.arange(n_local)
    stub._perm_gpu = np.arange(n_local)
    stub.low_mach_precond_enabled = False
    stub._cfl_controller = None

    def _update_cfl_controller(residual_norm):
        return None
    stub._update_cfl_controller = _update_cfl_controller

    calls = {"count": 0}

    def _compute_turbulence_source_distributed(dt):
        return None
    stub._compute_turbulence_source_distributed = _compute_turbulence_source_distributed

    def _compute_total_residual_gpu(mu_t_field=None):
        calls["count"] += 1
        # 与生产同一约定：`compute_*_residual_fr_gpu` 返回的是 **dU/dt**
        # （CPU/GPU 交叉验证测试直接把它与 CPU `compute_inviscid_residual_fr`
        # 比较，单机 GPU 取负得 R）。dU/dt = -5 (U - target)，不动点 U=target。
        # 2026-09-25 更正：此前这里写成 `+5 (U - target)` 并注释为 R，与生产
        # 约定相反 —— 多 GPU 手工 RK 分支正是按这个错的约定写的 `L = -res`，
        # 于是在真实残差上**逆时间积分**，而这个替身让它看起来是对的。
        return -5.0 * (stub.U_gpu - target)
    stub._compute_total_residual_gpu = _compute_total_residual_gpu

    def _global_residual_norm(res_flat):
        return float(np.linalg.norm(res_flat))
    stub._global_residual_norm = _global_residual_norm

    stub._calls = calls
    return stub


class TestMultiGpuDistributedDualTime:
    def test_spatial_residual_closure_swaps_and_restores_u_gpu(self):
        """决定性验证本次改动真正的风险点：`_spatial_residual` 闭包
        必须（1）把传入的 trial U 正确临时写入 `self.U_gpu`，让
        `_compute_total_residual_gpu` 读到的是 trial 值而不是原值；
        （2）调用结束后把 `self.U_gpu` 恢复回原值，不泄漏到外部。

        只验证第一次 `_spatial_residual` 调用本身、以及调用前后
        `self.U_gpu` 的状态（收敛性不是本用例的对象）。"""
        n_local, n_sps, n_vars = 2, 3, 5
        rng = np.random.default_rng(13)
        target = _make_target(n_local, n_sps, rng)
        stub = _make_stub(n_local, n_sps, n_vars, target)
        original_U = stub.U_gpu.copy()

        captured = {}
        real_step_dual_time = stub.time_integrator.step_dual_time

        def _spy(solution, spatial_residual, *args, **kwargs):
            # 用一个与 self.U_gpu 当前值不同的 trial 调一次：
            # `residual_at_trial` 是否正确反映 trial（而不是外层
            # self.U_gpu 原值）证明了"调用期间确实临时替换成了 trial"
            # ——不能直接在 spatial_residual 内部拍照 self.U_gpu，
            # 因为 finally 块会在 spatial_residual *返回前* 就已经把
            # self.U_gpu 恢复，调用方在这里永远只能看到恢复后的值
            # （这正是我们想要的行为，见下面 res_before 判据）。
            trial = solution + 1.0
            res_before = stub.U_gpu.copy()
            captured["residual_at_trial"] = spatial_residual(trial)
            captured["u_gpu_restored"] = np.allclose(stub.U_gpu, res_before)
            return real_step_dual_time(solution, spatial_residual, *args, **kwargs)

        stub.time_integrator.step_dual_time = _spy
        gd_mod.MultiGPUDistributedSolver.step(stub, dt=1e-2)

        assert "residual_at_trial" in captured
        trial_flat = (original_U + 1.0).reshape(n_local * n_sps, n_vars)
        # spatial_residual 返回 R = -dU/dt = 5 (U - target)
        expected_residual_at_trial = 5.0 * (trial_flat - target.reshape(n_local * n_sps, n_vars))
        np.testing.assert_allclose(
            captured["residual_at_trial"], expected_residual_at_trial, rtol=1e-10, atol=1e-10,
        )
        assert captured["u_gpu_restored"], "spatial_residual 调用后必须把 self.U_gpu 恢复回调用前的值"

    def test_dual_time_u_prev_persists_across_two_steps(self):
        n_local, n_sps, n_vars = 2, 2, 5
        rng = np.random.default_rng(12)
        target = _make_target(n_local, n_sps, rng)
        stub = _make_stub(n_local, n_sps, n_vars, target)

        assert stub._dual_time_U_prev is None
        gd_mod.MultiGPUDistributedSolver.step(stub, dt=1e-2)
        assert stub._dual_time_U_prev is not None
        first_prev = stub._dual_time_U_prev.copy()

        gd_mod.MultiGPUDistributedSolver.step(stub, dt=1e-2)
        # 第二步的 solution_prev 必须更新为第一步*结束时*的状态
        # （BDF2 用的 U^{n-1}），不是恒等于第一步开始前的初值。
        assert not np.allclose(stub._dual_time_U_prev, first_prev)


class TestMultiGpuDistributedRkUsesSharedIntegrator:
    """2026-09-25：多 GPU 的 SSP-RK 分支此前手工展开了一份 RK stage（正性用
    逐点硬钳、"先钳后滤"，残差报的是最后一个 stage 的），现改走与 CPU 同一个
    `TimeIntegrator` stage 推进。决定性判据：

    1. 结果与按 Shu-Osher 系数手算的 SSP-RK3 逐位（到舍入）一致；
    2. 报告的残差范数是 **stage 0** 的物理残差 `R(U^n)`（与 CPU 单机/分布式、
       单机 GPU 同一口径）；
    3. 正性回调在每个 stage 都被调用（3 次）。
    """

    def test_rk3_matches_hand_rolled_and_reports_stage0_residual(self):
        from autoflowcfd.core.gpu.gpu_time_integration import GPUTimeIntegrator

        n_local, n_sps, n_vars = 3, 2, 5
        rng = np.random.default_rng(21)
        target = _make_target(n_local, n_sps, rng)
        stub = _make_stub(n_local, n_sps, n_vars, target)
        stub.time_integrator = GPUTimeIntegrator(scheme="ssp_rk3")
        calls = {"n": 0}

        def _pos(U):
            calls["n"] += 1
            return U
        stub._get_positivity_limiter_gpu = lambda: _pos

        U0 = stub.U_gpu.reshape(-1, n_vars).copy()
        T = target.reshape(-1, n_vars)
        dt = 1.0e-3                          # 与 stub 的局部步长替身一致

        def L(U):                            # dU/dt = -R(U) = -5 (U - T)
            return -5.0 * (U - T)
        U1 = U0 + dt * L(U0)
        U2 = 0.75 * U0 + 0.25 * U1 + 0.25 * dt * L(U1)
        U3 = U0 / 3.0 + 2.0 / 3.0 * U2 + 2.0 / 3.0 * dt * L(U2)

        norm = gd_mod.MultiGPUDistributedSolver.step(stub, dt=0.0)

        np.testing.assert_allclose(stub.U_gpu.reshape(-1, n_vars), U3, rtol=1e-13, atol=0.0)
        assert norm == pytest.approx(float(np.linalg.norm(5.0 * (U0 - T))), rel=1e-14),             "残差范数必须是 stage 0 的物理残差 R(U^n)"
        assert calls["n"] == 3, "正性回调必须在每个 RK stage 都被调用"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
