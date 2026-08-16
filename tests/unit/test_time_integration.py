"""AutoFlowCFD V2.0 - TimeIntegrator.step_dual_time 单元测试。

核心判据：双时间步法的定义是，收敛后的解必须（近似）满足增广的隐式
BDF 方程 R_spatial(U) + (时间导数项) = 0——这正是此前的 bug 所在：旧实现
传入的“物理残差”只是纯空间残差，不含任何时间导数项，伪时间内迭代只是
反复收敛到同一个稳态，dt_physical/solution_prev 形同虚设。这里直接检验
`step_dual_time` 返回的解是否满足它自己声称要驱动到零的那个增广方程，
不依赖任何解析参考解——这是最直接、无歧义的正确性判据。
"""

import numpy as np
import pytest

from autoflowcfd.core.time_integration import TimeIntegrator, TimeIntegrationScheme


def _make_state(rho=1.2, u=10.0, v=0.0, w=0.0, p=1.0e5, n_pts=4) -> np.ndarray:
    gamma = 1.4
    e = p / ((gamma - 1.0) * rho) + 0.5 * (u**2 + v**2 + w**2)
    U = np.zeros((n_pts, 5))
    U[:, 0] = rho
    U[:, 1] = rho * u
    U[:, 2] = rho * v
    U[:, 3] = rho * w
    U[:, 4] = rho * e
    return U


def _linear_spatial_residual(a: float):
    """人为构造的线性空间残差 R(U) = a * (U - U_ref)，用于让隐式 BDF
    方程有唯一、可直接代数核验的不动点，不依赖真实 FR 通量组装。"""
    U_ref = _make_state()

    def R(U: np.ndarray) -> np.ndarray:
        return a * (U - U_ref)

    return R, U_ref


def test_dual_time_bdf1_satisfies_implicit_equation():
    """BDF1（无历史层）：收敛解必须（近似）满足
    R_spatial(U) + (U - U_n)/dt = 0。"""
    ti = TimeIntegrator(scheme=TimeIntegrationScheme.DUAL_TIME)
    R, _ = _linear_spatial_residual(a=0.01)
    U_n = _make_state(p=1.2e5)  # 与 R 的不动点不同，确保确实需要推进
    dt_physical = 1e-3
    pseudo_dt = np.full(U_n.shape[0], 1e-2)

    initial_dual_residual = R(U_n) + (U_n - U_n) / dt_physical
    initial_norm = np.linalg.norm(initial_dual_residual)

    U_new = ti.step_dual_time(
        U_n, R, pseudo_dt, dt_physical=dt_physical, solution_prev=None,
        max_inner_iter=500, tol=1e-12,
    )

    dual_residual = R(U_new) + (U_new - U_n) / dt_physical
    # 与 step_dual_time 自身的内层收敛判据一致：增广伪残差必须相对初始值
    # 大幅下降（旧 bug 下，因为完全没有时间项，"收敛"判据会被空间残差
    # 本身的稳态提前满足，且这个判据无论 dt_physical 取何值结果都一样）。
    assert np.linalg.norm(dual_residual) < initial_norm * 1e-5
    # U_new 必须确实相对 U_n 发生了变化（否则说明时间项没有生效）
    assert np.max(np.abs(U_new - U_n)) > 0.0


def test_dual_time_bdf2_satisfies_implicit_equation():
    """BDF2（有历史层）：收敛解必须（近似）满足
    R_spatial(U) + (3U - 4U_n + U_{n-1})/(2 dt) = 0。"""
    ti = TimeIntegrator(scheme=TimeIntegrationScheme.DUAL_TIME)
    R, _ = _linear_spatial_residual(a=0.02)
    U_nm1 = _make_state(p=1.1e5)
    U_n = _make_state(p=1.15e5)
    dt_physical = 1e-3
    pseudo_dt = np.full(U_n.shape[0], 1e-2)

    initial_dual_residual = R(U_n) + (3.0 * U_n - 4.0 * U_n + U_nm1) / (2.0 * dt_physical)
    initial_norm = np.linalg.norm(initial_dual_residual)

    U_new = ti.step_dual_time(
        U_n, R, pseudo_dt, dt_physical=dt_physical, solution_prev=U_nm1,
        max_inner_iter=500, tol=1e-12,
    )

    dual_residual = R(U_new) + (3.0 * U_new - 4.0 * U_n + U_nm1) / (2.0 * dt_physical)
    assert np.linalg.norm(dual_residual) < initial_norm * 1e-5


def test_dual_time_advances_physical_clock_by_dt_physical_not_stale_self_dt():
    """current_time 必须按调用方传入的 dt_physical 前进，而不是
    TimeIntegrator 构造时的默认 self.dt（旧 bug：self.current_time +=
    self.dt，与调用方实际传入的物理步长无关）。"""
    ti = TimeIntegrator(scheme=TimeIntegrationScheme.DUAL_TIME, dt=1e-5)
    R, _ = _linear_spatial_residual(a=0.01)
    U_n = _make_state()
    dt_physical = 2.5e-3  # 与构造时的 self.dt=1e-5 明显不同
    pseudo_dt = np.full(U_n.shape[0], 1e-2)

    ti.step_dual_time(U_n, R, pseudo_dt, dt_physical=dt_physical, solution_prev=None, max_inner_iter=20)

    assert ti.current_time == pytest.approx(dt_physical)


def test_generic_step_rejects_dual_time_scheme():
    """通用 step() 入口不能悄悄地按错误语义处理 DUAL_TIME（缺少
    dt_physical/solution_prev 概念），必须显式报错引导调用方直接用
    step_dual_time。"""
    ti = TimeIntegrator(scheme=TimeIntegrationScheme.DUAL_TIME)
    R, _ = _linear_spatial_residual(a=0.01)
    U_n = _make_state()
    dt_local = np.full(U_n.shape[0], 1e-2)
    with pytest.raises(ValueError):
        ti.step(U_n, R, dt_local)
