"""
AutoFlowCFD V2.0 - GPU 版 Dual-Time Stepping 内层伪时间迭代

与 core/time_integration_dual.py 对应的 CuPy 版本。
物理方程 dU/dt = -R_spatial(U) 在物理时间步内用 BDF 隐式离散，
再靠伪时间迭代把增广后的伪残差 R_dual(U) = R_spatial(U) + (BDF 时间导数项)
收敛到 0（等价于隐式求解 BDF 方程）。

关键修复：伪残差必须包含物理时间导数项，否则伪时间内迭代只是在反复
收敛到同一个稳态，物理时间步之间不会有任何差异。
"""

from __future__ import annotations

from typing import Callable, Optional

from autoflowcfd.core.gpu import get_cupy


def step_dual_time_gpu(
    integrator,
    solution,
    spatial_residual: Callable,
    pseudo_dt,
    dt_physical: float,
    solution_prev=None,
    max_inner_iter: int = 5,
    tol: float = 1e-4,
    filter_func: Optional[Callable] = None,
):
    """执行一步 GPU 版 Dual-Time Stepping。

    Args:
        integrator: GPUTimeIntegrator 实例
        solution: CuPy 数组 (N, n_vars) 物理时间层 n 的状态 U^n
        spatial_residual: 纯空间残差函数 R_spatial(U)，返回 CuPy 数组 (N, n_vars)
        pseudo_dt: CuPy 数组 (N,) 伪时间迭代用的局部步长
        dt_physical: 真正的物理时间步长（标量）
        solution_prev: CuPy 数组 (N, n_vars) 物理时间层 n-1 的状态 U^{n-1}；
                      None 表示这是仿真的第一个物理步，退化为一阶 BDF1
        max_inner_iter: 最大内层迭代次数
        tol: 绝对收敛容差
        filter_func: 可选的模态滤波回调函数

    Returns:
        U_tau: CuPy 数组 (N, n_vars) 收敛后的伪时间解
    """
    cp = get_cupy()
    from autoflowcfd.core.gpu.gpu_time_integration import enforce_positivity_gpu

    # SSP-RK3 系数表（用于伪时间推进）
    _SSP_RK3_GPU = {
        "stages": 3,
        "alpha": [[1.0],
                  [0.75, 0.25],
                  [1.0/3.0, 0.0, 2.0/3.0]],
        "beta": [1.0, 0.25, 2.0/3.0],
    }

    U_n = solution  # 物理时间步 n 的状态
    U_tau = U_n.copy()  # 伪时间初始猜测

    # 构造增广伪残差（含物理时间导数项）
    if solution_prev is None:
        # BDF1（一阶后向欧拉）：dU/dt ≈ (U - U_n) / dt_physical
        def dual_residual(U):
            return spatial_residual(U) + (U - U_n) / dt_physical
    else:
        # BDF2（二阶后向差分）：dU/dt ≈ (3U - 4U_n + U_{n-1}) / (2 dt_physical)
        def dual_residual(U):
            return spatial_residual(U) + (3.0 * U - 4.0 * U_n + solution_prev) / (2.0 * dt_physical)

    # 初始伪残差
    R_phys_initial = dual_residual(U_tau)
    initial_res_norm = float(cp.linalg.norm(R_phys_initial))

    # CFL 自适应参数
    cfl_current = 0.1
    cfl_min = 0.1
    cfl_max = 10.0

    for k in range(max_inner_iter):
        # 计算增广伪残差（含物理时间导数项）
        R_phys = dual_residual(U_tau)
        current_res_norm = float(cp.linalg.norm(R_phys))

        # 检查伪残差收敛
        # 相对判据：不受初始范数影响，k=0 时比值恒为 1，不可能满足 <1e-6
        if current_res_norm < initial_res_norm * 1e-6:
            break
        # 绝对判据：必须要求 k>=1（至少真正做过一次伪时间迭代）才允许触发
        if k >= 1 and current_res_norm < tol:
            break

        # CFL 自适应：根据残差变化调整伪时间步长
        if k > 0:
            res_ratio = current_res_norm / prev_res_norm
            if res_ratio < 0.5:
                # 残差快速下降，增加 CFL
                cfl_current = min(cfl_current * 1.5, cfl_max)
            elif res_ratio > 1.0:
                # 残差上升，减小 CFL
                cfl_current = max(cfl_current * 0.5, cfl_min)

        prev_res_norm = current_res_norm

        # 调整伪时间步长
        adjusted_pseudo_dt = pseudo_dt * cfl_current

        # 伪时间推进: dU/dtau = -R_dual，用 SSP-RK3 stage 推进
        # R_phys 已经是 U_tau 处的 dual_residual，作为 residual0 传入避免重复计算
        U_next = integrator._ssp_rk_stage_step_gpu(
            U_tau, dual_residual, adjusted_pseudo_dt, residual0=R_phys, table=_SSP_RK3_GPU, filter_func=filter_func
        )
        U_next = enforce_positivity_gpu(U_next)
        if filter_func is not None:
            U_next = filter_func(U_next)

        # 检查更新幅度
        update_norm = float(cp.linalg.norm(U_next - U_tau))
        if update_norm < 1e-10:
            break

        U_tau = U_next

    integrator.n_steps += 1
    integrator.current_time += dt_physical
    return U_tau
