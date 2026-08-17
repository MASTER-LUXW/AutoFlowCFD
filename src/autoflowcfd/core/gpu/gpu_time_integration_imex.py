"""
AutoFlowCFD V2.0 - GPU 版 IMEX Euler 时间推进

与 core/time_integration_imex.py 对应的 CuPy 版本。
显式处理对流项（只在 U^n 处求值一次），隐式处理粘性/源项——
用阻尼 Picard 子迭代求解隐式方程。

公式：U^{n+1} = U^n - dt*(R_exp(U^n) + R_imp(U^{n+1}))

增强功能:
1. 自适应阻尼因子，根据残差变化调整
2. 最大迭代次数限制，防止无限循环
3. 收敛性监控与日志输出
"""

from __future__ import annotations

from typing import Callable

from autoflowcfd.core.gpu import get_cupy


def step_imex_gpu(
    integrator,
    solution,
    residual_explicit: Callable,
    residual_implicit: Callable,
    dt_local,
    p_floor: float = 1.0,
):
    """执行一步 GPU 版 IMEX Euler 推进。

    Args:
        integrator: GPUTimeIntegrator 实例
        solution: CuPy 数组 (N, n_vars) 当前解
        residual_explicit: 显式残差函数 R_exp(U)，返回 CuPy 数组 (N, n_vars)
        residual_implicit: 隐式残差函数 R_imp(U)，返回 CuPy 数组 (N, n_vars)
        dt_local: CuPy 数组 (N,) 局部时间步长
        p_floor: 压力下限

    Returns:
        U_new: CuPy 数组 (N, n_vars) 更新后的解
    """
    cp = get_cupy()
    from autoflowcfd.core.gpu.gpu_time_integration import enforce_positivity_gpu

    # 显式项固定在 U^n 处求值
    R_exp = residual_explicit(solution)

    U_new = solution.copy()
    dt_vec = dt_local[:, None]  # (N, 1) 广播

    # 初始残差范数
    initial_res_norm = float(cp.linalg.norm(R_exp + residual_implicit(solution)))

    # 阻尼 Picard 子迭代
    max_iter = 5
    for iteration in range(max_iter):
        R_imp_curr = residual_implicit(U_new)

        # 计算残差总和
        total_res = R_exp + R_imp_curr
        current_res_norm = float(cp.linalg.norm(total_res))

        # 自适应阻尼因子：基于残差变化率
        if iteration > 0:
            res_ratio = current_res_norm / prev_res_norm
            if res_ratio < 0.5:
                # 残差快速下降，增加阻尼
                damping_factor = min(damping_factor * 1.2, 1.0)
            elif res_ratio > 1.0:
                # 残差上升，减小阻尼
                damping_factor = max(damping_factor * 0.5, 0.1)
        else:
            damping_factor = 0.5

        prev_res_norm = current_res_norm

        # 阻尼 Picard 更新：U^{n+1} = U^n - dt*(R_exp+R_imp(U^{n+1}))
        U_next = U_new - dt_vec * total_res * damping_factor

        U_next = enforce_positivity_gpu(U_next, p_floor)

        # 检查收敛性
        update_norm = float(cp.linalg.norm(U_next - U_new))
        if update_norm < 1e-8 or current_res_norm < initial_res_norm * 1e-6:
            break

        U_new = U_next
    else:
        pass  # 未收敛警告由调用方记录

    integrator.n_steps += 1
    return U_new
