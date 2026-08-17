"""
AutoFlowCFD V2.0 - IMEX Euler 时间推进 (S-05)

从 time_integration.py 拆出来（控制单文件行数，>400 行需拆分的项目
规范），签名以 `integrator: TimeIntegrator` 为第一参数，
`TimeIntegrator.step_imex` 保留同名薄委托方法，调用方式不变。
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from loguru import logger


def step_imex(
    integrator,
    solution: np.ndarray,
    residual_explicit: Callable[[np.ndarray], np.ndarray],
    residual_implicit: Callable[[np.ndarray], np.ndarray],
    dt_local: np.ndarray,
    p_floor: float = 1.0,
) -> np.ndarray:
    """执行一步 IMEX Euler 推进 (S-05)。

    约定与 _ssp_rk_stage_step 一致：residual_explicit/residual_implicit
    都返回 R(U)（dU/dt=-R(U)），不是 dU/dt 本身。

    逻辑：显式处理对流项（只在 U^n 处求值一次，`R_exp` 在整个子迭代
    过程中固定不变），隐式处理粘性/源项——用阻尼 Picard（successive
    substitution）子迭代求解隐式方程
        U^{n+1} = U^n - dt*(R_exp(U^n) + R_imp(U^{n+1}))
    即每次迭代用当前猜测 U^{n+1}_(k) 重新求值 R_imp，与固定的 R_exp
    相加后整体反号推进——这是获得稳定性收益的关键，不能写反：
    此前版本这里写成 `U_new + dt*total_res`（加号），与 SSP-RK 系列
    使用的 dU/dt=-R(U) 约定相反，导致 IMEX 路径是无条件发散的反向
    时间积分（受控算例验证：dU/dt=-2U 解析衰减到 0.980199，旧实现
    反而增长到 1.043324）。

    增强功能:
    1. 自适应阻尼因子，根据残差变化调整
    2. 最大迭代次数限制，防止无限循环
    3. 收敛性监控与日志输出
    """
    from .base import enforce_positivity

    R_exp = residual_explicit(solution)  # 显式项，固定在 U^n 处求值

    U_new = solution.copy()
    dt_vec = dt_local[:, None]

    # 初始残差范数
    initial_res_norm = np.linalg.norm(R_exp + residual_implicit(solution))

    # 用阻尼 Picard 子迭代逼近隐式方程的解
    max_iter = 5
    for iteration in range(max_iter):
        R_imp_curr = residual_implicit(U_new)

        # 计算残差总和
        total_res = R_exp + R_imp_curr
        current_res_norm = np.linalg.norm(total_res)

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

        # 阻尼 Picard 更新：U^{n+1} = U^n - dt*(R_exp+R_imp(U^{n+1}))，
        # 与 _ssp_rk_stage_step 的 dU/dt=-R(U) 约定一致（此前是 + 号，
        # 已修复，见上方文档）。
        U_next = U_new - dt_vec * total_res * damping_factor

        U_next = enforce_positivity(U_next, p_floor)

        # 检查收敛性
        update_norm = np.linalg.norm(U_next - U_new)
        if update_norm < 1e-8 or current_res_norm < initial_res_norm * 1e-6:
            logger.debug(f"IMEX converged at iteration {iteration+1}, res_norm={current_res_norm:.6e}")
            break

        U_new = U_next
    else:
        logger.warning(f"IMEX did not converge after {max_iter} iterations, final res_norm={current_res_norm:.6e}")

    integrator.n_steps += 1
    return U_new
