"""
AutoFlowCFD V2.0 - Dual-Time Stepping 内层伪时间迭代 (S-05)

从 time_integration.py 拆出来（控制单文件行数，>400 行需拆分的项目
规范），签名以 `integrator: TimeIntegrator` 为第一参数，
`TimeIntegrator.step_dual_time` 保留同名薄委托方法，调用方式不变——
与代码库里 fr_solver_turbulence.py/solver_helpers.py 已经在用的拆分
模式一致。
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from loguru import logger


def step_dual_time(
    integrator,
    solution: np.ndarray,
    spatial_residual: Callable[[np.ndarray], np.ndarray],
    pseudo_dt: np.ndarray,
    dt_physical: float,
    solution_prev: Optional[np.ndarray] = None,
    max_inner_iter: int = 5,
    tol: float = 1e-4,
    filter_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """执行一步 Dual-Time Stepping (S-05)：真正时间精度的物理时间推进。

    物理方程 dU/dt = -R_spatial(U) 在物理时间步内用 BDF 隐式离散，
    再靠伪时间迭代把增广后的伪残差
        R_dual(U) = R_spatial(U) + (BDF 时间导数项)
    收敛到 0（等价于隐式求解 BDF 方程）——这是双时间步法的标准定义
    （Jameson 1991），伪残差里必须包含物理时间导数项，否则伪时间内
    迭代只是在反复收敛到同一个稳态，物理时间步之间不会有任何差异，
    `dt_physical`/`solution_prev` 形同虚设。此前的实现里，调用方传入
    的 `physical_residual` 只是纯空间残差、不含任何物理时间耦合项，
    是这个 bug 的根源，已在此修复：改为在这里根据 `solution_prev`
    是否提供，用 BDF1（仿真第一个物理步，只有一层历史可用）或 BDF2
    （此后每一步，二阶精度）构造真正的增广伪残差。

    Args:
        integrator: TimeIntegrator 实例（用它的 _ssp_rk_stage_step /
            n_steps / current_time）
        solution: 物理时间层 n 的状态 U^n
        spatial_residual: 纯空间残差 R_spatial(U)（不含任何时间导数项）
        pseudo_dt: 伪时间迭代用的局部（逐单元）步长，只用来加速内层
            收敛，与外层真正的物理时间步 dt_physical 是两个独立概念
        dt_physical: 真正的物理时间步长（此前被忽略、用
            self.dt 代替的 bug 已修复，见下方 current_time 更新）
        solution_prev: 物理时间层 n-1 的状态 U^{n-1}；None 表示这是
            仿真的第一个物理步，退化为一阶 BDF1（后向欧拉）
        max_inner_iter: 默认值见 TimeIntegrator.__init__ 的
            dual_time_steps 参数文档（此前恒为硬编码 3，真实测得默认
            保守 CFL 起点下明显不够，已提高默认值并通过 CLI/构造参数
            暴露给调用方调整）
    """
    from .time_integration import _SSP_RK3, enforce_positivity

    U_n = solution  # 物理时间步 n 的状态
    U_tau = U_n.copy()  # 伪时间初始猜测

    if solution_prev is None:
        # BDF1（一阶后向欧拉）：dU/dt ≈ (U - U_n) / dt_physical
        def dual_residual(U: np.ndarray) -> np.ndarray:
            return spatial_residual(U) + (U - U_n) / dt_physical
    else:
        # BDF2（二阶后向差分）：dU/dt ≈ (3U - 4U_n + U_{n-1}) / (2 dt_physical)
        def dual_residual(U: np.ndarray) -> np.ndarray:
            return spatial_residual(U) + (3.0 * U - 4.0 * U_n + solution_prev) / (2.0 * dt_physical)

    # 初始伪残差
    R_phys_initial = dual_residual(U_tau)
    initial_res_norm = np.linalg.norm(R_phys_initial)

    logger.debug(f"Dual-Time Stepping: initial pseudo-residual norm = {initial_res_norm:.6e}")

    # CFL 自适应参数：起点用 cfl_min 而不是一个乐观值，是有意为之——
    # 增大 CFL 只在连续观测到残差快速下降后才发生，反应天然滞后；
    # 减小 CFL 只有等残差真的上升了才触发，那时解往往已经被推到
    # 错误区域，需要后续很多步才能"还债"。从保守步长起步、按下降
    # 情况谨慎放大，比从乐观步长起步、按上升情况被动收缩更不容易
    # 过冲——真实复现：起点用 1.0 时，前 3 次内层迭代残差范数从
    # 1e3 冲到 1.5e6（放大 1500 倍）才找到稳定区间，之后即使残差
    # 单调下降也需要远超预算的迭代次数才能追平这个过冲。
    cfl_current = 0.1
    cfl_min = 0.1
    cfl_max = 10.0

    for k in range(max_inner_iter):
        # 计算增广伪残差（含物理时间导数项）
        R_phys = dual_residual(U_tau)
        current_res_norm = np.linalg.norm(R_phys)

        # 检查伪残差收敛。绝对阈值 tol 的判据必须要求 k>=1（至少真正
        # 做过一次伪时间迭代）才允许触发——这是一个真实复现过的 bug：
        # tol 是一个跟具体问题尺度/网格 SP 总数无关的固定绝对值，
        # 对一个 SP 数量大、边界强迫又局部集中的网格，初始状态的全域
        # L2 范数很容易恰好已经低于这个绝对值（即使边界附近真实存在
        # 需要演化的物理强迫），若在 k=0（还没做过任何一次真正更新）
        # 就用这个绝对判据跳出循环，U_tau 会原地不动地"假收敛"，物理
        # 时间步之间不会有任何演化——已用 Couette 合成算例复现：从
        # 静止流场（与壁面速度不匹配、真实需要演化）出发，80 个物理
        # 步后 dual-time-residual/max_err 与 k=0 时逐位精确相同。
        # 相对判据（current_res_norm < initial_res_norm*1e-6）不受这个
        # 问题影响——按定义 k=0 时 current_res_norm 恒等于
        # initial_res_norm，比值恒为 1，不可能满足 <1e-6，不需要额外
        # 加 k>=1 限制。
        if current_res_norm < initial_res_norm * 1e-6:
            logger.debug(f"Dual-Time converged (relative) at iteration {k+1}, res_norm={current_res_norm:.6e}")
            break
        if k >= 1 and current_res_norm < tol:
            logger.debug(f"Dual-Time converged (absolute) at iteration {k+1}, res_norm={current_res_norm:.6e}")
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

        # 伪时间推进: dU/dtau = -R_dual，用与 pseudo_dt 稳定性域匹配的
        # 真正 SSP-RK stage 推进（见 _ssp_rk_stage_step 文档：此前这里
        # 是纯前向欧拉，但 pseudo_dt 是按 SSP-RK 的稳定性域标定的 CFL
        # 步长，前向欧拉稳定性域小得多，直接复用会失稳）。R_phys 已经
        # 是 U_tau 处的 dual_residual，作为 residual0 传入避免重复计算。
        U_next = integrator._ssp_rk_stage_step(
            U_tau, dual_residual, adjusted_pseudo_dt, residual0=R_phys, table=_SSP_RK3, filter_func=filter_func
        )
        U_next = enforce_positivity(U_next)
        if filter_func is not None:
            U_next = filter_func(U_next)

        # 检查更新幅度
        update_norm = np.linalg.norm(U_next - U_tau)
        if update_norm < 1e-10:
            logger.debug(f"Dual-Time update too small at iteration {k+1}")
            break

        U_tau = U_next
    else:
        logger.warning(f"Dual-Time did not converge after {max_inner_iter} iterations, "
                      f"final res_norm={current_res_norm:.6e}, initial={initial_res_norm:.6e}")

    integrator.n_steps += 1
    integrator.current_time += dt_physical
    return U_tau
