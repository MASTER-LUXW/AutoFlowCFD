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

import numpy as np  # noqa: F401  (类型标注)
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
    positivity_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
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
    from autoflowcfd.core.fr_solver.residual_diagnostics import SolverDivergedError
    from autoflowcfd.core.utils.array_module import array_module

    from .base import _SSP_RK3

    # CPU 与 GPU（`GPUTimeIntegrator` 继承本类的同一个方法）共用这一份实现：
    # 数组运算按数组模块分派，范数统一取成 Python float。GPU 此前有一份独立
    # 拷贝（`gpu_time_integration_dual.py`），2026-08-23 这里修掉的四处 CFL
    # 缺陷（硬下限卡死、外层分支把 retry 砍下的步长拉回、零容忍拒绝、判据
    # 不一致）在那一份里**一处都没修**，2026-09-25 删除该拷贝。
    xp = array_module(solution)

    def _norm(a) -> float:
        return float(xp.linalg.norm(a))

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
    initial_res_norm = _norm(R_phys_initial)

    logger.debug(f"Dual-Time Stepping: initial pseudo-residual norm = {initial_res_norm:.6e}")

    # CFL 自适应参数：起点用一个保守值而不是乐观值，是有意为之——
    # 增大 CFL 只在连续观测到残差快速下降后才发生，反应天然滞后；
    # 减小 CFL 只有等残差真的上升了才触发，那时解往往已经被推到
    # 错误区域，需要后续很多步才能"还债"。从保守步长起步、按下降
    # 情况谨慎放大，比从乐观步长起步、按上升情况被动收缩更不容易
    # 过冲——真实复现：起点用 1.0 时，前 3 次内层迭代残差范数从
    # 1e3 冲到 1.5e6（放大 1500 倍）才找到稳定区间，之后即使残差
    # 单调下降也需要远超预算的迭代次数才能追平这个过冲。
    #
    # 真实 bug 修复（2026-08-23，等熵涡合成算例——涡核峰值切向速度
    # ~270 m/s，接近来流声速量级——逐迭代追踪 CFL/残差轨迹诊断出，
    # 前后共修复三处相互掩盖的问题，不是从代码走查一次看出来的）：
    #
    # 1. 此前 cfl_min=0.1 是一个硬下限——残差上升时把 cfl_current
    #    减半直到碰到 0.1 就不再继续减小，但*接受*的仍然是用"减小前"
    #    那个过大 CFL 算出的 U_next（CFL 调整只影响下一次迭代，不回滚
    #    这一步），对真正需要 CFL<0.1 才稳定的区域，残差会在 0.1 附近
    #    持续增长、max_inner_iter 耗尽也不收敛。改为标准的"步骤拒绝+
    #    重试"（pseudo-transient continuation 文献做法，如 Kelley &
    #    Keyes 1998）：算出 U_next 后先检验残差有没有恶化超过容忍幅度，
    #    真正恶化就*拒绝*这一步、把 CFL 砍半后用同一个 U_tau 重算。
    # 2. 第一次修复后发现：外层"残差上升"分支的 `max(cfl_current*0.5,
    #    cfl_min)` 用的还是旧的 cfl_min=0.1，每次迭代开始时都会把 retry
    #    循环刚辛苦砍下去的 cfl_current 强行拉回 0.1，等于每次都从头
    #    重新踩一遍"0.1 太大→retry 砍到底"的坑。删除独立的 cfl_min，
    #    外层分支和 retry 循环共用同一个 `_CFL_HARD_FLOOR`——cfl_current
    #    是跨迭代持续的单一状态，只有"残差快速下降"能把它调高。
    # 3. 修复 1/2 后用零容忍（trial_res_norm<=current_res_norm 才接受）
    #    发现新问题：等熵涡这类强非线性算例的伪时间轨迹在早期迭代本来
    #    就有一段正常的"残差先涨后落"暂态（pseudo-transient continuation
    #    文献里的标准现象，不代表不稳定），零容忍会让 retry 循环把步长
    #    一路砍到浮点噪声量级（U_next 与 U_tau 数值上不再可分辨）也不肯
    #    接受，实质上卡死在原地。改用有界容忍（`_GROWTH_TOLERANCE`）：
    #    允许残差有限度地暂时变差（暂态期间的正常现象），只有真正失控
    #    的恶化才触发拒绝重试，接受阈值和重试下限都取比原来更宽松、但
    #    仍远比"完全不设限"保守的数量级。
    cfl_current = 0.1
    cfl_max = 10.0
    _CFL_HARD_FLOOR = 1e-6
    _MAX_REJECT_RETRIES = 20  # 0.1 砍 20 次到约 1e-7，覆盖到硬下限有富余
    _GROWTH_TOLERANCE = 1.5  # 允许单次迭代残差最多恶化到 1.5 倍再拒绝

    k = 0
    while k < max_inner_iter:
        # 计算增广伪残差（含物理时间导数项）
        R_phys = dual_residual(U_tau)
        current_res_norm = _norm(R_phys)

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

        # CFL 自适应：根据上一次接受的迭代残差变化调整起始步长。
        #
        # 第四个真实 bug（修复 1-3 后仍复现）：这里原来用 `res_ratio>1.0`
        # 判断"要不要缩小 CFL"，跟 retry 循环的接受阈值
        # （`<=current_res_norm*_GROWTH_TOLERANCE`）不是同一个标准——
        # 任何哪怕 0.1%~1% 的轻微残差波动（暂态期间完全正常、且已经在
        # retry 循环里被判定为"可接受"）都会被这里判成"要缩小"，导致
        # CFL 每次迭代都被砍一半，哪怕每一步单独看都被接受、残差整体
        # 也没有真正失控——最终照样一路砍到硬下限附近停滞（等熵涡合成
        # 算例复现：cfl 在 18 次迭代内从 0.1 单调砍到 1e-6，之后残差
        # 变化量落入浮点噪声，40 次迭代几乎原地不动）。改为跟 retry 循环
        # 共用同一个 `_GROWTH_TOLERANCE` 判据：只有真正超出容忍幅度的
        # 恶化才缩小 CFL；轻微波动（无论涨跌）保持 CFL 不变，不再对
        # 噪声级波动过度反应。
        if k > 0:
            res_ratio = current_res_norm / prev_res_norm
            if res_ratio < 0.5:
                # 残差快速下降，增加 CFL
                cfl_current = min(cfl_current * 1.5, cfl_max)
            elif res_ratio > _GROWTH_TOLERANCE:
                # 残差恶化超出容忍幅度，减小 CFL——与下面拒绝重试用同一个
                # _CFL_HARD_FLOOR，不会把 retry 已经砍下去的值拉回去。
                cfl_current = max(cfl_current * 0.5, _CFL_HARD_FLOOR)
            # else: 轻微波动（涨跌都在容忍幅度内），CFL 保持不变。

        # 步骤拒绝 + 重试：只有真正降低残差的步才被接受。
        accepted = False
        U_next = U_tau
        for _retry in range(_MAX_REJECT_RETRIES):
            adjusted_pseudo_dt = pseudo_dt * cfl_current

            # 伪时间推进: dU/dtau = -R_dual，用与 pseudo_dt 稳定性域匹配
            # 的真正 SSP-RK stage 推进（见 _ssp_rk_stage_step 文档：此前
            # 这里是纯前向欧拉，但 pseudo_dt 是按 SSP-RK 的稳定性域标定
            # 的 CFL 步长，前向欧拉稳定性域小得多，直接复用会失稳）。
            # R_phys 已经是 U_tau 处的 dual_residual，作为 residual0
            # 传入避免重复计算。
            # 每个 stage 的滤波与正性保持已在 `_ssp_rk_stage_step` 内部完成
            # （`base._finish_stage`）。此前这里出来后又**再做一遍**正性钳制与
            # 滤波 —— 对非幂等的滤波档（sensor 的 0.99 有界衰减）等于每次伪时间
            # 迭代多衰减一次（2026-09-24 删除）。
            try:
                U_trial = integrator._ssp_rk_stage_step(
                    U_tau, dual_residual, adjusted_pseudo_dt, residual0=R_phys, table=_SSP_RK3,
                    filter_func=filter_func, positivity_func=positivity_func,
                )
            except SolverDivergedError:
                # 伪时间试探步本来就允许失败：冲出可容许集与"残差恶化"是同一类
                # 失败，按拒绝处理、缩小伪时间步重试。已经砍到硬下限仍不可容许，
                # 才是真正的发散，原样抛出（不接受一个不可容许的状态）。
                # 此前逐点硬钳把这类试探步"修"回可容许集（不守恒）再交给下面
                # 的残差判据，所以这条路径从来没有显式处理过。
                if cfl_current <= _CFL_HARD_FLOOR:
                    raise
                cfl_current = max(cfl_current * 0.5, _CFL_HARD_FLOOR)
                continue

            trial_res_norm = _norm(dual_residual(U_trial))
            if trial_res_norm <= current_res_norm * _GROWTH_TOLERANCE or cfl_current <= _CFL_HARD_FLOOR:
                # 接受：残差没有恶化超过容忍幅度（暂态期间的有限恶化是
                # 正常现象，不是失稳），或者已经砍到硬下限——再砍下去
                # 步长会小到失去数值意义，接受当前结果，让外层的 k 迭代/
                # max_inner_iter 预算和最终的"未收敛"告警去反映这个
                # 真实的收敛难度，而不是在这里无限重试掩盖它。
                U_next = U_trial
                accepted = True
                break
            cfl_current = max(cfl_current * 0.5, _CFL_HARD_FLOOR)

        if not accepted:
            # 循环耗尽 _MAX_REJECT_RETRIES 次仍未找到不增大残差的步长——
            # 用最后一次（已经砍到硬下限附近）的结果继续，如实记录，
            # 不静默循环到 max_inner_iter 预算耗尽却看起来像"正常收敛慢"。
            logger.warning(
                f"Dual-Time inner iteration {k+1}: step rejected {_MAX_REJECT_RETRIES} times "
                f"down to cfl={cfl_current:.3e}, still could not reduce residual "
                f"({current_res_norm:.6e} -> {_norm(dual_residual(U_next)):.6e})"
            )

        # 检查更新幅度
        update_norm = _norm(U_next - U_tau)
        if update_norm < 1e-10:
            logger.debug(f"Dual-Time update too small at iteration {k+1}")
            U_tau = U_next
            break

        U_tau = U_next
        prev_res_norm = current_res_norm
        k += 1
    else:
        logger.warning(f"Dual-Time did not converge after {max_inner_iter} iterations, "
                      f"final res_norm={current_res_norm:.6e}, initial={initial_res_norm:.6e}")

    integrator.n_steps += 1
    integrator.current_time += dt_physical
    return U_tau
