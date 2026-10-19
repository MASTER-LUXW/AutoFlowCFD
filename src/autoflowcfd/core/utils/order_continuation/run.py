"""AutoFlowCFD V2.0 - Order Continuation 顶层编排：P0 -> P1 -> ... -> 目标阶数（四个后端共用的唯一循环）。

2026-10-05 以前有两份：CPU 单机的 `run_order_continuation`（换阶步骤内联、日志完整、resume 时用 checkpoint 里的
阶段起始残差做种子）与 CPU MPI / 单 GPU / 多 GPU 共用的 `run_distributed_order_continuation`（换阶委托给求解器、
只打印残差、resume 不恢复阶段基准、换阶不复位自适应 CFL）。阶段预算、升阶与收敛判据、发散守卫两份逐行相同，
差异全是"只在一份里修过"的东西。现在后端差异只经求解器接口进入：

* `_interpolate_to_new_order(p)`：这一阶的状态与几何全部就绪（`p == 0` 表示用来流重置到 P0）；
* `_limit_prolongated_state()`：延拓后在新阶数的点集上施加守恒的正性限制；
* `step(dt)`：返回全域残差范数（分布式各 rank 一致，判定同步发生）；
* 每步日志与收尾摘要（`fr_solver/solver/solve_loop.py::format_step_line / report_pseudo_time_summary`，
  与定阶循环同一份；只在 root rank 打印）。
"""

import time as _time
from typing import Any, Optional

from autoflowcfd.core.fr_solver.convergence_history import record_step
from autoflowcfd.core.fr_solver.residual_diagnostics import check_residual_finite
from autoflowcfd.core.mpi import is_root

from .policy import PhaseGate, update_phase_baseline
from .turbulence_reset import _reset_turbulence_if_resumed_field_exploded

#: 非最终阶段至少走这么多步才允许提前升阶（避免过早提升）。
MIN_ITER_BEFORE_TRANSITION = 20


def run_order_continuation(solver: Any, max_iter: int, dt: float, tol: float,
                           checkpoint_callback=None,
                           phase_max_iter: Optional[int] = None,
                           residual_drop_threshold: float = 1e2):
    """从 P0（resume 时从 checkpoint 所在阶数）逐阶爬升到 `solver.order`。

    Args:
        solver: 任一后端的求解器（接口见模块文档）。
        max_iter: 总迭代次数。
        dt: 时间步长（隐式/显式稳态下被局部伪时间步覆盖）。
        tol: 目标阶数的相对收敛容差；低一阶放宽 10 倍（`phase_tol`）。
        checkpoint_callback: `callback(solver, iteration)`，每步调用（分布式下是集体操作，全部 rank 调用）。
        phase_max_iter: 非最终阶段各自的步数上限；None 时取 `max_iter // len(orders)`。**目标阶数永远不受
            它约束，吃掉剩余全部步数**（2026-09-05 修复：此前只有显式传值时才这样分配，默认路径上目标阶数
            被和非最终阶段一样均分）。
        residual_drop_threshold: 非最终阶段提前升阶所需的残差下降倍数。

    Returns:
        `SolverResult`
    """
    from autoflowcfd.core.fr_solver.solver.solve_loop import format_step_line, report_pseudo_time_summary
    from autoflowcfd.core.fr_solver.solver.threads import blas_threads_limited
    from autoflowcfd.core.fr_solver.state import SolverResult

    report = is_root()
    original_order = solver.order
    # resume 出来的状态是 checkpoint 里的真实解（`_resumed_from_checkpoint`），不能重置回 P0 均匀流——
    # 2026-08-22 真实复现过：P1 checkpoint 续算静默从 P0 重新开始整个爬升。
    resumed = getattr(solver, "_resumed_from_checkpoint", False)
    if resumed:
        # 产生项渐变的进度已由恢复路径按 checkpoint 续接（`fr_solver/turbulence/init.py::
        # restore_production_ramp`）：渐变已完成时不再重置残差基准（否则丢掉刚恢复的阶段起始残差）
        solver._ramp_baseline_reset_done = bool(solver._turb_production_ramp_complete)
        _reset_turbulence_if_resumed_field_exploded(solver)
    else:
        solver._ramp_baseline_reset_done = False
        if solver.current_order != 0:
            solver._interpolate_to_new_order(0)
    if report:
        print("\n=== Order Continuation Strategy ===")
        print(f"Starting from P{solver.current_order}, targeting P{original_order}")

    starting_order = solver.current_order
    orders = list(range(starting_order, original_order + 1))
    total_iter = 0
    final_residual = 1e10
    with blas_threads_limited(1):
        for target_p in orders:
            if total_iter >= max_iter:
                # 总步数已用完：停在当前阶数（不再做一次没有步数可跑的升阶——延拓与算子重建都不便宜，
                # 写出的 checkpoint 也会是一个从未推进过的高阶状态）
                if report:
                    print(f"[INFO] 总步数 {max_iter} 已用完，停在 P{solver.current_order}（未升到 P{target_p}）")
                break
            if report:
                print(f"\n--- Phase: P{target_p} ---")
            if target_p != solver.current_order:
                solver._interpolate_to_new_order(target_p)
                solver._limit_prolongated_state()
            solver.current_order = target_p
            # 阶数变化让残差跳变（延拓误差），不是解在恶化，不应触发 CFL 收缩：每阶从控制器初值起步
            # （此前两个分布式后端的控制器跨阶存活）
            if getattr(solver, "_cfl_controller", None) is not None:
                solver._cfl_controller.reset()

            is_final_stage = target_p == original_order
            effective_phase_max_iter = phase_max_iter if phase_max_iter is not None else max_iter // len(orders)
            # 非最终阶段的上限同样受总步数约束：`max_iter` 是本次调用的总步数（CLI `--max-iter`），2026-10-14 以前
            # `--phase-max-iter` 大于剩余步数时非最终阶段照跑满它——plate_demo `solve resume --max-iter 8
            # --phase-max-iter 100` 在 P1 跑了 30 步（被手动停下），目标阶数分到的剩余步数为负、一步不跑
            stage_iter_budget = max_iter - total_iter
            if not is_final_stage:
                stage_iter_budget = min(stage_iter_budget, effective_phase_max_iter)
            phase_tol = tol * (10 ** (original_order - target_p))
            required_drop = 1.0 / max(phase_tol, 1e-30)

            initial_residual = None
            if resumed and target_p == starting_order:
                # resume 恢复出的第一个阶段：用 checkpoint 里存的阶段起始残差做种子，升阶判据按真实阶段起点算
                persisted = getattr(solver, "_phase_initial_residual", None)
                if persisted is not None:
                    initial_residual = persisted
                    if report:
                        print(f"[INFO] P{target_p} resume：用 checkpoint 里保存的阶段起始残差 ({persisted:.6e}) 做种子")
                elif report:
                    print(f"[WARN] P{target_p} 从没有阶段起始残差记录的 checkpoint resume：升阶判据从这次 resume 的"
                          f"第一步重新起算，可能提前触发")
            phase_gate = PhaseGate()        # 湍流一起到位才升阶/判收敛，见 policy.py 模块文档
            converged = False
            last_finite = None

            # 本模块被 core.utils 包导入期加载，time_integration 在那时尚未初始化完，局部导入
            from autoflowcfd.core.time_integration.time_accurate import advance

            for i in range(stage_iter_budget):
                t_start = _time.time()
                res = advance(solver, dt)
                elapsed = _time.time() - t_start
                final_residual = res
                total_iter += 1
                # 发散守卫在 checkpoint 回调之前：不把发散那一步的非有限状态写盘
                check_residual_finite(res, i + 1, order=target_p, last_finite=last_finite,
                                      extra_hint=solver._divergence_hint())
                last_finite = res
                record_step(solver, res)
                phase_gate.observe(solver, res)
                initial_residual = update_phase_baseline(solver, res, initial_residual, report,
                                                         f"P{target_p} Iter {i + 1}")
                solver._phase_initial_residual = initial_residual     # 写入 checkpoint，供 resume 续接
                if report:
                    print(format_step_line(solver, target_p, i + 1, res, initial_residual, elapsed, total_iter))
                if checkpoint_callback is not None:
                    checkpoint_callback(solver, total_iter)

                if i >= 1 and phase_gate.reached(solver, res, initial_residual, required_drop):
                    converged = True
                    if report:
                        print(f"[OK] P{target_p} converged at iter {i + 1}（要求 {required_drop:.1e}x 或舍入误差："
                              f"{phase_gate.describe(solver, res, initial_residual)}）")
                    break
                if i >= 1 and phase_gate.stalled_at_roundoff():
                    # 平台只停止空转、不报收敛（policy.py 模块文档）：中间阶段升阶，最终阶段以未收敛结束
                    if report:
                        verdict = "未收敛，结束" if is_final_stage else "升阶"
                        print(f"[STOP] P{target_p} iter {i + 1} 停在平台，{verdict}："
                              f"{phase_gate.describe(solver, res, initial_residual)}")
                    break
                if (not is_final_stage and i >= MIN_ITER_BEFORE_TRANSITION and initial_residual > 0
                        and phase_gate.reached(solver, res, initial_residual, residual_drop_threshold)):
                    if report:
                        print(f"[OK] P{target_p} advancing to next order at iter {i + 1}（要求 "
                              f"{residual_drop_threshold:.0e}x 或舍入误差："
                              f"{phase_gate.describe(solver, res, initial_residual)}）")
                    break

            if is_final_stage and converged:
                if report:
                    print(f"\n[OK] Order Continuation completed: Final P{original_order} converged")
                    report_pseudo_time_summary(solver, total_iter)
                return SolverResult(converged=True, iterations=total_iter, final_residual=final_residual)

    if report:
        report_pseudo_time_summary(solver, total_iter)
    return SolverResult(converged=False, iterations=total_iter, final_residual=final_residual)
