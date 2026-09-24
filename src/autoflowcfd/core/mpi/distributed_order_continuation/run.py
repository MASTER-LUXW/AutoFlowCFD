"""AutoFlowCFD V2.0 - 顶层编排：按 phase 推进各阶数

从 `src/autoflowcfd/core/mpi/distributed_order_continuation.py`(原 640 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""


from typing import Optional


from autoflowcfd.core.fr_solver.residual_diagnostics import check_residual_finite

from autoflowcfd.core.mpi import is_root
from autoflowcfd.core.utils.order_continuation import (
    _reset_turbulence_if_resumed_field_exploded,
)


def run_distributed_order_continuation(
    solver, max_iter: int, dt: float, tol: float,
    checkpoint_callback=None,
    phase_max_iter: Optional[int] = None,
    residual_drop_threshold: float = 1e2,
):
    """CPU MPI（"传统模式"/"完全分布式加载"）与多 GPU 分布式共用的
    Order Continuation 迭代循环——与单机 `order_continuation.py::
    run_order_continuation` 同一套残差-下降判据/checkpoint 回调/阶段
    步数预算逻辑，只是：
    - 阶数切换委托给 `solver._interpolate_to_new_order(target_p)`
      （按构造方式分派到本模块的 `cpu_traditional_interpolate_to_new_
      order`/`redistribute_fully_distributed_for_new_order`/GPU 对应
      实现，见各自文档），不是直接调用某个具体重建函数。
    - 打印只在 root rank 上做（`is_root()` 门控），避免 N-rank 场景下
      重复输出。
    - "resume 时检测湍流场是否被上界大面积钳制、若是则重置到来流初值"
      这项单机版早就有的安全网，**2026-09-05 之前本函数完全没有**——
      同日已补齐（见 `_reset_turbulence_if_resumed_field_exploded`
      文档，含"为什么不能直接照搬单机版 np.mean/np.sum"一节完整推导：
      湍流场在这三条后端上都是按 rank/设备本地分片存储，必须用
      `allreduce_sum` 做全局归约才能得到正确、所有 rank/设备一致的
      钳制比例，不是机械复制单机实现）。
    - 没有单机路径的"aerodynamic 系数打印"这项与本次分布式移植无关的
      特性——分布式气动力系数打印是独立的既有范围边界（"分布式路径
      不报告气动力系数"是本项目一贯的既有做法，见
      solve_steady_command.py 对应分支），不在本次任务范围内引入。

    Args:
        solver: `DistributedFRSolver`（`_is_fully_distributed` 为
            True/False 均可，取决于 `_interpolate_to_new_order` 内部
            分派）或 `MultiGPUDistributedSolver` 实例。
        max_iter, dt, tol: 与 `solver.solve()` 同名参数同一含义。
        checkpoint_callback, phase_max_iter, residual_drop_threshold:
            与单机 `run_order_continuation` 同名参数同一含义。

    Returns:
        `SolverResult`
    """
    from autoflowcfd.core.fr_solver.state import SolverResult

    # resumed 检测（2026-09-02，与单机 run_order_continuation 同一处
    # 设计——见该函数文档"resume 恢复出的 solver 状态"一节）：非 resume
    # 场景下，`DistributedFRSolver` 目前总是直接在目标阶数构造（`mesh`/
    # `ops` 按 CLI `--order` 直接生成），`solver.current_order` 从构造起
    # 就等于目标阶数，必须先重置回 P0 才能真正从头爬坡——不重置的话
    # `orders = range(current_order, original_order+1)` 会退化成只有
    # 目标阶数这一个元素，完全不爬坡，这正是本次要修的问题本身。
    # resume 场景（`_resumed_from_checkpoint=True`，见
    # solve_distributed_checkpoint_io.py::rebuild_distributed_solver_
    # from_checkpoint）则保留 checkpoint 恢复出的真实解、从
    # `solver.current_order`（checkpoint 实际所在阶数）继续爬坡，不
    # 重置。
    resumed = getattr(solver, '_resumed_from_checkpoint', False)
    if not resumed and solver.current_order != 0:
        solver._interpolate_to_new_order(0)
    elif resumed:
        # 真实完整性缺口修复（2026-09-05）：见
        # `_reset_turbulence_if_resumed_field_exploded` 文档完整推导。
        _reset_turbulence_if_resumed_field_exploded(solver)

    if is_root():
        print("\n=== Distributed Order Continuation Strategy ===")
        print(f"Starting from P{solver.current_order}, targeting P{solver.order}")

    original_order = solver.order
    starting_order = solver.current_order
    orders = list(range(starting_order, original_order + 1))

    total_iter = 0
    final_residual = 1e10

    for target_p in orders:
        if is_root():
            print(f"\n--- Phase: P{target_p} ---")

        if target_p > 0 and target_p != solver.current_order:
            solver._interpolate_to_new_order(target_p)
        solver.current_order = target_p

        # 真实 bug 修复（2026-09-05，用户指出，与单机 `run_order_
        # continuation` 同一处同一个根因——见该函数文档 phase_max_iter
        # 参数说明完整推导）：`phase_max_iter` 未显式传入时只是取
        # `max_iter // len(orders)` 作为这一个数字本身的默认值，不是
        # 切换整套预算分配策略的开关。"目标阶数吃掉剩余全部步数"这条
        # 规则对默认值和显式值一视同仁、无条件生效，不再要求用户显式
        # 传 `--phase-max-iter` 才能享受。
        is_final_stage = (target_p == original_order)
        effective_phase_max_iter = (
            phase_max_iter if phase_max_iter is not None else max_iter // len(orders)
        )
        stage_iter_budget = (max_iter - total_iter) if is_final_stage else effective_phase_max_iter
        phase_tol = tol * (10 ** (original_order - target_p))

        initial_residual_this_order = None
        min_iter_before_transition = 20
        converged = False
        _last_finite = None

        for i in range(stage_iter_budget):
            res = solver.step(dt)
            final_residual = res
            total_iter += 1

            # 发散即中止（2026-09-16）：这条循环此前没有任何有限性
            # 检查，而下面的 `checkpoint_callback` 是无条件调用的
            # ——残差变 NaN 之后会把 NaN 状态如实写进分布式
            # checkpoint。`res` 是 `solver.step()` 的全域 allreduce
            # 结果，各 rank 一致，所以全部 rank 同时抛出、不会死锁。
            check_residual_finite(
                res, i + 1, order=target_p, last_finite=_last_finite,
                extra_hint='分布式路径：残差是全域 allreduce 值（各 rank 一致）',
            )
            _last_finite = res

            if initial_residual_this_order is None:
                initial_residual_this_order = res

            if getattr(solver, '_turb_production_ramp_complete', False):
                if not getattr(solver, '_ramp_baseline_reset_done', False):
                    old_baseline = initial_residual_this_order
                    initial_residual_this_order = res
                    solver._ramp_baseline_reset_done = True
                    if is_root():
                        print(f"[INFO] P{target_p} Iter {i + 1}: Production ramp complete, "
                              f"resetting residual baseline: {old_baseline:.6e} -> {res:.6e}")

            if is_root():
                drop_ratio = initial_residual_this_order / max(res, 1e-30)
                print(f"P{target_p} Iter {i + 1}: Residual = {res:.6e} | Drop: {drop_ratio:.1f}x")

            if checkpoint_callback is not None:
                checkpoint_callback(solver, total_iter)

            drop_for_convergence = initial_residual_this_order / max(res, 1e-30)
            required_drop = 1.0 / max(phase_tol, 1e-30)
            if i >= 1 and drop_for_convergence >= required_drop:
                converged = True
                if is_root():
                    print(f"[OK] P{target_p} converged at iter {i + 1} "
                          f"(residual dropped {drop_for_convergence:.1e}x >= {required_drop:.1e}x)")
                break

            if (target_p < original_order
                    and i >= min_iter_before_transition
                    and initial_residual_this_order > 0
                    and initial_residual_this_order / max(res, 1e-30) >= residual_drop_threshold):
                if is_root():
                    print(f"[OK] P{target_p} residual dropped "
                          f"{initial_residual_this_order / res:.1f}x "
                          f"(>= {residual_drop_threshold:.0e}x), advancing to next order "
                          f"at iter {i + 1}")
                break

        if target_p == original_order and converged:
            if is_root():
                print(f"\n[OK] Distributed Order Continuation completed: "
                      f"Final P{original_order} converged")
            return SolverResult(converged=True, iterations=total_iter, final_residual=final_residual)

    return SolverResult(converged=False, iterations=total_iter, final_residual=final_residual)
