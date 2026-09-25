"""AutoFlowCFD V2.0 - CPU MPI 分布式的求解循环（mixin，只含方法）。

从 `step.py` 拆出（2026-09-25，单文件不超 500 行）：单步推进在 `step.py`，
这里是迭代循环（收敛/发散判定、checkpoint 回调、Order Continuation 分派）。
"""

import time
from typing import Optional

from loguru import logger

from autoflowcfd.core.mpi import is_root


class _DistributedSolveMixin:
    """CPU MPI 分布式的求解循环"""

    def solve(self, n_steps: int, dt: float, output_interval: int = 100, checkpoint_callback=None,
              tol: float = 1e-6, phase_max_iter: Optional[int] = None,
              residual_drop_threshold: float = 1e2):
        """运行分布式求解循环。

        真实 bug 修复（2026-09-02，用户明确要求"不允许出现完成度不是
        100%的功能点"后排查发现）：此前 `output_interval` 只控制
        `logger.info` 进度打印的频率，从未触发任何中间 checkpoint
        保存——分布式路径此前只在 CLI 里 `solve()` 返回*之后*保存一次
        最终 checkpoint（见 `solve_steady_command.py`），跑到一半被
        杀掉/崩溃会丢失全部进度，且没有任何"从分布式 checkpoint 继续
        跑"的机制（`solve resume` 命令对 `--n-ranks`/`--multi-gpu`
        完全没有感知）。与单机路径 `FRSolver.solve(...,
        checkpoint_callback=...)` 同一个设计补上回调机制：调用方
        （CLI）传入的 `checkpoint_callback(solver, iteration)` 在每步
        结束后被调用，由回调自己决定何时/如何保存（通常内部判断
        `iteration % checkpoint_interval`），不在这里耦合具体的保存
        格式——与单机路径的分工完全一致。

        Order Continuation 自动分派（2026-09-02，见 core/mpi/
        distributed_order_continuation.py 模块文档）：与单机
        `FRSolver.solve()` 同一个判据（`core/utils/order_continuation/policy.py::
        uses_order_continuation`：目标阶数 >= 1 时委托给 `run_distributed_order_
        continuation`，不需要 CLI/调用方显式请求；为什么 P1 也要从 P0 起步见该
        模块文档的 A/B 数据）。

        Args:
            n_steps: 最大时间步数
            dt: 时间步长
            output_interval: 输出间隔
            checkpoint_callback: 可选，`callback(solver, iteration)`，
                每步结束后调用一次（与单机 `FRSolver.solve` 同名参数
                同一个约定）
            tol, phase_max_iter, residual_drop_threshold: 仅在触发
                Order Continuation（`uses_order_continuation`）时生效，与单机
                `run_order_continuation` 同名参数同一含义。
        """
        from autoflowcfd.core.utils.order_continuation.policy import uses_order_continuation

        if uses_order_continuation(self):
            from autoflowcfd.core.mpi.distributed_order_continuation import (
                run_distributed_order_continuation,
            )
            return run_distributed_order_continuation(
                self, n_steps, dt, tol,
                checkpoint_callback=checkpoint_callback,
                phase_max_iter=phase_max_iter,
                residual_drop_threshold=residual_drop_threshold,
            )

        from autoflowcfd.core.fr_solver.residual_diagnostics import check_residual_finite
        from autoflowcfd.core.fr_solver.state import SolverResult
        from autoflowcfd.core.time_integration.implicit.mean_flow_step import newton_monitor_suffix

        if is_root():
            logger.info(f"Starting distributed solve: {n_steps} steps, dt={dt}")

        # 与单机 `solve_loop.py`、多 GPU `solve()` 同一组语义（2026-09-25 补齐：
        # 此前这里既不判收敛——`tol` 形参没被用——也不做逐步发散检查、不返回
        # 结果，CLI 只能把 `max_iter` 当作实际步数写进 checkpoint）。残差是全域
        # allreduce 值，各 rank 一致，所以收敛/发散判定在全部 rank 上同步发生。
        converged = False
        residual_norm = float("nan")
        last_finite = None
        n_done = 0
        for step_idx in range(n_steps):
            t0 = time.time()
            residual_norm = self.step(dt)
            n_done = step_idx + 1
            if is_root() and (step_idx == 0 or n_done % output_interval == 0):
                print(f"P{self.current_order} Iter {n_done}: Residual = {residual_norm:.6e} | "
                      f"Time: {time.time() - t0:.2f}s" + newton_monitor_suffix(self))

            # 发散检查在 checkpoint 回调之前：不把发散那一步的非有限状态写进 checkpoint
            check_residual_finite(
                residual_norm, n_done, last_finite=last_finite,
                extra_hint=f"分布式路径：{self.n_ranks} 个 rank，残差是全域 allreduce 值")
            last_finite = residual_norm

            if checkpoint_callback is not None:
                # 全部 rank 都要调用（distributed_save_checkpoint 是集体操作：每个
                # rank 贡献自己的 local 单元，只在 root 调用会让其余 rank 永久阻塞）
                checkpoint_callback(self, n_done)

            if residual_norm < tol:
                converged = True
                break

        if is_root():
            logger.info("Distributed solve completed.")
        return SolverResult(converged=converged, iterations=n_done, final_residual=residual_norm)
