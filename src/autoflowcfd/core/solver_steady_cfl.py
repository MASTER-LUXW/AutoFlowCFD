"""FRSolver.solve() 迭代循环内的自适应 CFL 调整。

从 solver_steady.py 拆出来的 mixin：根据最近若干次迭代的残差变化趋势
（对数尺度增长/下降斜率）动态调大或调小伪时间步的 CFL 数，是 solve()
主循环里逻辑上自成一段、除了 `self.time_integrator.cfl_target` 之外不
影响循环其余部分控制流（没有 continue/break）的一块——拆成单独文件只
是为了控制单文件行数，不改变判定阈值或调整幅度。
"""

import numpy as np
from loguru import logger


class CFLTrendAdjustMixin:
    """提供 `_adjust_cfl_by_trend` 给 `FRSolver`。

    依赖宿主类（`FRSolver`）已有的 `self.time_integrator`/`self.config`
    属性，不独立维护状态；`last_cfl_cut_iteration` 由调用方（solve() 循环）
    维护并通过参数/返回值传递，因为它同时也被循环里另外两处安全机制
    （发散自动恢复、爆炸式增长防护）读写，不能变成这个 mixin 私有的状态。
    """

    def _adjust_cfl_by_trend(self, res_history, iteration: int,
                              cfl_cut_this_iter: bool,
                              last_cfl_cut_iteration: int,
                              cfl_cut_cooldown: int) -> int:
        """按残差趋势调整 `self.time_integrator.cfl_target`。

        Skipped entirely if a safety mechanism already cut CFL this
        iteration (cfl_cut_this_iter) - otherwise this rule could
        immediately increase CFL right back up in the very same step,
        since the 8-iteration trend window doesn't yet reflect the cut's
        effect.

        Returns:
            Updated `last_cfl_cut_iteration` (unchanged unless this call
            itself cut the CFL).
        """
        if cfl_cut_this_iter or iteration <= 10 or len(res_history) < 8:
            return last_cfl_cut_iteration

        # Use last 8 iterations for trend analysis (smoother signal)
        n_window = min(8, len(res_history))
        recent = res_history[-n_window:]

        # Compute log-scale trend (better for exponential decay/growth)
        # trend = ln(res_final/res_initial) / n_steps
        # Negative = decreasing, Positive = increasing
        if recent[0] > 1e-30 and recent[-1] > 1e-30:
            log_trend = np.log(recent[-1] / recent[0]) / (n_window - 1)
        else:
            log_trend = 0.0

        # Add hysteresis to prevent oscillation.
        # Only adjust if trend is significant AND sustained.
        cfl_adjusted = False

        # Check for divergence or rapid increase.
        if log_trend > 0.15:  # ~16% increase per step (aggressive threshold)
            old_cfl = self.time_integrator.cfl_target
            # More conservative reduction: x0.6 instead of x0.5
            self.time_integrator.cfl_target = max(old_cfl * 0.6, 0.01)
            last_cfl_cut_iteration = iteration
            logger.warning(
                f"  [CFL ADJUST] Residuals increasing (log_trend={log_trend:.3f}/step), "
                f"reducing CFL: {old_cfl:.3f} -> {self.time_integrator.cfl_target:.3f}"
            )
            cfl_adjusted = True

        # Check for good convergence (can increase CFL) - only once the
        # CFL has been stable (no safety cut) for a cooldown window, so an
        # increase can't immediately undo a cut made before the lower CFL
        # has had a chance to prove itself.
        elif (log_trend < -0.25
              and self.time_integrator.cfl_target < self.config.cfl_max
              and iteration - last_cfl_cut_iteration >= cfl_cut_cooldown):
            # Require sustained decrease over the window.
            decreases = sum(1 for i in range(len(recent) - 1)
                             if recent[i + 1] < recent[i])
            decrease_ratio = decreases / (len(recent) - 1)

            if decrease_ratio > 0.7:  # At least 70% of steps decreasing
                old_cfl = self.time_integrator.cfl_target
                # Moderate increase: x1.15 instead of x1.2
                self.time_integrator.cfl_target = min(old_cfl * 1.15, self.config.cfl_max)
                logger.info(
                    f"  [CFL ADJUST] Residuals decreasing well (log_trend={log_trend:.3f}/step, "
                    f"decrease_ratio={decrease_ratio:.0%}), "
                    f"increasing CFL: {old_cfl:.3f} -> {self.time_integrator.cfl_target:.3f}"
                )
                cfl_adjusted = True

        if not cfl_adjusted and iteration % 50 == 0:
            # Log status every 50 iterations even when not adjusting.
            logger.debug(
                f"  [CFL STATUS] log_trend={log_trend:.3f}/step, "
                f"CFL={self.time_integrator.cfl_target:.3f}, "
                f"no adjustment needed"
            )

        return last_cfl_cut_iteration
