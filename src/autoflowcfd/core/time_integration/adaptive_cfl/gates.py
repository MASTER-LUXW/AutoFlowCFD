"""AutoFlowCFD V2.0 - 自适应 CFL 的闸门：放大/温和收缩的绝对基准闸门、增长上限、趋势放大

从 `src/autoflowcfd/core/time_integration/adaptive_cfl.py` 的 `AdaptiveCFLController` 拆出（2026-09-24，项目「单文件不超
500 行」规范）。mixin 是本仓库既有惯例（`_SolverGeometryMixin`、
`_GPUSolverInitMixin` 等），沿用它而不是另发明一套。

**只含方法，没有状态**：全部属性由 `AdaptiveCFLController` 的 `__init__` 建立，
这里通过 `self` 访问。
"""

from __future__ import annotations
import math
from loguru import logger


class _CFLGatesMixin:
    """自适应 CFL 的闸门：放大/温和收缩的绝对基准闸门、增长上限、趋势放大"""

    def _growth_blocked(self, current_residual: float) -> bool:
        """当前残差比"自己到过的最好成绩"差 `grow_block_ratio` 倍以上时
        禁止放大（模块文档第 9 条 (a)）。

        这只**阻止放大**，从不主动收缩——所以它不可能引入第 5/6 条修掉的
        那种棘轮效应。健康轨迹上残差严格单调下降、当前值就是历史最好值，
        比值恒为 1.00，本函数恒返回 False，行为逐位不变（实测见第 9 条
        列出的逐算例数据：健康档 1.00，失败档 20 步内 17.5）。
        """
        if self._res_best is None or self._res_best <= 0:
            return False
        if not math.isfinite(current_residual):
            return True   # 非有限值绝不是放大的时机
        return current_residual / self._res_best > self.grow_block_ratio

    def _mild_shrink_blocked(self, current_residual: float) -> bool:
        """当前残差离"自己到过的最好成绩"还不到 `mild_shrink_block_ratio`
        倍时，禁止**轻度**收缩（模块文档第 13 条）。

        这是第 9 条 (a) 的对称一半：那条挡的是"比历史最好差一倍以上还要
        放大"，这条挡的是"离历史最好还很近就急着收缩"。启动暂态里不同
        区域先后进入调整，残差（时间导数的范数）会缓慢回升，而解本身在
        改善 —— 真实数据见第 13 条（残差升 5.77% 的同一段窗口里，Cd 在
        441 条记录上严格单调收敛）。

        只**阻止**轻度收缩，从不主动放大，所以不可能引入第 5/6 条修掉的
        那种棘轮效应；**重度**恶化（`ratio > shrink_threshold`，含
        NaN/inf）不经过本闸门 —— 真实失稳 20 步内就把这个比值推到 17.5，
        而且那种速度本来就走重度分支立即响应。

        健康轨迹上残差严格单调下降、`ratio > 1.0` 根本不成立，本函数不会
        被求值，行为逐位不变。
        """
        if self._res_best is None or self._res_best <= 0:
            return False
        if not math.isfinite(current_residual):
            return False  # 非有限值是真实失稳，交给重度分支，别在这里拦
        return current_residual / self._res_best <= self.mild_shrink_block_ratio

    def _growth_cap(self) -> float:
        """放大的实际上限：`cfl_max` 与软上限取小。

        软上限（`_cfl_ceiling`）由每次收缩时设置，见模块文档第 8 条。
        """
        if self._cfl_ceiling is None:
            return self.cfl_max
        return min(self.cfl_max, self._cfl_ceiling)

    def _maybe_grow_on_trend(self) -> None:
        """窗口趋势判据：最近 `trend_window` 步累计确有下降则放大 CFL。

        只在死区（单步比值 0.95~1.0）里调用——其余四个区间本来就已经
        各自动作。完整动机见模块文档第 4 条。

        要求窗口必须**攒满**才判断：窗口每次放大后清空，所以两次放大
        之间至少相隔 trend_window 步，放大速率因此自带上限（默认 20 步
        最多 ×1.1），不需要额外的振荡抑制。冷却期仍然叠加生效，保证与
        shrink/crawl 之间也不会挤在一起。
        """
        if self.legacy_mode:
            return   # 见 __init__ 里 AFCFD_CFL_LEGACY 的说明
        if self._res_window.maxlen is None or len(self._res_window) < self._res_window.maxlen:
            return
        if self._steps_since_last_change < self.cooldown_steps:
            return
        r_old, r_new = self._res_window[0], self._res_window[-1]
        if not (math.isfinite(r_old) and math.isfinite(r_new)) or r_old <= 0:
            return
        if r_new / r_old >= self.trend_threshold:
            return   # 窗口内没有实质进展：停滞或原地震荡，不放大
        # 绝对基准闸门（模块文档第 9 条 (a)）：窗口内确有下降、但整条轨迹
        # 已经比自己到过的最好残差差 grow_block_ratio 倍以上时不放大。
        # `_res_window[-1]` 就是本步刚 append 进去的当前残差。这一条正是
        # 第 9 条实测那两次不该发生的放大（15.8 倍处、4.6 倍处）的阻断点。
        if self._growth_blocked(r_new):
            return

        old_cfl = self.cfl_number
        self.cfl_number = min(self.cfl_number * self.trend_factor, self._growth_cap())
        self._steps_since_last_change = 0
        self._res_window.clear()
        if abs(self.cfl_number - old_cfl) > 1e-10:
            logger.info(
                f"[AdaptiveCFL] Step {self._step_count}: CFL {old_cfl:.3f} → "
                f"{self.cfl_number:.3f} (grow_trend, {self.trend_window} 步累计 "
                f"ratio={r_new / r_old:.5f})"
            )
