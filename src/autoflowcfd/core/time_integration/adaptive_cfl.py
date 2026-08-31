"""
AutoFlowCFD V2.0 - 稳态求解器自适应 CFL 控制器

稳态伪时间迭代中，固定 CFL 数（如此前硬编码 0.1）会导致收敛速度过慢
（实测 20 步仅下降 4 倍）。本模块根据残差历史自动调节 CFL 数，在保持
稳定性的前提下加速收敛。

设计原则（2026-08-24）：
    1. 保证不发散：CFL 上限保守（0.3，远低于 SSP-RK3 稳定极限 ~1.0），
       残差恶化时立即缩小 CFL，爬升阶段固定 CFL 不动。
    2. 五级调节策略：
       - grow：ratio < 0.9（残差快速下降），连续 5 步确认后 ×1.1
       - crawl：0.9 ≤ ratio < 0.95（缓慢收敛），连续 5 步确认后 ×1.05
       - 死区：0.95 ≤ ratio ≤ 1.0，CFL 不变
       - shrink(轻)：1.0 < ratio ≤ 1.1（轻微恶化），立即 ×0.9
       - shrink(重)：ratio > 1.1（明显恶化），立即 ×0.8
       所有调节均有 5 步冷却期，避免 CFL 振荡。
    3. 与 dual.py 的双时间步自适应逻辑独立——两者面向不同的迭代结构
       （稳态每步一次 RK3 vs 双时间每步多次内迭代），参数和策略不同。

与 dual.py 自适应 CFL 的关键差异：
    - dual.py 有步拒绝 + 重试（内层迭代代价低），本控制器不做步拒绝
      （稳态一步代价高 ~8s，保存/恢复状态 + 重算残差不划算）
    - dual.py 对单步残差变化立即反应（内迭代中），本控制器要求连续
      多步确认后才调节（跨步反馈，天然滞后但更平滑）
    - dual.py 的 CFL 范围 [1e-6, 10.0]，本控制器 [0.05, 0.3]（保守）
"""

from __future__ import annotations

import math
from typing import List, Tuple

from loguru import logger


class AdaptiveCFLController:
    """稳态求解器自适应 CFL 控制器。

    根据连续两步残差范数的变化率，动态调节 CFL 数。

    调节逻辑：
        1. 爬升阶段（前 ramp_steps 步）：CFL 固定为 cfl_start，不调节。
           建立稳定的残差基线，避免初始暂态触发误调节。
        2. 爬升结束后，根据残差比 ratio = res_current / res_previous：
           - ratio < 0.9（残差快速下降 >10%）：
             连续 5 步确认后 CFL ×1.1（grow）
           - 0.9 ≤ ratio < 0.95（缓慢收敛）：
             连续 5 步确认后 CFL ×1.05（crawl）
           - 0.95 ≤ ratio ≤ 1.0（死区）：CFL 不变
           - 1.0 < ratio ≤ 1.1（轻微恶化）：立即 CFL ×0.9（shrink）
           - ratio > 1.1（明显恶化）：立即 CFL ×0.8（shrink）
        3. 冷却期：两次 CFL 调节之间至少间隔 cooldown_steps 步，
           避免 CFL 在相邻两步间来回振荡。

    Attributes:
        cfl_number: 当前 CFL 数（cfl.py 读取此值替代硬编码 0.1）
    """

    def __init__(
        self,
        cfl_start: float = 0.1,
        cfl_max: float = 0.3,
        cfl_min: float = 0.05,
        growth_factor: float = 1.1,
        shrink_factor: float = 0.8,
        growth_threshold: float = 0.9,
        shrink_threshold: float = 1.1,
        growth_confirm_steps: int = 5,
        cooldown_steps: int = 5,
        ramp_steps: int = 3,
        crawl_threshold: float = 0.95,
        crawl_factor: float = 1.05,
        crawl_confirm_steps: int = 5,
        mild_shrink_factor: float = 0.9,
    ):
        """初始化自适应 CFL 控制器。

        Args:
            cfl_start: 初始 CFL 数（已验证稳定的保守值）。
            cfl_max: CFL 上限。SSP-RK3 稳定极限 ~1.0，实际可用上限取决于
                网格/物理问题，0.3 是安全保守值。
            cfl_min: CFL 下限。低于此值说明问题本身很难，继续缩小意义不大。
            growth_factor: 快速放大因子（1.1 = 每次放大 10%）。
            shrink_factor: 重度缩小因子（0.8 = 每次缩小 20%，ratio > 1.1）。
            growth_threshold: 快速下降阈值（ratio < 0.9 即残差下降 >10%）。
            shrink_threshold: 重度恶化阈值（ratio > 1.1 即残差增长 >10%）。
            growth_confirm_steps: 快速放大前需连续满足条件的步数。
            cooldown_steps: 两次 CFL 调节之间的最小间隔步数。
            ramp_steps: 初始爬升步数。此期间 CFL 固定为 cfl_start，不调节。
            crawl_threshold: 慢速爬升阈值上界（ratio < 0.95 视为缓慢收敛）。
            crawl_factor: 慢速放大因子（1.05 = 每次放大 5%）。
            crawl_confirm_steps: 慢速放大前需连续满足条件的步数。
            mild_shrink_factor: 轻度缩小因子（0.9 = 每次缩小 10%，1.0 < ratio ≤ 1.1）。
        """
        # 参数
        self.cfl_start = cfl_start
        self.cfl_max = cfl_max
        self.cfl_min = cfl_min
        self.growth_factor = growth_factor
        self.shrink_factor = shrink_factor
        self.growth_threshold = growth_threshold
        self.shrink_threshold = shrink_threshold
        self.growth_confirm_steps = growth_confirm_steps
        self.cooldown_steps = cooldown_steps
        self.ramp_steps = ramp_steps
        self.crawl_threshold = crawl_threshold
        self.crawl_factor = crawl_factor
        self.crawl_confirm_steps = crawl_confirm_steps
        self.mild_shrink_factor = mild_shrink_factor
    
        # 状态
        self.cfl_number: float = cfl_start
        self._prev_residual: float = 0.0
        self._step_count: int = 0
        self._consecutive_good: int = 0      # 连续"快速下降"步数计数
        self._consecutive_crawl: int = 0     # 连续"缓慢收敛"步数计数
        self._steps_since_last_change: int = 0  # 距上次 CFL 调节的步数
        self._history: List[Tuple[int, float, float]] = []  # (step, cfl, residual)

    def update(self, current_residual: float) -> float:
        """根据当前步残差更新 CFL，返回下一步使用的 CFL 值。

        调用时机：step() 末尾，残差范数已计算出来之后。

        流程：
            1. 记录历史
            2. 首步 / 爬升阶段：仅记录，不调节
            3. 计算残差比 ratio = current / previous
            4. 判断是否满足调节条件（死区 + 冷却期 + 连续确认）
            5. 执行调节（放大 / 缩小 / 不变）

        Args:
            current_residual: 当前步的残差 RMS 范数

        Returns:
            cfl_number: 下一步使用的 CFL 值
        """
        self._step_count += 1
        self._steps_since_last_change += 1
        self._history.append((self._step_count, self.cfl_number, current_residual))

        # 首步：无前值可比，仅记录
        if self._prev_residual <= 0:
            self._prev_residual = current_residual
            return self.cfl_number

        # 爬升阶段：CFL 固定不动，建立残差基线
        if self._step_count <= self.ramp_steps:
            self._prev_residual = current_residual
            return self.cfl_number

        ratio = current_residual / max(self._prev_residual, 1e-30)
        # NaN/inf 防护（2026-08-25 代码审查）：残差发散成 NaN/inf 时，
        # 五区间判据的所有比较对 NaN 都为 False，会永久落入死区分支、
        # 控制器对发散完全无反应。基线或当前值非有限时一律视为最严重
        # 恶化（ratio=+inf → 立即重度缩小，CFL 快速降至下限）。
        prev_ok = math.isfinite(self._prev_residual) and self._prev_residual > 0
        if not math.isfinite(current_residual) or not prev_ok:
            ratio = float("inf")

        # --- 调节判断 ---

        if ratio < self.growth_threshold:
            # 快速下降（ratio < 0.9）→ grow
            self._consecutive_good += 1
            self._consecutive_crawl = 0
        elif ratio < self.crawl_threshold:
            # 缓慢收敛（0.9 ≤ ratio < 0.95）→ crawl
            self._consecutive_crawl += 1
            self._consecutive_good = 0
            if (self._consecutive_crawl >= self.crawl_confirm_steps
                    and self._steps_since_last_change >= self.cooldown_steps):
                old_cfl = self.cfl_number
                self.cfl_number = min(
                    self.cfl_number * self.crawl_factor, self.cfl_max
                )
                self._consecutive_crawl = 0
                self._steps_since_last_change = 0
                if abs(self.cfl_number - old_cfl) > 1e-10:
                    logger.info(
                        f"[AdaptiveCFL] Step {self._step_count}: "
                        f"CFL {old_cfl:.3f} → {self.cfl_number:.3f} "
                        f"(crawl, ratio={ratio:.3f})"
                    )
            self._prev_residual = current_residual
            return self.cfl_number
        elif ratio > 1.0:
            # 恶化：分轻度 (1.0 < ratio ≤ 1.1) 和重度 (ratio > 1.1)
            self._consecutive_good = 0
            self._consecutive_crawl = 0
            factor = (self.shrink_factor if ratio > self.shrink_threshold
                      else self.mild_shrink_factor)
            label = "shrink" if ratio > self.shrink_threshold else "shrink_mild"
            if self._steps_since_last_change >= self.cooldown_steps:
                old_cfl = self.cfl_number
                self.cfl_number = max(self.cfl_number * factor, self.cfl_min)
                self._steps_since_last_change = 0
                # 真实 bug 修复（2026-08-31，用户直接观察到真实日志报出
                # "CFL 0.050 → 0.050 (shrink_mild, ratio=1.001)"发现）：
                # CFL 已经触到下限 cfl_min 时，`max(cfl*factor, cfl_min)`
                # 会把结果 clip 回与 old_cfl 完全相同的值——数值上什么都
                # 没变，却无条件打印出"已调节"的日志，具有误导性（看起来
                # 像是控制器在正常工作、CFL 却诡异地不降低，实际上是已经
                # 到底、调节动作是空操作）。crawl（180行）/grow（222行）
                # 两个分支早就有 `abs(new-old)>1e-10` 才打印的判断，唯独
                # shrink 分支遗漏了同一道防护，是同一函数内三个并列分支
                # 彼此不一致的真实疏漏，不是三处案例中特意如此设计。
                if abs(self.cfl_number - old_cfl) > 1e-10:
                    logger.info(
                        f"[AdaptiveCFL] Step {self._step_count}: CFL {old_cfl:.3f} → "
                        f"{self.cfl_number:.3f} ({label}, ratio={ratio:.3f})"
                    )
            self._prev_residual = current_residual
            return self.cfl_number
        else:
            # 死区（0.95 ≤ ratio ≤ 1.0）：CFL 不变。同时复位连续计数，
            # 否则 good/crawl 步隔着死区步交替也能凑满确认步数，与文档的
            # "连续 N 步确认"语义不符。
            self._consecutive_good = 0
            self._consecutive_crawl = 0
            self._prev_residual = current_residual
            return self.cfl_number

        # --- 快速放大 CFL（需要连续确认 + 冷却期）---

        if (self._consecutive_good >= self.growth_confirm_steps
                and self._steps_since_last_change >= self.cooldown_steps):
            old_cfl = self.cfl_number
            self.cfl_number = min(self.cfl_number * self.growth_factor, self.cfl_max)
            self._consecutive_good = 0
            self._steps_since_last_change = 0
            if abs(self.cfl_number - old_cfl) > 1e-10:
                logger.info(
                    f"[AdaptiveCFL] Step {self._step_count}: CFL {old_cfl:.3f} → "
                    f"{self.cfl_number:.3f} (grow, ratio={ratio:.3f})"
                )

        self._prev_residual = current_residual
        return self.cfl_number

    def reset(self):
        """重置控制器状态。

        调用时机：Order Continuation 阶数切换时。阶数变化导致残差跳变
        （插值误差），不应触发 CFL 缩小。重置后重新开始爬升。
        """
        logger.info(
            f"[AdaptiveCFL] Reset at step {self._step_count} "
            f"(CFL was {self.cfl_number:.3f})"
        )
        self.cfl_number = self.cfl_start
        self._prev_residual = 0.0
        self._step_count = 0
        self._consecutive_good = 0
        self._consecutive_crawl = 0
        self._steps_since_last_change = 0
        # 不清空 _history（保留诊断信息）

    @property
    def step_count(self) -> int:
        """已处理的步数。"""
        return self._step_count

    @property
    def history(self) -> List[Tuple[int, float, float]]:
        """CFL 调节历史：(step, cfl_used, residual) 列表。"""
        return list(self._history)

    def summary(self) -> str:
        """返回当前状态的单行摘要（供日志输出）。"""
        return (
            f"CFL={self.cfl_number:.3f} "
            f"(steps={self._step_count}, good_streak={self._consecutive_good}, "
            f"crawl_streak={self._consecutive_crawl}, "
            f"since_change={self._steps_since_last_change})"
        )
