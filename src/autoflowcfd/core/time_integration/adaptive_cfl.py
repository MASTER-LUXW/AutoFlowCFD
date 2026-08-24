"""
AutoFlowCFD V2.0 - 稳态求解器自适应 CFL 控制器

稳态伪时间迭代中，固定 CFL 数（如此前硬编码 0.1）会导致收敛速度过慢
（实测 20 步仅下降 4 倍）。本模块根据残差历史自动调节 CFL 数，在保持
稳定性的前提下加速收敛。

设计原则（2026-08-24）：
    1. 保证不发散：CFL 上限保守（0.3，远低于 SSP-RK3 稳定极限 ~1.0），
       残差恶化时立即缩小 CFL，爬升阶段固定 CFL 不动。
    2. CFL 变动不要太频繁：设置“死区”（残差比 0.95~1.5 之间 CFL 不变）
       和“连续确认”（需要连续 2 步残差下降才放大 CFL），避免对单步
       波动过度反应。冷却期（至少间隔 2 步才允许再次调节）进一步抑制
       CFL 振荡。
    3. 与 dual.py 的双时间步自适应逻辑独立——两者面向不同的迭代结构
       （稳态每步一次 RK3 vs 双时间每步多次内迭代），参数和策略不同。

与 dual.py 自适应 CFL 的关键差异：
    - dual.py 有步拒绝 + 重试（内层迭代代价低），本控制器不做步拒绝
      （稳态一步代价高 ~8s，保存/恢复状态 + 重算残差不划算）
    - dual.py 对单步残差变化立即反应（内迭代中），本控制器要求连续
      多步确认后才调节（跨步反馈，天然滞后但更平滑）
    - dual.py 的 CFL 范围 [1e-6, 10.0]，本控制器 [0.1, 0.5]（保守）
"""

from __future__ import annotations

from typing import List, Tuple

from loguru import logger


class AdaptiveCFLController:
    """稳态求解器自适应 CFL 控制器。

    根据连续两步残差范数的变化率，动态调节 CFL 数。

    调节逻辑：
        1. 爬升阶段（前 ramp_steps 步）：CFL 固定为 cfl_start，不调节。
           建立稳定的残差基线，避免初始暂态触发误调节。
        2. 爬升结束后，根据残差比 ratio = res_current / res_previous：
           - ratio < growth_threshold（残差持续下降，至少下降 10%）：
             需要连续 growth_confirm_steps 步都满足条件才放大 CFL
           - ratio > shrink_threshold（残差恶化，增长超过 3 倍）：
             立即缩小 CFL（安全优先，不等确认）
           - 其余（死区 [0.9, 3.0]）：CFL 保持不变
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
        growth_factor: float = 1.15,
        shrink_factor: float = 0.5,
        growth_threshold: float = 0.95,
        shrink_threshold: float = 1.5,
        growth_confirm_steps: int = 2,
        cooldown_steps: int = 2,
        ramp_steps: int = 3,
    ):
        """初始化自适应 CFL 控制器。
    
        Args:
            cfl_start: 初始 CFL 数（已验证稳定的保守值）。
            cfl_max: CFL 上限。实测 cube_demo 791k 网格 CFL=0.144 时残差开始
                振荡（step 9 ratio=1.01, step 10 ratio=1.27），取 0.3 留
                一倍裕度。SSP-RK3 稳定极限 ~1.0，但实际可用上限取决于网格
                /物理问题，0.3 是安全保守值。
            cfl_min: CFL 下限。低于此值说明问题本身很难，继续缩小意义不大。
            growth_factor: 残差持续下降时的 CFL 放大因子（1.15 = 每次放大 15%）。
                比此前的 1.2 更保守，因为实测 CFL>0.12 后残差对步长敏感。
            shrink_factor: 残差恶化时的 CFL 缩小因子（0.5 = 每次减半）。
            growth_threshold: 残差比低于此值视为“持续下降”（ratio < 0.95 即
                残差至少下降 5%）。
            shrink_threshold: 残差比高于此值视为“恶化”（ratio > 1.5 即
                残差增长超过 50%）。实测 CFL=0.144 时残差比 1.27 已是不稳定
                信号，阈值从 3.0 降到 1.5 可以更早察觉并回退。
            growth_confirm_steps: 放大 CFL 前需要连续多少步满足“持续下降”条件。
                防止单步残差骤降（可能是初始暂态而非真正收敛）触发过度放大。
            cooldown_steps: 两次 CFL 调节之间的最小间隔步数。防止 CFL 在
                相邻步之间来回振荡（放大→恶化→缩小→恢复→放大→...）。
            ramp_steps: 初始爬升步数。此期间 CFL 固定为 cfl_start，不调节。
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

        # 状态
        self.cfl_number: float = cfl_start
        self._prev_residual: float = 0.0
        self._step_count: int = 0
        self._consecutive_good: int = 0  # 连续"快速下降"步数计数
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

        # --- 调节判断 ---

        if ratio < self.growth_threshold:
            # 残差持续下降（ratio < growth_threshold）
            self._consecutive_good += 1
        elif ratio > self.shrink_threshold:
            # 残差恶化（ratio > shrink_threshold）
            # 安全优先：立即缩小 CFL，不等连续确认
            self._consecutive_good = 0
            if self._steps_since_last_change >= self.cooldown_steps:
                old_cfl = self.cfl_number
                self.cfl_number = max(self.cfl_number * self.shrink_factor, self.cfl_min)
                self._steps_since_last_change = 0
                logger.info(
                    f"[AdaptiveCFL] Step {self._step_count}: CFL {old_cfl:.3f} → "
                    f"{self.cfl_number:.3f} (shrink, ratio={ratio:.2f})"
                )
            self._prev_residual = current_residual
            return self.cfl_number
        else:
            # 死区：残差变化在可接受范围内，CFL 不变
            # 但连续"好"计数不重置（轻微波动不应抹杀之前的积累）
            # 仅当残差真正恶化（ratio > 1.0）时才重置
            if ratio > 1.0:
                self._consecutive_good = 0
            self._prev_residual = current_residual
            return self.cfl_number

        # --- 放大 CFL（需要连续确认 + 冷却期）---

        if (self._consecutive_good >= self.growth_confirm_steps
                and self._steps_since_last_change >= self.cooldown_steps):
            old_cfl = self.cfl_number
            self.cfl_number = min(self.cfl_number * self.growth_factor, self.cfl_max)
            self._consecutive_good = 0  # 重置计数器，需要再积累
            self._steps_since_last_change = 0
            if abs(self.cfl_number - old_cfl) > 1e-10:
                logger.info(
                    f"[AdaptiveCFL] Step {self._step_count}: CFL {old_cfl:.3f} → "
                    f"{self.cfl_number:.3f} (grow, ratio={ratio:.2f}, "
                    f"consecutive_good={self._consecutive_good})"
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
            f"since_change={self._steps_since_last_change})"
        )
