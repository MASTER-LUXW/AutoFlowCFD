"""AutoFlowCFD V2.0 - 自适应 CFL 的逐步更新（放大/收缩判据与步拒绝）

从 `src/autoflowcfd/core/time_integration/adaptive_cfl.py` 的 `AdaptiveCFLController` 拆出（2026-09-24，项目「单文件不超
500 行」规范）。mixin 是本仓库既有惯例（`_SolverGeometryMixin`、
`_GPUSolverInitMixin` 等），沿用它而不是另发明一套。

**只含方法，没有状态**：全部属性由 `AdaptiveCFLController` 的 `__init__` 建立，
这里通过 `self` 访问。
"""

from __future__ import annotations
import math
from loguru import logger


class _CFLUpdateMixin:
    """自适应 CFL 的逐步更新（放大/收缩判据与步拒绝）"""

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
        if getattr(self, "fixed_cfl", False):
            # 固定 CFL：仍然记录历史（诊断/日志要用），但不调节。
            self._step_count += 1
            self.cfl_number = self.cfl_start
            self._history.append(
                (self._step_count, self.cfl_number, current_residual))
            return self.cfl_number

        self._step_count += 1
        self._steps_since_last_change += 1
        self._steps_since_shrink += 1
        # 绝对基准（见模块文档第 9 条）：先更新"到过的最好残差"，本步
        # 后续所有判断都能用上它。非有限值不入基准（那是发散本身，
        # 由 ratio=+inf 的 severe 分支处理）。
        if math.isfinite(current_residual) and current_residual > 0:
            if self._res_best is None or current_residual < self._res_best:
                self._res_best = current_residual

        # 软上限的缓慢释放（见模块文档第 8 条与 ceiling_release_steps）。
        # 释放额外要求"当前残差不差于上限设定当时"（第 9 条 (b)）：释放的
        # 理由是"问题随收敛变容易了"，残差比当时更差时这个理由不成立，
        # 释放就只是去重探一个已知失败的 CFL——实测 step 129 那次正是
        # 这样回到了 step 24 已经失败过的 0.064，并在 45 步后发散。
        _release_justified = (
            not math.isfinite(self._res_at_ceiling)
            or (math.isfinite(current_residual)
                and current_residual <= self._res_at_ceiling))
        if (self._cfl_ceiling is not None
                and _release_justified
                and self._steps_since_shrink >= self.ceiling_release_steps):
            self._cfl_ceiling = min(
                self.cfl_max, self._cfl_ceiling / self.ceiling_backoff)
            self._steps_since_shrink = 0
            if self._cfl_ceiling >= self.cfl_max - 1e-12:
                self._cfl_ceiling = None   # 已放开到 cfl_max，不再需要
        self._history.append((self._step_count, self.cfl_number, current_residual))
        self._res_window.append(current_residual)

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
            self._consecutive_mild_bad = 0
        elif ratio < self.crawl_threshold:
            # 缓慢收敛（0.9 ≤ ratio < 0.95）→ crawl
            self._consecutive_crawl += 1
            self._consecutive_good = 0
            self._consecutive_mild_bad = 0
            if (self._consecutive_crawl >= self.crawl_confirm_steps
                    and self._steps_since_last_change >= self.cooldown_steps
                    and not self._growth_blocked(current_residual)):
                old_cfl = self.cfl_number
                self.cfl_number = min(
                    self.cfl_number * self.crawl_factor, self._growth_cap()
                )
                self._consecutive_crawl = 0
                self._steps_since_last_change = 0
                self._res_window.clear()
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
            severe = ratio > self.shrink_threshold
            factor = self.shrink_factor if severe else self.mild_shrink_factor
            label = "shrink" if severe else "shrink_mild"
            # **轻度恶化需要连续确认**（2026-09-14，见模块文档第 5 条）：
            # 单步残差上升在稳态显式迭代里是正常噪声，旧实现对它立刻
            # ×0.9，使 CFL 在任何残差略带噪声的算例里单向棘轮下滑到下限
            # ——真实实测（128 单元压力扰动算例，4000 步）：grow 触发 114
            # 次、shrink_mild 触发 112 次，净效果 1.1^114 × 0.9^112 ≈ 0.39，
            # CFL 从 0.1 一路被压到下限 0.05。同函数内 grow/crawl 本来都
            # 要求连续 5 步确认，唯独这一支不要求，是不对称的疏漏。
            # 重度恶化（含 NaN/inf，ratio=+inf）不受确认约束、仍然立即
            # 收缩——那是真实失稳，必须第一时间响应。
            if severe:
                self._consecutive_mild_bad = 0
            else:
                self._consecutive_mild_bad += 1
            # 轻度恶化还要过**窗口判据**（2026-09-14 同批第三处，见模块
            # 文档第 6 条）：只要窗口内（自上次 CFL 调节以来的
            # trend_window 步）残差**确有累计下降**，就说明当前 CFL 整体
            # 是在起作用的，个别连续几步的小幅上升不构成收缩理由。窗口
            # 还没攒满时退回纯连续确认（保留调节后头 20 步的快速保护）。
            window_full = (self._res_window.maxlen is not None
                           and len(self._res_window) >= self._res_window.maxlen)
            # 迟滞带（见模块文档第 7 条）：放大要求窗口累计 < trend_threshold
            # (0.995)，收缩要求窗口累计 > trend_shrink_threshold (1.0)，两者
            # 之间留出空档。共用一个阈值时 CFL 必然在稳定边界上无限抖动。
            window_worsened = False
            if window_full:
                r_old, r_new = self._res_window[0], self._res_window[-1]
                window_worsened = (
                    math.isfinite(r_old) and math.isfinite(r_new) and r_old > 0
                    and r_new / r_old > self.trend_shrink_threshold)
            if severe or self.legacy_mode:
                confirmed = True
            elif self._mild_shrink_blocked(current_residual):
                # 绝对基准闸门（模块文档第 13 条）：离历史最好残差还很近，
                # 缓慢回升是启动暂态的区域轮换，不是失稳。
                confirmed = False
            elif window_full:
                # 与放大同一个时间尺度、同一个窗口量，只是阈值留了迟滞
                confirmed = (window_worsened
                             and self._consecutive_mild_bad
                             >= self.mild_shrink_confirm_steps)
            else:
                # 窗口盲区（刚调节过、还没攒满 trend_window 步）：这里如果
                # 退回"连续 3 步确认"，噪声会稳定地在盲区内凑出 3 连升，
                # 于是每次放大后马上被收缩抵消——实测就是 CFL 在
                # 0.050 <-> 0.055 之间无限往复（见模块文档第 6 条）。
                # 盲区内只对**持续**恶化反应：连续 trend_window//2 步。
                # 噪声几乎不可能连出这么多步，而真实的单调上升 10 步内
                # 就会被抓到；更快的失稳本来就走 severe 分支。
                confirmed = (self._consecutive_mild_bad
                             >= max(self.mild_shrink_confirm_steps,
                                    self.trend_window // 2))
            if not confirmed:
                self._prev_residual = current_residual
                return self.cfl_number
            self._consecutive_mild_bad = 0
            if self._steps_since_last_change >= self.cooldown_steps:
                old_cfl = self.cfl_number
                # 软上限：记住"这个 CFL 失败过"，此后放大不得越过它
                # （见模块文档第 8 条）。取收缩前的值而不是收缩后的值，
                # 这样上限随每次失败几何式下降、从上方收敛到稳定边界，
                # 而不是一步跳到远低于边界的位置。
                # 软上限的下界**只有在 cfl_start 本身尚未被证伪时**才取
                # cfl_start：那是调用方断言过稳定的保守起始值（见
                # cfl_start 参数文档），把"能否回到起点"也禁掉就是把第
                # 5/6 条修掉的棘轮换个形式引回来（实测会让带噪声的收敛
                # 轨迹滑到 0.034 << cfl_start=0.1）。
                #
                # **真实 bug 修复（2026-09-15）**：但一旦收缩发生在**低于**
                # cfl_start 的 CFL 上，就已经有直接证据说明 cfl_start 对
                # 这个问题不稳定，再用它兜底等于把上限抬到一个**已知不稳**
                # 的值上，第 8 条这道保护被完全抵消。真实复现（79 万单元
                # cube_demo，P1 真实内容、模态滤波器关闭，稳定 CFL 约
                # 0.063 而 cfl_start=0.1）：上限被钉在 0.1 形同不存在，
                # 控制器 step 66 爬到 0.0697 -> step 79 被判定过高退回
                # 0.0627 -> step 100 **又爬回 0.0690** -> step 102 残差
                # 一步从 2.17e9 跳到 1.87e25，step 103 发散。CFL 历史在
                # 修复前后的两次运行里逐位相同，死亡步号也相同（103），
                # 这正是第 8 条要消灭的"周期性穿越稳定边界"本身。
                #
                # 改法是最小的、有证据支撑的：`old_cfl >= cfl_start` 时
                # 行为逐位不变（那正是第 8 条原本设计的工况）；
                # `old_cfl < cfl_start` 时下界改用**收缩后的 CFL**——它
                # 保证上限不会挡住"停在当前值"（不构成棘轮），同时绝不
                # 高于刚刚失败的那个值。恢复仍由 ceiling_release_steps
                # 这条时间性释放负责，不依赖 cfl_start。
                #
                # 注意这只约束**放大**：真实持续恶化时 cfl_number 仍会
                # 被收缩到 cfl_min 以下界限，不受软上限影响。
                if old_cfl >= self.cfl_start:
                    ceiling_floor = self.cfl_start
                else:
                    ceiling_floor = min(
                        max(old_cfl * factor, self.cfl_min), self.cfl_max)
                ceiling = max(old_cfl * self.ceiling_backoff, ceiling_floor)
                self._cfl_ceiling = (ceiling if self._cfl_ceiling is None
                                     else min(self._cfl_ceiling, ceiling))
                # 记住"上限是在多差的残差上设的"，供时间性释放做条件判断
                # （见模块文档第 9 条 (b)）。上限只会下降，所以这里也只
                # 记录更严格（更小）的那个残差基准。
                if math.isfinite(current_residual):
                    self._res_at_ceiling = min(self._res_at_ceiling,
                                               current_residual)
                self._steps_since_shrink = 0
                # 双向钳制（见模块文档第 11 条）：下界 cfl_min 之外还必须
                # 钳住上界 cfl_max——否则 cfl_number 低于 cfl_min 时这一行
                # 会把"收缩"变成放大、并突破调用方给定的硬上限。
                self.cfl_number = min(
                    max(self.cfl_number * factor, self.cfl_min), self.cfl_max)
                self._steps_since_last_change = 0
                self._res_window.clear()
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
            # 死区（0.95 ≤ ratio ≤ 1.0）：单步比值分不出"慢速收敛"和
            # "停滞"，所以不按单步比值动 CFL；但也不能就此不动——显式
            # 稳态迭代的渐近段本来就长期停在这个区间（实测每步
            # ratio≈0.999 且持续单调下降），旧实现在这里直接 return，
            # CFL 于是被永久钉在 cfl_start，是一个真实的收敛速度缺陷。
            # 改由窗口趋势判据接管，见 `_maybe_grow_on_trend`。
            # 连续计数照旧复位：否则 good/crawl 步隔着死区步交替也能凑满
            # 确认步数，与文档的"连续 N 步确认"语义不符。
            self._consecutive_good = 0
            self._consecutive_crawl = 0
            self._consecutive_mild_bad = 0
            self._maybe_grow_on_trend()
            self._prev_residual = current_residual
            return self.cfl_number

        # --- 快速放大 CFL（需要连续确认 + 冷却期）---

        if (self._consecutive_good >= self.growth_confirm_steps
                and self._steps_since_last_change >= self.cooldown_steps
                and not self._growth_blocked(current_residual)):
            old_cfl = self.cfl_number
            self.cfl_number = min(self.cfl_number * self.growth_factor,
                                  self._growth_cap())
            self._consecutive_good = 0
            self._steps_since_last_change = 0
            self._res_window.clear()
            if abs(self.cfl_number - old_cfl) > 1e-10:
                logger.info(
                    f"[AdaptiveCFL] Step {self._step_count}: CFL {old_cfl:.3f} → "
                    f"{self.cfl_number:.3f} (grow, ratio={ratio:.3f})"
                )

        self._prev_residual = current_residual
        return self.cfl_number
