"""AutoFlowCFD V2.0 - AdaptiveCFLController 主类：构造、重置与状态查询

从 `src/autoflowcfd/core/time_integration/adaptive_cfl.py` 拆出（2026-09-24）。方法按职责分到同目录的 mixin 里，
这里只留构造与对外接口。
"""

from __future__ import annotations
import os
from collections import deque
from typing import Deque, List, Tuple
from loguru import logger
from .gates import _CFLGatesMixin
from .update import _CFLUpdateMixin


class AdaptiveCFLController(_CFLGatesMixin, _CFLUpdateMixin):
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
           - 0.95 ≤ ratio ≤ 1.0（死区）：单步比值无信息，交给窗口趋势
             判据（最近 trend_window 步累计下降 < trend_threshold 则
             ×trend_factor），见模块文档第 4 条
           - 1.0 < ratio ≤ 1.1（轻微恶化）：立即 CFL ×0.9（shrink）
           - ratio > 1.1（明显恶化）：立即 CFL ×0.8（shrink）
        3. 冷却期：两次 CFL 调节之间至少间隔 cooldown_steps 步，
           避免 CFL 在相邻两步间来回振荡。

    Attributes:
        cfl_number: 当前 CFL 数（cfl.py 读取此值替代硬编码 0.1）
    """

    def __init__(
        self,
        cfl_start: float = 0.03,
        cfl_max: float = 0.06,
        cfl_min: float = 0.01,
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
        trend_window: int = 20,
        trend_threshold: float = 0.995,
        trend_factor: float = 1.1,
        mild_shrink_confirm_steps: int = 3,
        trend_shrink_threshold: float = 1.0,
        ceiling_backoff: float = 0.95,
        ceiling_release_steps: int = 100,
        grow_block_ratio: float = 2.0,
        mild_shrink_block_ratio: float = 1.5,
    ):
        """初始化自适应 CFL 控制器。

        Args:
            cfl_start: 初始 CFL 数（调用方认为已验证稳定的保守值）。
                **注意它只是一个断言、不是事实**：软上限（第 8 条）原先
                无条件把 cfl_start 当下界，结果在真实稳定 CFL 低于
                cfl_start 的工况下那道保护被完全抵消（79 万单元 cube_demo
                实测：稳定 CFL 约 0.063、cfl_start=0.1，控制器爬到 0.0697
                失败后又爬回 0.0690 并在 2 步内发散）。现在只有**收缩发生
                在 cfl_start 之上**时才用它当下界；一旦在它以下发生收缩，
                就说明这个断言对当前问题不成立，下界改用收缩后的 CFL，
                详见 `_shrink` 分支里该处的完整说明。
            cfl_max: CFL 上限。SSP-RK3 线性稳定极限 ~1.0，实际可用上限
                取决于网格/物理问题（AUSM+up 低马赫预处理激活时更低）。
                默认 0.5（2026-09-07 从 0.3 上调——0.3 对本项目多数网格
                过于保守、稳态收敛慢）；不稳定时用 CLI `--cfl-max` 回调
                到 0.3 或更低。
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
            trend_window: 窗口趋势判据的窗口长度（步）。取 20 是因为渐近段
                每步只降千分之几，单步比值被死区吞掉，必须跨足够多步才
                能把真实趋势与噪声分开。
            trend_threshold: 窗口内累计残差比的上界。0.995 对应"20 步至少
                累计下降 0.5%"，比实测的健康值（0.99916^20≈0.983）宽松
                一个量级，因此不会把停滞误判成进展。
            trend_factor: 窗口趋势成立时的放大因子（1.1，与 growth_factor
                同——每次只放大 10%，且每次放大后窗口清空重新积累，所以
                从 0.1 爬到 0.5 约需 17 次、~340 步，属于单向温和试探）。
            mild_shrink_confirm_steps: **轻度**恶化（1.0 < ratio <= 1.1）需要
                连续满足的步数。取 3 的理由见模块文档第 5 条：单步残差上升
                在稳态显式迭代里是正常噪声，据此立刻收缩会让 CFL 在任何
                残差略带噪声的算例里单向棘轮下滑到下限。重度恶化
                （ratio > shrink_threshold，含 NaN/inf）**不受这个确认约束**，
                仍然立即收缩——那是真实失稳信号，必须第一时间响应。
            ceiling_release_steps: 连续这么多步没有发生任何收缩时，把软
                上限**放开一档**（乘 1/ceiling_backoff）。没有这条释放
                机制，带噪声的收敛轨迹里偶发的收缩会把上限不断压低、
                放大再也回不去——实测（噪声 sigma=0.4%、均值每步降 0.05%
                的 6000 步轨迹）CFL 会一路滑到 0.034，比 cfl_start=0.1
                还低，等于把第 5/6 条修掉的棘轮效应换了个形式又引回来。
                默认 100（= 5 倍 trend_window）：贴着稳定边界时收缩频繁、
                上限稳稳压住；真正安静的区段才缓慢放开。
            ceiling_backoff: 每次发生收缩时，把"软上限"设为
                `ceiling_backoff * 收缩前的 CFL`（默认 0.95），此后放大
                不得越过这个上限。理由见模块文档第 8 条：没有它，控制器
                会周期性地重新探过稳定边界，每次探过都要付一段残差过冲
                的代价——79 万单元真实网格实测因此变成净负收益。
            trend_shrink_threshold: 轻度收缩的**窗口**阈值，与 trend_threshold
                之间构成迟滞带（默认 0.995 ~ 1.0）。两个方向共用一个阈值时
                CFL 会在稳定边界上无限抖动——实测 128 单元算例稳定后 CFL
                在 0.106 <-> 0.120 之间每约 26 步往复一次（见模块文档第 7
                条）。落在带内（窗口有下降但幅度不够）时 CFL 保持不动。
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
        self.trend_window = trend_window
        self.trend_threshold = trend_threshold
        self.trend_factor = trend_factor
        self.mild_shrink_confirm_steps = mild_shrink_confirm_steps
        self.trend_shrink_threshold = trend_shrink_threshold
        self.ceiling_backoff = ceiling_backoff
        self.ceiling_release_steps = ceiling_release_steps
        self.grow_block_ratio = grow_block_ratio
        self.mild_shrink_block_ratio = mild_shrink_block_ratio

        # 环境变量 `AFCFD_CFL_LEGACY=1`：整体退回 2026-09-14 之前的行为
        # （死区里 CFL 完全不动 + 轻度恶化单步立即收缩），供现场排查与
        # 受控 A/B 用。与 `AFCFD_LOW_MACH_PRECOND` 同一原则："把这个新
        # 机制单独关掉再跑一遍"必须是一条随时可用的路径、不需要改代码。
        # 这条口子不是可选的锦上添花：本次做真实网格 CFL 对照时，"修复
        # 前"的基线只能靠"进程在改动落盘之前启动"这种巧合来获得，没有
        # 干净的 A/B 手段——这就是需要它的直接证据。
        # 三个 CFL 边界参数的协调（见模块文档第 11 条）：`cfl_min > cfl_max`
        # 是自相矛盾的配置，而 `cfl_max` 是调用方更强的意图声明（"这个问题
        # 在这个值以上不稳定"），所以以它为准把 `cfl_min` 钳下去。不静默：
        # 打 warning 说明发生了什么、为什么。
        if self.cfl_min > self.cfl_max:
            logger.warning(
                f"[AdaptiveCFL] cfl_min={self.cfl_min} 高于 cfl_max="
                f"{self.cfl_max}，是自相矛盾的配置；以 cfl_max 为准把 "
                f"cfl_min 钳到 {self.cfl_max}。低 CFL 稳定边界扫描时请显式"
                f"传 cfl_min，否则默认下限会顶住你想测的值。"
            )
            self.cfl_min = self.cfl_max
        if not (self.cfl_min <= self.cfl_start <= self.cfl_max):
            _clamped = min(max(self.cfl_start, self.cfl_min), self.cfl_max)
            logger.warning(
                f"[AdaptiveCFL] cfl_start={self.cfl_start} 不在 "
                f"[{self.cfl_min}, {self.cfl_max}] 内，钳到 {_clamped}。"
            )
            self.cfl_start = _clamped

        # ===== `cfl_start == cfl_max` 表示"固定 CFL"（2026-09-17）=====
        #
        # 这条此前是**偶然**成立的：控制器默认 `cfl_min=0.05` 高于用户请求
        # 的 0.045/0.03/0.02，于是 `min(max(cfl*0.9, cfl_min), cfl_max)` 把
        # 收缩结果顶回 cfl_max，看起来像"钉住"。2026-09-17 把 cfl_min 默认
        # 从 0.05 降到 0.01（真实可用工作点通过 CLI 到不了，见本模块文档
        # 第 11 条）之后，那个偶然的顶住消失了，定 CFL 探针会自己往下滑
        # （实测 0.045 -> 0.03645），而"探针全程恒定"正是稳定边界扫描的
        # 前提——扫出来的数才是所请求的那个 CFL。
        #
        # 所以这里把它变成**显式语义**：两端相等就是用户在要求固定 CFL，
        # 控制器整条调节逻辑旁路。这也是它唯一自洽的读法（一个上下限相等
        # 的区间里没有任何可调空间）。
        # 必须用**钳之前**的请求值判断，不能用钳之后的 `self.cfl_start`：
        # 传 `cfl_start=0.3` 而 `cfl_max=0.06` 时 start 会被钳到 0.06，
        # 与 max 相等，那是"请求越界被纠正"，**不是**"请求固定 CFL"，
        # 按固定 CFL 处理会把收缩机制整个关掉（第一版就这么错了，被
        # test_adaptive_cfl_trend_growth.py 的三条收缩测试当场抓到）。
        self.fixed_cfl = abs(self.cfl_max - float(cfl_start)) <= 1e-12 * max(
            self.cfl_max, 1.0)
        if self.fixed_cfl:
            logger.info(
                f"[AdaptiveCFL] cfl_start == cfl_max == {self.cfl_start:g}"
                f"，按固定 CFL 处理：自适应调节全程旁路（这是稳定边界扫描"
                f"所需的语义；要让控制器工作请让 cfl_max > cfl_start）"
            )

        self.legacy_mode = os.environ.get("AFCFD_CFL_LEGACY") == "1"
        if self.legacy_mode:
            logger.warning(
                "[AdaptiveCFL] AFCFD_CFL_LEGACY=1：已退回 2026-09-14 之前的"
                "调节策略（死区不调节 + 轻度恶化单步立即收缩）。稳态收敛"
                "步数会明显增多，仅用于对照/排查。"
            )
    
        # 状态
        # 用**钳制后**的 self.cfl_start，不是局部形参 cfl_start——第 11 条的
        # 协调逻辑改的是 self.cfl_start，若这里仍读形参，初值就会绕过协调
        # （cfl_start=0.5 / cfl_max=0.08 时第一步就跑在 0.5 上）。
        self.cfl_number: float = self.cfl_start
        self._prev_residual: float = 0.0
        self._step_count: int = 0
        self._consecutive_good: int = 0      # 连续"快速下降"步数计数
        self._consecutive_crawl: int = 0     # 连续"缓慢收敛"步数计数
        self._consecutive_mild_bad: int = 0  # 连续"轻度恶化"步数计数
        # 软上限：放大不得越过它。None 表示尚未发生过收缩、只受 cfl_max
        # 约束。见模块文档第 8 条。
        self._cfl_ceiling = None
        self._steps_since_shrink: int = 0
        # 绝对基准（见模块文档第 9 条）：自构造/reset 以来见过的最小
        # **有限**残差，以及"软上限设定当时"的残差（用于给时间性释放
        # 加条件）。None 表示尚无有限残差样本。
        self._res_best = None
        self._res_at_ceiling: float = float("inf")
        self._steps_since_last_change: int = 0  # 距上次 CFL 调节的步数
        self._history: List[Tuple[int, float, float]] = []  # (step, cfl, residual)
        # 窗口趋势判据的残差滑动窗口。长度 trend_window+1，使 [0] 与 [-1]
        # 正好相隔 trend_window 步。任何恶化或一次放大都会清空它。
        self._res_window: Deque[float] = deque(maxlen=max(2, trend_window + 1))

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
        self._res_window.clear()
        self._consecutive_mild_bad = 0
        # 阶数切换 = 换了一个离散问题，旧的稳定边界不再适用，软上限作废
        self._cfl_ceiling = None
        self._steps_since_shrink = 0
        # 阶数切换 = 换了一个离散问题，旧的残差量级不可比，绝对基准同样
        # 作废（否则新阶数的第一步残差会被拿去和旧阶数的最好值比）
        self._res_best = None
        self._res_at_ceiling = float("inf")
        self._prev_residual = 0.0
        self._step_count = 0
        self._consecutive_good = 0
        self._consecutive_crawl = 0
        self._steps_since_last_change = 0

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
