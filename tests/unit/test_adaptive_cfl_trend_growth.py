"""自适应 CFL 的「窗口趋势放大」回归测试（2026-09-14）。

修的是什么真实缺陷：`AdaptiveCFLController` 的五区间判据里，
`0.95 <= ratio <= 1.0` 是死区、CFL 完全不动。这套阈值是按**瞬态**式的
快速残差下降标定的（grow 要求单步降 >10%），但显式稳态迭代进入渐近段
后每步只降千分之几是正常且健康的——79 万单元 cube_demo P1 + 低马赫数
预处理的真实受控 A/B 实测每步 ratio≈0.99916 且残差持续单调下降，全程
稳稳落在死区里，于是 CFL 被永久钉在 cfl_start=0.100（30 步日志里一次
都没变过），cfl_max=0.5 从来到不了，白白丢掉约 5 倍步长。这是用户反馈
"从开始计算到收敛需要数万步"的一个直接成因。

修法与本文件的验证目标：死区里改看**累计**趋势（最近 trend_window 步
的残差比），确有下降就放大 CFL；停滞/震荡则不放大。下面的用例按
"必须放大"与"绝不能放大"两类分别钉住，特别是第一个用例直接复刻了
实测的 ratio=0.99916 轨迹——它在修复前必然断言失败。

## 为什么这些用例都显式传 `cfl_max`（2026-09-17）

本文件测的是**收缩/放大机制本身**，与默认上限无关。此前它们只传
`cfl_start=0.3/0.4` 而吃默认 `cfl_max`——当默认从 0.5 降到 0.06
（见 `cli/solve_steady_command.py` 的 --cfl-max 帮助：按直接谱测量与两张
真实网格的失效点重定）之后，`cfl_start` 被钳到 0.06，可调区间只剩
[cfl_min=0.05, 0.06]，机制根本展开不了，三条收缩测试当场失败。
把上限显式写进构造参数，测试从此与调参默认值解耦。
"""

import math

import numpy as np
import pytest

from autoflowcfd.core.time_integration.adaptive_cfl import AdaptiveCFLController


# 真实网格实测的每步残差比（cube_demo P1 + 预处理，30 步 A/B）
REAL_SLOW_RATIO = 0.99916


def _feed(controller, ratios):
    """按给定的逐步残差比喂一条残差轨迹，返回 CFL 历史。"""
    r = 1.0
    cfls = []
    for q in ratios:
        r *= q
        cfls.append(controller.update(r))
    return cfls


class TestGrowsOnSlowButRealProgress:
    def test_real_measured_slow_convergence_raises_cfl(self):
        """复刻实测轨迹：每步只降 0.084%，CFL 必须能涨起来。"""
        c = AdaptiveCFLController(cfl_start=0.1, cfl_max=0.5)
        cfls = _feed(c, [REAL_SLOW_RATIO] * 200)
        assert c.cfl_number > 0.1 + 1e-12, (
            "每步 ratio=0.99916 且持续单调下降 200 步，CFL 仍停在 cfl_start"
            "——死区把真实收敛误判成了无进展（本次修复针对的缺陷）")
        # 单调不降（这条轨迹里不该出现任何 shrink）
        assert all(b >= a - 1e-12 for a, b in zip(cfls, cfls[1:]))

    def test_reaches_cfl_max_on_sustained_progress(self):
        c = AdaptiveCFLController(cfl_start=0.1, cfl_max=0.5)
        _feed(c, [REAL_SLOW_RATIO] * 2000)
        assert c.cfl_number == pytest.approx(0.5, rel=1e-9)

    def test_growth_rate_is_bounded_by_window(self):
        """放大必须是温和试探：每次 ×trend_factor，且两次之间至少隔
        trend_window 步——否则控制器会在几步内跳到上限，失去"单向试探、
        随时可退"的安全性。"""
        c = AdaptiveCFLController(cfl_start=0.1, cfl_max=10.0,
                                  trend_window=20, trend_factor=1.1)
        _feed(c, [REAL_SLOW_RATIO] * 105)
        # 105 步最多 5 次放大（第一次要攒满 21 个样本）
        assert c.cfl_number <= 0.1 * 1.1 ** 5 + 1e-12
        assert c.cfl_number > 0.1


class TestDoesNotGrowWithoutRealProgress:
    def test_exact_stagnation_never_grows(self):
        """残差完全不动（ratio=1.0，仍在死区内）：绝不能放大。"""
        c = AdaptiveCFLController(cfl_start=0.1, cfl_max=0.5)
        _feed(c, [1.0] * 500)
        assert c.cfl_number == pytest.approx(0.1)

    def test_oscillation_without_net_progress_never_grows(self):
        """在死区内来回震荡、净进展为零：绝不能放大。

        轨迹：ratio 交替 0.98 / (1/0.98)。后者 >1 会走 shrink 分支，
        因此这里实际验证的是"震荡不会被趋势判据误读成收敛"。"""
        c = AdaptiveCFLController(cfl_start=0.2, cfl_max=0.5)
        ratios = [0.98, 1.0 / 0.98] * 250
        _feed(c, ratios)
        assert c.cfl_number <= 0.2 + 1e-12, "净零进展的震荡把 CFL 放大了"

    def test_progress_just_below_threshold_does_not_grow(self):
        """累计下降刚好达不到 trend_threshold：不放大（判据是单侧的，
        不允许"差一点也算"）。"""
        # 20 步累计 = q^20，要求恰好略大于 trend_threshold=0.995
        q = 0.995 ** (1.0 / 20.0) * 1.0000005
        c = AdaptiveCFLController(cfl_start=0.1, cfl_max=0.5,
                                  trend_window=20, trend_threshold=0.995)
        _feed(c, [q] * 300)
        assert c.cfl_number == pytest.approx(0.1), (
            f"20 步累计比 {q**20:.6f} >= 阈值 0.995，却放大了 CFL")


class TestWorseningInvalidatesTrend:
    def test_shrink_clears_window_no_immediate_regrow(self):
        """出现恶化后，不能凭恶化之前那段下降历史立刻把 CFL 放回去。"""
        c = AdaptiveCFLController(cfl_start=0.2, cfl_max=0.5,
                                  trend_window=20, cooldown_steps=5)
        _feed(c, [REAL_SLOW_RATIO] * 60)          # 先攒出下降趋势
        cfl_before = c.cfl_number
        c.update(c._history[-1][2] * 1.2)          # 明显恶化 -> shrink
        assert c.cfl_number < cfl_before, "明显恶化没有触发 shrink"
        after_shrink = c.cfl_number
        # 紧接着 19 步"死区"（不足以攒满新窗口）：不允许放大
        _feed(c, [1.0] * 19)
        assert c.cfl_number == pytest.approx(after_shrink), (
            "恶化后窗口没有作废，凭失效的历史又把 CFL 放大了")

    def test_divergence_to_nan_still_shrinks(self):
        """趋势判据不能干扰既有的 NaN/inf 发散保护。"""
        c = AdaptiveCFLController(cfl_start=0.3, cfl_min=0.05, cfl_max=0.5)
        _feed(c, [REAL_SLOW_RATIO] * 40)
        before = c.cfl_number
        for _ in range(10):
            c.update(float("nan"))
        assert c.cfl_number < before
        assert c.cfl_number >= 0.05 - 1e-12


class TestExistingBehaviourPreserved:
    def test_fast_drop_path_unchanged(self):
        """快速下降仍走原来的 grow 分支、仍受 cfl_max 约束。"""
        c = AdaptiveCFLController(cfl_start=0.1, cfl_max=0.5,
                                  ramp_steps=0, cooldown_steps=0,
                                  growth_confirm_steps=1)
        _feed(c, [0.5] * 50)
        assert c.cfl_number == pytest.approx(0.5)

    def test_reset_clears_trend_state(self):
        c = AdaptiveCFLController(cfl_start=0.1, cfl_max=0.5)
        _feed(c, [REAL_SLOW_RATIO] * 100)
        assert c.cfl_number > 0.1
        c.reset()
        assert c.cfl_number == pytest.approx(0.1)
        # reset 后必须重新攒满窗口才可能放大
        _feed(c, [REAL_SLOW_RATIO] * 19)
        assert c.cfl_number == pytest.approx(0.1)


class TestMildShrinkNeedsConfirmation:
    """轻度恶化必须连续确认，否则 CFL 会被残差噪声单向棘轮压到下限。

    症状与证据（与 `TestGrowsOnSlowButRealProgress` 是同一个"CFL 到不了
    cfl_max"问题的另一半成因）：`shrink_mild` 原先对**单步** ratio > 1.0
    就 ×0.9。稳态显式迭代的残差范数并不单调——即使迭代稳定且在收敛，
    约有一半的步会小幅上升。128 单元压力扰动算例实测 4000 步：grow 触发
    114 次、shrink_mild 触发 112 次，净 1.1^114 × 0.9^112 ≈ 0.39，CFL 从
    0.1 被压到下限 0.05 卡住。

    但**重度**恶化（含发散成 NaN/inf）的即时响应是安全底线，不能因为
    加了确认而变慢——下面用例把这两条一起钉住。
    """

    def test_noisy_but_converging_trajectory_keeps_cfl_up(self):
        """带噪声但整体收敛的轨迹：CFL 必须涨而不是被噪声压到下限。"""
        rng = np.random.default_rng(20260914)
        c = AdaptiveCFLController(cfl_start=0.1, cfl_max=0.5, cfl_min=0.05)
        # 每步平均降 0.1%，叠加 ±0.3% 的噪声（因此约半数步 ratio>1）
        ratios = [0.999 * float(np.exp(rng.normal(0.0, 0.003))) for _ in range(4000)]
        n_up = sum(1 for q in ratios if q > 1.0)
        assert n_up > 1000, f"本用例需要足够多的噪声上升步，实际 {n_up}"
        _feed(c, ratios)
        assert c.cfl_number > 0.1, (
            f"带噪声的收敛轨迹把 CFL 压到了 {c.cfl_number:.4f}（起始 0.1）"
            "——单步恶化就收缩的棘轮效应（本次修复针对的缺陷）")

    def test_isolated_worsening_steps_do_not_shrink(self):
        """孤立的单步恶化（前后都在下降）不得触发收缩。"""
        c = AdaptiveCFLController(cfl_start=0.2, cfl_max=0.5, cfl_min=0.05,
                                  mild_shrink_confirm_steps=3)
        # 下降 5 步、恶化 1 步，循环——恶化永远凑不满连续 3 步
        _feed(c, ([0.99] * 5 + [1.02]) * 60)
        assert c.cfl_number >= 0.2, (
            f"孤立噪声步触发了收缩，CFL 降到 {c.cfl_number:.4f}")

    def test_sustained_mild_worsening_still_shrinks(self):
        """持续的轻度恶化（真实的缓慢失稳）仍必须收缩——确认机制不能
        把保护本身关掉。"""
        c = AdaptiveCFLController(cfl_start=0.3, cfl_min=0.05, cfl_max=0.5,
                                  mild_shrink_confirm_steps=3)
        _feed(c, [1.05] * 200)
        assert c.cfl_number < 0.3
        assert c.cfl_number == pytest.approx(0.05), "持续恶化应一路收缩到下限"

    def test_real_measured_monotone_rise_still_shrinks_promptly(self):
        """复刻真实实测的"预处理关闭"轨迹（79 万单元 P1，残差逐步单调
        上升）：这种情况必须**及时**节流，与修复前的行为保持一致。"""
        c = AdaptiveCFLController(cfl_start=0.1, cfl_min=0.05)
        # 实测 30 步：3.2657e8 -> 3.3295e8，每步约 +0.066%
        _feed(c, [1.00066] * 30)
        assert c.cfl_number < 0.1, "单调上升 30 步却没有节流"

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_severe_and_nonfinite_shrink_immediately(self, bad):
        """重度恶化 / NaN / inf 不受确认约束，必须立即收缩。"""
        c = AdaptiveCFLController(cfl_start=0.4, cfl_min=0.05, cfl_max=0.5,
                                  ramp_steps=0, cooldown_steps=0,
                                  mild_shrink_confirm_steps=3)
        c.update(1.0)
        c.update(0.99)
        before = c.cfl_number
        c.update(bad if not math.isnan(bad) else bad)
        assert c.cfl_number < before, (
            "非有限残差没有被立即节流——发散保护被确认机制削弱了")

    def test_severe_worsening_shrinks_on_first_step(self):
        c = AdaptiveCFLController(cfl_start=0.4, cfl_min=0.05, cfl_max=0.5,
                                  ramp_steps=0, cooldown_steps=0,
                                  mild_shrink_confirm_steps=3)
        c.update(1.0)
        before = c.cfl_number
        c.update(1.5)            # ratio=1.5 > shrink_threshold=1.1
        assert c.cfl_number == pytest.approx(before * 0.8)


class TestNoRatchetDown:
    """棘轮效应必须从机制上消失：CFL 不能在带噪声的收敛段单向滑到下限。

    为什么单靠"连续确认"不够（实测，见模块文档第 6 条）：收缩最快每
    cooldown_steps(5) 步触发、幅度 ×0.8/×0.9；放大最快每 trend_window(20)
    步触发、幅度 ×1.1。收缩的"事件速率×幅度"是放大的数倍，于是任何
    带噪声的区段都单向下滑，最终卡在 cfl_min（比 cfl_start 还低）。
    128 单元算例实测：CFL 先爬到 0.2975 又塌回 0.05 并卡住，净效果
    1.1^183 × 0.9^138 × 0.8^18 ≈ 0.32。
    现在两个方向共用同一个窗口判据，下面用例把"不得滑到下限以下"和
    "真实持续恶化仍要一路收缩"这对相反要求一起钉住。
    """

    def test_never_ends_below_start_on_a_converging_noisy_run(self):
        rng = np.random.default_rng(7)
        c = AdaptiveCFLController(cfl_start=0.1, cfl_max=0.5, cfl_min=0.05)
        # 整体收敛（每步均值 -0.05%）但噪声幅度大于均值漂移
        ratios = [0.9995 * float(np.exp(rng.normal(0.0, 0.004))) for _ in range(6000)]
        _feed(c, ratios)
        assert c.cfl_number >= 0.1, (
            f"整体收敛的带噪声轨迹让 CFL 滑到了 {c.cfl_number:.4f} < cfl_start=0.1"
            "（棘轮效应）")

    def test_mild_worsening_is_ignored_while_window_still_improving(self):
        """窗口内仍有累计下降时，连续几步的小幅上升不得收缩。"""
        c = AdaptiveCFLController(cfl_start=0.25, cfl_max=0.5, cfl_min=0.05,
                                  trend_window=20, mild_shrink_confirm_steps=3)
        # 先攒满一个"确有下降"的窗口（累计 ~0.97），再来连续 4 步小幅上升
        _feed(c, [0.9985] * 21 + [1.005] * 4)
        assert c.cfl_number >= 0.25, (
            f"窗口仍在下降却因为 4 步小幅上升收缩到 {c.cfl_number:.4f}")

    def test_window_not_improving_plus_confirmation_does_shrink(self):
        """窗口内没有累计下降 + 连续确认 -> 必须收缩（保护不能被关掉）。"""
        c = AdaptiveCFLController(cfl_start=0.25, cfl_min=0.05,
                                  trend_window=20, mild_shrink_confirm_steps=3)
        _feed(c, [1.002] * 60)     # 一路缓慢恶化，窗口累计是上升的
        assert c.cfl_number < 0.25

    def test_severe_shrink_never_gated_by_window(self):
        """即使窗口显示在下降，重度恶化也必须立即收缩。"""
        c = AdaptiveCFLController(cfl_start=0.4, cfl_min=0.05, cfl_max=0.5,
                                  trend_window=20, cooldown_steps=0)
        _feed(c, [0.998] * 25)     # 窗口里是明确的下降
        before = c.cfl_number
        c.update(c._history[-1][2] * 1.5)   # ratio=1.5 -> 重度
        assert c.cfl_number == pytest.approx(before * 0.8), (
            "重度恶化被窗口判据拦住了——安全底线被破坏")


class TestHysteresisNoLimitCycle:
    """迟滞带：CFL 不得在稳定边界上形成极限环。

    实测（128 单元压力扰动算例，做完前三处修复后）：CFL 稳定下来之后
    在 0.106 <-> 0.120 之间每约 26 步往复一次，grow_trend 与 shrink_mild
    交替刷日志。原因是两个方向共用同一个窗口阈值 0.995——累计 < 0.995
    放大、>= 0.995 收缩，边界上必然抖动。现在收缩改用独立阈值 1.0（窗口
    **确实变差**才收缩），带内保持不动。
    """

    def test_no_oscillation_in_hysteresis_band(self):
        """窗口累计稳定落在 0.995~1.0 带内：一次调节都不应发生。"""
        c = AdaptiveCFLController(cfl_start=0.2, cfl_max=0.5, cfl_min=0.05)
        # 每步 ratio 使 20 步累计 ≈ 0.997（落在带内），且逐步交替越过 1.0
        # 以便同时压到 shrink_mild 的连续确认路径上
        q = 0.997 ** (1.0 / 20.0)
        ratios = []
        for i in range(2000):
            ratios.append(q * (1.0006 if i % 3 == 0 else 0.9997))
        _feed(c, ratios)
        assert c.cfl_number == pytest.approx(0.2), (
            f"落在迟滞带内却调节了 CFL：{c.cfl_number:.4f}（极限环）")
        assert len(c.history) == 2000

    def test_grow_still_fires_above_band(self):
        """累计下降超过放大阈值时仍然放大（迟滞不能把放大也关掉）。"""
        c = AdaptiveCFLController(cfl_start=0.1, cfl_max=0.5)
        _feed(c, [REAL_SLOW_RATIO] * 400)
        assert c.cfl_number > 0.1

    def test_shrink_still_fires_below_band(self):
        """窗口累计真的变差（> 1.0）且连续确认满足时仍然收缩。"""
        c = AdaptiveCFLController(cfl_start=0.3, cfl_min=0.05, cfl_max=0.5)
        _feed(c, [1.003] * 200)
        assert c.cfl_number < 0.3


class TestSoftCeilingPreventsRepeatedOvershoot:
    """软上限：控制器必须记住失败过的 CFL，不再周期性穿越稳定边界。

    这一条是被 79 万单元真实网格的长程对照数据逼出来的（见模块文档
    第 8 条）：只做到第 4~7 条时，控制器每隔约 trend_window 步就重新
    探过稳定边界一次，每次都要付一段残差过冲的代价。实测该算例上
    CFL 0.100 -> 0.195（过冲、残差涨 9%）-> 0.175 -> 0.174（再过冲）
    -> 0.156，残差长期在 3.2e8 附近 churn；而 CFL 冻结在 0.100 的对照
    同期稳步降到 2.961e8——也就是说那时这套机制在该算例上是**净负**的。

    下面用例把"失败过的 CFL 不会被再次越过"和"上限从上方收敛而不是
    一步退太远"一起钉住，并保留"真实持续恶化仍要一路收缩到下限"这条
    安全底线。
    """

    def test_growth_never_exceeds_failed_cfl(self):
        """一次收缩之后，放大不得再回到收缩前的那个 CFL。

        `ceiling_release_steps` 取一个大到不会触发的值：这里要验证的是
        **上限约束**本身；上限的缓慢释放是另一条独立行为，由
        `test_ceiling_is_released_after_long_quiet_period` 单独验证。
        两者混在一个用例里会互相掩盖。
        """
        c = AdaptiveCFLController(cfl_start=0.1, cfl_max=1.0, cfl_min=0.01,
                                  ceiling_backoff=0.95,
                                  ceiling_release_steps=10 ** 9)
        # 先靠持续的慢速下降把 CFL 爬起来
        _feed(c, [REAL_SLOW_RATIO] * 300)
        cfl_peak = c.cfl_number
        assert cfl_peak > 0.12, "本用例需要 CFL 先真的爬起来"

        # 制造一次明显恶化 -> 收缩，并记下软上限
        c.update(c._history[-1][2] * 1.5)
        assert c._cfl_ceiling is not None, "收缩没有设置软上限"
        expected_ceiling = cfl_peak * 0.95
        assert c._cfl_ceiling == pytest.approx(expected_ceiling), (
            f"软上限应为收缩前 CFL 的 0.95 倍（{expected_ceiling:.5f}），"
            f"实际 {c._cfl_ceiling:.5f}")

        # 之后再怎么持续下降，也不能越过软上限
        _feed(c, [REAL_SLOW_RATIO] * 3000)
        assert c.cfl_number <= expected_ceiling + 1e-12, (
            f"放大越过了失败过的 CFL：{c.cfl_number:.5f} > "
            f"{expected_ceiling:.5f}——会重新穿越稳定边界")

    def test_ceiling_ratchets_down_geometrically(self):
        """反复失败时软上限应几何式下降（从上方逼近边界），而不是一次
        退到远低于边界的位置。"""
        c = AdaptiveCFLController(cfl_start=0.2, cfl_max=1.0, cfl_min=0.01,
                                  ceiling_backoff=0.95,
                                  ceiling_release_steps=10 ** 9)
        ceilings = []
        for _ in range(4):
            _feed(c, [REAL_SLOW_RATIO] * 200)      # 爬到上限附近
            c.update(c._history[-1][2] * 1.5)      # 再失败一次
            ceilings.append(c._cfl_ceiling)
        assert all(b < a for a, b in zip(ceilings, ceilings[1:])), (
            f"软上限没有随反复失败继续下降：{ceilings}")
        # 每次只降一点（不是崩到下限）
        for a, b in zip(ceilings, ceilings[1:]):
            assert b > 0.5 * a, f"软上限一次降得太狠：{a:.5f} -> {b:.5f}"

    def test_ceiling_does_not_block_shrinking(self):
        """软上限只约束**放大**；真实持续恶化仍必须一路收缩到下限。"""
        c = AdaptiveCFLController(cfl_start=0.3, cfl_min=0.05, cfl_max=0.5,
                                  ceiling_backoff=0.95)
        _feed(c, [1.05] * 300)
        assert c.cfl_number == pytest.approx(0.05), (
            "软上限把收缩也挡住了——安全底线被破坏")

    def test_ceiling_is_released_after_long_quiet_period(self):
        """长期没有收缩时软上限必须**缓慢放开**一档。

        为什么必须有释放（实测逼出来的）：只有"永久上限"时，带噪声的
        收敛轨迹里偶发的收缩会把上限不断压低、放大再也回不去——噪声
        sigma=0.4%、均值每步降 0.05% 的 6000 步轨迹实测 CFL 一路滑到
        0.034（< cfl_start=0.1），等于把第 5/6 条修掉的棘轮效应换了个
        形式又引回来。
        """
        c = AdaptiveCFLController(cfl_start=0.1, cfl_max=1.0, cfl_min=0.01,
                                  ceiling_backoff=0.95,
                                  ceiling_release_steps=50)
        _feed(c, [REAL_SLOW_RATIO] * 200)
        c.update(c._history[-1][2] * 1.5)          # 触发一次收缩 -> 设上限
        ceiling0 = c._cfl_ceiling
        assert ceiling0 is not None

        # 之后一段安静（持续下降、无恶化）区段：上限应被放开
        _feed(c, [REAL_SLOW_RATIO] * 60)
        released = c._cfl_ceiling
        assert released is None or released > ceiling0 + 1e-12, (
            f"安静 60 步（> ceiling_release_steps=50）后软上限没有放开："
            f"{ceiling0:.5f} -> {released}")

    def test_ceiling_stays_at_cfl_start_while_shrinks_come_from_above(self):
        """**只要每次收缩都发生在 cfl_start 之上**，软上限不得低于
        cfl_start——那是调用方断言过稳定的保守起始值，禁止 CFL 回到自己
        的起点又是一种棘轮。

        这是原 `test_ceiling_never_below_cfl_start` 的**受限版本**：那条
        用例断言的是"上限在任何情况下都不低于 cfl_start"，已被真实数据
        证伪，见下一条用例的说明。这里把它仍然成立的那一半保留下来。
        """
        c = AdaptiveCFLController(cfl_start=0.2, cfl_max=1.0, cfl_min=0.01,
                                  ceiling_backoff=0.95,
                                  ceiling_release_steps=10 ** 9)
        _feed(c, [REAL_SLOW_RATIO] * 200)
        assert c.cfl_number > 0.2, "本用例需要 CFL 先爬到 cfl_start 之上"
        for _ in range(3):
            if c.cfl_number < 0.2:
                break                      # 已经掉到 cfl_start 以下，超出本用例范围
            c.update(c._history[-1][2] * 1.5)
            assert c._cfl_ceiling >= 0.2 - 1e-12, (
                f"收缩发生在 cfl_start 之上，软上限却跌到了 "
                f"{c._cfl_ceiling:.5f} < 0.2")
            _feed(c, [REAL_SLOW_RATIO] * 30)

    def test_ceiling_follows_down_when_shrink_happens_below_cfl_start(self):
        """一旦收缩发生在 **cfl_start 以下**，软上限必须跟着下来——不能
        继续钉在一个**已经被证伪**的 cfl_start 上。

        **真实 bug 修复（2026-09-15）**：原实现是
        `ceiling = max(old_cfl*backoff, cfl_start)`，于是当真实稳定 CFL
        落在 cfl_start 以下时，上限恒等于 cfl_start、第 8 条那道保护被
        完全抵消。79 万单元 cube_demo 上（P1 真实内容、模态滤波器关闭，
        稳定 CFL 约 0.063 而 cfl_start=0.1）复现出来的就是第 8 条本该
        消灭的"周期性穿越稳定边界"：

            step 66  CFL 爬到 0.0697
            step 79  被判定过高 -> 退回 0.0627（连续 23 步单调收敛）
            step 100 **又爬回 0.0690**
            step 102 残差一步从 2.17e9 跳到 1.87e25
            step 103 发散

        修复前后两次运行的 CFL 历史逐位相同、死亡步号也相同（103），
        是这条因果最直接的证据。

        判据同时钉住"不构成新棘轮"：上限必须 >= 收缩后的 cfl_number，
        也就是绝不挡住"停在当前值"。恢复由 ceiling_release_steps 负责
        （见 test_ceiling_is_released_after_long_quiet_period）。
        """
        c = AdaptiveCFLController(cfl_start=0.2, cfl_max=1.0, cfl_min=0.01,
                                  ceiling_backoff=0.5,
                                  ceiling_release_steps=10 ** 9)
        _feed(c, [REAL_SLOW_RATIO] * 100)
        saw_shrink_below_start = False
        for _ in range(6):
            old = c.cfl_number
            c.update(c._history[-1][2] * 1.5)
            if old < 0.2:
                saw_shrink_below_start = True
                assert c._cfl_ceiling < 0.2, (
                    f"收缩发生在 cfl_start 以下（old_cfl={old:.5f}），软上限"
                    f"却仍是 {c._cfl_ceiling:.5f} >= cfl_start=0.2——那个值"
                    f"已经被证伪，继续用它兜底就是把第 8 条的保护抵消掉")
                assert c._cfl_ceiling >= c.cfl_number - 1e-12, (
                    f"软上限 {c._cfl_ceiling:.5f} 低于收缩后的 CFL "
                    f"{c.cfl_number:.5f}——挡住了'停在当前值'，构成新棘轮")
            _feed(c, [REAL_SLOW_RATIO] * 10)
        assert saw_shrink_below_start, (
            "这个算例没有制造出'在 cfl_start 以下收缩'的情形，判据失去意义")

    def test_reset_clears_ceiling(self):
        """阶数切换后软上限必须作废（换了离散问题，旧边界不再适用）。"""
        c = AdaptiveCFLController(cfl_start=0.1, cfl_max=0.5)
        _feed(c, [REAL_SLOW_RATIO] * 200)
        c.update(c._history[-1][2] * 1.5)
        assert c._cfl_ceiling is not None
        c.reset()
        assert c._cfl_ceiling is None, "reset 没有清掉软上限"
        _feed(c, [REAL_SLOW_RATIO] * 2000)
        assert c.cfl_number == pytest.approx(0.5), (
            "reset 之后 CFL 仍被旧软上限卡住，到不了 cfl_max")
