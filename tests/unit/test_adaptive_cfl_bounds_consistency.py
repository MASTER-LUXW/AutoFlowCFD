"""自适应 CFL 三个边界参数的自洽性（`adaptive_cfl.py` 模块文档第 11 条）。

## 修的是什么

收缩分支原先写的是

    self.cfl_number = max(self.cfl_number * factor, self.cfl_min)

只钳下界、**完全不看 `cfl_max`**。于是当调用方刻意把 `cfl_max`（连带
`cfl_start`）设到默认 `cfl_min=0.05` 以下时——做低 CFL 稳定边界扫描正需要
这样——第一次收缩就会

    max(0.045 * 0.9, 0.05) = 0.05 > cfl_max = 0.045

把 CFL **调高**，日志还照旧标成 `shrink_mild`。真实运行日志原文：

    [AdaptiveCFL] Step 25: CFL 0.045 → 0.050 (shrink_mild, ratio=1.057)

后果不只是标签不对：软上限（模块文档第 8 条）只约束**放大**路径，所以这条
通道能绕过全部越界保护，把 CFL 推到调用方明确禁止的值上。实测让一条
791,492 单元的 CFL 0.045 探针从 step 25 起变成了 0.05 的运行，整段作废。

与 2026-08-31 那次修复是**同一个表达式的两种病**：那次是 `cfl_number` 已经
等于 `cfl_min` 时结果被 clip 成原值、却照样打印"已调节"（空操作）；这次是
`cfl_number` 低于 `cfl_min` 时结果被 clip 成**更大**的值（反向操作）。

## 判据

`cfl_max` 是硬上限：**任何**路径、**任何**配置下 `cfl_number` 都不得超过它。
本文件对收缩/放大/趋势放大/软上限四条路径分别钉住这一点。
"""

import numpy as np
import pytest

from autoflowcfd.core.time_integration.adaptive_cfl import AdaptiveCFLController


def _rising(n, start=2.0e8, rate=1.06):
    """持续上升的残差序列——驱动 shrink 分支。"""
    return [start * rate ** k for k in range(n)]


def _falling(n, start=1.0e9, rate=0.9):
    """快速下降的残差序列——驱动 grow 分支。"""
    return [start * rate ** k for k in range(n)]


class TestConstructorReconcilesBounds:
    """`cfl_min > cfl_max` 是自相矛盾的配置，以 cfl_max 为准。"""

    def test_cfl_min_is_clamped_to_cfl_max(self):
        c = AdaptiveCFLController(cfl_start=0.045, cfl_max=0.045)
        assert c.cfl_min == pytest.approx(0.045)
        assert c.cfl_max == pytest.approx(0.045)

    def test_warning_is_emitted_not_silent(self):
        """本项目不接受静默兜底——必须留下痕迹。"""
        import inspect

        from autoflowcfd.core.time_integration import adaptive_cfl
        src = inspect.getsource(adaptive_cfl.AdaptiveCFLController.__init__)
        assert "cfl_min > self.cfl_max" in src.replace("self.", "self.")
        assert "logger.warning" in src

    def test_cfl_start_clamped_into_range(self):
        c = AdaptiveCFLController(cfl_start=0.5, cfl_min=0.01, cfl_max=0.08)
        assert c.cfl_start == pytest.approx(0.08)
        assert c.cfl_number == pytest.approx(0.08)

    def test_cfl_start_below_min_clamped_up(self):
        c = AdaptiveCFLController(cfl_start=0.001, cfl_min=0.01, cfl_max=0.08)
        assert c.cfl_start == pytest.approx(0.01)

    def test_consistent_config_is_untouched(self):
        """正常配置必须逐位不变（这条修复只在矛盾配置下生效）。"""
        c = AdaptiveCFLController(cfl_start=0.1, cfl_min=0.05, cfl_max=0.5)
        assert (c.cfl_start, c.cfl_min, c.cfl_max) == (0.1, 0.05, 0.5)


class TestShrinkNeverRaisesCfl:
    """名为"收缩"的分支绝不能把 CFL 调高。"""

    def test_reproduces_the_real_log_scenario(self):
        """复现真实日志那一行的场景：cfl_max 低于默认 cfl_min，残差上升。

        这是 fail-then-pass 的 pass 半边；fail 半边见
        `test_unfixed_expression_would_have_raised_it` 用同一表达式手算。
        """
        c = AdaptiveCFLController(cfl_start=0.045, cfl_max=0.045)
        hist = []
        for r in [3.0e8, 2.0e8] + _rising(20):
            hist.append(c.update(r))
        assert max(hist) <= 0.045 + 1e-12, f"CFL 越过 cfl_max：max={max(hist)}"

    def test_unfixed_expression_would_have_raised_it(self):
        """把原表达式单独算一遍，确认这个场景确实会触发缺陷——否则上面
        那条测试可能根本没走到收缩分支，形成假通过。"""
        cfl_number, cfl_min, factor = 0.045, 0.05, 0.9
        unfixed = max(cfl_number * factor, cfl_min)
        assert unfixed == pytest.approx(0.05)
        assert unfixed > 0.045, "原表达式在此配置下确实会把 CFL 调高"
        fixed = min(max(cfl_number * factor, cfl_min), 0.045)
        assert fixed == pytest.approx(0.045)

    @pytest.mark.parametrize("cfl", [0.045, 0.03, 0.02, 0.01])
    def test_pinned_probe_stays_pinned(self, cfl):
        """`cfl_start == cfl_max` 的定 CFL 探针必须全程恒定——这是稳定
        边界扫描的前提，否则测出来的不是所请求的那个 CFL。"""
        c = AdaptiveCFLController(cfl_start=cfl, cfl_max=cfl)
        vals = [c.update(r) for r in [3.0e8, 2.0e8] + _rising(30)]
        assert min(vals) == pytest.approx(cfl)
        assert max(vals) == pytest.approx(cfl)

    def test_shrink_still_works_when_room_exists(self):
        """反向确认：下界留有空间时收缩必须照常发生（没有把机制修死）。"""
        c = AdaptiveCFLController(cfl_start=0.1, cfl_min=0.01, cfl_max=0.5)
        for r in [3.0e8, 2.0e8] + _rising(30):
            c.update(r)
        assert c.cfl_number < 0.1


class TestCflMaxIsRespectedOnEveryPath:
    """四条路径（grow / crawl / grow_trend / shrink）都不得越过 cfl_max。"""

    @pytest.mark.parametrize("series_name", ["falling", "rising", "flat", "noisy"])
    def test_never_exceeds_cfl_max(self, series_name):
        rng = np.random.default_rng(5)
        series = {
            "falling": _falling(300),
            "rising": [2.0e8] + _rising(299),
            "flat": [1.0e9 * (1.0 - 1e-4) ** k for k in range(300)],
            "noisy": list(np.cumprod(
                np.concatenate([[1.0e9], 0.999 * (1 + rng.uniform(-0.004, 0.004, 299))]))),
        }[series_name]
        c = AdaptiveCFLController(cfl_start=0.04, cfl_min=0.01, cfl_max=0.06)
        peak = max(c.update(float(r)) for r in series)
        assert peak <= 0.06 + 1e-12, f"{series_name}: CFL 峰值 {peak} 越过 cfl_max"

    def test_soft_ceiling_floor_also_capped(self):
        """软上限自身的地板（第 8 条）同样要被 cfl_max 收口，否则上限能
        被顶到 cfl_max 之上、再由放大路径跟上去。"""
        c = AdaptiveCFLController(cfl_start=0.045, cfl_max=0.045)
        for r in [3.0e8, 2.0e8] + _rising(20):
            c.update(r)
        if c._cfl_ceiling is not None:
            assert c._cfl_ceiling <= 0.045 + 1e-12

    def test_shrink_expression_is_two_sided_in_source(self):
        """结构判据：收缩那一行必须同时出现 min 与 max 的双向钳制。"""
        import inspect

        from autoflowcfd.core.time_integration import adaptive_cfl
        src = inspect.getsource(adaptive_cfl)
        assert "max(self.cfl_number * factor, self.cfl_min)), self.cfl_max" not in src
        # 双向钳制的实际形态（跨行）
        flat = " ".join(src.split())
        assert ("self.cfl_number = min( max(self.cfl_number * factor, self.cfl_min), "
                "self.cfl_max)") in flat, "收缩分支没有双向钳制"
