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


class TestEveryBackendConstructsAController:
    """五条求解器路径都必须构造自适应 CFL 控制器，并且都能接 cfl_min。

    2026-09-15 发现两个真实缺口：
      1. `cfl_min` 在**全部五处**控制器构造点都没有被传递，恒用默认 0.05
         ——而那个值高于真 P1 在 79 万单元 cube_demo 上实测稳定的 0.03，
         使一个已验证可用的工作点通过 CLI/API 根本到不了；
      2. 多 GPU **完全分布式**路径根本没有控制器
         （`gpu_distributed_fully_distributed.py` 只设
         `GPUTimeIntegrator(cfl=1.0)`，而 `_current_cfl()` 的回退正是
         `time_integrator.cfl`），于是它以 CFL=1.0 运行——远超 P>=1 的
         SSP-RK3 稳定极限；同一个类的"传统模式"一直是有控制器的。
    """

    def test_all_solver_entry_points_accept_cfl_min(self):
        import inspect

        from autoflowcfd.core.fr_solver.solver import FRSolver
        from autoflowcfd.core.gpu.distributed.gpu_distributed import (
            MultiGPUDistributedSolver,
        )
        from autoflowcfd.core.gpu.solver.gpu_solver import GPUFRSolver
        for cls in (FRSolver, GPUFRSolver, MultiGPUDistributedSolver):
            assert 'cfl_min' in inspect.signature(cls.__init__).parameters, cls.__name__

    def test_checkpoint_rebuild_accepts_cfl_min(self):
        """resume 低 CFL 工况同样需要它。"""
        import inspect

        from autoflowcfd.cli.solve_checkpoint_io import (
            rebuild_solver_from_checkpoint,
        )
        assert 'cfl_min' in inspect.signature(
            rebuild_solver_from_checkpoint).parameters

    def test_cli_exposes_cfl_min(self):
        from click.testing import CliRunner

        from autoflowcfd.cli.solve_steady_command import solve_steady
        out = CliRunner().invoke(solve_steady, ['--help']).output
        assert '--cfl-min' in out

    def test_distributed_package_carries_cfl_bounds(self):
        """完全分布式：三个量必须进 rank package，否则 CLI 选项在这条
        路径上被静默丢弃（与本模块记录过的 turb_model 同类缺口）。"""
        import inspect

        from autoflowcfd.core.mpi import distributed_mesh_loader as dml
        for fn in (dml.build_fully_distributed_rank_package,
                   dml.distributed_mesh_load_v2):
            params = inspect.signature(fn).parameters
            for k in ('cfl_start', 'cfl_max', 'cfl_min'):
                assert k in params, f"{fn.__name__} 缺 {k}"
        src = inspect.getsource(dml.build_fully_distributed_rank_package)
        for k in ('cfl_start', 'cfl_max', 'cfl_min'):
            assert f"'{k}': {k}" in src, f"package 里没写入 {k}"

    def test_multi_gpu_fully_distributed_builds_a_controller(self):
        """这条路径此前完全没有控制器，会以 CFL=1.0 运行。"""
        import inspect

        from autoflowcfd.core.gpu.distributed import (
            gpu_distributed_fully_distributed as fd,
        )
        src = inspect.getsource(fd)
        assert 'AdaptiveCFLController' in src
        assert '_cfl_controller' in src
        # 且必须是 None 感知地从 package 取三个边界值
        assert "('cfl_start', 0.1), ('cfl_max', 0.5), ('cfl_min', None)" in src

    def test_all_five_controller_sites_pass_cfl_min(self):
        """五处构造点逐一核对，防止将来新增后端时又漏一处。"""
        import inspect

        from autoflowcfd.core.fr_solver import solver as cpu_single
        from autoflowcfd.core.gpu.distributed import gpu_distributed as gpu_multi
        from autoflowcfd.core.gpu.distributed import (
            gpu_distributed_fully_distributed as gpu_multi_fd,
        )
        from autoflowcfd.core.gpu.solver import gpu_solver as gpu_single
        from autoflowcfd.core.mpi import distributed_solver as cpu_mpi
        for mod in (cpu_single, gpu_single, gpu_multi, gpu_multi_fd, cpu_mpi):
            src = inspect.getsource(mod)
            assert 'AdaptiveCFLController(' in src, mod.__name__
            assert 'cfl_min' in src, f"{mod.__name__} 的控制器构造没有接 cfl_min"


class TestConfigLayerCflDefaultsAreConsistent:
    """配置层与 CLI 是同一个物理量的两个入口，默认值不能互相矛盾。

    2026-09-15 发现：`SteadyConfig.cfl_max` 的默认值是 **10.0**，而 CLI
    `--cfl-max` 的默认是 0.5——差 20 倍；10.0 还是 SSP-RK3 线性稳定极限
    （~1.0）的 10 倍，而自适应控制器会真的往上限爬（软上限只在失败之后
    才收紧，见 adaptive_cfl.py 模块文档第 8 条），于是 YAML 驱动的算例会
    反复穿越稳定边界。文档示例里当时还写着 `cfl_max=5.0`。
    同时配置层完全没有 `cfl_min` 字段。
    """

    def test_cfl_max_default_matches_cli_and_stability_limit(self):
        from autoflowcfd.config.solver_config import SteadyConfig
        assert SteadyConfig().cfl_max == pytest.approx(0.5)
        # SSP-RK3 线性稳定极限 ~1.0：默认上限不得超过它
        assert SteadyConfig().cfl_max <= 1.0

    def test_cfl_min_field_exists_and_leaves_shrink_room(self):
        """下限必须严格小于初始值，否则控制器一步也收缩不了。"""
        from autoflowcfd.config.solver_config import SteadyConfig
        c = SteadyConfig()
        assert c.cfl_min < c.cfl_init, "cfl_min == cfl_init 会让收缩失效"

    def test_cfl_min_default_does_not_block_the_measured_stable_value(self):
        """0.03 是唯一在真实网格上被 250 步验证过稳定的 CFL；默认下限
        不得挡住它。"""
        from autoflowcfd.config.solver_config import SteadyConfig
        assert SteadyConfig().cfl_min <= 0.03

    @pytest.mark.parametrize("kw", [
        dict(cfl_min=0.2, cfl_init=0.05, cfl_max=0.5),   # min > init
        dict(cfl_min=0.6, cfl_init=0.6, cfl_max=0.5),    # min > max
        dict(cfl_min=0.0, cfl_init=0.05, cfl_max=0.5),   # min <= 0
    ])
    def test_inconsistent_ordering_rejected(self, kw):
        """矛盾配置要在配置层就拦下，不要等到控制器里变成"收缩把 CFL
        调高"（第 11 条那个缺陷）。"""
        from autoflowcfd.config.solver_config import SteadyConfig
        with pytest.raises(ValueError, match="CFL"):
            SteadyConfig(**kw)

    def test_low_cfl_config_is_expressible(self):
        """真 P1 的工作点必须能通过配置层表达出来。"""
        from autoflowcfd.config.solver_config import SteadyConfig
        c = SteadyConfig(cfl_min=0.01, cfl_init=0.03, cfl_max=0.5)
        assert (c.cfl_min, c.cfl_init, c.cfl_max) == (0.01, 0.03, 0.5)

    def test_api_forwards_cfl_min(self):
        import inspect

        from autoflowcfd import api
        src = inspect.getsource(api)
        assert 'kwargs.setdefault("cfl_min", config.cfl_min)' in src

    def test_docstring_example_is_not_above_the_stability_limit(self):
        """文档示例也不能给出一个越界的值——它会被照抄。"""
        from autoflowcfd.config.solver_config import SteadyConfig
        doc = SteadyConfig.__doc__
        assert 'cfl_max=5.0' not in doc
        assert 'cfl_max=0.5' in doc
