"""
AutoFlowCFD V2.0 - 残差变 inf/nan 时求解循环必须立刻中止。

## 真实事故（2026-09-16）

plate_demo_volume_les（179,237 单元）上一条 AUSM+up 预处理档对照运行：

    P1 Iter 102: Residual = 7.859027e+08  Cd=1.7473
    P1 Iter 103: Residual = 2.160498e+09  Cd=3.09950276e+48   <- 炸了
    P1 Iter 104: Residual = inf           Cd=nan
    P1 Iter 105: Residual = nan           Cd=nan
    P1 Iter 106..109: Residual = nan                          <- 还在跑

求解循环毫不在意地继续迭代，直到人工发现。两处真实损害：

1. 白烧机时（每步 13s，一条 400 步的运行会在 NaN 上空转近一小时）；
2. 每一步都照常调用 `checkpoint_callback`，把 NaN 状态写进 checkpoint
   并在收尾时用 NaN 覆盖 `final_state.pkl`——本来磁盘上那几个正常的
   中间 checkpoint 是唯一可用的残骸。

排查后发现这是**路径不对等**而非有意设计：GPU 单机（`gpu_solver.py`）与
多 GPU（`gpu_distributed.py`）本来就有 `np.isfinite(res)` 检查，而
三条 CPU 路径

    core/fr_solver/solver.py::FRSolver.solve                （P0/P1 常规循环）
    core/utils/order_continuation.py::run_order_continuation（全部 P2/P3 运行）
    core/mpi/distributed_order_continuation.py              （CPU MPI 分布式）

一处都没有。其中 `run_order_continuation` 是所有 P2/P3 运行实际走的路径
——项目记忆里那条"P2 第 4 步 inf"的运行就是在 inf 上继续迭代到预算耗尽。
多 GPU 那条虽然有检查，但顺序是**先保存 checkpoint 再检查**，所以发散
那一步的 NaN 依然会被如实写盘。

本文件把"必须抛 SolverDivergedError"与"检查必须早于 checkpoint 回调"
两件事都钉住。
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_solver.residual_diagnostics import (
    SolverDivergedError,
    check_residual_finite,
)


class TestCheckResidualFinite:
    """守卫函数本身的行为。"""

    @pytest.mark.parametrize('res', [0.0, 1e-30, 1.0, 8.4e8, 1e300])
    def test_finite_passes_silently(self, res):
        check_residual_finite(res, iteration=7, order=1, last_finite=1.0)

    @pytest.mark.parametrize('res', [np.inf, -np.inf, np.nan])
    def test_nonfinite_raises(self, res):
        with pytest.raises(SolverDivergedError):
            check_residual_finite(res, iteration=104, order=1, last_finite=7.859e8)

    def test_message_carries_iteration_order_and_last_finite(self):
        """报错信息必须自带定位信息。

        发散时用户手上通常只有一份日志尾巴，异常文本要能独立回答
        "第几步、什么阶数、上一个正常值是多少、接下来该动哪个旋钮"。
        """
        with pytest.raises(SolverDivergedError) as ei:
            check_residual_finite(np.nan, iteration=104, order=2,
                                  last_finite=7.859027e8)
        msg = str(ei.value)
        assert 'P2' in msg
        assert 'Iter 104' in msg
        assert '7.859027e+08' in msg
        # 三条常见成因都要在场（CFL / 网格质量门 / AUSM+up 预处理档）
        assert 'CFL' in msg
        assert '--skip-quality-check' in msg
        assert 'AFCFD_AUSM_PRECOND_MODE' in msg
        # 必须明说不会再写 checkpoint，否则用户会怀疑磁盘上的残骸是否已被污染
        assert 'checkpoint' in msg

    def test_extra_hint_is_included(self):
        with pytest.raises(SolverDivergedError, match='allreduce'):
            check_residual_finite(np.inf, iteration=3,
                                  extra_hint='分布式路径：残差是全域 allreduce 值')

    def test_order_optional(self):
        """不区分阶数的调用方（GPU 单机/多 GPU）可以不传 order。"""
        with pytest.raises(SolverDivergedError) as ei:
            check_residual_finite(np.nan, iteration=5)
        assert 'Iter 5' in str(ei.value)
        assert 'P' not in str(ei.value).split('Iter')[0].strip()

    def test_is_a_runtime_error(self):
        """继承 RuntimeError：已有的 `except Exception` / `except RuntimeError`
        兜底路径（CLI 各命令）不需要为它单独改动就能正确报错退出。"""
        assert issubclass(SolverDivergedError, RuntimeError)


class TestAllSolveLoopsAreGuarded:
    """静态检查：五条求解循环都必须调用这个守卫。

    用源码文本检查而不是跑真实求解——让一条求解循环真的发散需要一个
    真实网格和上百步迭代，不适合放进单元测试；而"有没有接这个守卫"
    是个结构性事实，文本检查就足够，且能在新增第六条循环时提醒作者。
    """

    #: 按**模块名**而不是文件路径：本项目把超 500 行的模块陆续拆成子包，
    #: 硬编码 `.py` 路径在拆包后直接 FileNotFoundError（2026-09-24 真实
    #: 踩到 distributed_order_continuation 这一项）。
    LOOPS = [
        'autoflowcfd.core.fr_solver.solver',
        'autoflowcfd.core.utils.order_continuation',
        'autoflowcfd.core.mpi.distributed_order_continuation',
        'autoflowcfd.core.gpu.solver.gpu_solver',
        'autoflowcfd.core.gpu.distributed.gpu_distributed',
    ]

    @pytest.mark.parametrize('rel', LOOPS)
    def test_loop_calls_guard(self, rel):
        from tests.unit._module_source import module_source

        src = module_source(rel)
        assert 'check_residual_finite(' in src, (
            f"{rel} 的求解循环没有接 check_residual_finite —— 残差变 NaN 后"
            f"会继续迭代并把 NaN 写进 checkpoint"
        )

    @pytest.mark.parametrize('rel', LOOPS)
    def test_guard_precedes_checkpoint_callback(self, rel):
        """守卫必须出现在 `checkpoint_callback(` 之前。

        多 GPU 路径此前就是顺序错了（先保存再检查），检查在也没用。

        判定从循环体里那条 `res = ....step(dt)` 起算，而不是从文件开头
        ——`gpu_distributed.py` 的 `solve` **文档字符串**里就提到了
        `checkpoint_callback(solver, iteration)` 这个回调约定，按文件
        开头找首次出现会命中文档而不是调用（第一版判据就踩了这个，
        报了一个不存在的顺序错误）。
        """
        import re

        from tests.unit._module_source import module_sources

        # **逐个子模块**看，而不是拼接后再找：顺序类断言在拼接后没有意义
        # （拼接顺序是 pkgutil 字母序，与代码语义无关）。先定位到真正含有
        # 求解循环的那个子模块，再在它内部判顺序。
        src, m = None, None
        for _name, _s in module_sources(rel):
            _m = re.search(r'res = (?:self|solver)\.step\(', _s)
            if _m is not None:
                src, m = _s, _m
                break
        assert m is not None, f'{rel} 里找不到求解循环的残差求值语句'
        body = src[m.start():]

        if 'checkpoint_callback(' not in body:
            pytest.skip(f'{rel} 的循环体里没有 checkpoint 回调')
        assert 'check_residual_finite(' in body, (
            f'{rel} 的循环体里没有 check_residual_finite'
        )
        assert body.index('check_residual_finite(') < body.index('checkpoint_callback('), (
            f"{rel} 里 check_residual_finite 出现在 checkpoint_callback 之后 —— "
            f"发散那一步的 NaN 状态仍会被写盘"
        )


class TestSolveLoopActuallyAborts:
    """行为级验证：不只是"源码里有那行调用"，而是循环真的停下来、
    且发散那一步**没有**调用 checkpoint 回调。

    用一个只实现 `FRSolver.solve` 循环体所需属性的替身 + 未绑定方法调用
    （`FRSolver.solve(stub, ...)`），而不是构造真实求解器：让真实求解器
    发散需要一个真实网格与上百步迭代（本项目实测 plate_demo_volume_les
    上是第 103 步），不适合放进单元测试。替身只需要提供循环体实际读到的
    那几个属性 —— 少一个就会 AttributeError，所以这个替身不会因为
    "假装得太少"而让测试失去意义（这是本项目"测试替身不完整"缺陷类的
    反面做法）。
    """

    def _make_stub(self, residuals):
        from types import SimpleNamespace

        from autoflowcfd.core.time_integration.base import (
            TimeIntegrationScheme,
            TimeIntegrator,
        )

        calls = {'step': 0, 'checkpoint': []}

        def step(dt):
            i = calls['step']
            calls['step'] += 1
            return residuals[i]

        stub = SimpleNamespace(
            step=step,
            residual_history=[],
            order=1,
            turb_model_name='NONE',
            order_continuation_enabled=False,
            time_integrator=TimeIntegrator(scheme=TimeIntegrationScheme.SSP_RK3),
            _cfl_controller=None,
            _reference_area=None,
            freestream=None,
            state=SimpleNamespace(),
        )
        return stub, calls

    def test_aborts_on_nan_and_skips_checkpoint(self):
        from autoflowcfd.core.fr_solver.solver import FRSolver

        stub, calls = self._make_stub([8.4e8, 7.9e8, float('nan'), 1.0, 1.0])
        seen = []

        with pytest.raises(SolverDivergedError) as ei:
            FRSolver.solve(stub, max_iter=5, dt=1e-4, tol=0.0,
                           checkpoint_callback=lambda s, it: seen.append(it))

        # 第 3 步（1 起算）是 nan，循环必须在那一步抛出
        assert calls['step'] == 3, f"发散后仍继续迭代：step 被调了 {calls['step']} 次"
        assert 'Iter 3' in str(ei.value)
        # 前两步正常，回调应当只收到 1 和 2 —— 第 3 步（nan）不能落盘
        assert seen == [1, 2], f"发散那一步仍调用了 checkpoint 回调：{seen}"
        # 上一个有限残差要如实报出来
        assert '7.900000e+08' in str(ei.value)

    def test_finite_run_is_unaffected(self):
        """全程有限时行为完全不变（守卫不能改变正常路径）。"""
        from autoflowcfd.core.fr_solver.solver import FRSolver

        stub, calls = self._make_stub([8.4e8, 7.9e8, 7.0e8, 6.5e8])
        seen = []
        result = FRSolver.solve(stub, max_iter=4, dt=1e-4, tol=0.0,
                                checkpoint_callback=lambda s, it: seen.append(it))
        assert calls['step'] == 4
        assert seen == [1, 2, 3, 4]
        assert result.iterations == 4
        assert result.final_residual == 6.5e8
        assert stub.residual_history == [8.4e8, 7.9e8, 7.0e8, 6.5e8]

    @pytest.mark.parametrize('bad', [float('inf'), float('-inf'), float('nan')])
    def test_aborts_on_every_nonfinite_kind(self, bad):
        from autoflowcfd.core.fr_solver.solver import FRSolver

        stub, calls = self._make_stub([8.4e8, bad, 1.0])
        with pytest.raises(SolverDivergedError):
            FRSolver.solve(stub, max_iter=3, dt=1e-4, tol=0.0)
        assert calls['step'] == 2


class TestCellVolumePercentileDiagnostic:
    """最大残差单元的体积分位必须出现在诊断行里。

    动机（2026-09-16 真实排查）：plate_demo_volume_les 上三条对照运行
    100 步里最大残差恒定落在同一个单元（cell18708），而判断"这是退化
    单元机制还是壁面处理机制"当时只能另写脚本重新加载整张体网格（约
    10 分钟）才算出该单元体积分位是 0.523%、超压点 81% 落在体积最小的
    1% 单元里。那个数字本该在日志里一眼看到。
    """

    def _diag(self, max_cell):
        import numpy as np

        from autoflowcfd.core.fr_solver.residual_diagnostics import ResidualDiagnostics

        return ResidualDiagnostics(
            rms_per_var=np.ones(5),
            scaled_rms_per_var=np.ones(5),
            max_abs=3.654e11,
            max_abs_cell=max_cell,
            max_abs_sp=6,
            max_abs_var=4,
        )

    def test_percentile_is_zero_for_smallest_and_hundred_for_largest(self):
        from autoflowcfd.core.fr_solver.residual_diagnostics import (
            cell_volume_percentile,
        )

        vols = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
        pct = cell_volume_percentile(vols)
        assert pct[1] == pytest.approx(0.0)      # 最小
        assert pct[0] == pytest.approx(100.0)    # 最大
        assert pct[3] == pytest.approx(25.0)
        assert pct[2] == pytest.approx(50.0)

    def test_line_includes_volume_percentile(self):
        from autoflowcfd.core.fr_solver.residual_diagnostics import (
            format_scaled_residual_line,
        )

        # 1000 个单元，第 7 个是全场最小 -> 分位 0.00%
        vols = np.linspace(1.0, 2.0, 1000)
        vols[7] = 1e-8
        line = format_scaled_residual_line(self._diag(7), cell_volumes=vols)
        assert 'cell7' in line
        assert 'vol 0.00%' in line

    def test_line_omits_percentile_without_volumes(self):
        """没有几何信息时不能报错，也不能编一个分位出来。"""
        from autoflowcfd.core.fr_solver.residual_diagnostics import (
            format_scaled_residual_line,
        )

        line = format_scaled_residual_line(self._diag(7))
        assert 'cell7' in line
        assert 'vol' not in line

    def test_percentile_handles_degenerate_inputs(self):
        """合成网格/测试替身可能不带 cell_volumes，或带一个空数组、
        多维数组 —— 都必须安静地退化为"没有信息"，而不是抛异常把整个
        求解循环打断（诊断行不该有能力弄崩求解）。"""
        from autoflowcfd.core.fr_solver.residual_diagnostics import (
            cell_volume_percentile,
        )

        assert cell_volume_percentile(None) is None
        assert cell_volume_percentile(np.array([])) is None
        assert cell_volume_percentile(np.zeros((3, 3))) is None
        # 单单元网格不能除以零
        assert cell_volume_percentile(np.array([1.0]))[0] == pytest.approx(0.0)

    def test_divergence_message_points_at_the_percentile_field(self):
        """发散报错必须告诉用户先去看那个字段，否则新增字段没人会用。"""
        with pytest.raises(SolverDivergedError) as ei:
            check_residual_finite(np.nan, iteration=104, order=1, last_finite=1.0)
        msg = str(ei.value)
        assert 'vol X%' in msg
        assert 'det(J)' in msg
