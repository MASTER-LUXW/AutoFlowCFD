"""`adaptive_cfl=False` 时请求的 CFL 必须真正生效，不能静默用 0.1。

## 缺口（真实缺陷，2026-09-17 发现）

`cfl.py::compute_local_time_step` 里这一行：

    CFL = _cfl_controller.cfl_number if _cfl_controller is not None else 0.1

`adaptive_cfl=False` 时构造函数不建控制器，于是 **CFL 静默回落到硬编码
0.1**，把调用方传进来的 `cfl_start/cfl_max/cfl_min` 整个丢掉。

发现它的方式：用平板边界层算例做 CFL 稳定边界扫描，"CFL 0.10" 与
"CFL 0.05" 两条运行给出**逐位相同**的残差轨迹、在同一步（3187）发散。
两个不同 CFL 不可能给出同一条轨迹——那是配置没生效的确证。

影响面：

  * 每一次 `adaptive_cfl=False` 的运行都跑在 0.1 上，与请求值无关；
  * `tests/validation/test_couette.py` 等全部关掉自适应的验证算例一直
    静默跑在 0.1（它们的结论"不发散"仍然成立，但"在 CFL X 下不发散"
    这半句此前是错的）；
  * 固定 CFL 是一条一等需求——稳定边界扫描与 A/B 对照都靠它，而它是
    本项目定默认值的唯一手段。

这正是项目标准禁止的那类**静默兜底**：一个明确的请求被无声丢弃，没有
任何日志痕迹。

## 修法

  * 构造函数在关闭自适应时记下 `solver.fixed_cfl_number = cfl_start`，
    并在启动日志里打印出来；
  * `cfl.py` 按"控制器 > `fixed_cfl_number` > 0.1"取值，且最后一档只在
    替身对象上才会走到，并且**打警告**，不再静默。

## 本文件覆盖

1. dt 与请求的 CFL 严格成正比（这是"生效"的定量判据，不是"不等于默认值"
   这种弱判据）
2. `fixed_cfl_number` 被设上且等于请求值
3. 自适应开启时仍然走控制器（没有把新分支塞到错误的优先级上）
4. 替身对象（既无控制器也无 fixed_cfl_number）会打警告
5. 源码级判据：那个裸 `else 0.1` 不得再出现
"""

import numpy as np
import pytest


def _channel_solver(cfl, adaptive):
    import sys
    from pathlib import Path

    tests_dir = str(Path(__file__).resolve().parents[1])
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    from validation._channel_mesh import build_channel_mesh_prism

    from autoflowcfd.core.fr_solver import FRSolver
    from autoflowcfd.core.time_integration import TimeIntegrationScheme

    mesh = build_channel_mesh_prism(1, 4, 3, 1, 1.0, 1.0, 0.25)
    s = FRSolver(mesh=mesh, order=1, turb_model_name="NONE", n_vars=5,
                 time_scheme=TimeIntegrationScheme.SSP_RK3,
                 rho_inf=1.225, vel_inf=30.0, p_inf=101325.0,
                 adaptive_cfl=adaptive,
                 cfl_start=cfl, cfl_max=cfl if not adaptive else 10 * cfl,
                 cfl_min=cfl / 10.0)
    s.order_continuation_enabled = False
    return s


def _dt_median(solver):
    from autoflowcfd.core.fr_solver.cfl import compute_local_time_step

    return float(np.median(np.asarray(compute_local_time_step(solver))))


class TestFixedCflActuallyChangesDt:
    """定量判据：dt 必须与请求的 CFL 严格成正比。"""

    @pytest.mark.parametrize("cfl", [0.01, 0.02, 0.05, 0.2])
    def test_dt_is_proportional_to_requested_cfl(self, cfl):
        ref_cfl = 0.1
        dt_ref = _dt_median(_channel_solver(ref_cfl, adaptive=False))
        dt = _dt_median(_channel_solver(cfl, adaptive=False))
        assert dt / dt_ref == pytest.approx(cfl / ref_cfl, rel=1e-10), (
            f"请求 CFL={cfl} 时 dt/dt(0.1) = {dt/dt_ref:.6f}，"
            f"应为 {cfl/ref_cfl:.6f}——CFL 没有真正生效")

    def test_two_different_cfls_never_give_the_same_dt(self):
        """这条直接对应发现缺陷的那个现象：两个不同 CFL 给出同一条轨迹。"""
        a = _dt_median(_channel_solver(0.05, adaptive=False))
        b = _dt_median(_channel_solver(0.10, adaptive=False))
        assert a != b
        assert b / a == pytest.approx(2.0, rel=1e-10)

    def test_fixed_cfl_number_is_recorded(self):
        s = _channel_solver(0.037, adaptive=False)
        assert getattr(s, "fixed_cfl_number", None) == pytest.approx(0.037)

    def test_adaptive_path_still_uses_the_controller(self):
        """没有把新分支塞到错误的优先级上：自适应开启时以控制器为准。"""
        s = _channel_solver(0.04, adaptive=True)
        assert s._cfl_controller is not None
        dt0 = _dt_median(s)
        # 手动改控制器的当前 CFL，dt 必须跟着变（说明读的是控制器）
        s._cfl_controller.cfl_number = 0.08
        dt1 = _dt_median(s)
        assert dt1 / dt0 == pytest.approx(2.0, rel=1e-10)


class TestStandInFallbackIsLoud:
    """既无控制器也无 fixed_cfl_number 的替身对象：可以回退，但必须出声。"""

    def test_fallback_warns_once(self, monkeypatch):
        from loguru import logger

        from autoflowcfd.core.fr_solver.cfl import compute_local_time_step

        s = _channel_solver(0.05, adaptive=False)
        # 拆掉两个来源，模拟诊断脚本/测试 stub
        s._cfl_controller = None
        del s.fixed_cfl_number

        seen = []
        sink_id = logger.add(lambda m: seen.append(str(m)), level="WARNING")
        try:
            compute_local_time_step(s)
            compute_local_time_step(s)
        finally:
            logger.remove(sink_id)
        hits = [m for m in seen if "fixed_cfl_number" in m]
        assert len(hits) == 1, f"应当只警告一次，实际 {len(hits)} 次"

    def test_fallback_value_is_still_0_1_for_back_compat(self):
        """回退值本身不变（向后兼容诊断脚本），变的只是"不再静默"。"""
        s_fb = _channel_solver(0.05, adaptive=False)
        s_fb._cfl_controller = None
        del s_fb.fixed_cfl_number
        s_ref = _channel_solver(0.1, adaptive=False)
        assert _dt_median(s_fb) == pytest.approx(_dt_median(s_ref), rel=1e-12)


class TestSourceLevelGuard:
    """结构判据：那个裸 `else 0.1` 不得再出现。"""

    def test_no_bare_else_0_1_in_cfl_module(self):
        import inspect

        from autoflowcfd.core.fr_solver import cfl as cfl_mod
        src = inspect.getsource(cfl_mod.compute_local_time_step)
        # **先剥掉注释**：那段修复说明里原样引用了旧表达式用于记录，
        # 不剥的话护栏会把文档当成代码命中（第一版就是这么假失败的）。
        code = " ".join(
            ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
        assert "if _cfl_controller is not None else 0.1" not in code, (
            "裸的 `else 0.1` 回来了——它会把调用方请求的固定 CFL 静默丢弃")
        # 2026-09-25 起取值规则统一在 adaptive_cfl/policy.py::current_cfl_number
        # （全部后端共用），这里必须调用它而不是另写一份。
        assert "current_cfl_number(solver)" in code
        from autoflowcfd.core.time_integration.adaptive_cfl import policy
        psrc = inspect.getsource(policy.current_cfl_number)
        assert "fixed_cfl_number" in psrc

    def test_constructor_prints_the_fixed_cfl(self):
        """启动日志必须能看出这次跑的是哪个 CFL（"这份日志是哪个配置跑出来
        的"不该变成事后考古，同一原则见 solver.py 里那段启动日志说明）。"""
        import inspect

        from autoflowcfd.core.fr_solver.solver.setup import _SolverSetupMixin
        from autoflowcfd.core.time_integration.adaptive_cfl.policy import (
            describe_cfl_policy,
        )
        # 2026-09-25 起 CFL 策略在 `FRSolver.__init__` 的装配阶段里建立
        src = inspect.getsource(_SolverSetupMixin._setup_turbulence_time_and_runtime)
        assert "fixed_cfl_number" in src and "describe_cfl_policy(" in src
        assert "fixed CFL = 0.037" in describe_cfl_policy(None, 0.037)
