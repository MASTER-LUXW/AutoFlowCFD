"""累计伪时间预算 —— 区分"收敛慢"与"根本还没走到"。

## 缺口（一次真实的、代价很大的误判）

plate_demo（0.5m 方板正对来流，363,392 单元）P1+SST、固定 CFL 0.03 下
Cd 从 0 线性爬到 5.77（文献 1.10~1.18）、滞止面压力系数中位 +3.41（定常流
的上界是 +1）、350 步零曲率。这被当成最高优先级的壁面处理缺陷追了好几天，
甚至被记成"C 类默认值决策的前置条件"。

用求解器自己的 dt 算一下累计伪时间就推翻了它：

    局部对流 h/U        tau/T = 0.877    （刚走完约一个）
    绕板 L_body/U       tau/T = 0.018    需要约  19,200 步
    全域 L_domain/U     tau/T = 0.0017   需要约 211,000 步

350 步只走完一个绕板特征时间的 **1.8%**。"单调线性上升、零曲率"恰恰是启动
暂态最开头的必然形态；而"定常流总压不可能超过来流滞止压力"这条判据**对
暂态不适用**（非定常 Bernoulli 有 ∂φ/∂t 项）。决定性的一条是壁面法向速度
在**单调趋零**（0.369 → 0.206 → −0.004 m/s）、与升压完全脱钩，所以升压不
是壁面通量产生的。

这个数一直算得出来，只是**从来没被打印出来过**。本文件钉住它现在是求解器
报告的一等诊断量，以及它的三层时标语义不会被悄悄改掉。

## 本文件覆盖

1. 比值的定义（tau/T = tau_median / (L/u)）与三层时标各自的口径
2. 逐单元累计（局部时间步进下不存在单一"当前时间"）
3. 拿不到某个时标时报 `nan` 而不是拿别的冒充
4. 格式化输出：compact 字段、完整摘要、`tau/T_body < 0.5` 的显式警示
5. `FRSolver` 真的在每步累加 `tau_accum`，且 `solve()` 会重置它
"""

import numpy as np
import pytest

from autoflowcfd.core.fr_solver.pseudotime_budget import (
    _extent,
    format_pseudo_time_budget,
    pseudo_time_budget,
)


class TestRatioDefinition:
    def test_tau_over_body_is_median_tau_over_L_div_u(self):
        dt = np.array([1.0e-6, 2.0e-6, 3.0e-6])
        b = pseudo_time_budget(dt, vel_inf=30.0, body_length=0.5,
                               n_steps=100)
        # tau = 100*dt -> 中位 2e-4；T = 0.5/30
        assert b["tau_median"] == pytest.approx(2.0e-4)
        assert b["tau_over_body"] == pytest.approx(2.0e-4 / (0.5 / 30.0))

    def test_steps_for_body_uses_median_dt(self):
        dt = np.full(5, 1.0e-6)
        b = pseudo_time_budget(dt, vel_inf=30.0, body_length=0.5, n_steps=1)
        assert b["steps_for_body"] == pytest.approx((0.5 / 30.0) / 1.0e-6)

    def test_reproduces_the_measured_plate_numbers(self):
        """钉住那次推翻误判用的三个数（plate_demo，350 步，CFL 0.03）。

        dt 中位 8.690e-7 s、h 中位 1.045e-2 m、U=30、L_body=0.5、
        L_domain=5.5 —— 实测 tau/T 分别是 0.877 / 0.018 / 0.0017。
        """
        dt = np.array([8.690e-7])
        vol = np.array([(1.045e-2) ** 3])
        b = pseudo_time_budget(dt, vel_inf=30.0, cell_volumes=vol,
                               body_length=0.5, domain_length=5.5,
                               n_steps=350)
        assert b["tau_over_local"] == pytest.approx(0.877, rel=0.02)
        assert b["tau_over_body"] == pytest.approx(0.0182, rel=0.02)
        assert b["tau_over_domain"] == pytest.approx(0.00166, rel=0.02)
        assert b["steps_for_body"] == pytest.approx(19180, rel=0.02)


class TestPerCellAccumulation:
    def test_accepts_2d_dt_and_takes_first_sp(self):
        """dt 可以是 (n_cells, n_sps)：同一单元各解点共用一个 dt。"""
        dt2 = np.tile(np.array([[1.0e-6], [3.0e-6]]), (1, 8))
        b = pseudo_time_budget(dt2, vel_inf=30.0, n_steps=10)
        assert b["dt_median"] == pytest.approx(2.0e-6)
        assert b["tau_median"] == pytest.approx(2.0e-5)

    def test_real_accumulation_beats_the_estimate_under_varying_dt(self):
        """自适应 CFL 下必须传真实累加量——`n_steps*dt` 会算错。

        这条不是风格问题：CFL 在收缩/放大时 dt 逐步变化，用"当前 dt 乘步数"
        会把整段历史按最后一步的步长重算，可以错到几倍。
        """
        dt_now = np.array([1.0e-6, 1.0e-6])
        tau_real = np.array([5.0e-5, 5.0e-5])      # 历史上 dt 曾大得多
        est = pseudo_time_budget(dt_now, vel_inf=30.0, body_length=0.5,
                                 n_steps=10)
        real = pseudo_time_budget(dt_now, vel_inf=30.0, body_length=0.5,
                                  tau_accum=tau_real, n_steps=10)
        assert est["estimated"] == 1.0
        assert real["estimated"] == 0.0
        assert est["tau_median"] == pytest.approx(1.0e-5)
        assert real["tau_median"] == pytest.approx(5.0e-5)
        assert real["tau_over_body"] > 4.0 * est["tau_over_body"]


class TestMissingScalesAreNaNNotSubstituted:
    def test_no_body_length_gives_nan_not_domain(self):
        b = pseudo_time_budget(np.full(3, 1e-6), vel_inf=30.0,
                               domain_length=5.5, n_steps=10)
        assert np.isnan(b["tau_over_body"])
        assert np.isfinite(b["tau_over_domain"])

    def test_no_cell_volumes_gives_nan_local(self):
        b = pseudo_time_budget(np.full(3, 1e-6), vel_inf=30.0, n_steps=10)
        assert np.isnan(b["tau_over_local"])

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"),
                                     float("inf")])
    def test_non_positive_or_non_finite_length_is_nan(self, bad):
        b = pseudo_time_budget(np.full(3, 1e-6), vel_inf=30.0,
                               body_length=bad, n_steps=10)
        assert np.isnan(b["tau_over_body"])
        assert np.isnan(b["steps_for_body"])


class TestFormatting:
    def test_compact_prefers_body_scale(self):
        b = pseudo_time_budget(np.full(3, 1e-6), vel_inf=30.0,
                               body_length=0.5, domain_length=5.5,
                               n_steps=350)
        out = format_pseudo_time_budget(b, compact=True)
        assert out.startswith("tau/T_body=")

    def test_compact_falls_back_to_domain_and_says_so(self):
        b = pseudo_time_budget(np.full(3, 1e-6), vel_inf=30.0,
                               domain_length=5.5, n_steps=350)
        out = format_pseudo_time_budget(b, compact=True)
        assert out.startswith("tau/T_dom=")

    def test_compact_reports_na_when_no_scale_available(self):
        b = pseudo_time_budget(np.full(3, 1e-6), vel_inf=30.0, n_steps=10)
        assert format_pseudo_time_budget(b, compact=True) == "tau/T=n/a"

    def test_full_summary_warns_below_half_a_body_time(self):
        """tau/T_body < 0.5 必须显式警示。

        这条警示就是那次误判缺失的东西：日志里只有残差在降，看不出物理场
        只走了 1.8%，于是启动暂态的 Cd 被拿去和文献值比。
        """
        b = pseudo_time_budget(np.full(3, 1e-6), vel_inf=30.0,
                               body_length=0.5, n_steps=350)
        out = format_pseudo_time_budget(b)
        assert b["tau_over_body"] < 0.5
        assert "启动暂态" in out
        assert "不能拿去与文献值比" in out

    def test_full_summary_has_no_warning_once_developed(self):
        b = pseudo_time_budget(np.full(3, 1e-3), vel_inf=30.0,
                               body_length=0.5, n_steps=350)
        out = format_pseudo_time_budget(b)
        assert b["tau_over_body"] > 0.5
        assert "启动暂态" not in out

    def test_estimated_flag_is_visible_in_the_summary(self):
        """按当前 dt 外推的数必须标出来，不能看起来像真实累加量。"""
        est = format_pseudo_time_budget(pseudo_time_budget(
            np.full(3, 1e-6), vel_inf=30.0, body_length=0.5, n_steps=10))
        real = format_pseudo_time_budget(pseudo_time_budget(
            np.full(3, 1e-6), vel_inf=30.0, body_length=0.5,
            tau_accum=np.full(3, 1e-5), n_steps=10))
        assert "外推" in est
        assert "外推" not in real

    def test_summary_names_the_length_it_used(self):
        b = pseudo_time_budget(np.full(3, 1e-6), vel_inf=30.0,
                               body_length=0.5, domain_length=5.5,
                               n_steps=10)
        out = format_pseudo_time_budget(b)
        assert "L = 0.5 m" in out
        assert "L = 5.5 m" in out


class TestDomainExtent:
    def test_reads_node_coords(self):
        class M:
            _node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 3.0, 2.0]])

        assert _extent(M()) == pytest.approx(3.0)

    def test_returns_none_without_nodes(self):
        class M:
            pass

        assert _extent(M()) is None

    def test_returns_none_for_empty_nodes(self):
        class M:
            _node_coords = np.zeros((0, 3))

        assert _extent(M()) is None


class TestSolverAccumulatesTau:
    """`FRSolver` 必须真的在每步累加，并在 `solve()` 开头重置。"""

    def _tiny_solver(self):
        """4x3x1 小棱柱通道网格 —— 与 `test_flow_direction.py` 同一个夹具，
        构造代价很低但走的是真实 `step()` 路径（累加发生在那里）。"""
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
                     rho_inf=1.225, vel_inf=33.33, p_inf=101325.0,
                     adaptive_cfl=False, cfl_start=0.05, cfl_max=0.05,
                     cfl_min=0.05)
        s.order_continuation_enabled = False
        return s

    def test_tau_accum_grows_monotonically(self):
        try:
            solver = self._tiny_solver()
        except Exception as exc:                        # pragma: no cover
            pytest.skip(f"小网格夹具不可用：{exc}")
        assert getattr(solver, "tau_accum", None) is None
        solver.step(1e-4)
        t1 = np.array(solver.tau_accum)
        assert t1.shape == (solver.mesh.n_cells,)
        assert np.all(t1 > 0.0)
        solver.step(1e-4)
        t2 = np.array(solver.tau_accum)
        assert np.all(t2 > t1), "第二步必须继续累加，不能被覆盖"

    def test_reported_dt_is_the_step_dt_not_the_accumulated_tau(self):
        """回归：`dt_median` 必须是**本步**的 dt，不能是累计 tau。

        第一版把 `tau` 同时当 `dt_cells` 和 `tau_accum` 传进去，于是摘要里
        "到 tau/T = 1 还需要多少步"（= T / dt_median）被按步数成比例低估
        ——12 步后报"需要 105 步"，真实是 1250 步，而且越跑越错。
        """
        try:
            solver = self._tiny_solver()
        except Exception as exc:                        # pragma: no cover
            pytest.skip(f"小网格夹具不可用：{exc}")
        solver._reference_area = 1.0
        for _ in range(8):
            solver.step(1e-4)
        b = solver._pseudo_time_budget(n_steps=8)
        assert b is not None
        # 8 步之后累计量必须明显大于单步 dt（固定 CFL 下约 8 倍）
        assert b["tau_median"] > 5.0 * b["dt_median"], (
            f"tau 中位 {b['tau_median']:.3e} 相对 dt 中位 "
            f"{b['dt_median']:.3e} 只大了 "
            f"{b['tau_median']/b['dt_median']:.2f} 倍——dt 被累计量污染了")
        assert b["tau_median"] == pytest.approx(
            8.0 * b["dt_median"], rel=0.05)
        # "还需要多少步"与 tau/T 必须自洽：tau/T * steps_for = n_steps
        assert b["tau_over_body"] * b["steps_for_body"] == pytest.approx(
            8.0, rel=0.05)

    def test_budget_helper_returns_finite_body_ratio(self):
        try:
            solver = self._tiny_solver()
        except Exception as exc:                        # pragma: no cover
            pytest.skip(f"小网格夹具不可用：{exc}")
        solver.step(1e-4)
        solver._reference_area = 1.0
        b = solver._pseudo_time_budget(n_steps=1)
        assert b is not None
        assert b["estimated"] == 0.0, "求解器必须传真实累加量而不是外推"
        assert np.isfinite(b["tau_over_body"])
        assert np.isfinite(b["tau_over_local"])

    def test_helper_returns_none_before_any_step(self):
        try:
            solver = self._tiny_solver()
        except Exception as exc:                        # pragma: no cover
            pytest.skip(f"小网格夹具不可用：{exc}")
        assert solver._pseudo_time_budget(n_steps=0) is None
