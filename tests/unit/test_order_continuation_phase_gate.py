"""Order Continuation 阶段判据 `PhaseGate`：湍流必须一起到位才升阶 / 判收敛，已收敛到
舍入误差也算到位（`core/utils/order_continuation/policy.py` 模块文档）。"""

from types import SimpleNamespace

import numpy as np

from autoflowcfd.core.utils.order_continuation.policy import (
    ROUNDOFF_DROP, PhaseGate, production_ramp_complete,
)


def _solver(turb_res=None, ramp_done=True, with_ramp=True):
    model = SimpleNamespace(production_factor=1.0) if with_ramp else SimpleNamespace()
    info = None if turb_res is None else {"res_norm_turbulence": turb_res}
    return SimpleNamespace(turb_model=model, _turb_production_ramp_complete=ramp_done, _newton_last_info=info)


def _turb(s, r):
    s._newton_last_info["res_norm_turbulence"] = r


def test_ramp_incomplete_blocks_advance_even_with_large_mean_drop():
    s = _solver(turb_res=1.0, ramp_done=False)
    gate = PhaseGate()
    gate.observe(s, 1e6)
    assert not production_ramp_complete(s)
    assert not gate.reached(s, res=1.0, baseline=1e6, required=1e3)


def test_turbulence_must_drop_from_post_ramp_baseline():
    s = _solver(turb_res=100.0, ramp_done=False)
    gate = PhaseGate()
    gate.observe(s, 1e3)               # 斜坡期间：湍流首值 100，但不作为斜坡后基准
    s._turb_production_ramp_complete = True
    _turb(s, 50.0)
    gate.observe(s, 1e3)               # 斜坡完成后的首个值 50 是湍流基准
    assert not gate.reached(s, res=0.5, baseline=1e3, required=1e3)
    _turb(s, 0.1)                       # 相对基准只降 500 倍
    assert not gate.reached(s, res=0.5, baseline=1e3, required=1e3)
    _turb(s, 0.04)                      # 降 1250 倍
    assert gate.reached(s, res=0.5, baseline=1e3, required=1e3)
    assert not gate.reached(s, res=10.0, baseline=1e3, required=1e3)   # 平均流没降够同样不行


def test_roundoff_level_counts_as_reached():
    """斜坡完成前已收敛：斜坡后重置的基准本身已很小，相对它降不够，但相对本阶段第一步
    已到舍入误差（plate_demo P0 实测形态）。"""
    s = _solver(turb_res=1e2)
    gate = PhaseGate()
    gate.observe(s, 4e8)                              # 本阶段第一步
    _turb(s, 1e-10)
    gate.observe(s, 2e-3)                             # 湍流基准也已经在舍入误差附近
    first_drop_res = 4e8 / (ROUNDOFF_DROP * 2)
    assert gate.reached(s, res=first_drop_res, baseline=2e-3, required=1e3)
    assert not gate.reached(s, res=1e-1, baseline=2e-3, required=1e3)


def test_without_implicit_turbulence_only_mean_flow_and_ramp_count():
    s = _solver(turb_res=None)
    gate = PhaseGate()
    gate.observe(s, 2e3)
    assert gate.reached(s, res=1.0, baseline=2e3, required=1e3)
    s2 = _solver(turb_res=None, ramp_done=False)
    assert not PhaseGate().reached(s2, res=1.0, baseline=2e3, required=1e3)
    laminar = SimpleNamespace(turb_model=None)
    assert PhaseGate().reached(laminar, res=1.0, baseline=2e3, required=1e3)


def test_nk_stall_is_detected_but_not_counted_as_reached():
    """NK：CFL 在上限、每步完整接受、残差在窗口内不再下降 -> 平台（`stalled_at_roundoff`，用于停止空转，
    plate_demo P0 曾在 4.2e-6 平台上空转 240 步）。平台**不算到位**：它只看平坦、不看绝对水平，真实的
    停滞同样满足它（I-89，此前在最终阶段把未收敛报成收敛）。"""
    from autoflowcfd.core.time_integration.adaptive_cfl.ser import SERCFLController
    from autoflowcfd.core.utils.order_continuation.policy import STALL_WINDOW

    s = _solver(turb_res=3e-3, ramp_done=True)
    s._cfl_controller = SERCFLController(cfl_start=1e4, cfl_max=1e4)
    s._newton_last_info = {"theta": 1.0}
    g = PhaseGate()
    rng = np.random.default_rng(0)
    for i in range(STALL_WINDOW + 1):
        _turb(s, 3e-3 * (1 + 0.01 * rng.standard_normal()))
        r = 4.2e-6 * (1 + 0.01 * rng.standard_normal())
        g.observe(s, r)
        if i < STALL_WINDOW:
            assert not g.reached(s, r, 2.2e-3, 1000.0)
    assert g.stalled_at_roundoff() and not g.reached(s, r, 2.2e-3, 1000.0)

    # CFL 不在上限（仍是伪时间推进）：不是舍入平台
    s._cfl_controller.cfl_number = 5e3
    g.observe(s, r)
    assert not g.stalled_at_roundoff()


def test_nk_still_decreasing_is_not_a_stall():
    from autoflowcfd.core.time_integration.adaptive_cfl.ser import SERCFLController
    from autoflowcfd.core.utils.order_continuation.policy import STALL_WINDOW

    s = _solver(turb_res=3e-3, ramp_done=True)
    s._cfl_controller = SERCFLController(cfl_start=1e4, cfl_max=1e4)
    s._newton_last_info = {"theta": 1.0}
    g = PhaseGate()
    for i in range(STALL_WINDOW + 1):
        _turb(s, 3e-3 * 0.9 ** i)
        g.observe(s, 1e-3 * 0.8 ** i)          # 窗口内下降 ~9 倍
    assert not g.stalled_at_roundoff()


def _plateau_stub(residuals):
    """定阶求解循环的最小替身：NK 的 SER 控制器停在上限、每步完整接受，残差按给定序列。"""
    from autoflowcfd.core.time_integration.adaptive_cfl.ser import SERCFLController
    from autoflowcfd.core.time_integration.base import TimeIntegrationScheme, TimeIntegrator

    from autoflowcfd.core.fr_solver.convergence_history import ConvergenceHistory

    calls = {"step": 0}

    def step(dt):
        r = residuals[min(calls["step"], len(residuals) - 1)]
        calls["step"] += 1
        return r

    stub = SimpleNamespace(
        step=step, convergence_history=ConvergenceHistory(), order=1, turb_model_name="NONE", turb_model=None,
        order_continuation_enabled=False,
        time_integrator=TimeIntegrator(scheme=TimeIntegrationScheme.NEWTON_KRYLOV),
        _cfl_controller=SERCFLController(cfl_start=1e4, cfl_max=1e4), _newton_last_info={"theta": 1.0},
        _turb_production_ramp_complete=True, _ramp_baseline_reset_done=True,
        _reference_area=None, freestream=None, state=SimpleNamespace(),
        _loop_monitor_suffix=lambda: "", _scaled_residual_field=lambda: None, _divergence_hint=lambda: "",
        _pseudo_time_budget=lambda n_steps: None)
    return stub, calls


def test_fixed_order_loop_stops_on_plateau_without_claiming_convergence(monkeypatch):
    """残差从 1 降到 1e-3 后停在平台（要求 1e6 倍）：在平台窗口满时结束，converged=False。"""
    from autoflowcfd.core.fr_solver.solver import FRSolver, solve_loop

    monkeypatch.setattr(solve_loop, "format_step_line", lambda *a, **k: "")     # 监控行不是被测对象
    from autoflowcfd.core.utils.order_continuation.policy import STALL_WINDOW

    seq = [1.0, 0.1, 0.01] + [1e-3] * 40
    stub, calls = _plateau_stub(seq)
    result = FRSolver.solve(stub, max_iter=40, dt=1e-4, tol=1e-6)
    assert result.converged is False
    assert calls["step"] == 3 + STALL_WINDOW + 1      # 平台从第 4 步起，窗口需要 STALL_WINDOW+1 个平台值


def test_fixed_order_loop_converges_by_the_relative_criterion(monkeypatch):
    from autoflowcfd.core.fr_solver.solver import FRSolver, solve_loop

    monkeypatch.setattr(solve_loop, "format_step_line", lambda *a, **k: "")

    stub, calls = _plateau_stub([10.0 ** -k for k in range(10)])
    result = FRSolver.solve(stub, max_iter=10, dt=1e-4, tol=1e-6)
    assert result.converged is True and calls["step"] == 7



def test_phase_baseline_from_an_older_residual_definition_is_not_reused():
    """阶段基准随 checkpoint 带上残差范数定义版本；旧定义（或没有版本）写出的基准与当前残差不可比，不做种子。"""
    from autoflowcfd.core.time_integration.residual_norm import RESIDUAL_NORM_VERSION
    from autoflowcfd.core.utils.order_continuation.checkpoint_state import (
        phase_state_metadata, restore_phase_state,
    )

    meta = phase_state_metadata(SimpleNamespace(_phase_initial_residual=3.5))
    assert meta["residual_norm_version"] == RESIDUAL_NORM_VERSION
    fresh = SimpleNamespace()
    restore_phase_state(fresh, meta)
    assert fresh._phase_initial_residual == 3.5
    for old in ({"phase_initial_residual": 3.5}, {"phase_initial_residual": 3.5, "residual_norm_version": 1}):
        s = SimpleNamespace()
        restore_phase_state(s, old)
        assert not hasattr(s, "_phase_initial_residual")
