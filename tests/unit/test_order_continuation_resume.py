"""Regression test for a real bug found 2026-08-22 (real-mesh resume,
asked directly by the user: "如果当前是P1，resume是从P1开始运算吗"):

`run_order_continuation` unconditionally reinitializes `solver.state` to a
uniform freestream and restarts the P0->target ramp from scratch whenever
`solver.state.U.shape[1] != 1` (not P0). This branch exists to handle a
*freshly constructed* FRSolver, whose default initial state is a uniform
freestream sized at the *target* order (not P0) - reinitializing it down
to a proper P0 starting point before ramping is correct there.

But `solve resume` reuses the exact same code path. A checkpoint saved
mid-ramp (e.g. genuinely converged physics at P1) also has
`state.U.shape[1] != 1`, and would hit the *identical* branch - silently
discarding the real, resumed solution and restarting the whole ramp from
a uniform P0 freestream, with no error or warning. Resuming a P1/P2
checkpoint would be indistinguishable from not resuming at all.

Fix: `rebuild_solver_from_checkpoint` now marks the solver with
`_resumed_from_checkpoint = True` right after loading the real state;
`run_order_continuation` skips the reinit-to-P0 branch when this flag is
set, and starts the `orders` ramp range from `solver.current_order`
(the checkpoint's real order) instead of always from P0.
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from autoflowcfd.core.utils.order_continuation import run_order_continuation


def _n_sps(order):
    return (order + 1) ** 3


def _fake_solver(current_order, target_order, resumed):
    n_sps = _n_sps(current_order)
    # Recognizable sentinel value - if the reinit-to-P0 branch fires, the
    # array gets replaced wholesale by a differently-shaped uniform state,
    # so any check that these exact values persist would fail.
    U = np.full((2, n_sps, 7), 42.0)

    mesh = SimpleNamespace(
        _order_geometry_cache={},
        set_order=lambda p: mesh.set_order_calls.append(p),
    )
    mesh.set_order_calls = []

    # 相对收敛判据需要残差真正下降：每个阶段的第一步返回 1e10（被捕获为
    # 初始残差），第二步起返回 1e-10（下降 1e20 倍，远超任何阶段的阈值）。
    # 旧代码用 step=lambda dt: 1e-9 配合绝对判据 res < 1e-4 立即收敛，
    # 相对判据下恒定残差不会产生任何下降比。
    # 用周期性模式确保跨阶段重置：每个阶段的第一步恰好落在高值上。
    _call_count = {"n": 0}

    def _decreasing_step(dt):
        n = _call_count["n"]
        _call_count["n"] += 1
        # 每个阶段 2 步收敛：奇数步高值，偶数步低值
        return 1e10 if n % 2 == 0 else 1e-10

    solver = SimpleNamespace(
        order=target_order,
        current_order=current_order,
        ops=SimpleNamespace(D_3d=np.zeros((n_sps, n_sps))),
        mesh=mesh,
        state=SimpleNamespace(U=U, n_cells=2, n_vars=7),
        freestream={"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0},
        turb_model=None,
        wall_distance=None,
        sgs_model=None,
        _resumed_from_checkpoint=resumed,
        step=_decreasing_step,
    )

    def _fake_interpolate(new_order):
        solver.current_order = new_order
        solver.state.U = np.full((2, _n_sps(new_order), 7), 42.0)

    solver._interpolate_to_new_order = _fake_interpolate
    return solver


def _fake_generate_ops(p, flux_point_type='radau'):
    n = _n_sps(p)
    return SimpleNamespace(D_3d=np.zeros((n, n)))


class TestResumeSkipsP0Reinit:
    def test_resumed_p1_checkpoint_ramp_starts_at_p1_not_p0(self):
        solver = _fake_solver(current_order=1, target_order=2, resumed=True)

        with patch(
            "autoflowcfd.fr.operators.generate_fr_operators", side_effect=_fake_generate_ops,
        ), patch(
            "autoflowcfd.core.fr_solver.state.FRState"
        ) as mock_frstate:
            result = run_order_continuation(solver, max_iter=10, dt=1e-3, tol=1e-6)

        # The P0-reinit branch must never have run: FRState() is only
        # constructed there.
        mock_frstate.assert_not_called()
        # The ramp must never have visited P0 at all.
        assert 0 not in solver.mesh.set_order_calls
        assert solver.mesh.set_order_calls == [1, 2]
        assert result.converged is True

    def test_non_resumed_fresh_solver_still_restarts_from_p0(self):
        """Guards the primary (non-resume) `solve steady` path: unchanged
        behaviour when _resumed_from_checkpoint is absent/False."""
        solver = _fake_solver(current_order=2, target_order=2, resumed=False)

        with patch(
            "autoflowcfd.fr.operators.generate_fr_operators", side_effect=_fake_generate_ops,
        ):
            run_order_continuation(solver, max_iter=30, dt=1e-3, tol=1e-6)

        # P0 appears twice: once from the reinit-to-P0 branch itself, once
        # from the ramp loop's own first (P0) phase - pre-existing,
        # harmless redundancy, not something this fix changes.
        assert solver.mesh.set_order_calls == [0, 0, 1, 2]


class TestResumeCeilingFractionResetHeuristic:
    """真实 bug 回归测试（2026-09-05，cube_demo 791,492 单元真实网格
    `solve resume` 长程验证决定性发现）：`run_order_continuation` 的
    "resume 时检测湍流场是否从未真正恢复"启发式，此前用
    `k_mean > max(1%*k_max, 10*k_inf)` 判断——这不是注释一直描述的
    "统计有多大比例单元被钳制在上界附近"，是对该意图的错误实现。
    真实复现：cube_demo 这类强分离钝体绕流，健康充分发展的
    k_mean=38.11（k_inf=0.167 的 228 倍）被这个公式误判成"爆炸残留"，
    resume 第一次调用就把整个 k_field/omega_field 清零重置回自由流
    初值，销毁真实演化的湍流场（表现为 CLI 观测到的"k 场几步内从
    38 崩溃到 0.166≈k_inf"——0.166 这个数字不是巧合，就是被强制
    重置成的 k_inf 本身）。现在改成真正统计 `k>=0.9*k_max`/
    `omega>=0.9*omega_max` 的比例，超过 10% 才判定为真爆炸。"""

    def _fake_solver_with_turb(self, current_order, target_order, k_field, omega_field,
                                k_max=555.44, omega_max=1e6):
        solver = _fake_solver(current_order=current_order, target_order=target_order, resumed=True)
        n_cells, n_sps = solver.state.n_cells, _n_sps(current_order)
        solver.turb_model = SimpleNamespace(
            k_field=np.full((n_cells, n_sps), k_field, dtype=np.float64),
            omega_field=np.full((n_cells, n_sps), omega_field, dtype=np.float64),
            k_max=k_max, omega_max=omega_max,
            nu_t=np.zeros((n_cells, n_sps)),
            production_factor=0.0,
        )
        return solver

    def test_healthy_high_turbulence_field_is_not_falsely_reset(self):
        """真实复现场景：k_mean=38.11，远超旧公式的 reset_threshold≈5.55，
        但只是钝体尾流健康发展的高湍流度场，不应该被重置。"""
        solver = self._fake_solver_with_turb(
            current_order=1, target_order=2, k_field=38.11, omega_field=28459.5,
        )
        with patch(
            "autoflowcfd.fr.operators.generate_fr_operators", side_effect=_fake_generate_ops,
        ):
            run_order_continuation(solver, max_iter=10, dt=1e-3, tol=1e-6)

        # 健康场必须原封不动地保留，不能被静默清零重置成来流初值。
        np.testing.assert_allclose(solver.turb_model.k_field, 38.11)
        np.testing.assert_allclose(solver.turb_model.omega_field, 28459.5)

    def test_genuinely_clamped_field_still_gets_reset(self):
        """真正的爆炸残留（大面积钉在上界）必须仍然触发重置——这个
        修复只是改判据的统计口径，不是把安全网整个拆掉。"""
        solver = self._fake_solver_with_turb(
            current_order=1, target_order=2, k_field=555.44, omega_field=1e6,
        )
        with patch(
            "autoflowcfd.fr.operators.generate_fr_operators", side_effect=_fake_generate_ops,
        ):
            run_order_continuation(solver, max_iter=10, dt=1e-3, tol=1e-6)

        # 100% 的单元都钉在上界，必须被重置到自由流初值，不能保持在 555.44/1e6。
        k_inf_expected = 1.5 * (33.33 * 0.01) ** 2
        assert not np.allclose(solver.turb_model.k_field, 555.44)
        np.testing.assert_allclose(solver.turb_model.k_field, k_inf_expected, rtol=1e-6)

    def test_partial_clamping_below_threshold_is_not_reset(self):
        """只有少数单元（低于10%）触及上界——健康网格上真实观测到的
        比例是 4.36%（k）/0.07%（omega）——不应该触发重置。用足够多的
        伪单元数（100）才能有意义地表达"5%"这种小比例（_fake_solver
        默认只有 2 个单元，容不下 <50% 的粒度）。"""
        solver = self._fake_solver_with_turb(
            current_order=1, target_order=2, k_field=1.0, omega_field=100.0,
        )
        n_cells = 100
        n_sps = _n_sps(1)
        solver.state.n_cells = n_cells
        solver.turb_model.k_field = np.full((n_cells, n_sps), 1.0)
        solver.turb_model.omega_field = np.full((n_cells, n_sps), 100.0)
        solver.turb_model.nu_t = np.zeros((n_cells, n_sps))
        # 5% 的单元钳制在上界（低于10%阈值）。
        n_clamped = 5
        solver.turb_model.k_field[:n_clamped] = solver.turb_model.k_max

        with patch(
            "autoflowcfd.fr.operators.generate_fr_operators", side_effect=_fake_generate_ops,
        ):
            run_order_continuation(solver, max_iter=10, dt=1e-3, tol=1e-6)

        # 未被重置：绝大多数单元应该还是原来的 1.0，不是被清零的 k_inf。
        assert np.any(np.isclose(solver.turb_model.k_field, 1.0))


def _fake_solver_with_residual_sequence(current_order, target_order, resumed,
                                         residuals, phase_initial_residual=None):
    """Like _fake_solver, but `step()` returns a scripted residual sequence
    (one value per call, held at the last value once exhausted) instead of
    a constant - needed to exercise the residual-drop promotion criterion
    (CL-02, `residual_drop_threshold`/`min_iter_before_transition`), not
    just the absolute-tolerance one."""
    solver = _fake_solver(current_order=current_order, target_order=target_order, resumed=resumed)
    calls = {"i": 0}

    def _scripted_step(dt):
        i = calls["i"]
        calls["i"] += 1
        return residuals[min(i, len(residuals) - 1)]

    solver.step = _scripted_step
    if phase_initial_residual is not None:
        solver._phase_initial_residual = phase_initial_residual
    return solver


class TestResumeSeedsPhaseInitialResidualFromCheckpoint:
    """Regression test for a real bug found 2026-08-23 (same live resume
    investigation as the P0 prism-quad-face dedup fix): `initial_residual_
    this_order` - the reference value the CL-02 residual-drop promotion
    criterion measures progress against - is a pure local variable, reset
    to `None` (then captured fresh from this call's first `solver.step()`)
    every time `run_order_continuation` is entered. A continuous
    `solve steady` run only ever calls this function once for its whole
    budget, so the reference is naturally captured at the *true* start of
    each order phase. `solve resume` is a brand new process making a brand
    new call, restarting the checkpointed order's phase reference from
    whatever the residual happens to be at THIS call's first step -
    completely disconnected from how much genuine progress this order had
    already made before the checkpoint was saved, so the "dropped >=100x"
    judgement no longer reflects the phase's real history.

    Fix: `solver._phase_initial_residual` persists this reference as a
    real solver attribute (round-tripped through write_checkpoint/
    rebuild_solver_from_checkpoint, see solve_checkpoint_io.py). When
    resuming into the first phase of a ramp, `run_order_continuation` seeds
    `initial_residual_this_order` from it when present, so "dropped >=100x"
    is judged against the phase's true historical start rather than this
    resume call's own first step.
    """

    def test_seeded_baseline_lets_a_resume_recognize_already_earned_promotion(self):
        # True history: this P0 phase started at residual 10000 (long
        # before the checkpoint that's being resumed from). By checkpoint
        # time it had already dropped to 100 (a real 100x drop) but the
        # checkpoint was saved before that got acted on. This resume's own
        # trajectory only creeps down slowly from there (100 -> ~9 by
        # iter 20) - on its own, nowhere near a fresh 100x drop.
        residuals = [100.0 / (1 + 0.5 * i) for i in range(30)]
        solver = _fake_solver_with_residual_sequence(
            current_order=0, target_order=2, resumed=True,
            residuals=residuals, phase_initial_residual=10000.0,
        )

        with patch(
            "autoflowcfd.fr.operators.generate_fr_operators", side_effect=_fake_generate_ops,
        ):
            run_order_continuation(solver, max_iter=90, dt=1e-3, tol=1e-6)

        # Seeded from the true historical start (10000), the phase's real
        # >=100x drop is correctly recognized once the residual-drop check
        # becomes eligible (i >= min_iter_before_transition=20) - the ramp
        # must advance past P0 well within this call's iteration budget.
        assert solver.mesh.set_order_calls[-1] in (1, 2)
        assert 1 in solver.mesh.set_order_calls or 2 in solver.mesh.set_order_calls

    def test_without_persisted_baseline_falls_back_to_first_step_and_warns(self, capsys):
        """Same trajectory, but an older checkpoint with no
        `_phase_initial_residual` recorded - must not crash, must fall back
        to capturing the reference from this resume's own first step (the
        pre-fix, imperfect-but-safe behaviour), and must warn.

        The P0 phase's local iteration budget (`phase_max_iter`) is finite
        and *always* forces a transition once exhausted regardless of
        whether a promotion criterion fired (a separate, pre-existing
        behaviour this fix doesn't touch) - so "never promotes" can't be
        asserted directly. Instead this compares *how many* P0 iterations
        it takes to leave the phase against the seeded case above: seeded,
        the true >=100x drop is recognized quickly (~21 iterations, once
        i >= min_iter_before_transition); unseeded, the drop-ratio never
        fires on its own (~31x max against the fresh baseline) and P0 only
        gets abandoned once its entire local budget (100 iterations) is
        exhausted - a clear, measurable difference proving the seeded path
        is genuinely exercised."""
        residuals = [100.0 / (1 + 0.5 * i) for i in range(30)]
        solver = _fake_solver_with_residual_sequence(
            current_order=0, target_order=2, resumed=True,
            residuals=residuals, phase_initial_residual=None,
        )
        assert not hasattr(solver, "_phase_initial_residual")

        p0_iters = {"n": 0}

        def _count_p0_iters(solver_ref, local_iter):
            if solver_ref.mesh.set_order_calls == [0]:
                p0_iters["n"] += 1

        with patch(
            "autoflowcfd.fr.operators.generate_fr_operators", side_effect=_fake_generate_ops,
        ):
            run_order_continuation(solver, max_iter=300, dt=1e-3, tol=1e-6,
                                    checkpoint_callback=_count_p0_iters)

        assert p0_iters["n"] == 100  # full phase_max_iter budget, criterion never fired
        assert "旧版本 checkpoint" in capsys.readouterr().out

    def test_solver_phase_initial_residual_attribute_tracks_current_phase(self):
        """`solver._phase_initial_residual` must be kept in sync every step
        (not just seeded once) so a mid-phase checkpoint_callback always
        persists the phase's real reference, and must reset for the next
        phase once the ramp actually advances - not leak the P0 phase's
        reference into P1's judgement.

        Uses the real checkpoint_callback hook (fires right after
        run_order_continuation sets the attribute each iteration, see
        order_continuation.py) rather than wrapping solver.step, since the
        attribute is only assigned *after* step() returns for that
        iteration - wrapping step would observe last iteration's value."""
        residuals = [50.0, 40.0, 30.0]  # 3 steps at P0 (phase_max_iter small)
        solver = _fake_solver_with_residual_sequence(
            current_order=0, target_order=1, resumed=False,
            residuals=residuals,
        )

        seen_values = []

        def _checkpoint_cb(solver_ref, local_iter):
            seen_values.append(getattr(solver_ref, "_phase_initial_residual", None))

        with patch(
            "autoflowcfd.fr.operators.generate_fr_operators", side_effect=_fake_generate_ops,
        ):
            run_order_continuation(solver, max_iter=6, dt=1e-3, tol=1e-6,
                                    checkpoint_callback=_checkpoint_cb)

        # First 3 recorded values are all P0's reference (50.0, captured at
        # P0's own first step); once the ramp reaches P1 the reference is
        # recaptured fresh for that phase (not left over from P0).
        assert seen_values[0] == 50.0
        assert seen_values[1] == 50.0
        assert seen_values[2] == 50.0
        assert seen_values[3] == 30.0  # P1's own first step, not 50.0 leaked from P0
        assert seen_values[4] == 30.0
        assert seen_values[5] == 30.0


class TestPhaseMaxIterDefaultGivesFinalStageRemainingBudget:
    """Regression test for a real bug found 2026-09-05 (user directly
    pointed out: "phase_max_iter 的默认值应该是 max_iter // len(orders)，
    两者不是或的关系"): the final (target) stage's "eat the rest of the
    budget, don't get diluted by len(orders)" rule — the entire point of
    the `phase_max_iter` parameter (2026-09-01) — was gated behind
    `phase_max_iter is not None`, i.e. only active when a caller explicitly
    passes `--phase-max-iter`. Callers who never pass it (the common CLI
    case) got the *old*, pre-fix behaviour by construction: the final
    stage capped at the same `max_iter // len(orders)` share as every
    other stage, silently truncated regardless of how much budget earlier
    stages left unused.

    Fix: `phase_max_iter`'s absence only supplies *this one number's*
    default value (`max_iter // len(orders)`); "final stage gets whatever
    budget remains" applies unconditionally, default or explicit alike.

    This test constructs a scenario where the difference is observable
    without relying on any residual-drop/convergence coincidence: P0
    (non-final) promotes early via the CL-02 residual-drop criterion,
    leaving unused budget; P1 (final) is scripted to never trigger any
    exit condition on its own, so it runs for exactly its assigned
    `stage_iter_budget` - directly exposing which budget it was given.
    """

    def test_final_stage_gets_leftover_budget_without_explicit_phase_max_iter(self):
        # P0: constant residual (drop=1) for 20 steps (i=0..19, below
        # min_iter_before_transition=20), then a single big drop at i=20
        # (drop=1000/10=100, exactly meeting the default
        # residual_drop_threshold=100) - promotes at i=20, having run 21
        # iterations (i=0..20 inclusive), well short of its 50-iteration
        # default share (max_iter=100, len(orders)=2 -> 100//2=50).
        residuals = [1000.0] * 20 + [10.0]
        solver = _fake_solver_with_residual_sequence(
            current_order=0, target_order=1, resumed=False, residuals=residuals,
        )

        with patch(
            "autoflowcfd.fr.operators.generate_fr_operators", side_effect=_fake_generate_ops,
        ):
            result = run_order_continuation(solver, max_iter=100, dt=1e-3, tol=1e-6)

        # P0 used 21 of its 50-iteration share, leaving 29 unused. P1
        # (final) never converges on its own (residual held flat at 10.0
        # once the scripted sequence is exhausted, see
        # _fake_solver_with_residual_sequence) - it must run for its full
        # assigned budget, then the whole call ends (not converged).
        # Old (buggy) behaviour: P1 capped at the same 50-iteration share
        # as P0 -> total_iter = 21 + 50 = 71, wasting the 29 P0 left
        # behind. Fixed behaviour: P1 gets *all* of max_iter's remainder
        # -> total_iter = 21 + (100 - 21) = 100, using the full budget the
        # caller asked for.
        assert result.iterations == 100
        assert result.converged is False
