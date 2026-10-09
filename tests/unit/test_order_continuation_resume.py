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
    # 可识别的哨兵值——如果触发了"重置回 P0"的分支，数组会被整体换成
    # 另一种形状的均匀状态，任何检查这些值是否保留的断言都会失败。
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
        # 统一循环（2026-10-05）的日志/守卫钩子：替身没有这些诊断
        _divergence_hint=lambda: "",
        _loop_monitor_suffix=lambda: "",
        _scaled_residual_field=lambda: None,
        _pseudo_time_budget=lambda n_steps: None,
        # 产生项渐变计数器（真实求解器构造时由 init_production_ramp 设置）
        _turb_ramp_step=0, _turb_production_ramp_steps=50, _turb_production_ramp_complete=False,
    )

    def _fake_interpolate(new_order):
        # 统一换阶接口：这一阶的状态与几何全部就绪（真实求解器在这里切网格阶数）
        solver.current_order = new_order
        solver.state.U = np.full((2, _n_sps(new_order), 7), 42.0)
        mesh.set_order(new_order)

    solver._interpolate_to_new_order = _fake_interpolate
    # 延拓后的正性限制：替身状态是常数 42，本来就可容许，钩子无事可做
    solver._limit_prolongated_state = lambda: None
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

        # 重置回 P0 的分支绝不能执行过：FRState() 只在那里构造。
        mock_frstate.assert_not_called()
        # 爬坡绝不能经过 P0；续算出的 P1 已经就位（不重新进入）。
        assert 0 not in solver.mesh.set_order_calls
        assert solver.mesh.set_order_calls == [2]
        assert result.converged is True

    def test_non_resumed_fresh_solver_still_restarts_from_p0(self):
        """守住主要的（非续算）`solve steady` 路径：_resumed_from_checkpoint
        不存在/为 False 时行为不变。
        """
        solver = _fake_solver(current_order=2, target_order=2, resumed=False)

        with patch(
            "autoflowcfd.fr.operators.generate_fr_operators", side_effect=_fake_generate_ops,
        ):
            run_order_continuation(solver, max_iter=30, dt=1e-3, tol=1e-6)

        # 重置到 P0 一次（统一循环不再在 P0 阶段开头重复切一次）
        assert solver.mesh.set_order_calls == [0, 1, 2]


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
        from autoflowcfd.core.fr_solver.turbulence import _set_freestream_turbulence
        from autoflowcfd.core.turbulence.sst import SSTModelFR

        solver = _fake_solver(current_order=current_order, target_order=target_order, resumed=True)
        n_cells, n_sps = solver.state.n_cells, _n_sps(current_order)
        k_inf, omega_inf = _set_freestream_turbulence(solver)
        model = SSTModelFR(n_cells, n_sps, k_inf=k_inf, omega_inf=omega_inf)
        model.k_field = np.full((n_cells, n_sps), k_field, dtype=np.float64)
        model.omega_field = np.full((n_cells, n_sps), omega_field, dtype=np.float64)
        model.k_max, model.omega_max = k_max, omega_max
        model.production_factor = 0.0
        solver.turb_model = model
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
    """与 _fake_solver 类似，但 `step()` 返回一个脚本化的残差序列（每次调用
    一个值，用完后保持最后一个值）而不是常数——用来测残差下降升阶判据
    （CL-02，`residual_drop_threshold`/`min_iter_before_transition`），而不只是
    绝对容差判据。
    """
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
    """2026-08-23 发现的真实缺陷的回归测试（与 P0 棱柱四边形面去重修复是同一次
    真实续算排查）：`initial_residual_this_order`——CL-02 残差下降升阶判据衡量
    进展所用的基准——是纯局部变量，每次进入 `run_order_continuation` 都重置为
    `None`（然后从这次调用的第一个 `solver.step()` 重新取得）。连续的
    `solve steady` 运行在整个预算内只调用这个函数一次，所以基准自然在每个阶数
    阶段*真正*的起点取得。`solve resume` 是全新的进程、全新的一次调用，把
    checkpoint 所在阶数的阶段基准从**这次**调用第一步的残差重新起算——与该阶数
    在 checkpoint 保存之前已经取得的真实进展完全脱节，"下降 >=100 倍"的判断
    不再反映该阶段的真实历史。

    修复：`solver._phase_initial_residual` 把这个基准作为求解器的真实属性持久化
    （经 write_checkpoint/rebuild_solver_from_checkpoint 往返，见
    solve_checkpoint_io.py）。续算进入爬坡的第一个阶段时，
    `run_order_continuation` 在它存在时用它给 `initial_residual_this_order`
    做种子，于是"下降 >=100 倍"按该阶段真实的历史起点判断，而不是按这次续算
    调用自己的第一步。
    """

    def test_seeded_baseline_lets_a_resume_recognize_already_earned_promotion(self):
        # 真实历史：这个 P0 阶段从残差 10000 开始（远早于被续算的那个
        # checkpoint）。到 checkpoint 时已经降到 100（真实的 100 倍下降），但
        # checkpoint 在据此动作之前就保存了。这次续算自己的轨迹只是从那里缓慢
        # 下降（100 -> 第 20 步约 9）——单看它自己，远不到新的 100 倍下降。
        residuals = [100.0 / (1 + 0.5 * i) for i in range(30)]
        solver = _fake_solver_with_residual_sequence(
            current_order=0, target_order=2, resumed=True,
            residuals=residuals, phase_initial_residual=10000.0,
        )

        with patch(
            "autoflowcfd.fr.operators.generate_fr_operators", side_effect=_fake_generate_ops,
        ):
            run_order_continuation(solver, max_iter=90, dt=1e-3, tol=1e-6)

        # 以真实的历史起点（10000）做种子，一旦残差下降判据可以生效
        # （i >= min_iter_before_transition=20），这个阶段真实的 >=100 倍下降就被
        # 正确识别——爬坡必须在这次调用的迭代预算之内早早越过 P0。
        assert solver.mesh.set_order_calls[-1] in (1, 2)
        assert 1 in solver.mesh.set_order_calls or 2 in solver.mesh.set_order_calls

    def test_without_persisted_baseline_falls_back_to_first_step_and_warns(self, capsys):
        """同一条轨迹，但用的是没有记录 `_phase_initial_residual` 的旧 checkpoint
        ——不能崩溃，必须退回到从这次续算自己的第一步取基准（修复之前的、不完美
        但安全的行为），并且必须给出警告。

        P0 阶段的局部迭代预算（`phase_max_iter`）是有限的，用完之后不论升阶判据
        是否触发都**一定**强制换阶（这是另一个早已存在、本修复没有触碰的行为）
        ——所以无法直接断言"从不升阶"。这里改为把离开 P0 阶段所需的迭代数与上面
        带种子的情形比较：带种子时真实的 >=100 倍下降很快被识别（约 21 步，
        一旦 i >= min_iter_before_transition）；不带种子时下降比自己从不触发
        （相对新基准最多约 31 倍），P0 只有在整个局部预算（100 步）用完之后才被
        放弃——一个清楚、可测的差别，证明带种子的路径确实被走到了。
        """
        residuals = [100.0 / (1 + 0.5 * i) for i in range(30)]
        solver = _fake_solver_with_residual_sequence(
            current_order=0, target_order=2, resumed=True,
            residuals=residuals, phase_initial_residual=None,
        )
        assert not hasattr(solver, "_phase_initial_residual")

        p0_iters = {"n": 0}

        def _count_p0_iters(solver_ref, local_iter):
            if solver_ref.current_order == 0:
                p0_iters["n"] += 1

        with patch(
            "autoflowcfd.fr.operators.generate_fr_operators", side_effect=_fake_generate_ops,
        ):
            run_order_continuation(solver, max_iter=300, dt=1e-3, tol=1e-6,
                                    checkpoint_callback=_count_p0_iters)

        assert p0_iters["n"] == 100  # 用满 phase_max_iter 预算，判据从未触发
        assert "没有阶段起始残差记录" in capsys.readouterr().out

    def test_solver_phase_initial_residual_attribute_tracks_current_phase(self):
        """`solver._phase_initial_residual` 必须每步都保持同步（不只是做一次种子），
        这样阶段中途的 checkpoint_callback 才总是持久化该阶段真实的基准；爬坡真的
        前进之后必须为下一阶段重置——不能把 P0 阶段的基准漏进 P1 的判断。

        用真实的 checkpoint_callback 钩子（每次迭代在 run_order_continuation 设置
        该属性之后立刻触发，见 order_continuation.py）而不是包装 solver.step：
        该属性是在这次迭代的 step() 返回*之后*才赋值的——包装 step 看到的是上一次
        迭代的值。
        """
        residuals = [50.0, 40.0, 30.0]  # 在 P0 上走 3 步（phase_max_iter 很小）
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

        # 前 3 个记录值都是 P0 的基准（50.0，在 P0 自己的第一步取得）；爬坡到
        # P1 之后基准为该阶段重新取得（不是 P0 遗留的）。
        assert seen_values[0] == 50.0
        assert seen_values[1] == 50.0
        assert seen_values[2] == 50.0
        assert seen_values[3] == 30.0  # P1 自己的第一步，不是从 P0 漏过来的 50.0
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
        # P0：残差恒定（下降 1 倍）20 步（i=0..19，低于
        # min_iter_before_transition=20），然后在 i=20 一次大降
        # （下降 1000/10=100，恰好达到默认的 residual_drop_threshold=100）——
        # 在 i=20 升阶，共走了 21 步（i=0..20），远少于它默认的 50 步份额
        # （max_iter=100，len(orders)=2 -> 100//2=50）。
        residuals = [1000.0] * 20 + [10.0]
        solver = _fake_solver_with_residual_sequence(
            current_order=0, target_order=1, resumed=False, residuals=residuals,
        )

        with patch(
            "autoflowcfd.fr.operators.generate_fr_operators", side_effect=_fake_generate_ops,
        ):
            result = run_order_continuation(solver, max_iter=100, dt=1e-3, tol=1e-6)

        # P0 用了 50 步份额里的 21 步，剩 29 步没用。P1（最终阶段）自己从不收敛
        # （脚本序列用完后残差保持 10.0，见 _fake_solver_with_residual_sequence）
        # ——它必须跑满分配给它的全部预算，然后整个调用结束（未收敛）。
        # 旧的（有缺陷的）行为：P1 被限制在与 P0 相同的 50 步份额 -> total_iter =
        # 21 + 50 = 71，浪费了 P0 剩下的 29 步。修复后的行为：P1 得到 max_iter 的
        # **全部**剩余 -> total_iter = 21 + (100 - 21) = 100，用满调用方要求的预算。
        assert result.iterations == 100
        assert result.converged is False


class TestPerPhaseCflReset:
    def test_every_phase_starts_from_the_controller_initial_value(self):
        """换阶时残差的跳变是延拓误差、不是解在恶化：每阶第一步都从控制器初值起步。复位在四个后端共用的循环里
        （2026-10-05 以前只有 CPU 单机循环与单 GPU 的换阶方法各自复位，两个分布式后端不复位）。"""
        from autoflowcfd.core.time_integration.adaptive_cfl import AdaptiveCFLController

        solver = _fake_solver(current_order=0, target_order=2, resumed=False)
        ctrl = AdaptiveCFLController(cfl_start=0.02, cfl_max=0.5, cfl_min=0.01, ramp_steps=0)
        solver._cfl_controller = ctrl
        inner = solver.step
        first_step_cfl = {}

        def _step(dt):
            first_step_cfl.setdefault(solver.current_order, ctrl.cfl_number)
            ctrl.cfl_number = 0.3          # 模拟本阶内控制器已经爬起来
            return inner(dt)
        solver.step = _step
        with patch("autoflowcfd.fr.operators.generate_fr_operators", side_effect=_fake_generate_ops):
            run_order_continuation(solver, max_iter=30, dt=1e-3, tol=1e-6)
        assert first_step_cfl == {0: 0.02, 1: 0.02, 2: 0.02}
