"""真实缺陷回归测试（2026-08-22 真实网格续算时发现，用户直接问："如果当前是P1，resume是从P1开始运算吗"）。

`run_order_continuation`（`core/utils/order_continuation/run.py`）在 `solver.state.U.shape[1] != 1`（不是 P0）时
无条件把 `solver.state` 重置为均匀来流、从 P0 重新爬坡。这条分支是给**全新构造**的 FRSolver 准备的：它的初始
状态是按**目标**阶数开的均匀来流，爬坡前降到 P0 起点是对的。

但 `solve resume` 走同一条路径。爬坡中途保存的 checkpoint（例如 P1 上已有真实收敛进展）同样满足
`state.U.shape[1] != 1`，会进入同一条分支——静默丢掉续算恢复的真实解、从均匀 P0 来流重新开始，不报错也不
警告。续算 P1 / P2 的 checkpoint 与根本没有续算无从区分。

修复：`rebuild_solver_from_checkpoint` 在恢复真实状态后给求解器打上 `_resumed_from_checkpoint = True`；
`run_order_continuation` 见到这个标记时跳过重置到 P0 的分支，阶数序列从 `solver.current_order`（checkpoint 的
真实阶数）起，而不是总从 P0 起。
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from autoflowcfd.core.fr_solver.convergence_history import ConvergenceHistory
from autoflowcfd.core.time_integration.base import TimeIntegrationScheme
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
        ops=SimpleNamespace(n_sps=n_sps),
        mesh=mesh,
        state=SimpleNamespace(U=U, n_cells=2, n_vars=7),
        freestream={"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0},
        mu_molecular=1.8e-5, _turbulence_intensity=0.01, _viscosity_ratio=5.0,
        turb_model=None,
        wall_distance=None,
        sgs_model=None,
        _resumed_from_checkpoint=resumed,
        step=_decreasing_step,
        # 真实求解器恒有时间积分器（advance 按格式推进物理时间）
        time_integrator=SimpleNamespace(scheme=TimeIntegrationScheme.SSP_RK3, current_time=0.0),
        convergence_history=ConvergenceHistory(),
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
    return SimpleNamespace(n_sps=n)


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
    """真实缺陷回归测试（2026-09-05，用户直接指出："phase_max_iter 的默认值应该是 max_iter // len(orders)，
    两者不是或的关系"）：最终（目标）阶段"吃掉剩余全部步数、不被 len(orders) 均分"这条规则——`phase_max_iter`
    参数（2026-09-01）存在的全部意义——此前被 `phase_max_iter is not None` 门住，只在调用方显式传
    `--phase-max-iter` 时生效。不传它的调用方（CLI 的常见情形）按构造得到修复前的旧行为：最终阶段与其他阶段
    一样被截在 `max_iter // len(orders)`，不管前面的阶段剩下多少预算。

    修复：不传 `phase_max_iter` 只决定**这一个数**的默认值（`max_iter // len(orders)`）；"最终阶段拿剩余全部
    预算"无条件成立，默认值与显式值都一样。

    本测试构造一个不依赖任何残差下降 / 收敛巧合就能看出差别的场景：P0（非最终阶段）按 CL-02 残差下降判据提前
    升阶、留下未用的预算；P1（最终阶段）被脚本化为自己永远不触发任何退出条件，于是恰好跑满分到的
    `stage_iter_budget`——直接暴露它拿到的是哪一份预算。
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


class TestTotalIterationBudget:
    """`max_iter` 是总步数：`phase_max_iter` 大于剩余步数时非最终阶段也不能越过它；用完时停在当前阶数，
    不再做一次没有步数可跑的升阶（2026-10-14：plate_demo `solve resume --max-iter 8 --phase-max-iter 100`
    在 P1 跑了 30 步）。"""

    @staticmethod
    def _never_converging_solver(current_order, target_order):
        solver = _fake_solver(current_order=current_order, target_order=target_order, resumed=True)
        solver.step = lambda dt: 1.0          # 残差不降：只有步数上限能让阶段结束
        return solver

    def test_non_final_phase_is_capped_by_the_total_budget(self):
        solver = self._never_converging_solver(1, 2)
        with patch("autoflowcfd.fr.operators.generate_fr_operators", side_effect=_fake_generate_ops):
            result = run_order_continuation(solver, max_iter=8, dt=1e-3, tol=1e-6, phase_max_iter=100)
        assert result.iterations == 8
        assert solver.current_order == 1 and solver.mesh.set_order_calls == []

    def test_remaining_budget_still_reaches_the_target_order(self):
        solver = self._never_converging_solver(1, 2)
        with patch("autoflowcfd.fr.operators.generate_fr_operators", side_effect=_fake_generate_ops):
            result = run_order_continuation(solver, max_iter=12, dt=1e-3, tol=1e-6, phase_max_iter=5)
        assert result.iterations == 12
        assert solver.mesh.set_order_calls == [2]          # P1 跑 5 步后升阶，目标阶数吃掉剩下的 7 步
