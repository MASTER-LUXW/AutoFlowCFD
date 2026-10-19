"""双时间步的隐式内迭代（2026-10-10，`time_integration/implicit/newton/physical_time.py`）。

每个物理步解 BDF 方程 `G(U) = R(U) + (a0 U - b)/dt = 0`：平均流与带输运方程的湍流模型紧耦合，内迭代是
带物理时间项的 Newton-Krylov 步。此前内迭代是显式伪时间推进（平均流 `dual.py`、湍流
`PhysicalTimeTerm`）：伪时间步受最小单元的显式稳定极限约束，带壁面边界层的网格上默认 20 次内迭代只把
伪残差降不到 1%，得到的不是 BDF 方程的解。本文件验证：

* 内迭代收敛到的就是 BDF1/BDF2 方程的解（可手算的线性模型）；
* 物理步起点已满足 BDF 方程到舍入水平时按更新量判收敛；
* 真实求解器（速度进口 / 压力出口 / 无滑移壁通道）每个物理步的内迭代在远高于显式稳定极限的 dt 下收敛，
  湍流场有界；
* 产生项渐变按物理步计；
* 续算与连续计算逐位相同（平均流与湍流的上一时间层随 checkpoint 持久化），缺了湍流那一层就接不上；
* 单 GPU、CPU MPI、多 GPU 与单机 CPU 一致。
"""

from types import SimpleNamespace

import numpy as np
import pytest

from autoflowcfd.core.time_integration import TimeIntegrationScheme
from autoflowcfd.core.time_integration.base import DUAL_TIME_UPDATE_FLOOR
from autoflowcfd.core.time_integration.implicit import step_newton_krylov
from autoflowcfd.core.time_integration.implicit.newton.physical_time import (
    BdfTerm, advance_physical_step, inner_iteration_suffix, previous_newton_state, store_newton_state,
)
from autoflowcfd.core.turbulence.dual_time import reset_dual_time_history
from autoflowcfd.core.utils.checkpoint_time import TURBULENCE_PREVIOUS_FIELD
from tests.unit._implicit_jfnk_common import _N, _NV, _SCALES, _linear_system
from tests.unit.test_gpu_solver_order_continuation import _patch_gpu_modules  # noqa: F401（自动夹具）
from tests.validation._channel_mesh import build_channel_mesh_prism, channel_wall_source

DUAL = TimeIntegrationScheme.DUAL_TIME
LX, H, LZ = 0.4, 0.1, 0.08
_BC = {"x_min": "VELOCITY_INLET", "x_max": "PRESSURE_OUTLET", "wall_bottom": "WALL", "wall_top": "WALL",
       "z_min": "SYMMETRY", "z_max": "SYMMETRY"}


# ---------------------------------------------------------------------------
# 1. 内迭代收敛到 BDF 方程的解
# ---------------------------------------------------------------------------

def _owner(max_inner=20, tolerance=1e-10, controller=None):
    owner = SimpleNamespace(time_integrator=SimpleNamespace(dual_time_steps=max_inner, dual_time_tolerance=tolerance),
                            _cfl_controller=controller, _newton_forcing=object())
    reset_dual_time_history(owner)
    return owner


def _advance(owner, residual, state, dt, dtau=1e3):
    """用真实的 `step_newton_krylov` 推进一个物理步；`state["u"]` 原地换成新解。"""
    x_now = state["u"].copy()

    def newton_step(term, first):
        state["first"].append(first)
        state["u"], info = step_newton_krylov(residual, state["u"], np.full(_N, dtau), _SCALES,
                                              gmres_max_iter=80, gmres_restart=40, physical_time=term)
        return info

    summary = advance_physical_step(owner, x_now, previous_newton_state(owner, np, _N, 1), newton_step, dt)
    store_newton_state(owner, x_now, _N, 1, _NV)
    return summary


def test_inner_iterations_solve_the_bdf_equations():
    """线性系统 `R(U) = A (U - U*)`：BDF1 的解是 `(I + dt A) U1 = U0 + dt A U*`，第二步 BDF2。伪时间步
    取到物理步的 1e4 倍（伪时间项可忽略）也收敛。"""
    residual, u_star, a = _linear_system()
    dt = 0.1
    state = {"u": u_star * 1.3, "first": []}
    owner = _owner()
    u0 = state["u"].copy()
    eye = np.eye(_NV)

    s1 = _advance(owner, residual, state, dt)
    u1 = state["u"]
    expected1 = np.linalg.solve(eye + dt * a, (u0 + dt * u_star @ a.T).T).T
    np.testing.assert_allclose(u1 / _SCALES, expected1 / _SCALES, rtol=0, atol=1e-9)
    assert s1["converged"] and s1["a0"] == 1.0 and s1["iterations"] <= 12      # 线性求解只到 0.1（无 forcing）
    np.testing.assert_array_equal(owner._dual_time_U_prev, u0)
    assert owner._dual_time_turb_prev is None and owner._newton_forcing is None

    s2 = _advance(owner, residual, state, dt)
    u2 = state["u"]
    expected2 = np.linalg.solve(1.5 * eye + dt * a, (2.0 * u1 - 0.5 * u0 + dt * u_star @ a.T).T).T
    np.testing.assert_allclose(u2 / _SCALES, expected2 / _SCALES, rtol=0, atol=1e-9)
    assert s2["converged"] and s2["a0"] == 1.5
    # 每个物理步只有第一次内迭代带"推进产生项渐变"的标记
    assert [f for f in state["first"] if f] == [True, True]
    assert "inner=%d " % s2["iterations"] in inner_iteration_suffix(owner)


def test_bdf_term_coefficients_and_padded_rows():
    x_now = np.arange(12.0).reshape(4, 3)
    x_prev = x_now - 1.0
    bdf1, bdf2 = BdfTerm(x_now, None, 0.5), BdfTerm(x_now, x_prev, 0.5)
    assert (bdf1.a0, bdf1.shift, bdf2.a0, bdf2.shift) == (1.0, 2.0, 1.5, 3.0)
    np.testing.assert_array_equal(bdf2.b_over_dt, (2.0 * x_now - 0.5 * x_prev) / 0.5)
    real = np.array([True, True, False, True])
    g = bdf1.bind(lambda x: 10.0 * x, real)(x_now + 1.0)
    np.testing.assert_array_equal(g[real], (10.0 * (x_now + 1.0) + 2.0 * 1.0)[real])
    np.testing.assert_array_equal(g[2], 0.0)                       # 零填充槽位：残差与物理时间项都不参与
    with pytest.raises(ValueError, match="物理时间步长必须为正"):
        BdfTerm(x_now, None, 0.0)
    with pytest.raises(ValueError, match="形状"):
        BdfTerm(x_now, x_prev[:2], 0.5)


def test_state_already_solving_the_bdf_equations_converges_on_the_update_norm():
    """从定常解出发（`R = 0`）：`||G||` 的初值就是舍入噪声，相对下降判据不成立，按 Newton 更新量判收敛。
    修复前这种物理步用完全部内迭代并报"未收敛"。"""
    residual, u_star, _a = _linear_system()
    state = {"u": u_star.copy(), "first": []}
    owner = _owner(max_inner=12)
    summary = _advance(owner, residual, state, 0.1)
    assert summary["converged"] and summary["iterations"] <= 2
    assert np.abs(state["u"] / u_star - 1.0).max() <= 10 * DUAL_TIME_UPDATE_FLOOR


def test_pseudo_time_cfl_restarts_every_physical_step_and_bad_inputs_are_rejected():
    from autoflowcfd.core.time_integration.adaptive_cfl.policy import DUAL_TIME_PSEUDO_CFL_START, build_cfl_policy
    from autoflowcfd.core.time_integration.adaptive_cfl.ser import SERCFLController

    controller, fixed = build_cfl_policy(DUAL)
    assert isinstance(controller, SERCFLController) and fixed is None
    assert controller.cfl_start == DUAL_TIME_PSEUDO_CFL_START
    residual, u_star, _a = _linear_system()
    owner = _owner(controller=controller)
    state = {"u": u_star * 1.3, "first": []}
    controller.cfl_number = 7.0                                    # 上一个物理步留下的值
    seen = []

    def newton_step(term, first):
        seen.append(controller.cfl_number)
        state["u"], info = step_newton_krylov(residual, state["u"], np.full(_N, 1e3), _SCALES,
                                              gmres_max_iter=80, gmres_restart=40, physical_time=term)
        return info

    advance_physical_step(owner, state["u"].copy(), None, newton_step, 0.1)
    assert seen[0] == DUAL_TIME_PSEUDO_CFL_START

    with pytest.raises(ValueError, match="内迭代次数"):
        advance_physical_step(_owner(max_inner=0), u_star, None, newton_step, 0.1)
    # 上一时间层的列数与当前不符（换了湍流模型）：退回 BDF1
    s = advance_physical_step(_owner(), state["u"].copy(), np.zeros((_N, _NV + 1)), newton_step, 0.1)
    assert s["a0"] == 1.0


def test_krylov_operator_takes_the_pseudo_time_term_from_this_step():
    """Krylov 算子的对角项 `1/dtau` 由本次求解的 `dtau` 给出，不向预处理对象要。此前它取自预处理对象
    （`add_ptc_term`），而块预处理对象跨步复用、带着装配时的 `dtau`：物理步换了 `dt` 或伪时间 CFL 变了
    之后解的是另一个线性系统，BDF2 步的内迭代每次只降到 0.45 倍。"""
    residual, u_star, a = _linear_system()
    u0 = u_star * 1.2
    built_with = []

    class _StaleBlocks:
        """只做恒等预处理，并记录被要求按哪个 `dtau` 构造。"""
        flexible = False

        def begin_step(self, *args):
            pass

        def preconditioner(self, dtau, n_var):
            built_with.append(float(np.max(dtau)))
            return self

        @staticmethod
        def apply(v):
            return v

        def stale_budget(self):
            return None

        recoverable = False

        restart_cap = None

        def recover(self, *args):
            return False

        def record(self, *args, **kwargs):
            pass

    class _TightForcing:
        """线性系统解到 1e-11（默认 0.1 只够定方向，看不出算子对不对）。"""

        @staticmethod
        def next_eta(res_norm, tol_nonlinear):
            return 1e-11

        @staticmethod
        def record_step(*args, **kwargs):
            pass

    dtau, dt = 0.05, 0.2
    term = BdfTerm(u0, None, dt)
    u1, info = step_newton_krylov(residual, u0, np.full(_N, dtau), _SCALES, gmres_max_iter=200, gmres_restart=60,
                                  block_precond=_StaleBlocks(), physical_time=term, forcing=_TightForcing())
    # 一个 Newton 步精确解 (1/dtau + 1/dt + A) dU = -G(U0)，G(U0) = R(U0)
    expected = u0 + np.linalg.solve((1.0 / dtau + 1.0 / dt) * np.eye(_NV) + a, -residual(u0).T).T
    np.testing.assert_allclose(u1 / _SCALES, expected / _SCALES, rtol=0, atol=1e-6)
    assert built_with and built_with[0] == pytest.approx(dtau / (1.0 + dtau / dt))
    assert info["update_norm"] > 0.0


# ---------------------------------------------------------------------------
# 2. 单机 CPU：内迭代收敛、有界、渐变按物理步计、续算逐位相同
# ---------------------------------------------------------------------------

def _cpu(model, ny=6, vel=30.0, inner=20):
    from autoflowcfd.core.fr_solver import FRSolver
    from autoflowcfd.core.fr_solver.turbulence.wall_distance import apply_wall_distance_source

    mesh = build_channel_mesh_prism(1, ny, ny, 2, LX, H, LZ)
    mesh.boundary_bc_types = dict(_BC)
    s = FRSolver(mesh=mesh, order=1, turb_model_name=model, time_scheme=DUAL, vel_inf=vel,
                 dual_time_inner_iter=inner)
    s.order_continuation_enabled = False
    apply_wall_distance_source(s, channel_wall_source(LX, H, LZ))
    return s


@pytest.mark.parametrize("model, dt", [("NONE", 1e-3), ("SST", 1e-3), ("SA", 1e-3), ("DDES", 1e-4)])
def test_inner_iterations_converge_on_a_wall_bounded_channel(model, dt):
    """均匀来流突然加上无滑移壁，dt 远高于近壁单元的显式稳定极限（约 1e-7 s）。每个物理步的 BDF 方程都解到
    容差，湍流场有界。

    此前两条路都走不通：显式内迭代下湍流只按物理 dt 做一次显式更新时 k 十步内到 -1e9；隐式内迭代在进/出口
    幽灵态按本侧法向速度做间断切换时（`boundary/fr_ghost_state.py`）30 次内迭代 `||G||` 不降反升
    （6.9 -> 1.1e3）或直接报错。"""
    s = _cpu(model)
    has_fields = s.turb_model is not None
    initial = [np.array(f, copy=True) for f in s.turb_model.transported_fields()] if has_fields else []
    for _ in range(4):
        s.step(dt)
        inner = s._dual_time_inner
        assert inner["converged"], inner
        assert inner["iterations"] <= 10, inner
    assert np.isfinite(s.state.U).all()
    if has_fields:
        for name, f0, f in zip(s.turb_model.TRANSPORTED_FIELDS, initial, s.turb_model.transported_fields()):
            assert np.isfinite(f).all(), name
            assert np.abs(f).max() <= 1e3 * max(np.abs(f0).max(), 1e-12), (name, float(np.abs(f).max()))
            if model != "SA":                          # SA-neg 允许 nu_tilde 为负
                assert f.min() > 0.0, (name, float(f.min()))


def test_production_ramp_advances_once_per_physical_step():
    s = _cpu("SST", ny=3, vel=10.0)
    for _ in range(3):
        s.step(2e-4)
    assert s._turb_ramp_step == 3


def _perturbed(model="SST"):
    s = _cpu(model, ny=3, vel=10.0, inner=40)
    # BDF 方程解到 1e-10：续算与连续计算的差只剩线性求解的舍入，远小于"少一个时间层"的差
    s.time_integrator.dual_time_tolerance = 1e-10
    rng = np.random.default_rng(7)
    s.state.U[..., 1:4] *= 1.0 + 0.05 * rng.standard_normal(s.state.U[..., 1:4].shape)
    s.state._update_primitives()
    return s


@pytest.mark.parametrize("model", ["SST", "SA"])
def test_resumed_run_solves_the_same_bdf_equations_including_turbulence(tmp_path, model):
    """续算的第一个物理步与连续计算解的是同一个 BDF2 方程（平均流与湍流的上一时间层、物理时间步长、
    产生项渐变进度都随 checkpoint 恢复）。两者不逐位相同：隐式步跨步复用的块预处理与伪时间步缩放不进
    checkpoint，线性求解的迭代路径不同，解在内迭代容差以内一致。"""
    from autoflowcfd.cli.solve.checkpoint_io.restore import restore_solver_state_from_fields
    from autoflowcfd.cli.solve.checkpoint_io.write import write_checkpoint
    from autoflowcfd.core.utils.checkpoint import CheckpointManager

    dt = 2e-4
    reference = _perturbed(model)
    reference.solve(max_iter=3, dt=dt, tol=0.0)

    first = _perturbed(model)
    first.solve(max_iter=2, dt=dt, tol=0.0)
    path = write_checkpoint(first, str(tmp_path), 2, "mesh.nas", order=1, turbulence_model=model.lower(),
                            backend="cpu", quiet=True)
    meta = CheckpointManager().load(path)[3]
    n_fields = len(first.turb_model.TRANSPORTED_FIELDS)
    assert meta["fields"][TURBULENCE_PREVIOUS_FIELD].shape == first.state.U.shape[:2] + (n_fields,)

    resumed = _perturbed(model)
    restore_solver_state_from_fields(resumed, meta["fields"], meta)
    resumed.solve(max_iter=1, dt=dt, tol=0.0)
    assert resumed._dual_time_inner["a0"] == 1.5 and resumed._dual_time_inner["converged"]
    assert resumed._turb_ramp_step == reference._turb_ramp_step

    def distance(solver):
        """与连续计算的最大相对差（平均流按各变量量级，湍流按各场最大值）。"""
        ref = reference.state.U
        scale = np.abs(ref).max(axis=(0, 1))
        scale[1:4] = np.abs(ref[..., 1:4]).max()
        d = [float((np.abs(solver.state.U - ref) / scale).max())]
        d += [float(np.abs(a - b).max() / np.abs(b).max())
              for a, b in zip(solver.turb_model.transported_fields(), reference.turb_model.transported_fields())]
        return max(d)

    resumed_distance = distance(resumed)
    assert resumed_distance <= 1e-7, resumed_distance

    # 对照：不恢复湍流的上一时间层（列数与当前未知量不符，续算第一步退回 BDF1）就接不上
    bdf1 = _perturbed(model)
    fields = {k: v for k, v in meta["fields"].items() if k != TURBULENCE_PREVIOUS_FIELD}
    restore_solver_state_from_fields(bdf1, fields, meta)
    bdf1.solve(max_iter=1, dt=dt, tol=0.0)
    assert bdf1._dual_time_inner["a0"] == 1.0
    assert distance(bdf1) >= 100.0 * max(resumed_distance, 1e-9), (distance(bdf1), resumed_distance)


def test_previous_level_of_another_model_is_not_restored():
    """换了湍流模型（未知量个数不同）时上一时间层没有意义：不恢复，第一个物理步用 BDF1。"""
    from autoflowcfd.core.utils.checkpoint_time import restore_turbulence_previous

    s = _cpu("SA", ny=3, vel=10.0)
    n_cells, n_sps = s.turb_model.transported_fields()[0].shape
    restore_turbulence_previous(s, np.zeros((n_cells, n_sps, 2)))
    assert s._dual_time_turb_prev is None
    restore_turbulence_previous(s, np.ones((n_cells, n_sps, 1)))
    assert s._dual_time_turb_prev[0].shape == (n_cells, n_sps)


# ---------------------------------------------------------------------------
# 3. 后端一致
# ---------------------------------------------------------------------------

def _assert_turbulence_matches(got_model, ref_model, tol, what):
    for name in tuple(ref_model.TRANSPORTED_FIELDS) + ("nu_t",):
        a, b = np.asarray(getattr(got_model, name)), np.asarray(getattr(ref_model, name))
        r = np.abs(a[:b.shape[0]] - b).max() / max(np.abs(b).max(), 1e-300)
        assert r <= tol, f"{what}：{name} 相对差 {r:.2e}"


def _assert_mean_flow_matches(got, ref, tol, what):
    scale = np.abs(ref).max(axis=(0, 1))
    scale[1:4] = np.abs(ref[..., 1:4]).max()                 # 动量三分量共用动量模的尺度
    r = (np.abs(np.asarray(got)[:ref.shape[0], :, :5] - ref[..., :5]) / scale[:5]).max()
    assert r <= tol, f"{what}：平均流相对差 {r:.2e}"


@pytest.mark.parametrize("model", ["NONE", "SST", "SA"])
def test_single_gpu_matches_cpu(model):
    from autoflowcfd.core.gpu.solver.gpu_solver import GPUFRSolver

    cpu = _cpu(model, ny=3, vel=10.0)
    mesh = build_channel_mesh_prism(1, 3, 3, 2, LX, H, LZ)
    mesh.boundary_bc_types = dict(_BC)
    gpu = GPUFRSolver(mesh=mesh, order=1, turb_model_name=model, vel_inf=10.0, time_scheme=DUAL,
                      wall_distance_source=channel_wall_source(LX, H, LZ))
    gpu.order_continuation_enabled = False
    for _ in range(3):
        cpu.step(2e-4)
        gpu.step(2e-4)
        assert gpu._dual_time_inner["converged"] and cpu._dual_time_inner["converged"]
        assert gpu._dual_time_inner["iterations"] == cpu._dual_time_inner["iterations"]
    # 两侧只差浮点重结合，经每步约 1 次的 GMRES 传播后在 1e-7 量级（同 test_multi_gpu_coupled_nk.py 的实测）
    shape = cpu.state.U.shape
    _assert_mean_flow_matches(gpu.U_gpu, cpu.state.U, 1e-6, "单 GPU")
    _assert_mean_flow_matches(np.asarray(gpu._dual_time_U_prev).reshape(shape),
                              np.asarray(cpu._dual_time_U_prev).reshape(shape), 1e-6, "单 GPU 上一时间层")
    if model == "NONE":
        assert gpu._dual_time_turb_prev is None
        return
    assert gpu._turb_ramp_step == cpu._turb_ramp_step == 3
    _assert_turbulence_matches(gpu.turb_model_gpu, cpu.turb_model, 2e-6, "单 GPU")
    for a, b in zip(gpu._dual_time_turb_prev, cpu._dual_time_turb_prev):
        np.testing.assert_allclose(np.asarray(a), b, rtol=2e-6, atol=1e-10)


@pytest.mark.parametrize("model", ["SST", "SA"])
def test_cpu_distributed_matches_single_machine(model):
    from tests.unit.test_distributed_turbulence_bc_parity import _assert_same, _pair

    single, dist = _pair(DUAL, 1, model)
    for _ in range(3):
        single.step(2e-4)
        dist.step(2e-4)
        assert dist._dual_time_inner["converged"] and single._dual_time_inner["converged"]
        assert dist._dual_time_inner["iterations"] == single._dual_time_inner["iterations"]
    assert dist._turb_ramp_step == single._turb_ramp_step == 3
    _assert_same(single, dist, "DUAL_TIME 三个物理步", tol=2e-6)
    for a, b in zip(dist._dual_time_turb_prev, single._dual_time_turb_prev):
        np.testing.assert_allclose(a[:b.shape[0]], b, rtol=2e-6, atol=1e-10)


@pytest.mark.parametrize("model", ["SST", "SA"])
def test_cpu_distributed_checkpoint_carries_turbulence_previous_level(tmp_path, model):
    from autoflowcfd.core.mpi.distributed_checkpoint import distributed_load_checkpoint, distributed_save_checkpoint
    from tests.unit.test_distributed_turbulence_bc_parity import _pair

    _single, a = _pair(DUAL, 1, model)
    a.order_continuation_enabled = False
    a.solve(max_iter=2, dt=2e-4, tol=0.0)
    path = distributed_save_checkpoint(a, str(tmp_path), 2, "mesh.nas", 1, model.lower(), "cpu")
    _single_b, b = _pair(DUAL, 1, model)
    distributed_load_checkpoint(path, b)
    assert len(b._dual_time_turb_prev) == len(a.turb_model.TRANSPORTED_FIELDS)
    n = a.partition.n_local_cells
    for x, y in zip(b._dual_time_turb_prev, a._dual_time_turb_prev):
        np.testing.assert_array_equal(x[:n], y[:n])


def test_time_budget_counts_physical_time_in_dual_time():
    """双时间步下"走过的时间"是物理时间 `n * dt`，每个单元相同。此前累加的是伪时间步（CFL 1e4 的局部步长），
    cube_demo 4 个 dt=1e-4 的物理步被报成"累计 tau 中位 0.27 s、物体尺度 18 个特征时间"。"""
    s = _cpu("NONE", ny=3, vel=10.0)
    for _ in range(3):
        s.step(2e-4)
    np.testing.assert_allclose(s.tau_accum, 3 * 2e-4, rtol=1e-12)


def test_inner_iterations_stop_with_an_error_when_the_linear_solves_keep_failing():
    """`a0/dt + J` 不定（dt 大于不稳定模态的时间尺度）时线性求解解不动：连续失败 3 次就报错并指向 dt，
    不把内迭代次数烧完、也不带着没解出来的状态进下一个物理步。cube_demo SA P1 dt=1e-3 的实测是每次内迭代
    GMRES 200 次、线性残差 1.0、1220 s。"""
    from autoflowcfd.core.time_integration.implicit.newton.physical_time import MAX_CONSECUTIVE_LINEAR_FAILURES

    calls = []

    def failing_step(term, first):
        calls.append(first)
        return dict(res_norm=1.0, res_norm_new=1.0, gmres_iters=50, linear_rel_residual=0.99, theta=1.0,
                    n_dtau_cuts=0, update_norm=1.0)

    x = np.ones((4, 5))
    with pytest.raises(RuntimeError, match="减小 --dt"):
        advance_physical_step(_owner(max_inner=30), x, None, failing_step, 1e-3)
    assert len(calls) == MAX_CONSECUTIVE_LINEAR_FAILURES

    # 失败不连续（中间有一次解到位）不触发
    seq = iter([0.99, 0.99, 0.05, 0.99, 0.99, 0.05] * 5)

    def flaky_step(term, first):
        return dict(res_norm=1.0, res_norm_new=0.9, gmres_iters=10, linear_rel_residual=next(seq), theta=1.0,
                    n_dtau_cuts=0, update_norm=1.0)

    summary = advance_physical_step(_owner(max_inner=12), x, None, flaky_step, 1e-3)
    assert summary["iterations"] == 12 and not summary["converged"]
