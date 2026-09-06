"""CPU MPI"传统模式"分布式 Order Continuation 决定性验证（2026-09-02）。

背景：`DistributedFRSolver` 之前完全没有 Order Continuation 机制——
`solve steady --order 2 --n-ranks>1` 会直接在目标阶数（P2）从均匀
自由流场开始求解，跳过单机路径自动执行的 P0->P1->P2 逐步爬坡（见
`core/mpi/distributed_order_continuation.py` 模块文档）。本文件验证：

1. `DistributedFRSolver.solve()` 在 `order>=2` 时真正自动分派到
   `run_distributed_order_continuation`（不是仍然走原来的单一阶数
   直接迭代循环）。
2. 阶数切换（`_interpolate_to_new_order`）正确重建 partition/
   dist_flat_face/state/halo_exchange，新阶数下 `step()` 仍能正常
   产出有限残差（不是插值/重建之后第一步就 NaN/形状不匹配崩溃）。
3. 从非 P0 直接构造的求解器（当前 CLI 生产路径的实际构造方式）真正
   被重置回 P0 重新爬坡，而不是从目标阶数原地卡住不动
   （`orders` 序列退化成单元素）。
"""

import types

import numpy as np
import pytest

from autoflowcfd.fr.operators import generate_fr_operators
from autoflowcfd.core.time_integration.base import TimeIntegrationScheme
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


@pytest.fixture()
def mesh_p2_and_ops_order1():
    """同 `mesh_p2_and_ops` 文档"不能用 module 级共享 fixture"的理由，
    只是阶数换成 1（`TestResumeCeilingFractionResetHeuristicDistributed`
    不需要 P2 的构造开销，只需要一个真实、形状/属性完整的
    `turb_model`）。"""
    order = 1
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    return mesh, ops


@pytest.fixture()
def mesh_p2_and_ops():
    # 每个测试独立构造一份新的 mesh/ops（不能用 module 级共享 fixture：
    # Order Continuation 会通过 `mesh.set_order` 原地改变 mesh 的活动
    # 阶数/`face_flux_points` 缓存，多个测试共享同一个 mesh 实例会互相
    # 污染彼此的阶数状态，与真实 CLI 场景"每次求解器构造都拿一份专属
    # mesh"不符）。
    order = 2
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    return mesh, ops


def _make_p2_solver(mesh, ops, turb_model_name="none"):
    from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver

    kwargs = dict(
        mesh=mesh, ops=ops, face_connectivity=mesh.face_connectivity,
        n_ranks=1, backend="cpu", order=2, turb_model_name=turb_model_name,
        time_scheme=TimeIntegrationScheme.SSP_RK3,
        mu_molecular=1.8e-5, rho_inf=1.225, vel_inf=33.33, p_inf=101325.0,
    )
    return DistributedFRSolver(mesh=mesh, ops=ops, **{k: v for k, v in kwargs.items() if k not in ("mesh", "ops")})


class TestDistributedOrderContinuationDispatch:
    def test_solve_at_order_2_auto_dispatches_to_order_continuation(self, mesh_p2_and_ops):
        """真实构造出的 solver（现有 CLI 生产路径同款：直接在目标阶数
        构造）调用 solve() 后必须真正走完 P0->P1->P2 爬坡——不是仍在
        P2 原地跑，也不是在 P0 卡住不前进。"""
        mesh, ops = mesh_p2_and_ops
        solver = _make_p2_solver(mesh, ops)

        assert solver.order == 2
        # 构造完成后 current_order 恒等于目标阶数（当前 CLI 生产路径
        # 的实际构造方式——mesh/ops 直接在目标阶数生成），这正是需要
        # 重置到 P0 才能真正爬坡的场景。
        assert solver.current_order == 2

        # dt=1e-9（不是分布式路径其余测试常用的 1e-6）：分布式路径用
        # 全局固定步长，没有单机路径 `_compute_local_time_step()` 那种
        # 逐 SP 自适应 CFL（见 step() 文档"dt 参数的语义"一节，既有的
        # 架构差异，不是本次改动引入）——P2 阶数下 CFL 稳定域比 P0/P1
        # 更窄（真实验证过：同一个网格/turb_model='none'，dt=1e-6 在
        # P0/P1 均能稳定运行，但 P2 阶段第 2 步残差就跳到 1.4e6，第 4
        # 步 inf；换成 dt=1e-9 后 P2 全程有限、数值稳定），与本次 Order
        # Continuation 重建逻辑本身是否正确无关，只是这个合成小网格在
        # 固定步长下的真实 CFL 约束。
        result = solver.solve(n_steps=60, dt=1e-9, output_interval=1000)

        # 爬坡走完后必须停在目标阶数 P2，不能停在中途的 P0/P1。
        assert solver.current_order == 2
        assert np.isfinite(result.final_residual)
        assert result.iterations > 0
        # 新阶数（P2，27 SPs/cell）下的 state 形状必须与 ops 一致——
        # 阶数切换后 partition/dist_flat_face/state 没有正确重建的话，
        # 这里会是旧阶数的形状。
        assert solver.state.U.shape[1] == ops.D_3d.shape[0] == 27
        assert np.all(np.isfinite(solver.state.U[:solver.partition.n_local_cells]))

    def test_order_1_does_not_trigger_order_continuation(self, mesh_p2_and_ops):
        """P1 不应该触发 Order Continuation（与单机 `self.order >= 2`
        同一个阈值），构造时的 current_order 应该保持不变，直接迭代。"""
        mesh_p2, ops_p2 = mesh_p2_and_ops
        order = 1
        mesh = _build_synthetic_mixed_mesh(order)
        ops = generate_fr_operators(order)
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver

        solver = DistributedFRSolver(
            mesh=mesh, ops=ops, face_connectivity=mesh.face_connectivity,
            n_ranks=1, backend="cpu", order=1, turb_model_name="none",
            time_scheme=TimeIntegrationScheme.SSP_RK3,
            mu_molecular=1.8e-5, rho_inf=1.225, vel_inf=33.33, p_inf=101325.0,
        )
        assert solver.current_order == 1
        solver.solve(n_steps=5, dt=1e-6, output_interval=1000)
        # 没有触发 Order Continuation：current_order 应该原地不变。
        assert solver.current_order == 1
        assert np.all(np.isfinite(solver.state.U[:solver.partition.n_local_cells]))


class TestResumeCeilingFractionResetHeuristicDistributed:
    """真实完整性缺口修复回归测试（2026-09-05）：CPU MPI 分布式路径
    此前完全没有单机版早就有的"resume 时检测湍流场是否被上界大面积
    钳制、若是则重置到来流初值"安全网，见
    `distributed_order_continuation.py::
    _reset_turbulence_if_resumed_field_exploded` 文档完整推导。直接
    测这个 helper 函数本身（不跑完整 `run_distributed_order_
    continuation` 迭代循环——那样后续真实物理 step() 会继续演化
    k_field/omega_field，让"重置前后是否等于 k_inf"这个断言变得不精确，
    与单机版 `TestResumeCeilingFractionResetHeuristic` 用 fake solver
    隔离测试是同一个理由，这里换成真实 `DistributedFRSolver`（n_ranks=1，
    单进程测试环境天然验证了 `allreduce_sum` 在 n_ranks=1 时的 no-op
    退化路径，多 rank 下的真正全局归约行为本身已经是 `allreduce_sum`
    自身的既有职责，不是本次要重新验证的范围）取得真实、形状/属性
    完整的 `turb_model`。"""

    def _make_solver_with_turb(self, mesh, ops, k_value, omega_value,
                                k_max=555.44, omega_max=1e6):
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver

        solver = DistributedFRSolver(
            mesh=mesh, ops=ops, face_connectivity=mesh.face_connectivity,
            n_ranks=1, backend="cpu", order=1, turb_model_name="sst",
            time_scheme=TimeIntegrationScheme.SSP_RK3,
            mu_molecular=1.8e-5, rho_inf=1.225, vel_inf=33.33, p_inf=101325.0,
        )
        solver.turb_model.k_field[:] = k_value
        solver.turb_model.omega_field[:] = omega_value
        solver.turb_model.k_max = k_max
        solver.turb_model.omega_max = omega_max
        return solver

    def test_healthy_high_turbulence_field_is_not_falsely_reset(self, mesh_p2_and_ops_order1):
        """真实复现场景同单机版：k_mean 远超"旧公式"会误判的阈值，但
        只是钝体尾流健康发展的高湍流度场，钳制占比很低，不应该被
        重置。"""
        from autoflowcfd.core.mpi.distributed_order_continuation import (
            _reset_turbulence_if_resumed_field_exploded,
        )
        mesh, ops = mesh_p2_and_ops_order1
        solver = self._make_solver_with_turb(mesh, ops, k_value=38.11, omega_value=28459.5)

        _reset_turbulence_if_resumed_field_exploded(solver)

        np.testing.assert_allclose(solver.turb_model.k_field, 38.11)
        np.testing.assert_allclose(solver.turb_model.omega_field, 28459.5)

    def test_genuinely_clamped_field_still_gets_reset(self, mesh_p2_and_ops_order1):
        """100% 的单元都钉在上界——必须仍然触发重置，这个修复只是改
        判据的统计口径（全局钳制比例），不是把安全网整个拆掉。"""
        from autoflowcfd.core.mpi.distributed_order_continuation import (
            _reset_turbulence_if_resumed_field_exploded,
        )
        mesh, ops = mesh_p2_and_ops_order1
        solver = self._make_solver_with_turb(mesh, ops, k_value=555.44, omega_value=1e6)

        _reset_turbulence_if_resumed_field_exploded(solver)

        k_inf_expected = 1.5 * (33.33 * 0.01) ** 2
        assert not np.allclose(solver.turb_model.k_field, 555.44)
        np.testing.assert_allclose(solver.turb_model.k_field, k_inf_expected, rtol=1e-6)
        assert np.all(solver.turb_model.nu_t == 0.0)

    def test_partial_clamping_below_threshold_is_not_reset(self, mesh_p2_and_ops_order1):
        """只有少数 SP（低于10%）触及上界——不应该触发重置。"""
        from autoflowcfd.core.mpi.distributed_order_continuation import (
            _reset_turbulence_if_resumed_field_exploded,
        )
        mesh, ops = mesh_p2_and_ops_order1
        solver = self._make_solver_with_turb(mesh, ops, k_value=1.0, omega_value=100.0)

        # 把 5% 的 (cell, sp) 元素钳制在上界（低于10%阈值）——用扁平化
        # 索引在整个 (n_local, n_sps) 数组上直接选取，不依赖具体单元数
        # 是否能整除出"5%"这种粒度。
        flat = solver.turb_model.k_field.reshape(-1)
        n_clamp = max(1, int(0.05 * flat.size))
        flat[:n_clamp] = solver.turb_model.k_max

        _reset_turbulence_if_resumed_field_exploded(solver)

        # 未被重置：绝大多数元素应该还是原来的 1.0，不是被清零的 k_inf。
        assert np.any(np.isclose(solver.turb_model.k_field, 1.0))


class TestDistributedOrderContinuationWithTurbulence:
    def test_sst_order_continuation_keeps_turb_fields_consistent(self, mesh_p2_and_ops):
        """湍流场（k_field/omega_field/nu_t）随阶数切换正确重塑形状，
        不在阶数切换后第一次 step() 就因为形状不匹配崩溃。"""
        mesh, ops = mesh_p2_and_ops
        solver = _make_p2_solver(mesh, ops, turb_model_name="sst")

        assert solver.turb_model is not None
        # dt=1e-9（不是分布式路径其余测试常用的 1e-6）：分布式路径用
        # 全局固定步长，没有单机路径 `_compute_local_time_step()` 那种
        # 逐 SP 自适应 CFL（见 step() 文档"dt 参数的语义"一节，既有的
        # 架构差异，不是本次改动引入）——P2 阶数下 CFL 稳定域比 P0/P1
        # 更窄（真实验证过：同一个网格/turb_model='none'，dt=1e-6 在
        # P0/P1 均能稳定运行，但 P2 阶段第 2 步残差就跳到 1.4e6，第 4
        # 步 inf；换成 dt=1e-9 后 P2 全程有限、数值稳定），与本次 Order
        # Continuation 重建逻辑本身是否正确无关，只是这个合成小网格在
        # 固定步长下的真实 CFL 约束。
        result = solver.solve(n_steps=60, dt=1e-9, output_interval=1000)

        assert solver.current_order == 2
        n_local = solver.partition.n_local_cells
        assert solver.turb_model.k_field.shape == (n_local, 27)
        assert solver.turb_model.omega_field.shape == (n_local, 27)
        assert np.isfinite(result.final_residual)
        assert np.all(np.isfinite(solver.turb_model.k_field))
        assert np.all(np.isfinite(solver.turb_model.omega_field))


class TestPhaseMaxIterDefaultGivesFinalStageRemainingBudgetDistributed:
    """Same real bug/fix as the single-machine `run_order_continuation`
    (see `test_order_continuation_resume.py::
    TestPhaseMaxIterDefaultGivesFinalStageRemainingBudget` for the full
    write-up) — `run_distributed_order_continuation` had the exact same
    `if phase_max_iter is not None: ... else: ...` bifurcation, gating
    "final stage eats the remaining budget" behind an explicit
    `--phase-max-iter`. Verified here with a minimal fake solver (the
    phase-budget arithmetic is pure orchestration logic, independent of
    any real distributed/GPU state) rather than a real
    `DistributedFRSolver`, mirroring the single-machine test's
    methodology."""

    def _make_fake_solver(self, residuals):
        calls = {"i": 0}

        def _scripted_step(dt):
            i = calls["i"]
            calls["i"] += 1
            return residuals[min(i, len(residuals) - 1)]

        def _fake_interpolate(new_order):
            solver.current_order = new_order

        solver = types.SimpleNamespace(
            order=1, current_order=0, step=_scripted_step,
            _resumed_from_checkpoint=False,
        )
        solver._interpolate_to_new_order = _fake_interpolate
        return solver

    def test_final_stage_gets_leftover_budget_without_explicit_phase_max_iter(self):
        from autoflowcfd.core.mpi.distributed_order_continuation import (
            run_distributed_order_continuation,
        )
        # Same construction as the single-machine test: P0 promotes early
        # (at i=20, the default residual_drop_threshold=100 met exactly),
        # using only 21 of its 50-iteration default share (max_iter=100,
        # len(orders)=2 -> 100//2=50); P1 (final) never triggers any exit
        # condition on its own (residual held flat at 10.0 once the
        # scripted sequence is exhausted) and must run for whatever budget
        # it's actually given.
        residuals = [1000.0] * 20 + [10.0]
        solver = self._make_fake_solver(residuals)

        result = run_distributed_order_continuation(solver, max_iter=100, dt=1e-3, tol=1e-6)

        # Old (buggy) behaviour: P1 capped at the same 50-iteration share
        # as P0 -> total_iter = 21 + 50 = 71. Fixed behaviour: P1 gets all
        # of max_iter's remainder -> total_iter = 21 + (100 - 21) = 100.
        assert result.iterations == 100
        assert result.converged is False
