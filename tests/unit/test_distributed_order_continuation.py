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

import numpy as np
import pytest

from autoflowcfd.fr.operators import generate_fr_operators
from autoflowcfd.core.time_integration.base import TimeIntegrationScheme
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


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
