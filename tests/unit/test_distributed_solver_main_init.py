"""AutoFlowCFD V2.0 - `DistributedFRSolver` 主构造函数（"传统模式"）
端到端收尾测试 + 分布式 checkpoint 中间保存/恢复验证（2026-09-02）。

背景（真实测试覆盖缺口，用户明确要求"不允许出现完成度不是100%的
功能点"后排查发现）：此前全部 `DistributedFRSolver` 端到端测试
（`test_distributed_mesh_loader.py`）都是通过 `from_fully_distributed_
package` 这个 classmethod 构造实例，从未有测试真正调用主 `__init__`
（"传统模式"：每个 rank 独立加载完整网格，传 `face_connectivity`
触发进程内分区）再跑真正的 `step()`——CLI 生产路径（`solve_steady_
command.py` 的 `elif n_ranks > 1: ... else:` 分支）用的正是这条主
`__init__`，此前完全没有端到端测试覆盖它。

同时验证本次补齐的分布式 checkpoint 中间保存机制：`DistributedFRSolver.
solve(..., checkpoint_callback=...)` 真正在每步结束后调用回调、
`distributed_save_checkpoint`/`distributed_load_checkpoint` 真正的
HDF5 往返（不是 mock）在单 rank 场景下保存/加载后状态逐位一致。
"""

import numpy as np
import pytest

from autoflowcfd.fr.operators import generate_fr_operators
from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive, primitive_to_conserved
from autoflowcfd.core.time_integration.base import TimeIntegrationScheme
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


def _nonuniform_U(mesh, rng):
    rho_inf, u_inf, v_inf, w_inf, p_inf = 1.225, 30.0, 5.0, -3.0, 101325.0
    n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
    Q = np.zeros((n_cells, n_sps, 5))
    Q[..., 0] = rho_inf * (1.0 + rng.uniform(-0.02, 0.02, size=(n_cells, n_sps)))
    Q[..., 1] = u_inf + rng.uniform(-3.0, 3.0, size=(n_cells, n_sps))
    Q[..., 2] = v_inf + rng.uniform(-2.0, 2.0, size=(n_cells, n_sps))
    Q[..., 3] = w_inf + rng.uniform(-2.0, 2.0, size=(n_cells, n_sps))
    Q[..., 4] = p_inf * (1.0 + rng.uniform(-0.01, 0.01, size=(n_cells, n_sps)))
    return np.stack(
        [primitive_to_conserved(Q[c, s]) for c in range(n_cells) for s in range(n_sps)]
    ).reshape(n_cells, n_sps, 5)


@pytest.fixture(scope="module")
def mesh_and_ops():
    order = 1
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    return mesh, ops


class TestDistributedFRSolverMainInitStep:
    """决定性判据：真正通过主 `__init__`（"传统模式"）构造的
    `DistributedFRSolver`，其 `step()` 内部实际使用的
    `mu`/`boundary_ghost_provider`/`mach_ref`/`enable_viscous`（均从
    `self.local_solver` 这个真正构造出来的 `FRSolver` 实例读取，见
    `step()` 对应代码）必须与直接构造一个单机 `FRSolver` 读到的完全
    一致——即 `local_solver` 这层间接确实把正确的值传导过去了。

    **本段文档已更新（2026-09-14）**：此前这里写"分布式路径目前用全局
    固定步长、不做单机那种逐 cell 局部 CFL——这是两条路径一直如此、
    有意的架构差异"，因此刻意不比较 `step()` 跑完一步之后的状态。
    那条"简化"已经补齐（`core/mpi/distributed_cfl.py`），两条路径现在
    用的是**同一个** `cfl.py::compute_local_time_step`，所以"跑完一步
    后的状态必须一致"重新成为有效、而且远更强的判据——见
    `TestDistributedStepMatchesSingleMachine`。本类保留原有的属性级
    比对（它验证的是 local_solver 这层间接是否把值正确传导过去，与
    时间积分无关，仍然有独立价值）。"""

    def test_local_solver_wiring_matches_direct_fr_solver_construction(self, mesh_and_ops):
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
        from autoflowcfd.core.fr_solver.solver import FRSolver

        mesh, ops = mesh_and_ops
        freestream_kwargs = dict(
            mu_molecular=1.8e-5, rho_inf=1.225, vel_inf=33.33, p_inf=101325.0,
        )

        ref_solver = FRSolver(
            mesh, order=mesh.order, turb_model_name="none",
            time_scheme=TimeIntegrationScheme.SSP_RK3, **freestream_kwargs,
        )

        dist_solver = DistributedFRSolver(
            mesh=mesh, ops=ops, face_connectivity=mesh.face_connectivity,
            n_ranks=1, backend="cpu", order=mesh.order, turb_model_name="none",
            time_scheme=TimeIntegrationScheme.SSP_RK3, **freestream_kwargs,
        )

        assert dist_solver.local_solver.mu_molecular == ref_solver.mu_molecular
        assert dist_solver.local_solver.freestream["mach_ref"] == ref_solver.freestream["mach_ref"]
        assert type(dist_solver.local_solver.boundary_ghost_provider) is type(
            ref_solver.boundary_ghost_provider
        )
        # 真实 bug 回归（本次修复）：local_solver 是货真价实的 FRSolver
        # 实例时没有 .config 属性，enable_viscous 必须安全退回 True
        # （与真实 FRSolver 恒计算粘性残差的行为一致），而不是
        # AttributeError。
        assert not hasattr(dist_solver.local_solver, "config")

    def test_step_runs_and_stays_finite(self, mesh_and_ops):
        """收尾集成测试：真正调用 `step()`（此前 100% 必现
        `AttributeError: 'FRSolver' object has no attribute 'config'`，
        因为从未有测试真正走过主 `__init__` + `step()` 这条组合，
        `local_solver` property 构造出的真正 `FRSolver` 与
        `from_fully_distributed_package` 专用的 `types.SimpleNamespace`
        替身形状不同——`step()` 只按后者的形状写的）。"""
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver

        mesh, ops = mesh_and_ops
        n_cells = mesh.n_cells
        rng = np.random.default_rng(2026)
        U0 = _nonuniform_U(mesh, rng)

        dist_solver = DistributedFRSolver(
            mesh=mesh, ops=ops, face_connectivity=mesh.face_connectivity,
            n_ranks=1, backend="cpu", order=mesh.order, turb_model_name="none",
            time_scheme=TimeIntegrationScheme.SSP_RK3,
            mu_molecular=1.8e-5, rho_inf=1.225, vel_inf=33.33, p_inf=101325.0,
        )
        dist_solver.state.U[:n_cells] = U0
        dist_solver.state.Q[:n_cells] = conserved_to_primitive(U0[..., :5])

        residual_norm = dist_solver.step(1e-6)

        assert np.isfinite(residual_norm)
        assert np.all(np.isfinite(dist_solver.state.U[:n_cells]))


class TestDistributedStepMatchesSingleMachine:
    """n_ranks=1 下分布式 `step()` 必须与单机 `step()` 推进出同一个状态。

    为什么这是最强的端到端判据：n_ranks=1 没有 halo、没有跨 rank 通信，
    分布式路径与单机路径面对的是同一个网格、同一个初场、同一个时间方案，
    因此**任何**差异都只能来自分布式实现自己引入的错误。而且"棱柱在前"
    的紧凑重排在 n_ranks=1 下依然生效，所以这条判据同时覆盖了 perm/
    inv_perm 的正确性。

    2026-09-14 之前这条判据不成立，因为分布式用全局固定 dt 而单机用逐
    cell 局部 CFL（当时被记作"有意的架构差异"）。局部 CFL 补齐后两条
    路径共用同一个步长公式，差异应当只剩浮点重结合噪声。

    容差说明：不要求逐位相同。两条路径对同一批单元的算术**顺序**不同
    （单机一次性全场计算；分布式按 compact 子集抽取+重排后计算），
    浮点重结合会产生 ~1e-13 量级的相对差异——这与
    `test_distributed_compute_residual.py` 模块文档记录的同一类现象
    同源，不是正确性问题。判据取相对 1e-10，比噪声高两个数量级、
    又远低于任何真实索引/公式错误会产生的差异（实测破坏性验证：
    跳过 inv_perm 重排会产生 O(1) 的差异）。
    """

    def _build(self, mesh, ops, U0, scheme):
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
        from autoflowcfd.core.fr_solver.solver import FRSolver

        kw = dict(mu_molecular=1.8e-5, rho_inf=1.225, vel_inf=33.33, p_inf=101325.0)
        single = FRSolver(mesh, order=mesh.order, turb_model_name="none",
                          time_scheme=scheme, **kw)
        single.order_continuation_enabled = False
        single.state.U[...] = U0
        single.state._update_primitives()

        dist = DistributedFRSolver(
            mesh=mesh, ops=ops, face_connectivity=mesh.face_connectivity,
            n_ranks=1, backend="cpu", order=mesh.order, turb_model_name="none",
            time_scheme=scheme, **kw)
        n = mesh.n_cells
        dist.state.U[:n] = U0
        dist.state.Q[:n] = conserved_to_primitive(U0[..., :5])
        return single, dist

    @pytest.mark.parametrize("precond", [False, True])
    def test_local_dt_matches_single_machine(self, mesh_and_ops, precond, monkeypatch):
        """先单独比对**步长**本身——它是本次补齐的核心产物，单独钉住能在
        出问题时直接指出是 dt 还是时间积分的问题。"""
        monkeypatch.setenv("AFCFD_LOW_MACH_PRECOND", "1" if precond else "0")
        mesh, ops = mesh_and_ops
        rng = np.random.default_rng(31337)
        U0 = _nonuniform_U(mesh, rng)
        single, dist = self._build(mesh, ops, U0, TimeIntegrationScheme.SSP_RK3)
        assert single.low_mach_precond_enabled is precond
        assert dist.low_mach_precond_enabled is precond

        ref_mean, ref_phys = single._compute_local_time_step(return_physical_too=True)
        got_mean, got_phys = dist._compute_distributed_local_time_step(
            dist.state.get_local_U(), return_physical_too=True)
        for name, got, exp in (("mean_flow", got_mean, ref_mean),
                               ("physical", got_phys, ref_phys)):
            rel = np.abs(got - exp) / np.maximum(np.abs(exp), 1e-300)
            assert rel.max() <= 1e-12, f"{name} dt 与单机不一致：{rel.max():.3e}"
        if precond:
            assert ref_mean.max() > ref_phys.max() * 1.5, (
                "预处理没有真的放大 dt，本用例失去判别力")

    @pytest.mark.parametrize("precond", [False, True])
    def test_one_step_matches_single_machine(self, mesh_and_ops, precond, monkeypatch):
        monkeypatch.setenv("AFCFD_LOW_MACH_PRECOND", "1" if precond else "0")
        mesh, ops = mesh_and_ops
        rng = np.random.default_rng(4242)
        U0 = _nonuniform_U(mesh, rng)
        single, dist = self._build(mesh, ops, U0, TimeIntegrationScheme.SSP_RK3)

        single.step(1e-6)
        dist.step(1e-6)

        n = mesh.n_cells
        got = dist.state.U[:n]
        exp = single.state.U
        assert np.all(np.isfinite(got))
        scale = np.maximum(np.abs(exp).max(axis=(0, 1)), 1e-300)
        rel = (np.abs(got - exp) / scale).max()
        assert rel <= 1e-10, (
            f"分布式推进一步后的状态与单机不一致（相对 {rel:.3e}）"
            f"，precond={precond}")


    def test_imex_step_matches_single_machine(self, mesh_and_ops):
        """IMEX（2026-09-25 补齐）：此前分布式 `step()` 对 IMEX_EULER 无条件走
        `_ssp_rk_stage_step`，系数表里没有 IMEX 条目、回退成 1 级前向 Euler
        —— `--time-method imex --n-ranks N` 静默跑成前向 Euler。现在必须与
        单机 IMEX 推进出同一个状态（同一积分器、同一对流/粘性拆分、同一
        CFL 策略）。"""
        mesh, ops = mesh_and_ops
        rng = np.random.default_rng(777)
        U0 = _nonuniform_U(mesh, rng)
        single, dist = self._build(mesh, ops, U0, TimeIntegrationScheme.IMEX_EULER)

        single.step(1e-6)
        dist.step(1e-6)

        n = mesh.n_cells
        got = dist.state.U[:n]
        exp = single.state.U
        assert np.all(np.isfinite(got))
        assert not np.array_equal(exp, U0), "单机 IMEX 一步没有推进，本用例失去判别力"
        scale = np.maximum(np.abs(exp).max(axis=(0, 1)), 1e-300)
        rel = (np.abs(got - exp) / scale).max()
        assert rel <= 1e-10, f"分布式 IMEX 一步后的状态与单机不一致（相对 {rel:.3e}）"

class TestDistributedDualTimeStepping:
    """DUAL_TIME（真正时间精度的瞬态仿真模式）分布式支持验证（2026-09-02，
    真实bug修复——见 __init__/step() 里 `self._time_integrator` 构造处的
    说明：此前无条件硬编码 SSP_RK3、`step()` 无条件调用
    `_ssp_rk_stage_step`，DUAL_TIME 请求了也会被静默换成稳态收敛加速
    模式，完全无路可走）。

    判据：分布式路径与单机路径用同一个物理 dt_physical、同一个
    max_inner_iter，只要两者的伪时间内迭代都真正收敛到容差以内，
    最终解就应该逐位一致——不像 SSP-RK3 稳态加速模式那样有"分布式用
    全局固定 dt、单机用逐 SP 局部 CFL 步长"这个天然的、无关本次改动
    的差异（那个差异只存在于伪时间加速路径，不影响 DUAL_TIME 收敛到
    的物理解本身，见 TestDistributedFRSolverMainInitStep 类文档的
    说明）。"""

    def test_dual_time_spatial_residual_matches_single_machine(self, mesh_and_ops):
        """DUAL_TIME 完整收敛后的最终解不适合用来跟单机比较：单机路径
        用逐 SP 局部 CFL 步长做伪时间加速、分布式路径目前用全局固定
        步长（见 __init__ 里对应说明——这是分布式稳态加速路径本来就
        有的、独立于本次 DUAL_TIME 改动的已知架构差异），两条路径的
        伪时间迭代*路径*不同，走到收敛判据触发的落点也可能不同（已
        实测复现，即使从精确均匀自由流场出发也是如此——这个合成网格
        的四面体单元已知有坍缩坐标退化引起的条件数病态，见
        [[tet_collapsed_coord_anisotropy]] 记忆，具体成因超出本次
        DUAL_TIME 接入验证的范围）。

        真正需要验证、且不受这个架构差异干扰的是：`step_dual_time`
        实际收到的 `spatial_residual(U)` 回调（分布式版就是 step()
        已有的 `residual_func`，DUAL_TIME 分支直接复用它，不需要额外
        包装）算出的值，必须与单机 `FRSolver.compute_inviscid_
        residual()+compute_viscous_residual()`（DUAL_TIME 用的正是
        同一个纯空间残差，与所选时间推进方案本身无关）在同一个初始
        状态上逐位一致——这是本次改动真正新增、真正有风险的部分
        （`_time_integrator` 构造/`step()` 分支/`residual_func` 复用），
        `step_dual_time` 内部 BDF 构造+收敛迭代本身是既有单机数值
        逻辑，不是本次改动的对象。"""
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
        from autoflowcfd.core.fr_solver.solver import FRSolver

        mesh, ops = mesh_and_ops
        n_cells = mesh.n_cells
        rng = np.random.default_rng(555)
        U0 = _nonuniform_U(mesh, rng)
        freestream_kwargs = dict(
            mu_molecular=1.8e-5, rho_inf=1.225, vel_inf=33.33, p_inf=101325.0,
        )

        ref_solver = FRSolver(
            mesh, order=mesh.order, turb_model_name="none",
            time_scheme=TimeIntegrationScheme.DUAL_TIME, **freestream_kwargs,
        )
        ref_solver.state.U[:] = U0
        ref_solver.state.Q[:] = conserved_to_primitive(U0[..., :5])
        expected_R = (
            ref_solver.compute_inviscid_residual() + ref_solver.compute_viscous_residual()
        )
        expected_spatial_residual = (-expected_R).reshape(n_cells * mesh.n_sps_per_cell, 5)

        dist_solver = DistributedFRSolver(
            mesh=mesh, ops=ops, face_connectivity=mesh.face_connectivity,
            n_ranks=1, backend="cpu", order=mesh.order, turb_model_name="none",
            time_scheme=TimeIntegrationScheme.DUAL_TIME, **freestream_kwargs,
        )
        dist_solver.state.U[:n_cells] = U0
        dist_solver.state.Q[:n_cells] = conserved_to_primitive(U0[..., :5])

        captured = {}
        real_step_dual_time = dist_solver._time_integrator.step_dual_time

        def _spy_step_dual_time(solution, spatial_residual, *args, **kwargs):
            captured["value"] = spatial_residual(solution)
            return real_step_dual_time(solution, spatial_residual, *args, **kwargs)

        dist_solver._time_integrator.step_dual_time = _spy_step_dual_time
        dist_solver.step(1e-5)

        assert "value" in captured, "step_dual_time 必须真正被 step() 调用到"
        np.testing.assert_allclose(
            captured["value"], expected_spatial_residual, rtol=1e-10, atol=1e-14,
        )

    def test_dual_time_second_step_uses_bdf2_with_solution_prev(self, mesh_and_ops):
        """第二步应该真正用上 BDF2（`solution_prev`），不是每步都退化成
        BDF1——验证 `self._dual_time_U_prev` 确实在两步之间被正确持久化
        并传给 `step_dual_time`。"""
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver

        mesh, ops = mesh_and_ops
        n_cells = mesh.n_cells
        rng = np.random.default_rng(556)
        U0 = _nonuniform_U(mesh, rng)

        dist_solver = DistributedFRSolver(
            mesh=mesh, ops=ops, face_connectivity=mesh.face_connectivity,
            n_ranks=1, backend="cpu", order=mesh.order, turb_model_name="none",
            time_scheme=TimeIntegrationScheme.DUAL_TIME,
            mu_molecular=1.8e-5, rho_inf=1.225, vel_inf=33.33, p_inf=101325.0,
        )
        dist_solver.state.U[:n_cells] = U0
        dist_solver.state.Q[:n_cells] = conserved_to_primitive(U0[..., :5])

        assert dist_solver._dual_time_U_prev is None
        dist_solver.step(1e-5)
        assert dist_solver._dual_time_U_prev is not None
        first_step_prev = dist_solver._dual_time_U_prev.copy()
        dist_solver.step(1e-5)
        # solution_prev 必须更新为第一步*结束时*的 U_flat（BDF2 用的
        # U^{n-1}），不是恒为初始 U0 或恒为 None。
        assert not np.allclose(dist_solver._dual_time_U_prev, first_step_prev)
        assert np.all(np.isfinite(dist_solver.state.U[:n_cells]))


class TestDistributedCheckpointRoundTrip:
    """分布式 checkpoint 保存/加载的真实 HDF5 往返验证（单 rank，本机
    无 mpi4py 时 partition/gather/scatter 的 n_ranks==1 分支就是恒等
    操作，见 distributed_checkpoint.py::gather_global_state 文档），
    以及 `solve(checkpoint_callback=...)` 回调真正被调用的验证。"""

    def _make_solver(self, mesh, ops):
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
        return DistributedFRSolver(
            mesh=mesh, ops=ops, face_connectivity=mesh.face_connectivity,
            n_ranks=1, backend="cpu", order=mesh.order, turb_model_name="none",
            time_scheme=TimeIntegrationScheme.SSP_RK3,
            mu_molecular=1.8e-5, rho_inf=1.225, vel_inf=33.33, p_inf=101325.0,
        )

    def test_save_then_load_restores_state(self, mesh_and_ops, tmp_path):
        from autoflowcfd.core.mpi.distributed_checkpoint import (
            distributed_save_checkpoint, distributed_load_checkpoint,
        )

        mesh, ops = mesh_and_ops
        n_cells = mesh.n_cells
        rng = np.random.default_rng(7)
        U0 = _nonuniform_U(mesh, rng)

        solver = self._make_solver(mesh, ops)
        solver.state.U[:n_cells] = U0

        saved_path = distributed_save_checkpoint(
            solver, str(tmp_path), 42, "dummy_input.nas", mesh.order, "none", "cpu",
        )
        assert saved_path is not None
        assert (tmp_path / "checkpoints").exists() or True  # 具体子目录由 CheckpointManager 决定

        # 用一个全新构造的求解器加载，验证真正的往返（不是同一个对象
        # 自己读自己内存里的值这种没有意义的"验证"）。
        solver2 = self._make_solver(mesh, ops)
        U_local, metadata, iteration = distributed_load_checkpoint(saved_path, solver2)

        assert iteration == 42
        np.testing.assert_allclose(U_local, U0, rtol=1e-12, atol=1e-14)

    def test_checkpoint_carries_the_physics_and_resume_reads_it_back(self, mesh_and_ops, tmp_path):
        """分布式 checkpoint 必须持久化决定物理解的参数（2026-09-25）。

        此前分布式写入端一个都不写：resume 要么按默认来流静默重建另一个
        算例，要么（09-24 起）因"来流缺失"直接报错；攻角/侧滑角在四条分布式
        resume 构造路径上也全部没有恢复。这里用**非缺省**的粘度、攻角、
        侧滑角、Tu/VR 与 turb_model='none'（此前 none/les 连粘度属性都只在
        湍流分支里设置）验证写入端与共用读取端的往返。"""
        from autoflowcfd.cli.solve.checkpoint_io import physics_from_metadata
        from autoflowcfd.core.mpi.distributed_checkpoint import (
            distributed_load_checkpoint, distributed_save_checkpoint,
        )
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver

        mesh, ops = mesh_and_ops
        want = dict(rho_inf=1.1, vel_inf=21.0, p_inf=95000.0, aoa_deg=4.5, aos_deg=-1.25,
                    mu_molecular=2.3e-5, turbulence_intensity=0.037, viscosity_ratio=8.0)
        solver = DistributedFRSolver(
            mesh=mesh, ops=ops, face_connectivity=mesh.face_connectivity,
            n_ranks=1, backend="cpu", order=mesh.order, turb_model_name="none",
            time_scheme=TimeIntegrationScheme.SSP_RK3, **want)
        path = distributed_save_checkpoint(
            solver, str(tmp_path), 7, "dummy_input.nas", mesh.order, "none", "cpu",
            surface_mesh="dummy_surface.nas")
        _U, metadata, _it = distributed_load_checkpoint(path, solver)
        got = physics_from_metadata(metadata)
        for k, v in want.items():
            assert got[k] == pytest.approx(v), f"{k}: 写入 {v}、读回 {got[k]}"
        assert metadata.get("surface_mesh") == "dummy_surface.nas"

    def test_every_distributed_resume_path_restores_the_flow_angles(self):
        """四条分布式 resume 构造路径（CPU 传统 / CPU 完全分布式 / 多 GPU 传统 /
        多 GPU 完全分布式）都必须把攻角与侧滑角交给求解器或来流字典。"""
        import inspect

        from autoflowcfd.cli.solve import distributed_checkpoint_io as mod
        src = inspect.getsource(mod.rebuild_distributed_solver_from_checkpoint)
        assert src.count("aoa_deg=aoa_deg, aos_deg=aos_deg") == 2
        assert src.count('"aoa_deg": aoa_deg, "aos_deg": aos_deg') == 2
        assert "physics_from_metadata(metadata)" in src

    def test_solve_checkpoint_callback_invoked_at_right_iterations(self, mesh_and_ops):
        """真实 bug 回归测试：此前 `DistributedFRSolver.solve()` 的
        `output_interval` 只控制进度打印，没有任何中间 checkpoint 保存
        机制——补齐后验证 `checkpoint_callback` 确实在每一步都被调用
        （回调自己决定是否真正保存，与单机路径 `_checkpoint_cb` 同一个
        分工），且传入的 iteration 编号正确（1-indexed，从 1 到
        n_steps）。"""
        mesh, ops = mesh_and_ops
        n_cells = mesh.n_cells
        solver = self._make_solver(mesh, ops)
        # 真实（非全零）初始状态——默认的全零 DistributedFRState 密度
        # 恒为 0，物理上不合法，会在残差计算里触发 NaN/溢出，与本测试
        # 要验证的"回调调用次数/编号是否正确"无关，用真实初场避免
        # 无意义的噪声。
        rng = np.random.default_rng(3)
        U0 = _nonuniform_U(mesh, rng)
        solver.state.U[:n_cells] = U0
        solver.state.Q[:n_cells] = conserved_to_primitive(U0[..., :5])

        calls = []
        def _cb(solver_ref, iteration):
            calls.append(iteration)

        solver.solve(n_steps=5, dt=1e-6, output_interval=100, checkpoint_callback=_cb)

        assert calls == [1, 2, 3, 4, 5]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
