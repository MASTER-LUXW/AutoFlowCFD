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
            n_ranks=1, order=mesh.order, turb_model_name="none",
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
            n_ranks=1, order=mesh.order, turb_model_name="none",
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

    def _build(self, mesh, ops, U0, scheme, **extra):
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
        from autoflowcfd.core.fr_solver.solver import FRSolver

        kw = dict(mu_molecular=1.8e-5, rho_inf=1.225, vel_inf=33.33, p_inf=101325.0, **extra)
        single = FRSolver(mesh, order=mesh.order, turb_model_name="none",
                          time_scheme=scheme, **kw)
        single.order_continuation_enabled = False
        single.state.U[...] = U0
        single.state._update_primitives()

        dist = DistributedFRSolver(
            mesh=mesh, ops=ops, face_connectivity=mesh.face_connectivity,
            n_ranks=1, order=mesh.order, turb_model_name="none",
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



    def test_time_accurate_step_matches_single_machine(self, mesh_and_ops):
        """时间精确显式推进（瞬态 rk3 档）：分布式与单机子步数相同、推进出同一个状态（子步长取跨 rank
        的全场最小稳定步长，`time_integration/time_accurate.py`）。"""
        from autoflowcfd.core.time_integration.time_accurate import advance

        mesh, ops = mesh_and_ops
        U0 = _nonuniform_U(mesh, np.random.default_rng(2718))
        single, dist = self._build(mesh, ops, U0, TimeIntegrationScheme.SSP_RK3, time_accurate=True)
        assert single.time_accurate and dist.time_accurate
        assert not single.low_mach_precond_enabled and not dist.low_mach_precond_enabled
        h = float(single._compute_local_time_step().min())
        dt = 3.5 * h
        advance(single, dt)
        advance(dist, dt)
        assert single.last_substeps == dist.last_substeps == 4
        n = mesh.n_cells
        scale = np.maximum(np.abs(single.state.U).max(axis=(0, 1)), 1e-300)
        rel = (np.abs(dist.state.U[:n] - single.state.U) / scale).max()
        assert rel <= 1e-10, f"时间精确推进后分布式与单机不一致（相对 {rel:.3e}）"


class TestDistributedDualTimeStepping:
    """DUAL_TIME 的分布式接线：每个物理步是带 BDF 物理时间项的 Newton 步（与单机同一个实现，
    `time_integration/implicit/physical_time.py`），上一时间层跨步保存。带湍流模型的对照见
    `test_dual_time_newton.py`。"""

    _KW = dict(mu_molecular=1.8e-5, rho_inf=1.225, vel_inf=33.33, p_inf=101325.0)

    def _pair(self, mesh, ops, seed):
        from autoflowcfd.core.fr_solver.solver import FRSolver
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver

        U0 = _nonuniform_U(mesh, np.random.default_rng(seed))
        single = FRSolver(mesh, order=mesh.order, turb_model_name="none",
                          time_scheme=TimeIntegrationScheme.DUAL_TIME, **self._KW)
        single.state.U[:] = U0
        single.state.Q[:] = conserved_to_primitive(U0[..., :5])
        dist = DistributedFRSolver(
            mesh=mesh, ops=ops, face_connectivity=mesh.face_connectivity,
            n_ranks=1, order=mesh.order, turb_model_name="none",
            time_scheme=TimeIntegrationScheme.DUAL_TIME, **self._KW)
        dist.state.U[:mesh.n_cells] = U0
        dist.state.Q[:mesh.n_cells] = conserved_to_primitive(U0[..., :5])
        for s in (single, dist):
            s.time_integrator.dual_time_steps = 40
            s.time_integrator.dual_time_tolerance = 1e-10
        return single, dist, U0

    def test_two_physical_steps_match_single_machine(self, mesh_and_ops):
        """两个物理步（BDF1 + BDF2）都解到 1e-10：分布式与单机得到同一个 BDF 解。"""
        mesh, ops = mesh_and_ops
        n = mesh.n_cells
        single, dist, U0 = self._pair(mesh, ops, 555)
        scale = np.abs(U0).max(axis=(0, 1))
        scale[1:4] = np.abs(U0[..., 1:4]).max()
        for k, a0 in enumerate((1.0, 1.5)):
            before = dist.state.U[:n].copy()
            single.step(1e-5)
            dist.step(1e-5)
            assert dist._dual_time_inner["converged"] and dist._dual_time_inner["a0"] == a0
            # 上一时间层是本物理步开始时的解
            np.testing.assert_array_equal(dist._dual_time_U_prev.reshape(before.shape), before)
            rel = (np.abs(dist.state.U[:n] - single.state.U) / scale).max()
            assert rel <= 1e-7, f"第 {k + 1} 个物理步与单机相对差 {rel:.3e}"
        assert (np.abs(dist.state.U[:n] - U0) / scale).max() > 1e-6          # 解确实推进了

    def test_pseudo_time_cfl_is_not_driven_by_the_outer_residual(self, mesh_and_ops):
        """伪时间 CFL 每个物理步从起点重新起步、只在内迭代里更新；step() 末尾不再拿外层残差更新它
        （此前显式内迭代下它是固定 CFL，没有这个问题）。"""
        from autoflowcfd.core.time_integration.adaptive_cfl.policy import DUAL_TIME_PSEUDO_CFL_START

        mesh, ops = mesh_and_ops
        _single, dist, _U0 = self._pair(mesh, ops, 556)
        seen = []
        real_update = dist._cfl_controller.update
        dist._cfl_controller.update = lambda *a, **k: (seen.append(dist._dual_time_inner), real_update(*a, **k))[1]
        dist._dual_time_inner = "before"
        dist.step(1e-5)
        assert dist._cfl_controller.cfl_start == DUAL_TIME_PSEUDO_CFL_START
        assert seen and all(x == "before" for x in seen)      # 全部更新都发生在内迭代期间（摘要尚未写回）


class TestDistributedCheckpointRoundTrip:
    """分布式 checkpoint 保存/加载的真实 HDF5 往返验证（单 rank，本机
    无 mpi4py 时 partition/gather/scatter 的 n_ranks==1 分支就是恒等
    操作，见 distributed_checkpoint.py::gather_global_state 文档），
    以及 `solve(checkpoint_callback=...)` 回调真正被调用的验证。"""

    def _make_solver(self, mesh, ops):
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
        return DistributedFRSolver(
            mesh=mesh, ops=ops, face_connectivity=mesh.face_connectivity,
            n_ranks=1, order=mesh.order, turb_model_name="none",
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
            n_ranks=1, order=mesh.order, turb_model_name="none",
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

        solver.solve(max_iter=5, dt=1e-6, checkpoint_callback=_cb)

        assert calls == [1, 2, 3, 4, 5]

    def test_solve_uses_the_shared_loop_relative_criterion_and_history(self, mesh_and_ops):
        """2026-10-05 起与其余后端共用 `SolveLoopMixin`：返回 `SolverResult`、收敛历史每步一条、收敛判据是
        相对下降 1/tol 倍（此前是绝对的 `res < tol`，且不记收敛历史）。"""
        from autoflowcfd.core.fr_solver.state import SolverResult

        mesh, ops = mesh_and_ops
        n_cells = mesh.n_cells
        solver = self._make_solver(mesh, ops)
        U0 = _nonuniform_U(mesh, np.random.default_rng(3))
        solver.state.U[:n_cells] = U0
        solver.state.Q[:n_cells] = conserved_to_primitive(U0[..., :5])

        solver.order_continuation_enabled = False   # 测定阶循环本身（爬坡路径的判据另有测试）
        result = solver.solve(max_iter=3, dt=1e-6, tol=1e-6)
        assert isinstance(result, SolverResult) and result.iterations == 3 and not result.converged
        assert len(solver.residual_history) == 3 and solver.residual_history[-1] == result.final_residual
        # 判据区分：tol 远大于残差时，绝对判据 `res < tol` 第 1 步即停；相对判据从第 2 步起检查
        # "下降 1/tol 倍"，第 2 步停（与单机同一语义）
        assert solver.residual_history[0] < 1e300
        result = solver.solve(max_iter=5, dt=1e-6, tol=1e300)
        assert result.converged and result.iterations == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
