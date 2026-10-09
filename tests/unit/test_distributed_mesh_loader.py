"""AutoFlowCFD V2.0 - 真正的"完全分布式网格加载"端到端验证
（2026-09-02，见 `distributed_mesh_loader.py::PrecompactedMeshData`/
`build_fully_distributed_rank_package` 模块文档）。

背景：只有 root 持有完整网格时，各 rank 的局部网格既没有全局面连接关系，也没法按
`compact_global_ids` 从全局数组重新抽取几何——旧的"分发局部网格数据"方案（已删除）在这两点上都不成立。

新设计：root rank 用它独有的完整全局网格，对每个 rank 分别调用纯函数
`build_fully_distributed_rank_package`（`build_distributed_partition`/
`build_distributed_flat_face` 本身都是纯函数，root 可以对任意 rank
调用），产出一个已经按该 rank 的 `compact_global_ids` 切好的紧凑包
（`PrecompactedMeshData` + `dist_fc` + 切好 `group_code` 的边界幽灵态
提供者），只有这个紧凑包（远小于完整网格）需要发给非 root rank。

核心判据（与本项目已建立的方法论一致）：
1. 紧凑包本身可以安全 pickle（真正跨进程发送的前提）。
2. `DistributedMeshAdapter(partition, dist_fc, precompacted_mesh, ops)`
   产出的 jacobians/cell_volumes 必须与"传统模式"
   `DistributedMeshAdapter(partition, dist_fc, full_mesh, ops)` 逐位
   相等（证明"root 预先切好"与"每个 rank 自己从全局数组切"是同一个
   结果，只是切的位置不同）。
3. 用非均匀流场（不是均匀自由流场——见 distributed_mpi_local_faces_
   critical_bug 记忆，均匀场无法探测"贡献丢失"类 bug）算出的分布式
   inviscid/viscous 残差，必须与单机路径逐位一致。
4. 边界幽灵态提供者的 `group_code` 重切片正确（与 `partition.
   local_faces` 对应）。
"""

import pickle
import numpy as np
import pytest

from autoflowcfd.core.mpi.distributed_compute import (
    DistributedMeshAdapter,
    distributed_compute_inviscid_residual,
    distributed_compute_viscous_residual,
)
from autoflowcfd.core.mpi.distributed_mesh_loader import (
    PrecompactedMeshData,
    build_fully_distributed_rank_package,
)
from autoflowcfd.core.fr_residual.inviscid import compute_inviscid_residual_fr
from autoflowcfd.core.fr_residual.viscous import compute_viscous_residual

from tests.unit._wall_source import synthetic_wall_source
from tests.unit._distributed_mesh_loader_common import _FakeHaloExchange, _build_all_packages, _nonuniform_U, mesh_and_ops


class TestRankPackagePicklable:
    def test_package_roundtrips_through_pickle(self, mesh_and_ops):
        """真正跨进程发送的前提：紧凑包必须能被 pickle 序列化/反序列化，
        且往返后数值不变。"""
        mesh, ops = mesh_and_ops
        packages, _, _ = _build_all_packages(mesh, ops)

        for pkg in packages:
            buf = pickle.dumps(pkg)
            pkg2 = pickle.loads(buf)

            assert isinstance(pkg2['precompacted_mesh'], PrecompactedMeshData)
            np.testing.assert_array_equal(
                pkg2['precompacted_mesh'].jacobians['det_jacs'],
                pkg['precompacted_mesh'].jacobians['det_jacs'],
            )
            np.testing.assert_array_equal(
                pkg2['dist_fc'].compact_global_ids, pkg['dist_fc'].compact_global_ids
            )
            assert pkg2['partition'].n_local_cells == pkg['partition'].n_local_cells


class TestPrecompactedMeshAdapterMatchesTraditionalMode:
    @pytest.mark.parametrize("rank", [0, 1])
    def test_jacobians_cell_volumes_match(self, mesh_and_ops, rank):
        mesh, ops = mesh_and_ops
        packages, _, cell_partition = _build_all_packages(mesh, ops)
        pkg = packages[rank]

        # "传统模式"参照：DistributedMeshAdapter 直接吃完整全局网格。
        adapter_traditional = DistributedMeshAdapter(
            pkg['partition'], pkg['dist_fc'], mesh, ops
        )
        # 新模式：DistributedMeshAdapter 吃 root 预先切好的紧凑数据。
        adapter_precompacted = DistributedMeshAdapter(
            pkg['partition'], pkg['dist_fc'], pkg['precompacted_mesh'], ops
        )

        assert adapter_precompacted.n_cells == adapter_traditional.n_cells
        assert adapter_precompacted.n_prism_cells == adapter_traditional.n_prism_cells
        np.testing.assert_array_equal(
            adapter_precompacted.jacobians['det_jacs'], adapter_traditional.jacobians['det_jacs']
        )
        np.testing.assert_array_equal(
            adapter_precompacted.jacobians['inv_jacs'], adapter_traditional.jacobians['inv_jacs']
        )
        np.testing.assert_array_equal(
            adapter_precompacted.cell_volumes, adapter_traditional.cell_volumes
        )


class TestBoundaryGhostProviderSlicing:
    @pytest.mark.parametrize("rank", [0, 1])
    def test_group_code_matches_local_faces_slice(self, mesh_and_ops, rank):
        mesh, ops = mesh_and_ops
        packages, boundary_ghost_provider_global, _ = _build_all_packages(mesh, ops)
        pkg = packages[rank]

        provider = pkg['boundary_ghost_provider']
        if boundary_ghost_provider_global is None:
            assert provider is None
            return

        expected = boundary_ghost_provider_global.group_code[pkg['partition'].local_faces]
        np.testing.assert_array_equal(provider.group_code, expected)


class TestFullyDistributedResidualMatchesSingleMachine:
    """决定性判据：用真正的"完全分布式"紧凑包（root 预先切好，非
    root rank 从未见过完整全局网格）算出的残差，必须与单机路径逐位
    一致——非均匀流场，覆盖对流/粘性的非平凡分支。"""

    @pytest.mark.parametrize("rank", [0, 1])
    def test_inviscid_residual_matches(self, mesh_and_ops, rank):
        mesh, ops = mesh_and_ops
        rng = np.random.default_rng(2026)
        U = _nonuniform_U(mesh, rng)
        mach_ref = 0.2

        packages, boundary_ghost_provider_global, _ = _build_all_packages(mesh, ops)
        # 单机参照必须用同一个边界幽灵态提供者（未切片的全局版本——单机
        # 路径没有分区，全部面都是"本地"面），否则"分布式用真实 provider
        # vs 单机用 None（镜像内部值）"是两种不同的边界处理，比较毫无
        # 意义（此前一版测试的真实错误：忘了把这个 provider 也传给单机
        # 参照，导致两边比较的是"不同边界条件下的残差"而不是"同一个
        # 边界条件下分布式 vs 单机"）。
        residual_single = compute_inviscid_residual_fr(
            U, mesh, ops, mach_ref=mach_ref,
            boundary_ghost_provider=boundary_ghost_provider_global,
        )

        pkg = packages[rank]
        partition = pkg['partition']
        dist_fc = pkg['dist_fc']

        n_halo = partition.n_halo
        native_ids = (
            np.concatenate([partition.local_cells, partition.halo_cells])
            if n_halo > 0 else partition.local_cells
        )
        fake_halo = _FakeHaloExchange(U[native_ids])
        U_local = U[partition.local_cells]

        residual_local = distributed_compute_inviscid_residual(
            U_local, partition, fake_halo, dist_fc, pkg['precompacted_mesh'], ops,
            boundary_ghost_provider=pkg['boundary_ghost_provider'], mach_ref=mach_ref,
        )
        expected_local = residual_single[partition.local_cells]
        np.testing.assert_allclose(residual_local, expected_local, rtol=1e-9, atol=1e-9)

    @pytest.mark.parametrize("rank", [0, 1])
    def test_viscous_residual_matches(self, mesh_and_ops, rank):
        mesh, ops = mesh_and_ops
        rng = np.random.default_rng(2027)
        U = _nonuniform_U(mesh, rng)
        mu = 1.8e-5

        packages, boundary_ghost_provider_global, _ = _build_all_packages(mesh, ops)
        # 见 test_inviscid_residual_matches 同一处注释：单机参照必须用
        # 同一个（未切片的全局）边界幽灵态提供者。
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
        Q = conserved_to_primitive(U[..., :5])
        residual_single = compute_viscous_residual(
            U, Q, ops, mesh, mu=mu, boundary_ghost_provider=boundary_ghost_provider_global,
        )

        pkg = packages[rank]
        partition = pkg['partition']
        dist_fc = pkg['dist_fc']

        n_halo = partition.n_halo
        native_ids = (
            np.concatenate([partition.local_cells, partition.halo_cells])
            if n_halo > 0 else partition.local_cells
        )
        fake_halo = _FakeHaloExchange(U[native_ids])
        U_local = U[partition.local_cells]

        residual_local = distributed_compute_viscous_residual(
            U_local, partition, fake_halo, dist_fc, pkg['precompacted_mesh'], ops,
            mu, boundary_ghost_provider=pkg['boundary_ghost_provider'],
        )
        expected_local = residual_single[partition.local_cells]
        np.testing.assert_allclose(residual_local, expected_local, rtol=1e-9, atol=1e-9)


class TestDistributedFRSolverFromFullyDistributedPackage:
    """收尾集成测试：`DistributedFRSolver.from_fully_distributed_
    package` 构造 + 真正调用 `step()`（完整 RK3 子迭代 + halo 交换 +
    残差组装），不是只测底层纯函数。n_ranks=1（本机无真实 MPI，无法
    模拟真正的跨进程 halo 通信，但 n_ranks=1 时 partition 没有任何
    邻居、`HaloExchange` 的 send/recv 列表本来就是空的，不需要真实
    MPI 也能正确工作）——验证的是"新构造路径 + 现有 step() 之间的
    接线是否正确"，残差数值本身的正确性已经由上面的测试类决定性
    验证过。"""

    def test_step_matches_single_machine_rk3(self, mesh_and_ops):
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
        from autoflowcfd.core.time_integration.base import TimeIntegrationScheme
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive

        mesh, ops = mesh_and_ops
        rng = np.random.default_rng(99)
        U0 = _nonuniform_U(mesh, rng)
        n_cells = mesh.n_cells
        mach_ref = 0.2
        mu = 1.8e-5
        dt = 1e-6

        cell_partition = np.zeros(n_cells, dtype=np.int32)  # 单 rank：全部 local
        freestream = {"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0}

        import types
        from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider
        root_stub = types.SimpleNamespace(
            mesh=mesh, freestream=freestream, turb_model_name="NONE", wmles_model=None,
        )
        boundary_ghost_provider = build_boundary_ghost_provider(root_stub, bc_overrides={})

        package = build_fully_distributed_rank_package(
            mesh, ops, mesh.face_connectivity, cell_partition, rank=0, n_ranks=1,
            boundary_ghost_provider_global=boundary_ghost_provider,
            freestream=freestream, mu_molecular=mu, mach_ref=mach_ref,
            order=mesh.order, enable_viscous=True,
            wall_distance_source=synthetic_wall_source(mesh),
        )

        solver = DistributedFRSolver.from_fully_distributed_package(package, n_ranks=1, rank=0)
        solver.state.U[:n_cells] = U0
        solver.state.Q[:n_cells] = conserved_to_primitive(U0[..., :5])

        solver.step(dt)
        U_distributed = solver.state.U[:n_cells].copy()

        # 单机参照：直接用**真正的** FRSolver.step()。
        #
        # 参照方式已更新（2026-09-14）：此前这里自己拼一套
        # `TimeIntegrator._ssp_rk_stage_step` + 残差函数，并把
        # `dt_flat = np.full(..., dt)` 当步长——那是当时分布式路径的真实
        # 行为（全局固定 dt，被记作"已接受的简化"）。那条简化已经补齐：
        # 分布式现在用与单机**同一个** `cfl.py::compute_local_time_step`
        # 算逐单元局部步长，并且同样施加模态滤波与低马赫数预处理。手工
        # 拼参照就必须把这三件事逐一复刻一遍，既冗余又容易与生产代码
        # 失去同步——直接用真正的单机 `FRSolver.step()` 作参照，判据更强
        # （覆盖步长/滤波/预处理全部环节），也不会再有这种同步风险。
        from autoflowcfd.core.fr_solver.solver import FRSolver

        single = FRSolver(
            mesh, order=mesh.order, turb_model_name="none",
            time_scheme=TimeIntegrationScheme.SSP_RK3,
            mu_molecular=mu, rho_inf=freestream["rho_inf"],
            vel_inf=freestream["vel_inf"], p_inf=freestream["p_inf"],
        )
        single.order_continuation_enabled = False
        # mach_ref 在本用例里被显式指定为 0.2（不是从 vel_inf/p_inf 推出来
        # 的那个值），分布式包里存的就是它——参照必须用同一个，否则
        # AUSM+up 的低马赫修正与 Gamma 的 beta^2 下限都会不一致。
        single.freestream["mach_ref"] = mach_ref
        single.state.U[...] = U0
        single.state._update_primitives()
        single.step(dt)
        U_single = single.state.U.copy()

        np.testing.assert_allclose(U_distributed, U_single, rtol=1e-9, atol=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
