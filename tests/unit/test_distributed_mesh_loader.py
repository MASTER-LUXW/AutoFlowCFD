"""AutoFlowCFD V2.0 - 真正的"完全分布式网格加载"端到端验证
（2026-09-02，见 `distributed_mesh_loader.py::PrecompactedMeshData`/
`build_fully_distributed_rank_package` 模块文档）。

背景：此前的 `distributed_mesh_load`/`build_local_mesh_from_data` 从未
真正跑通过——产出的 `local_mesh` 缺 `face_connectivity`/`face_flux_
points`，且 `DistributedMeshAdapter` 假设 `local_mesh` 是完整全局网格
（按 `compact_global_ids` 重新从全局数组抽取），两个前提在"只有 root
持有完整网格"这个真正的完全分布式场景下都不成立，从未被 CLI 实际
调用过（生产路径一直用"每 rank 独立加载完整网格"的回退方案）。

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

from autoflowcfd.core.mpi.partition import build_distributed_partition
from autoflowcfd.core.mpi.distributed_flat_face import build_distributed_flat_face
from autoflowcfd.core.mpi.distributed_compute import (
    DistributedMeshAdapter,
    distributed_compute_inviscid_residual,
    distributed_compute_viscous_residual,
)
from autoflowcfd.core.mpi.distributed_mesh_loader import (
    PrecompactedMeshData,
    build_fully_distributed_rank_package,
)
from autoflowcfd.core.fr_residual.inviscid import (
    compute_inviscid_residual_fr, primitive_to_conserved,
)
from autoflowcfd.core.fr_residual.viscous import compute_viscous_residual
from autoflowcfd.fr.operators import generate_fr_operators
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


class _FakeHaloExchange:
    def __init__(self, U_extended: np.ndarray):
        self._U_extended = U_extended

    def exchange(self, U_local: np.ndarray) -> np.ndarray:
        return self._U_extended


@pytest.fixture(scope="module")
def mesh_and_ops():
    order = 1
    mesh = _build_synthetic_mixed_mesh(order)
    ops = generate_fr_operators(order)
    assert mesh.n_prism_cells == 2
    assert mesh.n_cells == 4
    return mesh, ops


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


def _build_all_packages(mesh, ops, n_ranks=2):
    fc = mesh.face_connectivity
    cell_partition = np.array([0, 1, 0, 1], dtype=np.int32)
    freestream = {"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0}

    import types
    from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider
    root_stub = types.SimpleNamespace(
        mesh=mesh, freestream=freestream, turb_model_name="NONE", wmles_model=None,
    )
    boundary_ghost_provider_global = build_boundary_ghost_provider(root_stub, bc_overrides={})

    packages = [
        build_fully_distributed_rank_package(
            mesh, ops, fc, cell_partition, r, n_ranks,
            boundary_ghost_provider_global, freestream, mu_molecular=1.8e-5, mach_ref=0.2,
            order=mesh.order, enable_viscous=True,
        )
        for r in range(n_ranks)
    ]
    return packages, boundary_ghost_provider_global, cell_partition


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


class TestFullyDistributedSstTurbulence:
    """真正的"完全分布式加载"模式下 SST/DDES/IDDES 湍流支持（2026-09-02
    补齐——此前这条路径明确只支持 turbulence_model='none'，理由是
    "wall_distance/h_max/h_wn 需要完整全局网格"——但 root 本来就手握
    完整全局网格，只是当时没有一并把这几何量算出来发给各 rank）。

    核心判据：`build_fully_distributed_rank_package`/`distributed_mesh_
    load_v2`/`DistributedFRSolver.from_fully_distributed_package` 三层
    构造出的 wall_distance_compact/turb_model 初值，必须与直接独立调用
    对应底层函数（`compute_distributed_wall_distance`/`_set_freestream_
    turbulence`）算出的参照逐位一致——不是"看起来跑起来了"，是真正对照
    验证这条新链路每一步都没有引入偏差。
    """

    def test_wall_distance_and_turb_model_init_matches_direct_computation(self, mesh_and_ops):
        mesh, ops = mesh_and_ops
        rank, n_ranks = 0, 1
        cell_partition = np.zeros(mesh.n_cells, dtype=np.int32)
        freestream = {"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0}
        mu = 1.8e-5

        import types
        from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider
        root_stub = types.SimpleNamespace(
            mesh=mesh, freestream=freestream, turb_model_name="SST", wmles_model=None,
        )
        boundary_ghost_provider = build_boundary_ghost_provider(root_stub, bc_overrides={})

        package = build_fully_distributed_rank_package(
            mesh, ops, mesh.face_connectivity, cell_partition, rank=rank, n_ranks=n_ranks,
            boundary_ghost_provider_global=boundary_ghost_provider,
            freestream=freestream, mu_molecular=mu, mach_ref=0.2,
            order=mesh.order, enable_viscous=True,
            turb_model_name="SST", wall_node_indices=None,
            turbulence_intensity=0.02, viscosity_ratio=8.0,
        )

        assert package['turb_model_name'] == "SST"
        assert package['wall_distance_compact'] is not None

        # ---- 独立参照：直接调用 compute_distributed_wall_distance ----
        from autoflowcfd.core.mpi.partition import build_distributed_partition
        from autoflowcfd.core.mpi.distributed_flat_face import build_distributed_flat_face
        from autoflowcfd.core.mpi.distributed_turbulence import compute_distributed_wall_distance
        partition_ref = build_distributed_partition(mesh.face_connectivity, cell_partition, rank=rank, n_ranks=n_ranks)
        dist_fc_ref = build_distributed_flat_face(mesh, ops, partition_ref, cell_partition=cell_partition)
        expected_wall_distance = compute_distributed_wall_distance(partition_ref, dist_fc_ref, mesh, None)
        np.testing.assert_allclose(package['wall_distance_compact'], expected_wall_distance)

        # ---- 构造真正的 DistributedFRSolver ----
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
        solver = DistributedFRSolver.from_fully_distributed_package(package, n_ranks=n_ranks, rank=rank)

        assert solver.turb_model_name == "SST"
        assert solver.turb_model is not None
        assert solver.turb_model.k_field.shape == (partition_ref.n_local_cells, mesh.n_sps_per_cell)
        np.testing.assert_allclose(solver.wall_distance_compact, expected_wall_distance)

        # ---- 独立参照：k_inf/omega_inf 应该与直接调用 _set_freestream_
        # turbulence 算出的值逐位一致（同一套 Tu/VR 推导公式）。
        from autoflowcfd.core.fr_solver.turbulence import _set_freestream_turbulence
        ref_stub = types.SimpleNamespace(
            freestream=freestream, mu_molecular=mu,
            _turbulence_intensity=0.02, _viscosity_ratio=8.0,
        )
        k_inf_expected, omega_inf_expected = _set_freestream_turbulence(ref_stub)
        np.testing.assert_allclose(solver.turb_model.k_field, k_inf_expected)
        np.testing.assert_allclose(solver.turb_model.omega_field, omega_inf_expected)
        assert solver.turb_model.k_max == pytest.approx(0.5 * freestream["vel_inf"] ** 2)

    def test_step_with_sst_runs_and_stays_finite(self, mesh_and_ops):
        """收尾集成测试：真正调用 `step()`（含 SST 源项+输运+mean-flow
        RK3），不只是构造。n_ranks=1（本机无真实 MPI 的既定限制，与
        `TestDistributedFRSolverFromFullyDistributedPackage` 同一个
        约束）。"""
        mesh, ops = mesh_and_ops
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
        rank, n_ranks = 0, 1
        cell_partition = np.zeros(n_cells, dtype=np.int32)
        freestream = {"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0}
        mu = 1.8e-5

        import types
        from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider
        root_stub = types.SimpleNamespace(
            mesh=mesh, freestream=freestream, turb_model_name="SST", wmles_model=None,
        )
        boundary_ghost_provider = build_boundary_ghost_provider(root_stub, bc_overrides={})

        package = build_fully_distributed_rank_package(
            mesh, ops, mesh.face_connectivity, cell_partition, rank=rank, n_ranks=n_ranks,
            boundary_ghost_provider_global=boundary_ghost_provider,
            freestream=freestream, mu_molecular=mu, mach_ref=0.2,
            order=mesh.order, enable_viscous=True, turb_model_name="SST",
        )

        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
        solver = DistributedFRSolver.from_fully_distributed_package(package, n_ranks=n_ranks, rank=rank)

        rng = np.random.default_rng(7)
        U0 = _nonuniform_U(mesh, rng)
        solver.state.U[:n_cells] = U0
        solver.state.Q[:n_cells] = conserved_to_primitive(U0[..., :5])

        solver.step(1e-6)

        assert np.all(np.isfinite(solver.state.U[:n_cells]))
        assert np.all(np.isfinite(solver.turb_model.k_field))
        assert np.all(np.isfinite(solver.turb_model.omega_field))
        assert np.all(solver.turb_model.k_field > 0)
        assert np.all(solver.turb_model.omega_field > 0)

    def test_step_with_ddes_runs_and_stays_finite(self, mesh_and_ops):
        """同上 SST 收尾集成测试，换成 DDES——与 IDDES 共用
        `from_fully_distributed_package` 里同一个 `ddes_model`/
        `des_length_scale_halo_exchange` 构造分支，唯一区别是 DDES 不
        需要 h_wn（只有 IDDES 需要近壁法向间距）。2026-09-02 起 DDES
        也需要 h_max（`apply_to_sst_model` 改用各向异性感知的 max_edge
        网格尺度，见 des.py::DDESModel.compute_grid_scale 文档"Note"
        一节），单独测一遍确认这条分支正确要求 h_max_global、正确产出
        非 None 的 `iddes_h_max_compact`，同时仍然不需要 h_wn。"""
        mesh, ops = mesh_and_ops
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
        rank, n_ranks = 0, 1
        cell_partition = np.zeros(n_cells, dtype=np.int32)
        freestream = {"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0}
        mu = 1.8e-5

        import types
        from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider
        from autoflowcfd.core.turbulence.des import compute_h_max_and_h_wn
        root_stub = types.SimpleNamespace(
            mesh=mesh, freestream=freestream, turb_model_name="DDES", wmles_model=None,
        )
        boundary_ghost_provider = build_boundary_ghost_provider(root_stub, bc_overrides={})

        # 缺少 h_max_global 时必须显式报错，不能静默退化——决定性验证
        # 这条 fail-fast 护栏本身。
        with pytest.raises(RuntimeError):
            build_fully_distributed_rank_package(
                mesh, ops, mesh.face_connectivity, cell_partition, rank=rank, n_ranks=n_ranks,
                boundary_ghost_provider_global=boundary_ghost_provider,
                freestream=freestream, mu_molecular=mu, mach_ref=0.2,
                order=mesh.order, enable_viscous=True, turb_model_name="DDES",
            )

        h_max_global, _h_wn_global = compute_h_max_and_h_wn(mesh)
        package = build_fully_distributed_rank_package(
            mesh, ops, mesh.face_connectivity, cell_partition, rank=rank, n_ranks=n_ranks,
            boundary_ghost_provider_global=boundary_ghost_provider,
            freestream=freestream, mu_molecular=mu, mach_ref=0.2,
            order=mesh.order, enable_viscous=True, turb_model_name="DDES",
            h_max_global=h_max_global,
        )

        assert package['iddes_h_max_compact'] is not None
        assert package['iddes_h_wn_compact'] is None

        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
        solver = DistributedFRSolver.from_fully_distributed_package(package, n_ranks=n_ranks, rank=rank)

        assert solver.turb_model_name == "DDES"
        assert solver.ddes_model is not None
        assert solver.des_length_scale_halo_exchange is not None

        rng = np.random.default_rng(13)
        U0 = _nonuniform_U(mesh, rng)
        solver.state.U[:n_cells] = U0
        solver.state.Q[:n_cells] = conserved_to_primitive(U0[..., :5])

        solver.step(1e-6)
        solver.step(1e-6)

        assert np.all(np.isfinite(solver.state.U[:n_cells]))
        assert np.all(np.isfinite(solver.turb_model.k_field))
        assert np.all(np.isfinite(solver.turb_model.omega_field))
        assert np.all(solver.turb_model.k_field > 0)
        assert np.all(solver.turb_model.omega_field > 0)
        assert solver.turb_model.des_length_scale is not None
        assert np.all(np.isfinite(solver.turb_model.des_length_scale))

    def test_step_with_iddes_runs_and_stays_finite(self, mesh_and_ops):
        """同上一个 SST 收尾集成测试，换成 IDDES——补齐"完全分布式加载"
        对 IDDES 的 h_max/h_wn compact 切片端到端覆盖（此前只写了构造期
        代码，没有跑过真正的 step()，见本文件模块级 pending 记录）。
        顺带覆盖了本次修复的 `_compute_omega_wall_target` 无条件调用
        `get_flat_face_geometry` 的 bug（IDDES 与 SST 共用同一条 omega
        输运路径）。"""
        mesh, ops = mesh_and_ops
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
        rank, n_ranks = 0, 1
        cell_partition = np.zeros(n_cells, dtype=np.int32)
        freestream = {"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0}
        mu = 1.8e-5

        import types
        from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider
        from autoflowcfd.core.turbulence.des import compute_h_max_and_h_wn
        root_stub = types.SimpleNamespace(
            mesh=mesh, freestream=freestream, turb_model_name="IDDES", wmles_model=None,
        )
        boundary_ghost_provider = build_boundary_ghost_provider(root_stub, bc_overrides={})
        h_max_global, h_wn_global = compute_h_max_and_h_wn(mesh)

        package = build_fully_distributed_rank_package(
            mesh, ops, mesh.face_connectivity, cell_partition, rank=rank, n_ranks=n_ranks,
            boundary_ghost_provider_global=boundary_ghost_provider,
            freestream=freestream, mu_molecular=mu, mach_ref=0.2,
            order=mesh.order, enable_viscous=True, turb_model_name="IDDES",
            h_max_global=h_max_global, h_wn_global=h_wn_global,
        )

        assert package['iddes_h_max_compact'] is not None
        assert package['iddes_h_wn_compact'] is not None

        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
        solver = DistributedFRSolver.from_fully_distributed_package(package, n_ranks=n_ranks, rank=rank)

        assert solver.turb_model_name == "IDDES"
        assert solver.ddes_model is not None
        assert solver.des_length_scale_halo_exchange is not None

        rng = np.random.default_rng(11)
        U0 = _nonuniform_U(mesh, rng)
        solver.state.U[:n_cells] = U0
        solver.state.Q[:n_cells] = conserved_to_primitive(U0[..., :5])

        # 两步（与 CPU 分布式 DDES/IDDES 的既有回归测试一致），因为
        # des_length_scale 第二步才真正走 halo-exchange 读回路径（第一
        # 步是 None，跳过读入分支——见 distributed_turbulence.py 里
        # des_length_scale_halo_exchange 的文档）。
        solver.step(1e-6)
        solver.step(1e-6)

        assert np.all(np.isfinite(solver.state.U[:n_cells]))
        assert np.all(np.isfinite(solver.turb_model.k_field))
        assert np.all(np.isfinite(solver.turb_model.omega_field))
        assert np.all(solver.turb_model.k_field > 0)
        assert np.all(solver.turb_model.omega_field > 0)
        assert solver.turb_model.des_length_scale is not None
        assert np.all(np.isfinite(solver.turb_model.des_length_scale))

    def test_step_with_wmles_runs_and_stays_finite(self, mesh_and_ops):
        """"完全分布式加载"模式补齐 WMLES（2026-09-02 续接——此前
        fail-fast 拒绝，排查发现"需要额外分布式面外插"的理由不成立后
        真正实现，见 core/utils/solver_helpers.py 模块文档）。需要一个
        真实 WALL 组才能验证壁面剪应力修正真正生效（不是恰好没有 WALL
        面、correction 恒为 None 的平凡通过），手动给一个真实边界面的
        owner 单元打标签，与 test_wmles_wall_bc_wiring.py 同一个手法。"""
        mesh, ops = mesh_and_ops
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
        rank, n_ranks = 0, 1
        cell_partition = np.zeros(n_cells, dtype=np.int32)
        freestream = {"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0}
        mu = 1.8e-5

        fc = mesh.face_connectivity
        boundary_face = int(np.nonzero(fc.is_boundary)[0][0])
        wall_cell = int(fc.owner_cell[boundary_face])
        mesh.boundary_groups = {"wall_group": np.array([wall_cell], dtype=np.int64)}
        mesh.boundary_bc_types = {"wall_group": "WALL"}

        import types
        from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider
        root_stub = types.SimpleNamespace(
            mesh=mesh, freestream=freestream, turb_model_name="WMLES", wmles_model=object(),
        )
        boundary_ghost_provider = build_boundary_ghost_provider(root_stub, bc_overrides={})

        package = build_fully_distributed_rank_package(
            mesh, ops, mesh.face_connectivity, cell_partition, rank=rank, n_ranks=n_ranks,
            boundary_ghost_provider_global=boundary_ghost_provider,
            freestream=freestream, mu_molecular=mu, mach_ref=0.2,
            order=mesh.order, enable_viscous=True, turb_model_name="WMLES",
        )

        assert package['wall_distance_compact'] is not None

        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
        solver = DistributedFRSolver.from_fully_distributed_package(package, n_ranks=n_ranks, rank=rank)

        assert solver.turb_model_name == "WMLES"
        assert solver.wmles_model is not None
        assert solver.turb_model is None  # WMLES 没有 k/omega ODE 状态

        rng = np.random.default_rng(17)
        U0 = _nonuniform_U(mesh, rng)
        solver.state.U[:n_cells] = U0
        solver.state.Q[:n_cells] = conserved_to_primitive(U0[..., :5])

        solver.step(1e-6)

        assert np.all(np.isfinite(solver.state.U[:n_cells]))

    def test_step_with_les_runs_and_stays_finite(self, mesh_and_ops):
        """"完全分布式加载"模式补齐 LES（2026-09-02 续接）——WALE 纯
        代数模型，不需要 root 预计算任何几何量，构造应该比 SST 家族
        简单得多。"""
        mesh, ops = mesh_and_ops
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
        rank, n_ranks = 0, 1
        cell_partition = np.zeros(n_cells, dtype=np.int32)
        freestream = {"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0}
        mu = 1.8e-5

        import types
        from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider
        root_stub = types.SimpleNamespace(
            mesh=mesh, freestream=freestream, turb_model_name="LES", wmles_model=None,
        )
        boundary_ghost_provider = build_boundary_ghost_provider(root_stub, bc_overrides={})

        package = build_fully_distributed_rank_package(
            mesh, ops, mesh.face_connectivity, cell_partition, rank=rank, n_ranks=n_ranks,
            boundary_ghost_provider_global=boundary_ghost_provider,
            freestream=freestream, mu_molecular=mu, mach_ref=0.2,
            order=mesh.order, enable_viscous=True, turb_model_name="LES",
        )

        assert package['wall_distance_compact'] is None  # LES 不需要壁面距离

        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
        solver = DistributedFRSolver.from_fully_distributed_package(package, n_ranks=n_ranks, rank=rank)

        assert solver.turb_model_name == "LES"
        assert solver.sgs_model is not None
        assert solver.turb_model is None
        assert solver.wmles_model is None

        rng = np.random.default_rng(19)
        U0 = _nonuniform_U(mesh, rng)
        solver.state.U[:n_cells] = U0
        solver.state.Q[:n_cells] = conserved_to_primitive(U0[..., :5])

        solver.step(1e-6)

        assert np.all(np.isfinite(solver.state.U[:n_cells]))


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
        from autoflowcfd.core.time_integration.base import TimeIntegrator, TimeIntegrationScheme
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive

        mesh, ops = mesh_and_ops
        rng = np.random.default_rng(99)
        U0 = _nonuniform_U(mesh, rng)
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
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


class TestFullyDistributedDualTimeSupport:
    """"完全分布式加载" + DUAL_TIME 决定性验证（2026-09-02 续接，见
    distributed_mesh_loader.py 模块文档"DUAL_TIME 支持"一节）——此前
    `build_fully_distributed_rank_package` 从未把 `time_scheme`/
    `dual_time_inner_iter` 塞进 package，`--fully-distributed
    --time-method dual-time` 因此恒被 CLI 拒绝，不是设计上不支持。"""

    def test_package_carries_time_scheme_fields(self, mesh_and_ops):
        from autoflowcfd.core.time_integration.base import TimeIntegrationScheme

        mesh, ops = mesh_and_ops
        n_cells = mesh.n_cells
        cell_partition = np.zeros(n_cells, dtype=np.int32)
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
            freestream=freestream, mu_molecular=1.8e-5, mach_ref=0.2,
            order=mesh.order, enable_viscous=True,
            time_scheme=TimeIntegrationScheme.DUAL_TIME, dual_time_inner_iter=7,
        )
        assert package['time_scheme'] == TimeIntegrationScheme.DUAL_TIME
        assert package['dual_time_inner_iter'] == 7

        # package 本身仍然可以安全 pickle（真正跨进程发送的前提，与既有
        # TestRankPackagePicklable 同一个判据）——枚举值可安全序列化。
        pickle.loads(pickle.dumps(package))

    def test_default_time_scheme_is_ssp_rk3(self, mesh_and_ops):
        from autoflowcfd.core.time_integration.base import TimeIntegrationScheme

        mesh, ops = mesh_and_ops
        n_cells = mesh.n_cells
        cell_partition = np.zeros(n_cells, dtype=np.int32)
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
            freestream=freestream, mu_molecular=1.8e-5, mach_ref=0.2,
            order=mesh.order, enable_viscous=True,
        )
        assert package['time_scheme'] == TimeIntegrationScheme.SSP_RK3
        assert package['dual_time_inner_iter'] == 20

    def test_from_fully_distributed_package_step_uses_dual_time(self, mesh_and_ops):
        """真正调用 `step()`：DUAL_TIME 分支必须被走到（而不是静默退回
        SSP_RK3），且不崩溃、结果有限。"""
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
        from autoflowcfd.core.time_integration.base import TimeIntegrationScheme
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive

        mesh, ops = mesh_and_ops
        n_cells, n_sps = mesh.n_cells, mesh.n_sps_per_cell
        rng = np.random.default_rng(101)
        U0 = _nonuniform_U(mesh, rng)

        cell_partition = np.zeros(n_cells, dtype=np.int32)
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
            freestream=freestream, mu_molecular=1.8e-5, mach_ref=0.2,
            order=mesh.order, enable_viscous=True,
            time_scheme=TimeIntegrationScheme.DUAL_TIME, dual_time_inner_iter=3,
        )
        solver = DistributedFRSolver.from_fully_distributed_package(package, n_ranks=1, rank=0)
        assert solver._time_integrator.scheme == TimeIntegrationScheme.DUAL_TIME
        assert solver._time_integrator.dual_time_steps == 3

        solver.state.U[:n_cells] = U0
        solver.state.Q[:n_cells] = conserved_to_primitive(U0[..., :5])
        assert solver._dual_time_U_prev is None

        res = solver.step(1e-6)

        assert np.isfinite(res)
        assert np.all(np.isfinite(solver.state.U[:n_cells]))
        # DUAL_TIME 分支必须在 step() 结束时持久化上一物理时间层状态
        # （BDF2 需要），不是 SSP_RK3 分支（那条分支从不碰这个属性）。
        assert solver._dual_time_U_prev is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
