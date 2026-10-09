"""完全分布式网格加载：SST 湍流路径与单机逐项一致（从 test_distributed_mesh_loader.py 拆出，背景见该文件模块文档）。"""

import numpy as np
import pytest

from autoflowcfd.core.mpi.distributed_mesh_loader import build_fully_distributed_rank_package

from tests.unit._wall_source import synthetic_wall_source
from tests.unit._distributed_mesh_loader_common import _nonuniform_U, mesh_and_ops
from tests.unit._boundary_groups import tag_wall_cells


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
            turb_model_name="SST",
            turbulence_intensity=0.02, viscosity_ratio=8.0,
            wall_distance_source=synthetic_wall_source(mesh),
        )

        assert package['turb_model_name'] == "SST"
        assert package['wall_distance_compact'] is not None

        # ---- 独立参照：直接调用 compute_distributed_wall_distance ----
        from autoflowcfd.core.mpi.partition import build_distributed_partition
        from autoflowcfd.core.mpi.distributed_flat_face import build_distributed_flat_face
        from autoflowcfd.core.mpi.distributed_turbulence import compute_distributed_wall_distance
        partition_ref = build_distributed_partition(mesh.face_connectivity, cell_partition, rank=rank, n_ranks=n_ranks)
        dist_fc_ref = build_distributed_flat_face(mesh, ops, partition_ref, cell_partition=cell_partition)
        expected_wall_distance = compute_distributed_wall_distance(
            dist_fc_ref, mesh, synthetic_wall_source(mesh))
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
        n_cells = mesh.n_cells
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
            wall_distance_source=synthetic_wall_source(mesh),
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

    def test_sa_package_builds_model_pins_wall_points_and_steps(self, mesh_and_ops):
        """SA-neg 走完全分布式加载：root 为输运模型算壁距并放进包，rank 侧构造 SA 模型（来流值与
        单机同一个工厂），按本 rank 的 local 壁距识别壁面解点并置零，真正推进一步保持有限。"""
        mesh, ops = mesh_and_ops
        n_cells = mesh.n_cells
        cell_partition = np.zeros(n_cells, dtype=np.int32)
        freestream = {"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0}
        mu = 1.8e-5

        import types
        from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider
        from autoflowcfd.core.turbulence.sa import SAModel
        from autoflowcfd.core.turbulence.sa.constants import chi_for_viscosity_ratio
        root_stub = types.SimpleNamespace(mesh=mesh, freestream=freestream, turb_model_name="SA", wmles_model=None)
        package = build_fully_distributed_rank_package(
            mesh, ops, mesh.face_connectivity, cell_partition, rank=0, n_ranks=1,
            boundary_ghost_provider_global=build_boundary_ghost_provider(root_stub, bc_overrides={}),
            freestream=freestream, mu_molecular=mu, mach_ref=0.2, order=mesh.order, enable_viscous=True,
            turb_model_name="SA", viscosity_ratio=3.0, wall_distance_source=synthetic_wall_source(mesh),
        )
        assert package['wall_distance_compact'] is not None

        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
        from autoflowcfd.core.mpi.distributed_turbulence import local_part
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
        solver = DistributedFRSolver.from_fully_distributed_package(package, n_ranks=1, rank=0)
        m = solver.turb_model
        assert isinstance(m, SAModel)
        assert m.nu_tilde_inf == pytest.approx(chi_for_viscosity_ratio(3.0) * mu / 1.225, rel=1e-14)
        wall = np.asarray(local_part(solver, solver.wall_distance_compact)) == 0.0
        assert wall.any() and np.all(m.nu_tilde_field[wall] == 0.0)

        U0 = _nonuniform_U(mesh, np.random.default_rng(7))
        solver.state.U[:n_cells] = U0
        solver.state.Q[:n_cells] = conserved_to_primitive(U0[..., :5])
        solver.step(1e-6)
        assert np.all(np.isfinite(solver.state.U[:n_cells])) and np.all(np.isfinite(m.nu_tilde_field))
        assert np.all(m.nu_tilde_field[wall] == 0.0)

    def test_step_with_ddes_runs_and_stays_finite(self, mesh_and_ops):
        """同上 SST 收尾集成测试，换成 DDES——与 IDDES 共用
        `from_fully_distributed_package` 里同一个 `ddes_model` 构造分支，
        唯一区别是 DDES 不
        需要 h_wn（只有 IDDES 需要近壁法向间距）。2026-09-02 起 DDES
        也需要 h_max（`apply_to_sst_model` 改用各向异性感知的 max_edge
        网格尺度，见 des.py::DDESModel.compute_grid_scale 文档"Note"
        一节），单独测一遍确认这条分支正确要求 h_max_global、正确产出
        非 None 的 `iddes_h_max_compact`，同时仍然不需要 h_wn。"""
        mesh, ops = mesh_and_ops
        n_cells = mesh.n_cells
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
                wall_distance_source=synthetic_wall_source(mesh),
            )

        h_max_global, _h_wn_global = compute_h_max_and_h_wn(mesh)
        package = build_fully_distributed_rank_package(
            mesh, ops, mesh.face_connectivity, cell_partition, rank=rank, n_ranks=n_ranks,
            boundary_ghost_provider_global=boundary_ghost_provider,
            freestream=freestream, mu_molecular=mu, mach_ref=0.2,
            order=mesh.order, enable_viscous=True, turb_model_name="DDES",
            h_max_global=h_max_global,
            wall_distance_source=synthetic_wall_source(mesh),
        )

        assert package['iddes_h_max_compact'] is not None
        assert package['iddes_h_wn_compact'] is None

        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
        solver = DistributedFRSolver.from_fully_distributed_package(package, n_ranks=n_ranks, rank=rank)

        assert solver.turb_model_name == "DDES"
        assert solver.ddes_model is not None

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
        n_cells = mesh.n_cells
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
            wall_distance_source=synthetic_wall_source(mesh),
        )

        assert package['iddes_h_max_compact'] is not None
        assert package['iddes_h_wn_compact'] is not None

        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
        from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
        solver = DistributedFRSolver.from_fully_distributed_package(package, n_ranks=n_ranks, rank=rank)

        assert solver.turb_model_name == "IDDES"
        assert solver.ddes_model is not None

        rng = np.random.default_rng(11)
        U0 = _nonuniform_U(mesh, rng)
        solver.state.U[:n_cells] = U0
        solver.state.Q[:n_cells] = conserved_to_primitive(U0[..., :5])

        # 两步（与 CPU 分布式 DDES/IDDES 的既有回归测试一致），因为
        # des_length_scale 第二步才真正走 halo-exchange 读回路径（第一
        # 步是 None，跳过读入分支——见 distributed_turbulence.py 里
        # build_distributed_turbulence_view 的 des_length_scale 一段）。
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
        n_cells = mesh.n_cells
        rank, n_ranks = 0, 1
        cell_partition = np.zeros(n_cells, dtype=np.int32)
        freestream = {"rho_inf": 1.225, "vel_inf": 33.33, "p_inf": 101325.0}
        mu = 1.8e-5

        fc = mesh.face_connectivity
        boundary_face = int(np.nonzero(fc.is_boundary)[0][0])
        wall_cell = int(fc.owner_cell[boundary_face])
        tag_wall_cells(mesh, [wall_cell])

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
            wall_distance_source=synthetic_wall_source(mesh),
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
        n_cells = mesh.n_cells
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
            wall_distance_source=synthetic_wall_source(mesh),
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
