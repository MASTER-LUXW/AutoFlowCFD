"""完全分布式网格加载：双时间步（DUAL_TIME）支持（从 test_distributed_mesh_loader.py 拆出，背景见该文件模块文档）。"""

import pickle
import numpy as np

from autoflowcfd.core.mpi.distributed_mesh_loader import build_fully_distributed_rank_package

from tests.unit._wall_source import synthetic_wall_source
from tests.unit._distributed_mesh_loader_common import _nonuniform_U, mesh_and_ops


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
            wall_distance_source=synthetic_wall_source(mesh),
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
            wall_distance_source=synthetic_wall_source(mesh),
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
        n_cells = mesh.n_cells
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
            wall_distance_source=synthetic_wall_source(mesh),
        )
        solver = DistributedFRSolver.from_fully_distributed_package(package, n_ranks=1, rank=0)
        assert solver.time_integrator.scheme == TimeIntegrationScheme.DUAL_TIME
        assert solver.time_integrator.dual_time_steps == 3

        solver.state.U[:n_cells] = U0
        solver.state.Q[:n_cells] = conserved_to_primitive(U0[..., :5])
        assert solver._dual_time_U_prev is None

        res = solver.step(1e-6)

        assert np.isfinite(res)
        assert np.all(np.isfinite(solver.state.U[:n_cells]))
        # DUAL_TIME 分支必须在 step() 结束时持久化上一物理时间层状态
        # （BDF2 需要），不是 SSP_RK3 分支（那条分支从不碰这个属性）。
        assert solver._dual_time_U_prev is not None
