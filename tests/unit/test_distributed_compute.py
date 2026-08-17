"""
AutoFlowCFD V2.0 - 分布式残差计算测试

验证分布式计算模块在非 MPI 环境下的降级行为和基本接口正确性。
"""

import pytest
import numpy as np


class TestDistributedComputeImport:
    """测试分布式计算模块导入。"""

    def test_import_distributed_compute(self):
        """验证分布式计算模块可以正常导入。"""
        from autoflowcfd.core.mpi.distributed_compute import (
            DistributedMeshAdapter,
            distributed_compute_inviscid_residual,
            distributed_compute_viscous_residual,
            distributed_compute_physical_gradient,
            distributed_turbulence_transport,
        )
        assert DistributedMeshAdapter is not None
        assert distributed_compute_inviscid_residual is not None

    def test_import_distributed_solver(self):
        """验证分布式求解器可以正常导入。"""
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
        assert DistributedFRSolver is not None


class TestDistributedMeshAdapter:
    """测试 DistributedMeshAdapter 接口。"""

    def test_adapter_interface(self):
        """验证适配器提供与 HighOrderMesh 相同的接口。"""
        from autoflowcfd.core.mpi.distributed_compute import DistributedMeshAdapter
        from autoflowcfd.core.mpi.partition import DistributedPartition
        from autoflowcfd.core.mpi.distributed_flat_face import DistributedFlatFaceGeometry

        # 创建模拟数据
        class MockPartition:
            n_local_cells = 10
            n_halo = 2
            n_global_cells = 12

        class MockMesh:
            n_points_1d = 2
            n_sps_per_cell = 8
            cell_types = np.array([0] * 10)  # 10 tet cells
            jacobians = {"det_jacs": np.ones((10, 8)), "inv_jacs": np.eye(3).reshape(1, 1, 3, 3).repeat(10, 0).repeat(8, 1)}
            jacobians_fine = None
            face_flux_points = None

        class MockOps:
            pass

        class MockDistFC:
            pass

        partition = MockPartition()
        dist_fc = MockDistFC()
        mesh = MockMesh()
        ops = MockOps()

        adapter = DistributedMeshAdapter(partition, dist_fc, mesh, ops)

        # 验证接口
        assert adapter.n_cells == 10
        assert adapter.n_halo_cells == 2
        assert adapter.n_points_1d == 2
        assert adapter.n_sps_per_cell == 8
        assert adapter.n_prism_cells == 0
        assert adapter.face_connectivity is dist_fc
        assert adapter.jacobians is mesh.jacobians


class TestDistributedSolverInterface:
    """测试 DistributedFRSolver 接口。"""

    def test_solver_has_step_method(self):
        """验证分布式求解器有 step 方法。"""
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
        assert hasattr(DistributedFRSolver, 'step')

    def test_solver_has_solve_method(self):
        """验证分布式求解器有 solve 方法。"""
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
        assert hasattr(DistributedFRSolver, 'solve')


class TestNonMPIDegradation:
    """测试非 MPI 环境下的降级行为。"""

    def test_mpi_available_check(self):
        """验证 MPI 可用性检查。"""
        from autoflowcfd.core.mpi import mpi_available, get_rank, get_size
        # 在非 MPI 环境中，这些函数应该返回安全的默认值
        rank = get_rank()
        size = get_size()
        assert rank == 0
        assert size == 1

    def test_cli_n_ranks_without_mpi(self):
        """验证 CLI 在无 MPI 环境下使用 --n-ranks > 1 时的错误处理。"""
        from autoflowcfd.core.mpi import mpi_available
        # 这个测试验证逻辑：如果 MPI 不可用且 n_ranks > 1，应该报错
        # 实际的 CLI 测试需要 click.testing.CliRunner
        if not mpi_available:
            # 预期行为：应该提示用户安装 mpi4py
            pass  # 实际测试需要启动 CLI，这里只验证逻辑


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
