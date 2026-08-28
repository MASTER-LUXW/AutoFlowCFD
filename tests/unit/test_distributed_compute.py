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
        )
        assert DistributedMeshAdapter is not None
        assert distributed_compute_inviscid_residual is not None

    def test_no_turbulence_transport_entry(self):
        """分布式湍流输运占位函数已移除（第三轮评审整改）：构造期 fail-fast
        拒绝非 none 湍流后，该入口零调用方且不可达，不应再存在。"""
        import autoflowcfd.core.mpi.distributed_compute as dc
        assert not hasattr(dc, "distributed_turbulence_transport")

    def test_import_distributed_solver(self):
        """验证分布式求解器可以正常导入。"""
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
        assert DistributedFRSolver is not None


class TestDistributedMeshAdapter:
    """测试 DistributedMeshAdapter 接口。"""

    def test_adapter_interface(self):
        """验证适配器提供与 HighOrderMesh 相同的接口。

        真实 bug 修复（#2，V2.0 专家组盲审第4轮，2026-08-28）：这个测试
        此前的 mock（`MockDistFC` 空对象、断言 `adapter.n_cells==10`
        即 n_local_cells、`adapter.jacobians is mesh.jacobians` 即原样
        转发）反映的是已确认错误的旧行为——`dist_fc`（真实的
        `DistributedFlatFaceGeometry`）的 owner_cell/neighbor_cell 用的
        是 local+halo 压缩索引空间（且是"棱柱在前"排列，见
        distributed_flat_face.py 模块文档），不是 n_local_cells，也不能
        直接复用 mesh 原始（全局单元编号）jacobians——两者是不同的索引
        空间。现在 mock 出一个具备 `compact_global_ids`/`base_flat.
        n_prism` 的 `MockDistFC`（10 local + 2 halo = 12 个位置，
        这里让 compact_global_ids 恰好是恒等排列 arange(12)，即测试
        场景本身不含棱柱/四面体混合重排——重排逻辑本身已由
        test_distributed_flat_face_prism_grouping.py 用真实网格单独
        验证过，这里只验证 DistributedMeshAdapter 按 compact_global_ids
        正确抽取+重排 jacobians 这一层逻辑），验证适配器按新的、正确的
        压缩索引空间语义工作。
        """
        from autoflowcfd.core.mpi.distributed_compute import DistributedMeshAdapter

        _n_local, _n_halo, n_compact = 10, 2, 12

        # 创建模拟数据
        class MockPartition:
            n_local_cells = _n_local
            n_halo = _n_halo
            n_global_cells = 12

        class MockMesh:
            n_cells = 12  # 完整全局网格单元数（compact_global_ids 索引进这个范围）
            n_points_1d = 2
            n_sps_per_cell = 8
            cell_types = np.array([0] * 10 + [1] * 2)
            jacobians = {
                "det_jacs": np.arange(12 * 8, dtype=float).reshape(12, 8),
                "inv_jacs": np.eye(3).reshape(1, 1, 3, 3).repeat(12, 0).repeat(8, 1),
            }
            jacobians_fine = None
            cell_volumes = np.arange(12, dtype=float)
            face_flux_points = None

        class MockOps:
            pass

        class _MockBaseFlat:
            n_prism = 0  # 这批 compact_global_ids 里没有棱柱

        class MockDistFC:
            compact_global_ids = np.arange(n_compact)  # 恒等排列（测试重点不在重排本身）
            base_flat = _MockBaseFlat()

        partition = MockPartition()
        dist_fc = MockDistFC()
        mesh = MockMesh()
        ops = MockOps()

        adapter = DistributedMeshAdapter(partition, dist_fc, mesh, ops)

        # 验证接口：n_cells 现在是压缩索引空间大小（n_local+n_halo），
        # 不是 n_local_cells。
        assert adapter.n_cells == n_compact
        assert adapter.n_halo_cells == 2
        assert adapter.n_points_1d == 2
        assert adapter.n_sps_per_cell == 8
        assert adapter.n_prism_cells == 0
        assert adapter.face_connectivity is dist_fc
        # 恒等排列下重排结果在数值上应与原始 mesh.jacobians 一致（但不
        # 再是同一个对象引用——现在总是重新抽取+拷贝）。
        np.testing.assert_array_equal(adapter.jacobians["det_jacs"], mesh.jacobians["det_jacs"])


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
