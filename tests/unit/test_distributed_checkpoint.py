"""
AutoFlowCFD V2.0 - 分布式 Checkpoint 和结果保存测试

验证分布式 checkpoint/结果保存在非 MPI 环境下的降级行为和接口正确性。
"""

import pytest
import numpy as np


class TestDistributedCheckpointImport:
    """测试分布式 checkpoint 模块导入。"""

    def test_import_module(self):
        """验证模块可以正常导入。"""
        from autoflowcfd.core.mpi.distributed_checkpoint import (
            gather_global_state,
            scatter_local_state,
            distributed_save_checkpoint,
            distributed_load_checkpoint,
            distributed_save_results,
        )
        assert gather_global_state is not None
        assert scatter_local_state is not None


class TestGatherScatter:
    """测试 gather/scatter 函数。"""

    def test_scatter_local_state(self):
        """验证 scatter 从全局数组中提取 local cells。"""
        from autoflowcfd.core.mpi.distributed_checkpoint import scatter_local_state

        n_global = 10
        n_sps = 4
        n_vars = 5

        U_global = np.random.rand(n_global, n_sps, n_vars)
        local_cells = np.array([0, 2, 5, 7])  # 4 个 local cells

        U_local = scatter_local_state(U_global, local_cells)

        assert U_local.shape == (4, n_sps, n_vars)
        np.testing.assert_array_equal(U_local, U_global[local_cells])

    def test_gather_single_rank(self):
        """验证单 rank 时 gather 直接返回拷贝。"""
        from autoflowcfd.core.mpi.distributed_checkpoint import gather_global_state

        n_local = 5
        n_sps = 4
        n_vars = 5
        n_global = 5

        U_local = np.random.rand(n_local, n_sps, n_vars)
        local_cells = np.arange(n_local)

        U_global = gather_global_state(U_local, local_cells, n_global)

        # 单 rank 模式（非 MPI 环境），应返回拷贝
        assert U_global is not None
        assert U_global.shape == (n_global, n_sps, n_vars)
        np.testing.assert_array_equal(U_global, U_local)

    def test_gather_scatter_roundtrip(self):
        """验证 scatter(gather(U)) == U 的往返一致性。"""
        from autoflowcfd.core.mpi.distributed_checkpoint import (
            gather_global_state,
            scatter_local_state,
        )

        n_sps = 4
        n_vars = 5

        # 模拟 3 个 rank 的 local cells
        U_local = np.random.rand(5, n_sps, n_vars)
        local_cells = np.array([0, 3, 5, 7, 9])
        n_global = 10

        # gather（单 rank 模式）
        U_global = gather_global_state(U_local, local_cells, n_global)

        # scatter
        U_recovered = scatter_local_state(U_global, local_cells)

        np.testing.assert_array_equal(U_recovered, U_local)


class TestDistributedCheckpointInterface:
    """测试分布式 checkpoint 接口。"""

    def test_save_checkpoint_function_exists(self):
        """验证 distributed_save_checkpoint 函数存在。"""
        from autoflowcfd.core.mpi.distributed_checkpoint import distributed_save_checkpoint
        assert callable(distributed_save_checkpoint)

    def test_load_checkpoint_function_exists(self):
        """验证 distributed_load_checkpoint 函数存在。"""
        from autoflowcfd.core.mpi.distributed_checkpoint import distributed_load_checkpoint
        assert callable(distributed_load_checkpoint)

    def test_save_results_function_exists(self):
        """验证 distributed_save_results 函数存在。"""
        from autoflowcfd.core.mpi.distributed_checkpoint import distributed_save_results
        assert callable(distributed_save_results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
