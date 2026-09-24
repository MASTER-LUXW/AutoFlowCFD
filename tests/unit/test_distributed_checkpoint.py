"""
AutoFlowCFD V2.0 - 分布式 Checkpoint 和结果保存测试

验证分布式 checkpoint/结果保存在非 MPI 环境下的降级行为和接口正确性。
"""

from pathlib import Path

import pytest
import numpy as np
from tests.unit._gpu_cupy_shim import patch_module_get_cupy


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


class _NumpyAsCupy:
    """把 numpy 数组本身当成"CuPy 数组"：asnumpy/asarray 对 numpy 输入
    是恒等操作；`cuda.Device(id)` 返回一个 no-op 上下文管理器——足以
    验证 `MultiGPUDistributedSolver.save_checkpoint_distributed`/
    `load_checkpoint_distributed` 这层"GPU 下载/上传 + 复用 CPU
    distributed_save_checkpoint/distributed_load_checkpoint"round-trip
    逻辑本身是否透明无损，不需要真实 CUDA 设备。"""

    def asnumpy(self, x):
        return np.asarray(x)

    def asarray(self, x):
        return np.asarray(x)

    class cuda:
        class Device:
            def __init__(self, device_id):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False


class TestGpuDistributedCheckpointRoundtrip:
    """`MultiGPUDistributedSolver.save_checkpoint_distributed`/
    `load_checkpoint_distributed`（gpu_distributed_init.py，#4，
    2026-08-28）端到端往返测试——此前只有"函数存在"这类接口测试覆盖
    CPU 侧的 `distributed_save_checkpoint`/`distributed_load_checkpoint`
    本身，GPU 特有的这两个方法（下载 U_gpu 到 state.U、按真实签名调用
    CPU 版函数、加载后同步回 state.U/U_gpu）从未有过哪怕是 numpy 替身
    级别的验证。用最小鸭子类型对象（不构造完整 MultiGPUDistributedSolver
    ——那需要真实 CuPy 数组管理器）调用这两个*未绑定方法*本身，验证
    save->load 往返后 U 数据逐位不变。
    """

    def _build_fake_solver(self, tmp_path, n_cells=4, n_sps=8, n_vars=5):
        from autoflowcfd.core.mpi.partition import build_distributed_partition
        from autoflowcfd.core.mpi.distributed_state import DistributedFRState
        from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh

        mesh = _build_synthetic_mixed_mesh(order=1)
        fc = mesh.face_connectivity
        cell_partition = np.zeros(mesh.n_cells, dtype=np.int32)  # 单 rank：全部 local
        partition = build_distributed_partition(fc, cell_partition, rank=0, n_ranks=1)

        import types
        solver = types.SimpleNamespace()
        solver.partition = partition
        solver.state = DistributedFRState(partition, n_sps, n_vars)
        solver.mesh = mesh
        solver.device_id = 0
        n_local = partition.n_local_cells
        rng = np.random.default_rng(123)
        solver.U_gpu = rng.uniform(-1.0, 1.0, size=(n_local, n_sps, n_vars))
        return solver

    def test_save_then_load_roundtrip_preserves_state(self, tmp_path, monkeypatch):
        import autoflowcfd.core.gpu.distributed.gpu_distributed_init as gdi_mod
        from autoflowcfd.core.gpu.distributed.gpu_distributed import MultiGPUDistributedSolver

        shim = _NumpyAsCupy()
        patch_module_get_cupy(monkeypatch, gdi_mod, shim)

        solver = self._build_fake_solver(tmp_path)
        U_gpu_original = solver.U_gpu.copy()

        saved_path = MultiGPUDistributedSolver.save_checkpoint_distributed(
            solver, str(tmp_path), iteration=42, input_file="dummy.nas",
            order=1, turbulence_model="none", backend="gpu",
        )
        assert saved_path is not None
        assert Path(saved_path).exists()

        # save 之后 state.U[:n_local] 应该已经被同步（下载自 U_gpu）。
        n_local = solver.partition.n_local_cells
        np.testing.assert_allclose(solver.state.U[:n_local], U_gpu_original)

        # 构造一个"全新"的 fake solver（模拟真正的 resume 场景：U_gpu
        # 当前是任意初始值，load 之后必须被 checkpoint 里的值覆盖）。
        solver2 = self._build_fake_solver(tmp_path)
        solver2.U_gpu = np.zeros_like(U_gpu_original)  # 明显不同于原值

        metadata, iteration = MultiGPUDistributedSolver.load_checkpoint_distributed(
            solver2, saved_path,
        )
        assert iteration == 42
        # HDF5 往返后是 numpy bool_，不是 Python bool 单例，用 == 而非 is。
        assert metadata.get("distributed") == True  # noqa: E712

        np.testing.assert_allclose(solver2.U_gpu, U_gpu_original, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(
            solver2.state.U[:n_local], U_gpu_original, rtol=1e-10, atol=1e-12
        )


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
