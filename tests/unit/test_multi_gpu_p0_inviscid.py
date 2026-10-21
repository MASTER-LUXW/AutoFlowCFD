"""多 GPU 分布式 P0 无粘残差走有限体积核，在本 rank 面表上与 CPU 单机 P0 残差逐 local 单元一致（I-187）。

此前多 GPU 在 P0 也调用 FR 无粘残差：P0 没有过积分算子，直接抛异常（Order Continuation 起步的 P0 阶段跑不了）。
网格含混合拆分面（棱柱四边形侧面一半挨四面体、一半在边界）与真多源面（两半各挨一个四面体），按两 rank 划分，
逐 rank 构造真实的分布式面几何、限制到本 rank 面序的幽灵态提供者；CUDA RawKernel 在本机不能执行，换成逐面调用
CPU numba 核的替身（同一算法；替身同时检查内核要求的 int32 单元编号）。
"""

import numpy as np
import pytest

from autoflowcfd.boundary.fr_ghost_state import restrict_ghost_provider
from autoflowcfd.core.fr_residual.inviscid import compute_inviscid_residual_fr
from autoflowcfd.core.mpi.distributed_flat_face import build_distributed_flat_face
from autoflowcfd.core.mpi.partition import build_distributed_partition
from tests.unit._gpu_cupy_shim import patch_module_get_cupy
from tests.unit._numpy_as_cupy import NumpyAsCupy
from tests.unit.test_gpu_mixed_face_parity import _MESHES, _state
from tests.unit.test_p0_mixed_split_faces import _provider


@pytest.fixture(autouse=True)
def _gpu_shim(monkeypatch):
    import autoflowcfd.core.gpu as core_gpu_mod
    import autoflowcfd.core.gpu.distributed.gpu_distributed as gd_mod
    import autoflowcfd.core.gpu.residual.gpu_flux as gpu_flux_mod
    import autoflowcfd.core.gpu.residual.gpu_p0_inviscid as p0_mod
    from autoflowcfd.core.fr_residual.inviscid_p0_kernel import _p0_inviscid_kernel

    patch_module_get_cupy(monkeypatch, [core_gpu_mod, gd_mod, gpu_flux_mod, p0_mod], NumpyAsCupy())

    def kernel_stand_in(Q_flat, owner, neighbor, is_bnd, normal, area_w, volumes, Q_ghost, n_cells, n_faces,
                        mach_ref=0.1):
        assert owner.dtype == np.int32 and neighbor.dtype == np.int32       # CUDA 内核的参数是 int*
        res = np.zeros((n_cells, 5))
        for f in range(n_faces):                                              # 逐面调用，没有并行累加竞争
            _p0_inviscid_kernel(owner.astype(np.int64), neighbor.astype(np.int64), is_bnd, normal, area_w,
                                np.ascontiguousarray(Q_flat), Q_ghost, volumes, np.array([f], np.int64), res,
                                mach_ref)
        return res[:, None, :]

    monkeypatch.setattr(p0_mod, "compute_inviscid_residual_p0_cupy_gpu_resident", kernel_stand_in)


def _rank_solver(mesh, cell_partition, rank, U_global, provider):
    """一个 rank 的多 GPU 求解器替身：只带 P0 无粘残差用到的真实分布式数据。"""
    fc = mesh.face_connectivity
    partition = build_distributed_partition(fc, cell_partition, rank=rank, n_ranks=2)
    dist_fc = build_distributed_flat_face(mesh, mesh.operators, partition, cell_partition=cell_partition)
    from autoflowcfd.core.gpu.distributed.gpu_distributed.residual import _MultiGPUResidualMixin

    s = _MultiGPUResidualMixin()
    s.__dict__.update(
        mesh=mesh, ops=mesh.operators, partition=partition, dist_flat_face=dist_fc, device_id=0,
        flat_face_gpu=None,
        _perm_gpu=dist_fc.perm, _inv_perm_gpu=dist_fc.inv_perm,
        mesh_data={"cell_volumes": mesh.cell_volumes[dist_fc.compact_global_ids]},
        boundary_ghost_provider=restrict_ghost_provider(provider, partition.local_faces),
        freestream={"mach_ref": 0.1},
        # halo 交换协议原生排列：local 在前、halo 在后
        U_extended_gpu=U_global[np.concatenate([partition.local_cells, partition.halo_cells])],
    )
    return s


@pytest.mark.parametrize("kind", sorted(_MESHES))
def test_multi_gpu_p0_inviscid_matches_cpu_on_every_rank(kind):
    mesh = _MESHES[kind](0)
    U = _state(mesh)
    provider = _provider(mesh)
    ref = compute_inviscid_residual_fr(U, mesh, mesh.operators, boundary_ghost_provider=provider, mach_ref=0.1)
    cell_partition = (np.arange(mesh.n_cells) % 2).astype(np.int32)        # 棱柱与四面体分在不同 rank
    for rank in range(2):
        s = _rank_solver(mesh, cell_partition, rank, U, provider)
        got = np.asarray(s.compute_inviscid_residual_gpu())
        expected = ref[s.partition.local_cells]
        scale = np.abs(ref).max(axis=(0, 1))
        assert np.abs(got - expected).max(axis=(0, 1)).max() <= 1e-12 * scale.max(), (rank, got, expected)
        # 面几何按分布式面几何对象缓存
        assert s._p0_face_arrays_gpu() is s._p0_face_cache
