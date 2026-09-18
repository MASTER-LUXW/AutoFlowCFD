"""验证 `MultiGPUDistributedSolver._init_distributed_face_geometry` 传给
`build_gpu_flat_face` 的对象类型正确——真实 bug 修复（2026-09-02，排查
"--multi-gpu 仍拒绝 native"时发现，与 native 无关，collapsed 模式下同样
必现）：此前直接传 `DistributedFlatFaceGeometry`（只对部分字段提供
`@property` 转发到 `base_flat`），`GPUFlatFaceGeometry.__init__` 访问
`owner_cell`/`neighbor_cell`/`owner_adj_row_exact`/`color_face_indices`等
它没有转发的字段时必然 `AttributeError`——`MultiGPUDistributedSolver`
从未真正构造成功过。

本机没有真实 CuPy，但 `GPUFlatFaceGeometry.__init__` 只依赖 `cp.asarray`
（等价于 `np.asarray`）和 `cp.cuda.Device(device_id)` 上下文管理器
——用一个最小 numpy 假 cupy 模块可以在本机真实执行这个构造函数本身
（不是简单的 hasattr 属性名单，是真的跑一遍生产代码路径），决定性验证
`dist_flat_face.base_flat` 才是正确的输入。
"""

import numpy as np
import pytest
from contextlib import contextmanager
from unittest.mock import patch

from autoflowcfd.core.mpi.partition import build_distributed_partition
from autoflowcfd.core.mpi.distributed_flat_face import build_distributed_flat_face
from autoflowcfd.fr.operators import generate_fr_operators
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


class _FakeCudaDevice:
    def __init__(self, device_id):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeCuda:
    Device = _FakeCudaDevice


class _FakeCupy:
    """最小 numpy 假 cupy——只提供 GPUFlatFaceGeometry.__init__ 用到的
    两个接口（asarray + cuda.Device 上下文管理器），足够真实跑一遍
    该构造函数，不需要真实 CUDA 硬件。"""
    asarray = staticmethod(np.asarray)
    array = staticmethod(np.array)
    cuda = _FakeCuda


@pytest.fixture()
def dist_fc_native():
    order = 2
    mesh = _build_synthetic_mixed_mesh(order, tet_basis_mode="native")
    ops = generate_fr_operators(order)
    cell_partition = np.array([0, 1, 0, 1], dtype=np.int32)
    fc = mesh.face_connectivity
    partition = build_distributed_partition(fc, cell_partition, rank=0, n_ranks=2)
    return build_distributed_flat_face(mesh, ops, partition, cell_partition=cell_partition)


def test_wrapper_object_missing_gpu_required_attributes(dist_fc_native):
    """决定性确认此前的真实 bug 场景：直接传 DistributedFlatFaceGeometry
    包装对象本身缺少 GPUFlatFaceGeometry 需要的关键字段（不是转发属性）。"""
    missing = [
        name for name in (
            "owner_cell", "neighbor_cell", "owner_adj_row_exact",
            "neighbor_adj_row_exact", "owner_cube_face", "neighbor_cube_face",
            "true_area_weight", "ref_area_weight", "boundary_extrap_native", "lift_native",
            "color_face_indices", "n_colors", "mixed_nb_partner",
        )
        if not hasattr(dist_fc_native, name)
    ]
    assert missing, (
        "DistributedFlatFaceGeometry 现在似乎已经代理了这些字段——如果这是"
        "有意的新改动，请同步检查 gpu_distributed_init.py 是否还需要"
        "'.base_flat' 这个变通方案，并更新本测试。"
    )


def test_base_flat_is_gpu_compatible_native(dist_fc_native):
    """决定性验证：`dist_flat_face.base_flat`（修复后实际传给
    build_gpu_flat_face 的对象）能被 GPUFlatFaceGeometry.__init__ 真实
    构造成功（用假 cupy 在本机真实执行，不是属性名单走查），且包含
    native 四面体（路径C）字段。"""
    from autoflowcfd.core.gpu import gpu_face_geometry as gfg_module

    with patch.object(gfg_module, "get_cupy", return_value=_FakeCupy()):
        gpu_flat = gfg_module.build_gpu_flat_face(dist_fc_native.base_flat, device_id=0)

    assert gpu_flat.n_faces == dist_fc_native.base_flat.n_faces
    assert np.array_equal(gpu_flat.owner_cell, dist_fc_native.base_flat.owner_cell)
    assert np.array_equal(gpu_flat.owner_cube_face, dist_fc_native.base_flat.owner_cube_face)
    assert np.any(dist_fc_native.base_flat.owner_cube_face >= 6), (
        "native 网格的合成测试网格应该包含至少一个 native 四面体面，"
        "否则本测试没有真正覆盖 native 字段透传"
    )
    assert np.array_equal(gpu_flat.lift_native, dist_fc_native.base_flat.lift_native)


def test_wrapper_object_rejected_by_gpu_flat_face_constructor(dist_fc_native):
    """回归测试：直接把 DistributedFlatFaceGeometry 包装对象传给
    build_gpu_flat_face（此前的真实 bug 现场）必须失败——如果这个测试
    开始失败（不再抛 AttributeError），说明 DistributedFlatFaceGeometry
    已经代理了全部所需字段，`gpu_distributed_init.py` 里的 `.base_flat`
    变通方案可能不再需要，但在改回来之前必须先确认全部字段都已代理。"""
    from autoflowcfd.core.gpu import gpu_face_geometry as gfg_module

    with patch.object(gfg_module, "get_cupy", return_value=_FakeCupy()):
        with pytest.raises(AttributeError):
            gfg_module.build_gpu_flat_face(dist_fc_native, device_id=0)
