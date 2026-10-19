"""分布式网格适配器按（分区, 面几何, 网格状态）缓存（I-100）。

构造时按压缩索引空间复制全局 jacobians / 体积 / 过积分细点度量；此前每次残差求值（无粘、粘性、湍流、人工粘性、
隐式 J v）都新建一个。
"""

import inspect

import numpy as np
import pytest

from autoflowcfd.core.mpi import distributed_compute as dc
from autoflowcfd.core.mpi.distributed_compute import distributed_mesh_adapter
from autoflowcfd.core.mpi.distributed_flat_face import build_distributed_flat_face
from autoflowcfd.core.mpi.partition import build_distributed_partition
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


@pytest.fixture
def setup():
    mesh = _build_synthetic_mixed_mesh(1)
    part = np.array([0, 1, 0, 1], dtype=np.int32)
    partition = build_distributed_partition(mesh.face_connectivity, part, rank=0, n_ranks=2)
    dist_fc = build_distributed_flat_face(mesh, mesh.operators, partition, cell_partition=part)
    return mesh, partition, dist_fc


def test_adapter_is_built_once_per_state(setup):
    mesh, partition, dist_fc = setup
    a = distributed_mesh_adapter(partition, dist_fc, mesh, mesh.operators)
    assert distributed_mesh_adapter(partition, dist_fc, mesh, mesh.operators) is a
    np.testing.assert_array_equal(
        a.jacobians["det_jacs"],
        mesh.jacobians["det_jacs"].reshape(mesh.n_cells, -1)[dist_fc.compact_global_ids])


def test_replaced_mesh_geometry_rebuilds_the_adapter(setup):
    mesh, partition, dist_fc = setup
    a = distributed_mesh_adapter(partition, dist_fc, mesh, mesh.operators)
    mesh.jacobians = {k: v.copy() for k, v in mesh.jacobians.items()}       # 例如换阶后重建的几何
    assert distributed_mesh_adapter(partition, dist_fc, mesh, mesh.operators) is not a


def test_residual_entry_points_use_the_cache_and_require_the_physics_inputs():
    src = inspect.getsource(dc)
    assert src.count("DistributedMeshAdapter(partition, dist_fc, local_mesh, ops)") == 1   # 只在缓存函数里
    params = inspect.signature(dc.distributed_compute_inviscid_residual).parameters
    assert params["mach_ref"].default is inspect.Parameter.empty
    assert params["boundary_ghost_provider"].default is inspect.Parameter.empty
    params = inspect.signature(dc.distributed_compute_viscous_residual).parameters
    assert params["boundary_ghost_provider"].default is inspect.Parameter.empty
    assert not hasattr(dc, "distributed_compute_physical_gradient")
