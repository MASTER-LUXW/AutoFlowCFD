"""分区线切过混合拆分面时的分布式残差（I-98）。

混合拆分面：棱柱四边形侧面一半挨四面体、一半在边界上（`fr/face_flux_points/grouping.py`）。棱柱在 rank 0、
四面体在 rank 1 时，rank 1 以 halo_owner 面持有内部子面记录，此前它的配对边界子面记录（owner 为远端棱柱）
不在 rank 1 的面表里，分布式面几何构造报 RuntimeError（与分区相关的启动崩溃）。判据：两个 rank 的 local
单元残差与单机一致（非均匀流场；P1 无粘与粘性，P0 粘性——P0 无粘走汇集全局状态的另一条路径）。
"""

import numpy as np
import pytest

from autoflowcfd.boundary.fr_ghost_state import DefaultGhostProvider
from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.fr_residual.inviscid import compute_inviscid_residual_fr, primitive_to_conserved
from autoflowcfd.core.fr_residual.viscous_flux import compute_viscous_residual_fr
from autoflowcfd.core.fr_residual.viscous_flux.constants import PRANDTL
from autoflowcfd.core.mpi.distributed_compute import (
    distributed_compute_inviscid_residual,
    distributed_compute_viscous_residual,
)
from autoflowcfd.core.mpi.distributed_flat_face import build_distributed_flat_face
from autoflowcfd.core.mpi.partition import build_distributed_partition
from tests.unit.test_distributed_compute_residual import _FakeHaloExchange
from tests.unit.test_p0_mixed_split_faces import _mixed_mesh

_PART = np.array([0, 1], dtype=np.int32)      # 单元 0 为棱柱、1 为四面体
_P_REF = 1.0e5


def _state(mesh):
    rng = np.random.default_rng(11)
    Q = np.empty((mesh.n_cells, mesh.n_sps_per_cell, 5))
    Q[..., 0] = 1.2 * (1.0 + 0.05 * rng.standard_normal(Q.shape[:2]))
    Q[..., 1:4] = 30.0 * rng.standard_normal(Q.shape[:2] + (3,))
    Q[..., 4] = _P_REF * (1.0 + 0.02 * rng.standard_normal(Q.shape[:2]))
    return primitive_to_conserved(Q)


def _distributed(mesh, U, rank, compute, **kw):
    partition = build_distributed_partition(mesh.face_connectivity, _PART, rank=rank, n_ranks=2)
    dist_fc = build_distributed_flat_face(mesh, mesh.operators, partition, cell_partition=_PART)
    native = np.concatenate([partition.local_cells, partition.halo_cells])
    out = compute(U[partition.local_cells], partition, _FakeHaloExchange(U[native]), dist_fc, mesh, mesh.operators,
                  boundary_ghost_provider=DefaultGhostProvider(), **kw)
    return partition, out


@pytest.mark.parametrize("order", [0, 1])
def test_partition_cutting_a_mixed_face_matches_single_node(order):
    mesh = _mixed_mesh(order)
    assert int((get_flat_face_geometry(mesh, mesh.operators).mixed_nb_partner >= 0).sum()) == 1
    U = _state(mesh)
    inviscid = compute_inviscid_residual_fr(U, mesh, mesh.operators, mach_ref=0.2,
                                            boundary_ghost_provider=DefaultGhostProvider())
    viscous = compute_viscous_residual_fr(U, mesh, mesh.operators, 1.8e-5, PRANDTL,
                                          boundary_ghost_provider=DefaultGhostProvider())
    assert np.abs(inviscid).max() > 1.0                       # 非平凡流场
    for rank in (0, 1):
        if order >= 1:
            # P0 的分布式无粘残差走另一条路径（汇集成全局状态后调用单机函数，
            # `distributed_order_continuation/p0_residual.py`），不经过这个入口
            partition, r_inv = _distributed(mesh, U, rank, distributed_compute_inviscid_residual, mach_ref=0.2)
            np.testing.assert_allclose(r_inv, inviscid[partition.local_cells], rtol=1e-10,
                                       atol=1e-10 * np.abs(inviscid).max())
        partition, r_vis = _distributed(mesh, U, rank, distributed_compute_viscous_residual, mu=1.8e-5)
        np.testing.assert_allclose(r_vis, viscous[partition.local_cells], rtol=1e-10,
                                   atol=1e-10 * np.abs(viscous).max())
