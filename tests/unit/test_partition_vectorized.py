"""分区构造向量化后与逐面参照实现逐位相同；METIS 缺失时报错、n_cells 必需（I-96）。

参照实现即此前的逐面 Python 循环（邻接表按面序号填充、halo 判定、面分类）。
"""

import numpy as np
import pytest

from autoflowcfd.core.mpi.partition import build_cell_adjacency_graph, build_distributed_partition, partition_mesh
from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh
from tests.validation._channel_mesh import build_channel_mesh_prism


def _reference_adjacency(owner, neighbor, is_boundary, n_cells):
    rows = [[] for _ in range(n_cells)]
    for f in range(len(owner)):
        if is_boundary[f] or neighbor[f] < 0:
            continue
        rows[owner[f]].append(neighbor[f])
        rows[neighbor[f]].append(owner[f])
    indptr = np.concatenate([[0], np.cumsum([len(r) for r in rows])])
    return indptr, np.array([c for r in rows for c in r], dtype=np.int64)


def _reference_partition(fc, part, rank):
    halo = {}
    for f in range(fc.n_faces):
        oc, nc = fc.owner_cell[f], fc.neighbor_cell[f]
        if fc.is_boundary[f] or nc < 0:
            continue
        if part[oc] == rank and part[nc] != rank:
            halo[int(nc)] = int(part[nc])
        if part[nc] == rank and part[oc] != rank:
            halo[int(oc)] = int(part[oc])
    halo_cells = sorted(halo)
    recv = {}
    for c in halo_cells:
        recv.setdefault(halo[c], []).append(c)
    classes = []
    for f in range(fc.n_faces):
        oc, nc = fc.owner_cell[f], fc.neighbor_cell[f]
        owner_local = part[oc] == rank
        neighbor_local = nc >= 0 and part[nc] == rank
        if not (owner_local or neighbor_local):
            continue
        if not owner_local:
            classes.append("halo_owner")
        elif fc.is_boundary[f]:
            classes.append("physical")
        elif neighbor_local:
            classes.append("interior")
        else:
            classes.append("partition")
    return halo_cells, recv, classes


def _meshes():
    return [build_channel_mesh_prism(1, 6, 3, 2, 0.6, 0.1, 0.08), _build_synthetic_mixed_mesh(1)]


@pytest.mark.parametrize("mesh", _meshes(), ids=["channel", "mixed"])
def test_adjacency_matches_the_per_face_reference(mesh):
    fc = mesh.face_connectivity
    got = build_cell_adjacency_graph(fc.owner_cell, fc.neighbor_cell, fc.is_boundary, mesh.n_cells)
    ref = _reference_adjacency(fc.owner_cell, fc.neighbor_cell, fc.is_boundary, mesh.n_cells)
    for g, r in zip(got, ref):
        np.testing.assert_array_equal(g, r)


@pytest.mark.parametrize("mesh", _meshes(), ids=["channel", "mixed"])
@pytest.mark.parametrize("n_ranks", [2, 3])
def test_partition_structure_matches_the_per_face_reference(mesh, n_ranks):
    fc = mesh.face_connectivity
    part = partition_mesh(fc, n_ranks, n_cells=mesh.n_cells, method="block")
    for rank in range(n_ranks):
        p = build_distributed_partition(fc, part, rank, n_ranks)
        halo_cells, recv, classes = _reference_partition(fc, part, rank)
        np.testing.assert_array_equal(p.halo_cells, halo_cells)
        np.testing.assert_array_equal(p.halo_owners, [part[c] for c in halo_cells])
        assert list(p.recv_lists) == list(recv)
        for r, cells in recv.items():
            np.testing.assert_array_equal(p.recv_lists[r], cells)
        fcls = p.face_classification
        names = np.full(len(p.local_faces), "", dtype=object)
        names[fcls.halo_owner_mask] = "halo_owner"
        names[fcls.physical_boundary_mask] = "physical"
        names[fcls.interior_mask] = "interior"
        names[fcls.partition_boundary_mask] = "partition"
        assert list(names) == classes


def test_metis_is_required_for_more_than_one_part():
    try:
        import pymetis  # noqa: F401
        pytest.skip("本机装了 pymetis")
    except ImportError:
        pass
    mesh = build_channel_mesh_prism(1, 4, 2, 2, 0.4, 0.1, 0.08)
    with pytest.raises(ImportError, match="pymetis"):
        partition_mesh(mesh.face_connectivity, 2, n_cells=mesh.n_cells)
    # 单分区不需要分区器
    assert not partition_mesh(mesh.face_connectivity, 1, n_cells=mesh.n_cells).any()


def test_cell_count_is_required():
    mesh = build_channel_mesh_prism(1, 4, 2, 2, 0.4, 0.1, 0.08)
    with pytest.raises(TypeError):
        partition_mesh(mesh.face_connectivity, 2)
