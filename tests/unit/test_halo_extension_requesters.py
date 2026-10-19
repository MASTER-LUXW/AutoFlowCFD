"""面通量点跨引用的 halo 扩展：面的 owner 与 neighbor 所在的两个 rank 都是请求方（I-97）。

两个 rank 都持有该面（owner 一侧与 halo_owner 一侧）、都在两侧坐标系里做插值；neighbor 坐标系的 owner_src
依赖由 neighbor 所在 rank 使用。此前只取 owner 所在 rank 为请求方：依赖单元不与对侧单元相邻时，neighbor 所在
rank 缺这个 halo 单元。
"""

import numpy as np

from autoflowcfd.core.mpi.partition import (
    build_distributed_partition,
    extend_halo_for_flux_point_cross_references,
    partition_mesh,
)
from tests.validation._channel_mesh import build_channel_mesh_prism


def test_both_ranks_holding_a_face_receive_its_extra_dependency():
    mesh = build_channel_mesh_prism(1, 6, 2, 2, 0.6, 0.1, 0.08)
    fc = mesh.face_connectivity
    n_ranks = 3
    part = partition_mesh(fc, n_ranks, n_cells=mesh.n_cells, method="block")
    nb = np.asarray(fc.neighbor_cell)
    owner = np.asarray(fc.owner_cell)
    # 一个跨 rank 的内部面（owner 在 a、neighbor 在 b），依赖一个第三个 rank 上的单元
    cross = np.flatnonzero((nb >= 0) & (part[owner] != part[np.maximum(nb, 0)]))
    f = int(cross[0])
    a, b = int(part[owner[f]]), int(part[nb[f]])
    c_rank = ({0, 1, 2} - {a, b}).pop()
    base = {r: build_distributed_partition(fc, part, r, n_ranks) for r in (a, b)}
    # 依赖单元取不在两侧基础 1-ring halo 里的（否则无论请求方取谁都已经在 halo 里）
    c = int(next(x for x in np.flatnonzero(part == c_rank)
                 if x not in base[a].halo_cells and x not in base[b].halo_cells))
    dep = np.full(fc.n_faces, -1, dtype=np.int64)
    dep[f] = c

    parts = {}
    for r in range(n_ranks):
        p = build_distributed_partition(fc, part, r, n_ranks)
        parts[r] = extend_halo_for_flux_point_cross_references(p, part, fc, [dep])
    for r in (a, b):
        assert c in parts[r].halo_cells, f"rank {r} 持有面 {f}，但缺它的依赖单元 {c}"
        assert parts[r].recv_lists[c_rank].tolist().count(c) == 1
        assert c in parts[c_rank].local_to_global[parts[c_rank].send_lists[r]]
