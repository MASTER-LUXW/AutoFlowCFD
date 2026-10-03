"""分布式全局粗校正的后端适配（`core/mpi/distributed_coarse.py`、`distributed_coupling_graph`）。

本机没有 MPI，halo 交换器用假的（local 段原样、halo 段给定值），判据是映射的约定本身：

1. 逐单元值 -> 紧凑空间：同一个 `CompactCellValues` 在 numpy（CPU-MPI）与 cupy（多 GPU，这里
   用 numpy 冒充）两种数组模块下给出同一个结果 `[local; halo][perm]`；
2. P0 差分耦合图的跨 rank 单元对：一端本 rank、另一端 halo 的每个面给出一对（行为 local
   编号、列为紧凑编号），halo 列的颜色与单元类型取自紧凑空间。
"""

import types

import numpy as np

from autoflowcfd.core.mpi.distributed_coarse import CompactCellValues
from autoflowcfd.core.mpi.distributed_implicit import distributed_coupling_graph



class _FakeHalo:
    """local 段原样，halo 段取给定的逐单元值（真实交换器接受任意逐单元形状，这里是标量）。"""

    def __init__(self, halo_values):
        self.halo = np.asarray(halo_values, dtype=np.float64)

    def exchange(self, local):
        return np.concatenate([local, self.halo])


class _NumpyAsCupy:
    def __getattr__(self, name):
        return getattr(np, name)

    def asnumpy(self, x):
        return np.asarray(x)


def test_cpu_and_gpu_compact_values_agree_with_definition():
    local = np.array([10.0, 11.0, 12.0, 13.0])
    halo = _FakeHalo([20.0, 21.0])
    perm = np.array([4, 0, 2, 5, 1, 3])
    expected = np.concatenate([local, halo.halo])[perm]
    np.testing.assert_array_equal(CompactCellValues(halo, perm, np)(local), expected)
    np.testing.assert_array_equal(CompactCellValues(halo, perm, _NumpyAsCupy())(local), expected)


def test_coupling_graph_lists_cross_rank_pairs_with_halo_colors():
    # 紧凑空间 6 个单元（"棱柱在前"：0、1 是棱柱），本 rank 的 local 单元是紧凑 1、2、4，
    # 其余（0、3、5）是 halo。面：(1,2) 本地，(2,3)、(0,4) 跨 rank，(4,5) 跨 rank，(3,5) 两端都是 halo
    owner = np.array([1, 2, 0, 4, 3, 1])
    neigh = np.array([2, 3, 4, 5, 5, -1])
    inv_perm = np.array([1, 2, 4, 0, 3, 5])           # local 原生单元 0,1,2 -> 紧凑 1,2,4
    solver = types.SimpleNamespace(
        partition=types.SimpleNamespace(n_local_cells=3),
        dist_flat_face=types.SimpleNamespace(inv_perm=inv_perm, base_flat=types.SimpleNamespace(
            owner_cell=owner, neighbor_cell=neigh, n_prism=2)),
        _coupling_colors_local=np.array([0, 1, 2]))
    compact_colors = np.array([7.0, 0.0, 1.0, 8.0, 2.0, 9.0])   # halo 0/3/5 的颜色 7/8/9
    ctx = types.SimpleNamespace(compact_cell_values=lambda v: compact_colors)
    g = distributed_coupling_graph(solver, ctx)
    # 本地单元对：(1,2) 面 -> local (0,1)、(1,0)
    assert sorted(zip(g.rows.tolist(), g.cols.tolist())) == [(0, 1), (1, 0)]
    # 跨 rank：(2,3) -> local 1 与紧凑 3；(0,4) -> local 2 与紧凑 0；(4,5) -> local 2 与紧凑 5
    pairs = sorted(zip(g.halo_rows.tolist(), g.halo_cols.tolist()))
    assert pairs == [(1, 3), (2, 0), (2, 5)]
    order = np.lexsort((g.halo_cols, g.halo_rows))
    assert g.halo_colors[order].tolist() == [8, 7, 9]
    assert g.halo_col_is_prism[order].tolist() == [False, True, False]
