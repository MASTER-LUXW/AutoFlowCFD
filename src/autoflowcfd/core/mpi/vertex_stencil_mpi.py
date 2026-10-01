"""AutoFlowCFD V2.0 - 分布式顶点邻域模板（BJ 越界判据用，CPU MPI 与多 GPU 共用）。

单机顶点模板与它为什么必需见 `core/fr_operators/vertex_stencil.py`。分布式的
难点是：共享同一顶点的两个单元可能相隔 2 个以上面跳，1 层面邻居的 halo 拼不出
完整顶点邻域。这里不扩 halo，而是：

1. 每个 rank 只用**自己的 local 单元**建 `(顶点, 单元)` 对（顶点用全局编号，
   单元用 halo 交换的原生 local 下标）；
2. 构造时一次 `allgather` 找出"被不止一个 rank 的单元用到"的共享顶点（全部
   rank 得到同一份有序列表）；
3. 每次求值在逐顶点归约之后，对共享顶点做一次 MAX 与一次 MIN 全局归约
   （`VertexStencil.reduce_nodes`），再散射回单元。

max/min 与求值顺序无关，所以 local 单元得到的包络与单机**逐位相同**，与分区数
无关（halo 行不在模板里，它们的判据结果本来就丢弃）。通信量是共享顶点数
（O(n^(2/3))）× 2，远小于一次 halo 交换。

集合通信函数可注入（默认 `core/mpi/comm.py`），测试用线程模拟多 rank。
"""

from typing import Callable, Optional, Tuple

import numpy as np

from autoflowcfd.core.fr_operators.vertex_stencil import VertexStencil


def vertex_pairs_of_cells(mesh, global_cells: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """给定单元（全局下标，"棱柱在前"排列）的 `(顶点全局编号, 在 global_cells 中的位置)` 对。

    root（完全分布式加载）与传统模式各 rank（持有全局网格）都用它，所以
    两种加载方式得到同一份模板。
    """
    prism_conn = getattr(mesh, "_fixed_prism_conn", None)
    tet_conn = getattr(mesh, "_fixed_tet_conn", None)
    n_prism = int(getattr(mesh, "n_prism_cells", 0) or 0)
    if prism_conn is None and tet_conn is None:
        raise RuntimeError(
            "分布式 BJ 顶点模板需要网格的单元-顶点连接（_fixed_prism_conn/"
            "_fixed_tet_conn），当前网格没有")
    g = np.asarray(global_cells, dtype=np.int64)
    nodes, pos = [], []
    for sel, conn, offset in ((g < n_prism, prism_conn, 0),
                              (g >= n_prism, tet_conn, n_prism)):
        idx = np.flatnonzero(sel)
        if idx.size == 0:
            continue
        rows = np.asarray(conn, dtype=np.int64)[g[idx] - offset]
        nodes.append(rows.ravel())
        pos.append(np.repeat(idx, rows.shape[1]))
    return (np.ascontiguousarray(np.concatenate(nodes)),
            np.ascontiguousarray(np.concatenate(pos)))


def build_distributed_vertex_stencil(
    node_global: np.ndarray, cell_native: np.ndarray, *,
    to_device: Optional[Callable] = None,
    allgather: Optional[Callable] = None,
    allreduce_max: Optional[Callable] = None,
    allreduce_min: Optional[Callable] = None,
    n_ranks: Optional[int] = None,
) -> VertexStencil:
    """由本 rank local 单元的 `(顶点全局编号, 原生 local 下标)` 对建模板。

    **集合调用**：全部 rank 必须同时调用（内部有一次 allgather）；返回模板的
    `reduce_nodes` 同样是集合调用。

    Args:
        node_global, cell_native: `vertex_pairs_of_cells` 的结果（位置即原生
            local 下标，因为原生 local 排列就是 `partition.local_cells` 的顺序）
        to_device: 把 numpy 数组搬到计算设备（多 GPU）；默认留在主机
        allgather, allreduce_max, allreduce_min, n_ranks: 集合通信（默认
            `core/mpi/comm.py` 与当前通信子大小）
    """
    from autoflowcfd.core.mpi import comm as _comm

    allgather = allgather or _comm.allgather_array
    allreduce_max = allreduce_max or _comm.allreduce_max
    allreduce_min = allreduce_min or _comm.allreduce_min
    if n_ranks is None:
        from autoflowcfd.core.mpi import get_comm, mpi_available
        n_ranks = get_comm().Get_size() if mpi_available else 1
    to_device = to_device or (lambda a: a)

    nodes_u, node_local = np.unique(np.asarray(node_global, dtype=np.int64),
                                    return_inverse=True)
    reduce_nodes = None
    if n_ranks > 1:
        vals, counts = np.unique(allgather(nodes_u), return_counts=True)
        shared = vals[counts > 1]                     # 全部 rank 相同、有序
        mine = np.flatnonzero(np.isin(nodes_u, shared))
        pos = np.searchsorted(shared, nodes_u[mine])
        mine_dev = to_device(mine)
        n_shared = int(shared.size)

        def reduce_nodes(node_max, node_min):
            for arr, fill, reduce in ((node_max, -np.inf, allreduce_max),
                                      (node_min, np.inf, allreduce_min)):
                buf = np.full(n_shared, fill)
                buf[pos] = _host(arr[mine_dev])
                arr[mine_dev] = to_device(reduce(buf)[pos])

    return VertexStencil(
        node_of_pair=to_device(np.ascontiguousarray(node_local.astype(np.int64))),
        cell_of_pair=to_device(np.ascontiguousarray(np.asarray(cell_native, dtype=np.int64))),
        n_nodes=int(nodes_u.size),
        reduce_nodes=reduce_nodes)


def _host(a) -> np.ndarray:
    get = getattr(a, "get", None)
    return np.asarray(get() if get is not None and not isinstance(a, np.ndarray) else a)


def local_vertex_pairs(mesh, partition) -> Tuple[np.ndarray, np.ndarray]:
    """本 rank local 单元的顶点对：完全分布式加载随包下发的那份，否则由全局网格现算。"""
    pairs = getattr(mesh, "local_vertex_pairs", None)
    if pairs is not None:
        return pairs
    return vertex_pairs_of_cells(mesh, partition.local_cells)
