"""AutoFlowCFD V2.0 - 局部与全局逐单元数组之间的收集（写 checkpoint）与分发（读 checkpoint），都是点对点缓冲区通信。"""


from typing import Optional

import numpy as np


from autoflowcfd.core.mpi import get_comm, get_rank, get_size


def gather_global_state(
    U_local: np.ndarray,
    local_cells: np.ndarray,
    n_global_cells: int,
) -> Optional[np.ndarray]:
    """从所有 rank 收集 local cells 数据，在 root 组装全局状态。

    每个 rank 持有 local cells 的 (n_local, n_sps, n_vars) 数据，
    local_cells 给出这些 cell 的全局索引。Root rank 根据全局索引
    将各 rank 的数据放入全局数组的正确位置。

    Args:
        U_local: (n_local_cells, n_sps, n_vars) 本 rank 的 local cell 数据
        local_cells: (n_local_cells,) 本 rank 的 cell 全局索引
        n_global_cells: 全局 cell 总数

    Returns:
        U_global: (n_global_cells, n_sps, n_vars) 全局状态（仅 root rank 有值，
                  其他 rank 返回 None）
    """
    comm = get_comm()
    rank = get_rank()
    n_ranks = get_size()

    if n_ranks == 1:
        # 单 rank：直接按全局索引放置
        n_sps = U_local.shape[1]
        n_vars = U_local.shape[2]
        U_global = np.zeros((n_global_cells, n_sps, n_vars), dtype=np.float64)
        U_global[local_cells] = U_local
        return U_global

    # 各 rank 向 root 发送自己的 local cell 数据 + 全局索引
    if rank == 0:
        # Root: 初始化全局数组
        n_sps = U_local.shape[1]
        n_vars = U_local.shape[2]
        U_global = np.zeros((n_global_cells, n_sps, n_vars), dtype=np.float64)

        # 放入 root 自己的数据
        U_global[local_cells] = U_local

        # 接收其他 rank 的数据
        for r in range(1, n_ranks):
            # 先接收全局索引
            n_recv = np.empty(1, dtype=np.int64)
            comm.Recv(n_recv, source=r, tag=99)
            n_l = int(n_recv[0])
            idx_buf = np.empty(n_l, dtype=np.int64)
            comm.Recv(idx_buf, source=r, tag=100)

            # 接收数据
            shape_buf = np.empty(2, dtype=np.int64)
            comm.Recv(shape_buf, source=r, tag=101)
            n_s, n_v = int(shape_buf[0]), int(shape_buf[1])
            data_buf = np.empty((n_l, n_s, n_v), dtype=np.float64)
            comm.Recv(data_buf, source=r, tag=102)

            U_global[idx_buf] = data_buf
    else:
        # 非 root: 发送数据
        n_local = np.array([len(local_cells)], dtype=np.int64)
        comm.Send(n_local, dest=0, tag=99)
        comm.Send(local_cells.astype(np.int64), dest=0, tag=100)

        shape_buf = np.array([U_local.shape[1], U_local.shape[2]], dtype=np.int64)
        comm.Send(shape_buf, dest=0, tag=101)
        comm.Send(U_local, dest=0, tag=102)

        return None

    return U_global


def distribute_from_root(global_array: Optional[np.ndarray], local_cells: np.ndarray) -> np.ndarray:
    """root 上的全局逐单元数组 `(n_global, ...)` -> 本 rank 的 local 段 `(n_local, ...)`（集体调用，
    `gather_global_state` 的逆操作；非 root 的 `global_array` 不读，传 None）。

    root 收齐各 rank 的全局单元索引后逐个 rank 点对点发送它那一段（缓冲区接口）：各 rank 只分配自己的段，单条
    消息是一个 rank 的段而不是全局数组。2026-10-19 以前续算把全局数组用小写 `bcast`（pickle）发给每个 rank：每个
    rank 内存 O(全局)，单条消息超过 2 GiB 时多数 MPI 实现直接失败——千万单元级 P2（全局 U 约 10 GB）无法续算。
    """
    local_cells = np.ascontiguousarray(local_cells, dtype=np.int64)
    n_ranks = get_size()
    if n_ranks == 1:
        return np.ascontiguousarray(np.asarray(global_array, dtype=np.float64)[local_cells])
    comm = get_comm()
    rank = get_rank()
    tail = comm.bcast(tuple(np.asarray(global_array).shape[1:]) if rank == 0 else None, root=0)
    if rank == 0:
        g = np.asarray(global_array, dtype=np.float64)
        for r in range(1, n_ranks):
            n_recv = np.empty(1, dtype=np.int64)
            comm.Recv(n_recv, source=r, tag=110)
            idx = np.empty(int(n_recv[0]), dtype=np.int64)
            comm.Recv(idx, source=r, tag=111)
            comm.Send(np.ascontiguousarray(g[idx]), dest=r, tag=112)
        return np.ascontiguousarray(g[local_cells])
    comm.Send(np.array([local_cells.size], dtype=np.int64), dest=0, tag=110)
    comm.Send(local_cells, dest=0, tag=111)
    out = np.empty((local_cells.size,) + tuple(tail), dtype=np.float64)
    comm.Recv(out, source=0, tag=112)
    return out
