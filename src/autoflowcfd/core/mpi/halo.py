"""
AutoFlowCFD V2.0 - Halo 层管理与数据交换

分区后每个 rank 持有 local cells + halo cells（1 层邻居 cell）。交换的是逐单元数据
`(n_cells, ...)`：平均流 `(n_sps, n_vars)`、k/omega `(n_sps, 2)`、DES 长度尺度
`(n_sps,)`、P0 湍流源项的速度梯度 `(n_sps, 3, 3)`、逐单元标量 `()` 都经同一个实例，
send/recv buffer 按逐单元形状缓存（此前按形状各建一个交换器、另有一份标量专用的
`exchange_scalar`，同一协议三份实现）。

交换协议（非阻塞异步）:
1. 每个 rank 将 send_lists[r] 中的 cell 数据打包到连续 buffer
2. MPI.Isend 异步发送给 rank r
3. MPI.Irecv 异步接收来自 rank r 的 halo cell 数据
4. MPI.Waitall 等待所有完成
5. 将接收到的数据填入扩展数组的 halo 位置

内存布局:
- 扩展数组: (n_local_cells + n_halo_cells, ...)
  - [0, n_local_cells): local cells 的数据
  - [n_local_cells, n_total): halo cells 的数据（从邻居 rank 接收）
"""

import numpy as np
from typing import Dict

from loguru import logger

from autoflowcfd.core.mpi import get_comm, get_mpi, mpi_available
from autoflowcfd.core.mpi.partition import DistributedPartition


class HaloExchange:
    """Halo 数据交换管理器：按逐单元形状缓存的 send/recv buffer + 非阻塞通信。

    Attributes:
        partition: 分区信息
        n_sps: 每单元解点数（构造时预分配主形状 `(n_sps, n_vars)` 用）
        n_vars: 平均流变量数
    """

    def __init__(self, partition: DistributedPartition, n_sps: int, n_vars: int):
        """
        Args:
            partition: 本 rank 的分区信息
            n_sps: 每单元解点数
            n_vars: 平均流变量数（欧拉方程 5，SST 7 等）；`(n_sps, n_vars)` 在构造时预分配，
                其它逐单元形状首次交换时分配、之后复用
        """
        self.partition = partition
        self.n_sps = n_sps
        self.n_vars = n_vars
        self._buffers_by_shape: Dict[tuple, tuple] = {}
        main_send, main_recv = self._buffers((n_sps, n_vars))
        logger.debug(
            f"Rank {partition.rank}: Halo exchange initialized - "
            f"{partition.n_halo} halo cells, {len(partition.neighbor_ranks)} neighbors, "
            f"send bufs: {sum(b.size for b in main_send.values()) * 8 / 1e6:.1f} MB, "
            f"recv bufs: {sum(b.size for b in main_recv.values()) * 8 / 1e6:.1f} MB"
        )

    def _buffers(self, cell_shape: tuple) -> tuple:
        """逐单元形状 `cell_shape` 的 `(send, recv)` buffer（按邻居 rank 的字典），首次使用时分配。"""
        bufs = self._buffers_by_shape.get(cell_shape)
        if bufs is None:
            part = self.partition
            bufs = ({r: np.empty((len(c),) + cell_shape) for r, c in part.send_lists.items()},
                    {r: np.empty((len(c),) + cell_shape) for r, c in part.recv_lists.items()})
            self._buffers_by_shape[cell_shape] = bufs
        return bufs

    def exchange(self, local_data: np.ndarray) -> np.ndarray:
        """执行一次 halo 交换。

        Args:
            local_data: `(n_local_cells, ...)` 本 rank 的 local cell 数据，逐单元形状任意

        Returns:
            extended_data: `(n_total_cells, ...)` 扩展数组
                [0:n_local_cells] = local_data 的拷贝
                [n_local_cells:n_total] = 从邻居接收的 halo 数据
        """
        part = self.partition
        n_local = part.n_local_cells
        cell_shape = tuple(local_data.shape[1:])

        extended_data = np.empty((part.n_total_cells,) + cell_shape, dtype=np.float64)
        extended_data[:n_local] = local_data

        if not mpi_available or not part.neighbor_ranks:
            return extended_data

        comm = get_comm()
        MPI = get_mpi()
        send_buffers, recv_buffers = self._buffers(cell_shape)

        # 1. 打包发送数据
        for r, local_indices in part.send_lists.items():
            send_buffers[r][:] = local_data[local_indices]

        # 2. 发起非阻塞接收
        recv_requests = []
        for r in part.neighbor_ranks:
            if r in recv_buffers:
                recv_requests.append(comm.Irecv(recv_buffers[r], source=r, tag=0))

        # 3. 发起非阻塞发送
        send_requests = []
        for r in part.neighbor_ranks:
            if r in send_buffers:
                send_requests.append(comm.Isend(send_buffers[r], dest=r, tag=0))

        # 4. 等待所有通信完成
        if recv_requests:
            MPI.Request.Waitall(recv_requests)
        if send_requests:
            MPI.Request.Waitall(send_requests)

        # 5. 将接收到的数据填入 halo 位置（全局编号 -> halo 数组中的局部偏移）
        for r, global_cells in part.recv_lists.items():
            if r in recv_buffers:
                for i, gc in enumerate(global_cells):
                    halo_idx = np.searchsorted(part.halo_cells, gc)
                    if halo_idx < len(part.halo_cells) and part.halo_cells[halo_idx] == gc:
                        extended_data[n_local + halo_idx] = recv_buffers[r][i]

        return extended_data
