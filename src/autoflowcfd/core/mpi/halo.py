"""
AutoFlowCFD V2.0 - Halo 层管理与数据交换

FR 求解中需要交换的是 SPs 上的场值 (n_cells, n_sps, n_vars)。
分区后每个 rank 持有 local cells + halo cells（1 层邻居 cell）。

交换协议（非阻塞异步）:
1. 每个 rank 将 send_lists[r] 中的 cell SPs 值打包到连续 buffer
2. MPI.Isend 异步发送给 rank r
3. MPI.Irecv 异步接收来自 rank r 的 halo cell SPs 值
4. MPI.Waitall 等待所有完成
5. 将接收到的数据填入扩展数组的 halo 位置

内存布局:
- 扩展数组: (n_local_cells + n_halo_cells, n_sps, n_vars)
  - [0, n_local_cells): local cells 的数据
  - [n_local_cells, n_total): halo cells 的数据（从邻居 rank 接收）

优化:
- 预分配固定大小 send/recv buffer（避免每次交换重新分配）
- 非阻塞通信 Isend/Irecv（后续可与体积项计算重叠）
- 连续内存布局（按 send_list 索引打包）
"""

import numpy as np
from typing import Dict, Optional

from loguru import logger

from autoflowcfd.core.mpi import get_comm, get_mpi, mpi_available
from autoflowcfd.core.mpi.partition import DistributedPartition


class HaloExchange:
    """Halo 数据交换管理器。

    管理预分配的 send/recv buffer 和非阻塞通信请求。

    Attributes:
        partition: 分区信息
        n_sps: 每单元解点数
        n_vars: 变量数
        send_buffers: dict[rank] → 预分配的发送 buffer
        recv_buffers: dict[rank] → 预分配的接收 buffer
    """

    def __init__(self, partition: DistributedPartition, n_sps: int, n_vars: int):
        """初始化 halo 交换管理器。

        Args:
            partition: 本 rank 的分区信息
            n_sps: 每单元解点数
            n_vars: 变量数（欧拉方程 5，SST 7 等）
        """
        self.partition = partition
        self.n_sps = n_sps
        self.n_vars = n_vars

        # 预分配 send/recv buffer
        self.send_buffers: Dict[int, np.ndarray] = {}
        self.recv_buffers: Dict[int, np.ndarray] = {}

        for r, cells in partition.send_lists.items():
            self.send_buffers[r] = np.empty(
                (len(cells), n_sps, n_vars), dtype=np.float64
            )
        for r, cells in partition.recv_lists.items():
            self.recv_buffers[r] = np.empty(
                (len(cells), n_sps, n_vars), dtype=np.float64
            )

        logger.debug(
            f"Rank {partition.rank}: Halo exchange initialized - "
            f"{partition.n_halo} halo cells, {len(partition.neighbor_ranks)} neighbors, "
            f"send bufs: {sum(b.size for b in self.send_buffers.values()) * 8 / 1e6:.1f} MB, "
            f"recv bufs: {sum(b.size for b in self.recv_buffers.values()) * 8 / 1e6:.1f} MB"
        )

    def exchange(self, local_data: np.ndarray) -> np.ndarray:
        """执行一次 halo 交换。

        将 local_data 中 send_lists 指定的 cell 数据发送给邻居 rank，
        接收 halo cell 数据并返回扩展数组。

        Args:
            local_data: (n_local_cells, n_sps, n_vars) 本 rank 的 local cell 数据

        Returns:
            extended_data: (n_total_cells, n_sps, n_vars) 扩展数组
                [0:n_local_cells] = local_data 的拷贝
                [n_local_cells:n_total] = 从邻居接收的 halo 数据
        """
        part = self.partition
        n_local = part.n_local_cells
        n_total = part.n_total_cells

        # 构建扩展数组
        extended_data = np.empty((n_total, self.n_sps, self.n_vars), dtype=np.float64)
        extended_data[:n_local] = local_data

        if not mpi_available or not part.neighbor_ranks:
            return extended_data

        comm = get_comm()
        MPI = get_mpi()

        # 1. 打包发送数据
        for r, local_indices in part.send_lists.items():
            self.send_buffers[r][:] = local_data[local_indices]

        # 2. 发起非阻塞接收
        recv_requests = []
        for r in part.neighbor_ranks:
            if r in self.recv_buffers:
                req = comm.Irecv(self.recv_buffers[r], source=r, tag=0)
                recv_requests.append(req)

        # 3. 发起非阻塞发送
        send_requests = []
        for r in part.neighbor_ranks:
            if r in self.send_buffers:
                req = comm.Isend(self.send_buffers[r], dest=r, tag=0)
                send_requests.append(req)

        # 4. 等待所有通信完成
        if recv_requests:
            MPI.Request.Waitall(recv_requests)
        if send_requests:
            MPI.Request.Waitall(send_requests)

        # 5. 将接收到的数据填入 halo 位置
        for r, global_cells in part.recv_lists.items():
            if r in self.recv_buffers:
                # 将全局索引转为 halo 数组中的局部偏移
                halo_offset = part.n_local_cells
                for i, gc in enumerate(global_cells):
                    # 在 halo_cells 中找到 gc 的位置
                    halo_idx = np.searchsorted(part.halo_cells, gc)
                    if halo_idx < len(part.halo_cells) and part.halo_cells[halo_idx] == gc:
                        extended_data[halo_offset + halo_idx] = self.recv_buffers[r][i]

        return extended_data

    def exchange_scalar(self, local_scalar: np.ndarray) -> np.ndarray:
        """执行标量场的 halo 交换（单变量，如 k 或 omega）。

        Args:
            local_scalar: (n_local_cells, n_sps) 本 rank 的 local cell 标量数据

        Returns:
            extended_scalar: (n_total_cells, n_sps) 扩展数组
        """
        part = self.partition
        n_local = part.n_local_cells
        n_total = part.n_total_cells

        extended_scalar = np.empty((n_total, self.n_sps), dtype=np.float64)
        extended_scalar[:n_local] = local_scalar

        if not mpi_available or not part.neighbor_ranks:
            return extended_scalar

        comm = get_comm()
        MPI = get_mpi()

        # 打包/解包标量数据（复用 float64 buffer，按 n_sps 大小切片）
        send_bufs = {}
        recv_bufs = {}
        for r, indices in part.send_lists.items():
            send_bufs[r] = local_scalar[indices]  # (n_send, n_sps)
        for r, cells in part.recv_lists.items():
            recv_bufs[r] = np.empty((len(cells), self.n_sps), dtype=np.float64)

        # 非阻塞通信
        recv_requests = []
        for r in part.neighbor_ranks:
            if r in recv_bufs:
                req = comm.Irecv(recv_bufs[r], source=r, tag=1)
                recv_requests.append(req)

        send_requests = []
        for r in part.neighbor_ranks:
            if r in send_bufs:
                req = comm.Isend(np.ascontiguousarray(send_bufs[r]), dest=r, tag=1)
                send_requests.append(req)

        if recv_requests:
            MPI.Request.Waitall(recv_requests)
        if send_requests:
            MPI.Request.Waitall(send_requests)

        # 填入 halo 位置
        for r, global_cells in part.recv_lists.items():
            if r in recv_bufs:
                halo_offset = part.n_local_cells
                for i, gc in enumerate(global_cells):
                    halo_idx = np.searchsorted(part.halo_cells, gc)
                    if halo_idx < len(part.halo_cells) and part.halo_cells[halo_idx] == gc:
                        extended_scalar[halo_offset + halo_idx] = recv_bufs[r][i]

        return extended_scalar
