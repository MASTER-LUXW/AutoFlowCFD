"""
AutoFlowCFD V2.0 - Halo 层管理与数据交换

分区后每个 rank 持有 local cells + halo cells（1 层邻居 cell）。交换的是逐单元数据
`(n_cells, ...)`：平均流 `(n_sps, 5)`、湍流变量 `(n_sps, n_turb)`、DES 长度尺度
`(n_sps,)`、P0 湍流源项的速度梯度 `(n_sps, 3, 3)`、逐单元标量 `()` 都经同一个实例，
send/recv buffer 按逐单元形状与 dtype 缓存（此前按形状各建一个交换器、另有一份标量专用的
`exchange_scalar`，同一协议三份实现）。

交换协议（非阻塞异步）:
1. 每个 rank 将 send_lists[r] 中的 cell 数据打包到连续 buffer
2. MPI.Isend 异步发送给 rank r
3. MPI.Irecv 异步接收来自 rank r 的 halo cell 数据
4. MPI.Waitall 等待所有完成
5. 将接收到的数据一次写入扩展数组的 halo 位置（位置在构造期由 `halo_recv_positions` 算好）

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


def halo_recv_positions(partition: DistributedPartition) -> Dict[int, np.ndarray]:
    """每个邻居 rank 的接收 buffer 第 i 行在扩展数组里的行号 `n_local + halo 偏移`（CPU `HaloExchange` 与 GPU
    `GPUHaloExchange` 共用）。

    `recv_lists[r]` 里的每个全局单元必须是本 rank 的 halo 单元：找不到说明分区数据不一致，此时那一行 halo
    永远收不到数据（扩展数组是未初始化的 `empty`），必须硬失败。2026-10-17 以前 CPU 在每次交换里对每个 halo
    单元做一次 Python 级 `searchsorted`（隐式步每次 J v 都交换），GPU 每个 halo 单元一次设备赋值（一次 kernel
    启动），找不到时两者都静默跳过（I-95 / I-122）。
    """
    halo = np.asarray(partition.halo_cells)
    n_local = int(partition.n_local_cells)
    out: Dict[int, np.ndarray] = {}
    for r, global_cells in partition.recv_lists.items():
        gc = np.asarray(global_cells, dtype=np.int64)
        idx = np.searchsorted(halo, gc)
        found = idx < len(halo)
        found[found] = halo[idx[found]] == gc[found]
        if not found.all():
            raise ValueError(
                f"rank {partition.rank}: 来自 rank {r} 的接收列表里有 {int((~found).sum())} 个单元不在本 rank 的 "
                f"halo_cells 中（例如全局单元 {int(gc[~found][0])}），分区数据不一致")
        out[r] = (n_local + idx).astype(np.int64)
    return out


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
            n_vars: 平均流变量数（5）；`(n_sps, n_vars)` 在构造时预分配，其它逐单元形状首次交换时分配、之后复用
        """
        self.partition = partition
        self.n_sps = n_sps
        self.n_vars = n_vars
        self._recv_rows = halo_recv_positions(partition)
        self._buffers_by_shape: Dict[tuple, tuple] = {}
        main_send, main_recv = self._buffers((n_sps, n_vars), np.dtype(np.float64))
        logger.debug(
            f"Rank {partition.rank}: Halo exchange initialized - "
            f"{partition.n_halo} halo cells, {len(partition.neighbor_ranks)} neighbors, "
            f"send bufs: {sum(b.size for b in main_send.values()) * 8 / 1e6:.1f} MB, "
            f"recv bufs: {sum(b.size for b in main_recv.values()) * 8 / 1e6:.1f} MB"
        )

    def _buffers(self, cell_shape: tuple, dtype: np.dtype) -> tuple:
        """逐单元形状 `cell_shape`、类型 `dtype` 的 `(send, recv)` buffer（按邻居 rank 的字典），首次使用时分配。"""
        key = (cell_shape, dtype)
        bufs = self._buffers_by_shape.get(key)
        if bufs is None:
            part = self.partition
            bufs = ({r: np.empty((len(c),) + cell_shape, dtype) for r, c in part.send_lists.items()},
                    {r: np.empty((len(c),) + cell_shape, dtype) for r, c in part.recv_lists.items()})
            self._buffers_by_shape[key] = bufs
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

        # 类型跟随输入（此前强制 float64，整型/布尔场被静默转换）
        extended_data = np.empty((part.n_total_cells,) + cell_shape, dtype=local_data.dtype)
        extended_data[:n_local] = local_data

        if not part.neighbor_ranks:
            return extended_data
        if not mpi_available():
            raise RuntimeError(f"rank {part.rank} 有 {len(part.neighbor_ranks)} 个邻居 rank，但 MPI 不可用："
                               f"halo 单元收不到数据")

        comm = get_comm()
        MPI = get_mpi()
        send_buffers, recv_buffers = self._buffers(cell_shape, extended_data.dtype)

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

        # 5. 将接收到的数据填入 halo 位置（行号构造期算好）
        for r, rows in self._recv_rows.items():
            extended_data[rows] = recv_buffers[r]

        return extended_data
