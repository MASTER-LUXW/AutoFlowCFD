"""
AutoFlowCFD V2.0 - MPI 通信封装

封装常用 MPI 集合通信和点对点通信操作，提供统一的接口供分布式求解器使用。
非 MPI 环境下降级为本地操作（no-op），保证同一套代码在单进程和多进程下都能运行。

核心功能:
- allreduce_sum/max/min: 全局归约
- allgather_array: 收集各 rank 的数组
- barrier: 同步屏障
- Timer: 通信计时与性能统计
"""

import numpy as np
from typing import Optional

from autoflowcfd.core.mpi import get_comm, get_mpi, mpi_available


def allreduce_sum(local_value):
    """全局求和归约。

    Args:
        local_value: 标量或 numpy 数组（本 rank 的局部值）

    Returns:
        全局求和结果（所有 rank 得到相同值）
    """
    if not mpi_available:
        return local_value
    comm = get_comm()
    if comm.Get_size() == 1:
        return local_value
    if isinstance(local_value, np.ndarray):
        result = np.empty_like(local_value)
        comm.Allreduce(local_value, result, op=get_mpi().SUM)
        return result
    else:
        return comm.allreduce(local_value, op=get_mpi().SUM)


def allreduce_max(local_value):
    """全局最大值归约。"""
    if not mpi_available:
        return local_value
    comm = get_comm()
    if comm.Get_size() == 1:
        return local_value
    if isinstance(local_value, np.ndarray):
        result = np.empty_like(local_value)
        comm.Allreduce(local_value, result, op=get_mpi().MAX)
        return result
    else:
        return comm.allreduce(local_value, op=get_mpi().MAX)


def allreduce_min(local_value):
    """全局最小值归约。"""
    if not mpi_available:
        return local_value
    comm = get_comm()
    if comm.Get_size() == 1:
        return local_value
    if isinstance(local_value, np.ndarray):
        result = np.empty_like(local_value)
        comm.Allreduce(local_value, result, op=get_mpi().MIN)
        return result
    else:
        return comm.allreduce(local_value, op=get_mpi().MIN)


def allgather_array(local_array: np.ndarray) -> np.ndarray:
    """收集各 rank 的一维数组，拼接为全局数组。

    各 rank 的 local_array 长度可以不同（变长收集）。

    Args:
        local_array: (n_local,) 本 rank 的局部数据

    Returns:
        global_array: (n_global,) 所有 rank 数据拼接
    """
    if not mpi_available:
        return local_array.copy()
    comm = get_comm()
    if comm.Get_size() == 1:
        return local_array.copy()

    # 收集各 rank 的数组长度
    local_size = np.array([local_array.shape[0]], dtype=np.int64)
    all_sizes = np.empty(comm.Get_size(), dtype=np.int64)
    comm.Allgather(local_size, all_sizes)

    # 变长收集
    total_size = int(np.sum(all_sizes))
    global_array = np.empty(total_size, dtype=local_array.dtype)
    # 使用 Allgatherv 处理变长
    displs = np.zeros(comm.Get_size(), dtype=np.int64)
    displs[1:] = np.cumsum(all_sizes[:-1])
    comm.Allgatherv(
        local_array,
        [global_array, all_sizes, displs, get_mpi().DOUBLE]
        if local_array.dtype == np.float64 else
        [global_array, all_sizes, displs, get_mpi().INT64]
        if local_array.dtype == np.int64 else
        [global_array, all_sizes, displs, get_mpi().FLOAT]
    )
    return global_array


def barrier():
    """MPI 同步屏障。所有 rank 到达后才继续。"""
    if not mpi_available:
        return
    comm = get_comm()
    if comm.Get_size() > 1:
        comm.Barrier()


def bcast_from_root(obj):
    """从 root (rank 0) 广播对象到所有 rank。

    支持 numpy 数组和 pickle 可序列化对象。
    """
    if not mpi_available:
        return obj
    comm = get_comm()
    if comm.Get_size() == 1:
        return obj
    return comm.bcast(obj, root=0)


class MPITimer:
    """MPI 感知的计时器。

    使用 MPI.Wtime() 提供高精度计时，支持按 rank 记录并归约统计。
    """

    def __init__(self, name: str = ""):
        self.name = name
        self._start = None
        self._elapsed = 0.0

    def start(self):
        if mpi_available:
            self._start = get_mpi().Wtime()
        else:
            import time
            self._start = time.perf_counter()

    def stop(self) -> float:
        if mpi_available:
            self._elapsed = get_mpi().Wtime() - self._start
        else:
            import time
            self._elapsed = time.perf_counter() - self._start
        return self._elapsed

    @property
    def elapsed(self) -> float:
        return self._elapsed

    def allreduce_max_elapsed(self) -> float:
        """所有 rank 中最大耗时（用于性能报告——取最慢 rank）。"""
        return allreduce_max(self._elapsed)
