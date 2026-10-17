"""
AutoFlowCFD V2.0 - GPU 直接 Halo 交换

优化 halo 交换性能：避免 GPU→CPU→MPI→CPU→GPU 的完整往返。

两种模式（与 CPU `core/mpi/halo.py::HaloExchange` 同一协议、同一组接收位置 `halo_recv_positions`）：
1. CUDA-aware MPI：mpi4py 直接收发 cupy 数组（`__cuda_array_interface__`）。打包在当前 CUDA 流上异步执行，
   而 MPI 不等待 CUDA 流，所以发送前必须同步当前流。
2. 非 CUDA-aware MPI（staging）：打包后整块拷到主机 buffer，MPI 收发主机 buffer，再整块拷回设备。

两种模式都只传 send/recv 列表里的单元；buffer 按逐单元形状与 dtype 缓存复用；接收数据一次花式索引写入
扩展数组的 halo 行（行号在构造期算好、驻留设备）。

使用:
    from autoflowcfd.core.gpu.distributed.gpu_halo_exchange import GPUHaloExchange
    gpu_halo = GPUHaloExchange(partition, n_sps, n_vars, device_id=0)
    U_extended_gpu = gpu_halo.exchange(U_gpu)
"""

import numpy as np
from typing import Dict, Optional
from loguru import logger

from autoflowcfd.core.gpu import gpu_available, get_cupy
from autoflowcfd.core.mpi import get_comm, get_mpi, mpi_available
from autoflowcfd.core.mpi.halo import halo_recv_positions
from autoflowcfd.core.mpi.partition import DistributedPartition


def _check_cuda_aware_mpi() -> bool:
    """检测 MPI 实现是否支持 CUDA-aware 通信。

    CUDA-aware MPI 可以直接发送/接收 GPU buffer，无需经过 CPU 中转。
    检测方法：只让 rank 0/1 之间做一次真实的小规模 GPU buffer 通信测试，
    再把结果广播给所有 rank，保证全体 rank 得到同一个结论。

    第四次评审修复：此前 rank>=2 直接 `return True`，完全不参与检测、
    也不知道 rank 0/1 真实测出的结果——如果真实环境不支持 CUDA-aware
    MPI（rank 0/1 测出 False），rank>=2 仍会走 `_exchange_cuda_aware`
    （GPU 指针 Isend/Irecv），而 rank 0/1 走 `_exchange_staging`（CPU
    numpy buffer）——同一个通信域里两种消息格式不匹配，会挂起或产生
    垃圾数据。改为只由 rank 0（与 rank 1 配对测试，size==1 时跳过测试
    直接判 True）得出真实结论，再用 Allreduce（逻辑与）确保所有 rank
    看到完全一致的结果，不允许任何 rank 各自猜测。
    """
    if not mpi_available or not gpu_available:
        return False

    cp = get_cupy()
    comm = get_comm()
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size == 1:
        return True  # 单 rank 不需要实际通信

    # 只有 rank 0/1 参与真实测试，结果只在 rank 0 上确定
    local_result = False
    if rank == 0:
        try:
            test_gpu = cp.ones(10, dtype=cp.float64)
            req = comm.Isend(test_gpu, dest=1, tag=99)
            req.Wait()
            local_result = True
        except Exception:
            local_result = False
    elif rank == 1:
        try:
            recv_buf = cp.zeros(10, dtype=cp.float64)
            req = comm.Irecv(recv_buf, source=0, tag=99)
            req.Wait()
            local_result = True
        except Exception:
            local_result = False
    # rank>=2 不参与测试，local_result 保持 False——下面用 MAX（逻辑或）
    # 而不是 AND：真正的检测结果只由 rank 0/1 产生，rank>=2 的 False
    # 不应该拖累这个结果，只需要把 rank 0/1 的结论传播给它们。
    from mpi4py import MPI as _MPI
    global_result = comm.allreduce(local_result, op=_MPI.MAX)
    return bool(global_result)


# 全局缓存检测结果
_cuda_aware_mpi_cache: Optional[bool] = None


def is_cuda_aware_mpi() -> bool:
    """返回 MPI 是否支持 CUDA buffer 直接通信（带缓存）。"""
    global _cuda_aware_mpi_cache
    if _cuda_aware_mpi_cache is None:
        _cuda_aware_mpi_cache = _check_cuda_aware_mpi()
        if _cuda_aware_mpi_cache:
            logger.info("CUDA-aware MPI detected: GPU halo exchange enabled")
        else:
            logger.info("CUDA-aware MPI not available: using staging buffer halo exchange")
    return _cuda_aware_mpi_cache


class GPUHaloExchange:
    """GPU 直接 Halo 交换管理器。

    支持两种模式：
    1. CUDA-aware MPI：直接发送/接收 GPU buffer（零拷贝）
    2. Staging buffer：GPU→pinned CPU buffer→MPI→pinned CPU buffer→GPU

    Attributes:
        partition: 分区信息
        n_sps: 每单元解点数
        n_vars: 变量数
        device_id: GPU 设备 ID
        cuda_aware: 是否使用 CUDA-aware MPI
    """

    def __init__(
        self,
        partition: DistributedPartition,
        n_sps: int,
        n_vars: int,
        device_id: int = 0,
    ):
        """初始化 GPU halo 交换。

        Args:
            partition: 本 rank 的分区信息
            n_sps: 每单元解点数
            n_vars: 变量数
            device_id: GPU 设备 ID
        """
        if not gpu_available:
            raise RuntimeError("CuPy required for GPU halo exchange")
        if not mpi_available:
            raise RuntimeError("MPI required for halo exchange")

        cp = get_cupy()
        self.partition = partition
        self.n_sps = n_sps
        self.n_vars = n_vars
        self.device_id = device_id
        self.cuda_aware = is_cuda_aware_mpi()

        # 按逐单元形状与 dtype 缓存的 buffer（GPU send/recv；非 CUDA-aware MPI 时另有主机 staging
        # send/recv），与 CPU `mpi/halo.py::HaloExchange` 同一约定：构造时预分配平均流主形状
        # `(n_sps, n_vars)`，湍流变量、DES 长度尺度、P0 速度梯度、逐单元标量等其它形状首次
        # 交换时分配、之后复用
        self._buffers_by_shape: Dict[tuple, tuple] = {}
        self._buffers((n_sps, n_vars), np.dtype(np.float64))

        # 接收 buffer 各行在扩展数组里的行号（与 CPU 同一个函数）。2026-10-17 以前按 Python 列表对每个 halo
        # 单元单独做一次设备赋值（每个 halo 单元一次 kernel 启动），找不到的单元静默跳过（I-122）
        with cp.cuda.Device(device_id):
            self._recv_rows = {r: cp.asarray(rows) for r, rows in halo_recv_positions(partition).items()}
            self._send_rows = {r: cp.asarray(np.asarray(c, dtype=np.int64))
                               for r, c in partition.send_lists.items()}

        logger.debug(
            f"GPU Halo exchange initialized: rank {partition.rank}, "
            f"{partition.n_halo} halo cells, cuda_aware={self.cuda_aware}"
        )

    def _buffers(self, cell_shape: tuple, dtype: np.dtype) -> tuple:
        """逐单元形状 `cell_shape`、类型 `dtype` 的 `(send_gpu, recv_gpu, send_cpu, recv_cpu)`（按邻居 rank
        的字典；CUDA-aware MPI 下主机 staging 两个为空，staging 模式下 recv_gpu 为空），首次使用时分配。"""
        key = (cell_shape, dtype)
        bufs = self._buffers_by_shape.get(key)
        if bufs is None:
            cp = get_cupy()
            part = self.partition
            with cp.cuda.Device(self.device_id):
                send_gpu = {r: cp.empty((len(c),) + cell_shape, dtype) for r, c in part.send_lists.items()}
                recv_gpu = ({r: cp.empty((len(c),) + cell_shape, dtype) for r, c in part.recv_lists.items()}
                            if self.cuda_aware else {})
            send_cpu, recv_cpu = {}, {}
            if not self.cuda_aware:
                send_cpu = {r: np.empty((len(c),) + cell_shape, dtype) for r, c in part.send_lists.items()}
                recv_cpu = {r: np.empty((len(c),) + cell_shape, dtype) for r, c in part.recv_lists.items()}
            bufs = (send_gpu, recv_gpu, send_cpu, recv_cpu)
            self._buffers_by_shape[key] = bufs
        return bufs

    def exchange(self, U_gpu):
        """执行 GPU halo 交换。

        Args:
            U_gpu: CuPy 数组 `(n_local_cells, ...)` 本 rank 的 local cell 数据，逐单元形状任意

        Returns:
            extended_data: CuPy 数组 `(n_total_cells, ...)`
        """
        cp = get_cupy()
        part = self.partition
        n_local = part.n_local_cells
        n_total = part.n_total_cells
        cell_shape = tuple(U_gpu.shape[1:])

        with cp.cuda.Device(self.device_id):
            # 类型跟随输入（此前强制 float64）
            extended = cp.empty((n_total,) + cell_shape, dtype=U_gpu.dtype)
            extended[:n_local] = U_gpu

            if not part.neighbor_ranks:
                return extended

            send_gpu, recv_gpu, send_cpu, recv_cpu = self._buffers(cell_shape, np.dtype(U_gpu.dtype))
            for r, rows in self._send_rows.items():
                cp.take(U_gpu, rows, axis=0, out=send_gpu[r])
            # 打包在当前流上异步执行：MPI（CUDA-aware 直接读设备 buffer）与 D2H 之前都必须等它完成
            cp.cuda.get_current_stream().synchronize()
            if self.cuda_aware:
                self._sendrecv(send_gpu, recv_gpu)
                for r, rows in self._recv_rows.items():
                    extended[rows] = recv_gpu[r]
            else:
                for r, buf in send_gpu.items():
                    cp.asnumpy(buf, out=send_cpu[r])
                self._sendrecv(send_cpu, recv_cpu)
                for r, rows in self._recv_rows.items():
                    extended[rows] = cp.asarray(recv_cpu[r])

        return extended

    def _sendrecv(self, send_bufs, recv_bufs):
        """非阻塞收发（buffer 是 cupy 数组时依赖 CUDA-aware MPI 经 `__cuda_array_interface__` 直接读写设备内存）。
        2026-10-17 以前 CUDA-aware 路径传的是 `cp.cuda.MemoryPointer.from_device(...)`——CuPy 没有这个 API，
        该路径一执行就 AttributeError；打包后也没有流同步（I-122）。"""
        comm = get_comm()
        MPI = get_mpi()
        part = self.partition
        reqs = [comm.Irecv(recv_bufs[r], source=r, tag=0) for r in part.neighbor_ranks if r in recv_bufs]
        reqs += [comm.Isend(send_bufs[r], dest=r, tag=0) for r in part.neighbor_ranks if r in send_bufs]
        if reqs:
            MPI.Request.Waitall(reqs)
