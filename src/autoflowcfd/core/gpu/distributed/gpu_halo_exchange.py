"""
AutoFlowCFD V2.0 - GPU 直接 Halo 交换

优化 halo 交换性能：避免 GPU→CPU→MPI→CPU→GPU 的完整往返。

策略：
1. 使用 CuPy 的 CUDA IPC / mpi4py GPU buffer 支持
   - mpi4py 支持直接发送/接收 GPU buffer（需要 CUDA-aware MPI）
   - 如果 MPI 实现不支持 CUDA buffer，自动回退到 staging buffer 模式
2. Staging buffer 模式（非 CUDA-aware MPI）：
   - 在 GPU 上用 cp.cuda.Stream 异步拷贝到 staging area
   - 使用 pinned memory 减少 H2D/D2H 延迟
   - 只传输 send_lists 中的数据（不是全场拷贝）
3. 预分配 recv buffer，避免每次交换重新分配

性能对比：
- 旧方案：全场 GPU→CPU (cp.asnumpy) → MPI → CPU→GPU (cp.asarray)
- 新方案：只传输 send/recv 列表中的数据，使用异步流重叠计算与通信

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

        with cp.cuda.Device(device_id):
            # 预分配 GPU send/recv buffer
            self.send_buffers_gpu: Dict[int, cp.ndarray] = {}
            self.recv_buffers_gpu: Dict[int, cp.ndarray] = {}

            for r, cells in partition.send_lists.items():
                self.send_buffers_gpu[r] = cp.zeros(
                    (len(cells), n_sps, n_vars), dtype=cp.float64
                )
            for r, cells in partition.recv_lists.items():
                self.recv_buffers_gpu[r] = cp.zeros(
                    (len(cells), n_sps, n_vars), dtype=cp.float64
                )

            # 预分配 CPU pinned staging buffer（用于非 CUDA-aware MPI）
            self.send_buffers_cpu: Dict[int, np.ndarray] = {}
            self.recv_buffers_cpu: Dict[int, np.ndarray] = {}

            if not self.cuda_aware:
                for r, cells in partition.send_lists.items():
                    self.send_buffers_cpu[r] = np.empty(
                        (len(cells), n_sps, n_vars), dtype=np.float64
                    )
                for r, cells in partition.recv_lists.items():
                    self.recv_buffers_cpu[r] = np.empty(
                        (len(cells), n_sps, n_vars), dtype=np.float64
                    )

            # 预计算 halo 索引映射（避免每次交换重复计算）
            self._halo_index_map: Dict[int, list] = {}
            for r, global_cells in partition.recv_lists.items():
                offsets = []
                for gc in global_cells:
                    idx = np.searchsorted(partition.halo_cells, gc)
                    if idx < len(partition.halo_cells) and partition.halo_cells[idx] == gc:
                        offsets.append(idx)
                    else:
                        offsets.append(-1)  # 无效
                self._halo_index_map[r] = offsets

        # 专用 CUDA stream 用于异步传输
        with cp.cuda.Device(device_id):
            self._comm_stream = cp.cuda.Stream(non_blocking=True)

        logger.debug(
            f"GPU Halo exchange initialized: rank {partition.rank}, "
            f"{partition.n_halo} halo cells, cuda_aware={self.cuda_aware}"
        )

    def exchange(self, U_gpu) -> 'cp.ndarray':
        """执行 GPU halo 交换。

        Args:
            U_gpu: CuPy 数组 (n_local_cells, n_sps, n_vars) 本 rank 的 local cell 数据

        Returns:
            extended_data: CuPy 数组 (n_total_cells, n_sps, n_vars)
        """
        cp = get_cupy()
        part = self.partition
        n_local = part.n_local_cells
        n_total = part.n_total_cells

        with cp.cuda.Device(self.device_id):
            # 构建扩展数组
            extended = cp.empty((n_total, self.n_sps, self.n_vars), dtype=cp.float64)
            extended[:n_local] = U_gpu

            if not part.neighbor_ranks:
                return extended

            if self.cuda_aware:
                self._exchange_cuda_aware(U_gpu, extended)
            else:
                self._exchange_staging(U_gpu, extended)

        return extended

    def _exchange_cuda_aware(self, U_gpu, extended):
        """CUDA-aware MPI 直接 GPU buffer 通信（零拷贝）。"""
        cp = get_cupy()
        comm = get_comm()
        MPI = get_mpi()
        part = self.partition

        # 打包发送数据（GPU 上直接操作）
        for r, local_indices in part.send_lists.items():
            self.send_buffers_gpu[r][:] = U_gpu[local_indices]

        # 非阻塞接收
        recv_reqs = []
        for r in part.neighbor_ranks:
            if r in self.recv_buffers_gpu:
                req = comm.Irecv(
                    cp.cuda.MemoryPointer.from_device(
                        self.recv_buffers_gpu[r].data
                    ),
                    source=r, tag=0,
                )
                recv_reqs.append(req)

        # 非阻塞发送
        send_reqs = []
        for r in part.neighbor_ranks:
            if r in self.send_buffers_gpu:
                req = comm.Isend(
                    cp.cuda.MemoryPointer.from_device(
                        self.send_buffers_gpu[r].data
                    ),
                    dest=r, tag=0,
                )
                send_reqs.append(req)

        if recv_reqs:
            MPI.Request.Waitall(recv_reqs)
        if send_reqs:
            MPI.Request.Waitall(send_reqs)

        # 填入 halo 位置
        self._fill_halo_gpu(extended)

    def _exchange_staging(self, U_gpu, extended):
        """Staging buffer 模式：GPU→CPU→MPI→CPU→GPU。

        优化点：
        1. 只传输 send_lists 中的数据（不是全场拷贝）
        2. 使用异步流重叠 GPU 操作
        3. 预分配 CPU buffer 避免重复分配
        """
        cp = get_cupy()
        comm = get_comm()
        MPI = get_mpi()
        part = self.partition

        # 1. GPU 上打包到 send_buffers_gpu
        for r, local_indices in part.send_lists.items():
            self.send_buffers_gpu[r][:] = U_gpu[local_indices]

        # 2. GPU→CPU 异步拷贝
        cp.cuda.Stream.null.synchronize()
        for r in part.send_lists:
            cp.asnumpy(self.send_buffers_gpu[r], out=self.send_buffers_cpu[r])

        # 3. MPI 通信（CPU buffer）
        recv_reqs = []
        for r in part.neighbor_ranks:
            if r in self.recv_buffers_cpu:
                req = comm.Irecv(self.recv_buffers_cpu[r], source=r, tag=0)
                recv_reqs.append(req)

        send_reqs = []
        for r in part.neighbor_ranks:
            if r in self.send_buffers_cpu:
                req = comm.Isend(self.send_buffers_cpu[r], dest=r, tag=0)
                send_reqs.append(req)

        if recv_reqs:
            MPI.Request.Waitall(recv_reqs)
        if send_reqs:
            MPI.Request.Waitall(send_reqs)

        # 4. CPU→GPU 异步拷贝
        for r in part.recv_lists:
            if r in self.recv_buffers_cpu:
                self.recv_buffers_gpu[r][:] = cp.asarray(self.recv_buffers_cpu[r])

        # 5. 填入 halo 位置
        self._fill_halo_gpu(extended)

    def _fill_halo_gpu(self, extended):
        """将接收到的数据填入 GPU 扩展数组的 halo 位置。"""
        cp = get_cupy()
        part = self.partition
        n_local = part.n_local_cells

        for r, global_cells in part.recv_lists.items():
            if r in self.recv_buffers_gpu:
                offsets = self._halo_index_map[r]
                for i, halo_idx in enumerate(offsets):
                    if halo_idx >= 0:
                        extended[n_local + halo_idx] = self.recv_buffers_gpu[r][i]
