"""AutoFlowCFD V2.0 - 分布式块预处理全局粗校正的通信上下文（CPU-MPI 与多 GPU 共用）。

全局粗校正（`time_integration/implicit/coarse/global_coarse.py`）要三样东西：集合通信
（`allgather` / `allreduce`）、本 rank 编号与 rank 数、以及"本 rank 逐单元值 -> 紧凑空间
（local+halo，"棱柱在前"）逐单元值"的映射——耦合块的列单元编号就在紧凑空间里，halo
单元的全局粗节点编号、P0 差分装配要的 halo 单元颜色都经它取得。映射走与残差同一次
halo 交换与换序（`perm`），各后端只差交换器的形状约定：

* CPU-MPI：`HaloExchange.exchange_scalar`（`(n_local, n_sps)`）；
* 多 GPU：`GPUHaloExchange.exchange`（`(n_local, n_sps, 5)` 的 cupy 数组）。

映射对象随块缓存长期存活，做成类而不是闭包，只持有交换器与换序（项目规范）。
交换是集体调用：只在块缓存构造全局粗空间 / 耦合图时各 rank 同步调用。
"""

import numpy as np

from autoflowcfd.core.mpi.comm import allgather_array, allreduce_sum
from autoflowcfd.core.time_integration.implicit.coarse import CoarseCommContext


class CpuCompactCellValues:
    """CPU-MPI：逐单元值经 `exchange_scalar` 扩到 local+halo，再按 `perm` 换到紧凑空间。"""

    __slots__ = ("exchange_scalar", "perm", "n_sps")

    def __init__(self, halo_exchange, perm, n_sps: int):
        self.exchange_scalar = halo_exchange.exchange_scalar
        self.perm = np.asarray(perm, dtype=np.int64)
        self.n_sps = int(n_sps)

    def __call__(self, values_local):
        v = np.ascontiguousarray(np.repeat(np.asarray(values_local, dtype=np.float64)[:, None], self.n_sps, 1))
        return self.exchange_scalar(v)[self.perm][:, 0]


class GpuCompactCellValues:
    """多 GPU：逐单元值铺满 `(n_local, n_sps, 5)` 后经 GPU halo 交换、按 `perm` 换序，取回主机。"""

    __slots__ = ("exchange", "perm", "n_sps", "n_vars", "cp")

    def __init__(self, gpu_halo, perm_gpu, n_sps: int, cp):
        self.exchange = gpu_halo.exchange
        self.perm = perm_gpu
        self.n_sps = int(n_sps)
        self.n_vars = int(gpu_halo.n_vars)
        self.cp = cp

    def __call__(self, values_local):
        cp = self.cp
        v = cp.asarray(np.asarray(values_local, dtype=np.float64))
        full = cp.ascontiguousarray(cp.broadcast_to(v[:, None, None], (v.shape[0], self.n_sps, self.n_vars)))
        return cp.asnumpy(self.exchange(full)[self.perm][:, 0, 0])


def coarse_comm_context(partition, compact_cell_values) -> CoarseCommContext:
    """本 rank 的全局粗校正通信上下文（`allgather_array` / 数组 `allreduce_sum` 来自 `comm.py`）。"""
    return CoarseCommContext(rank=int(partition.rank), n_ranks=int(partition.n_ranks), allgather=allgather_array,
                             allreduce_sum=allreduce_sum, compact_cell_values=compact_cell_values)
