"""AutoFlowCFD V2.0 - 分布式块预处理全局粗校正的通信上下文（CPU-MPI 与多 GPU 共用）。

全局粗校正（`time_integration/implicit/coarse/global_coarse.py`）要三样东西：集合通信
（`allgather` / `allreduce`）、本 rank 编号与 rank 数、以及"本 rank 逐单元值 -> 紧凑空间
（local+halo，"棱柱在前"）逐单元值"的映射——耦合块的列单元编号就在紧凑空间里，halo
单元的全局粗节点编号、P0 差分装配要的 halo 单元颜色都经它取得。映射走与残差同一次
halo 交换与换序（`perm`）：CPU `HaloExchange` 与 GPU `GPUHaloExchange` 的 `exchange` 都接受任意
逐单元形状，逐单元标量直接交换（此前 CPU 走标量专用的 `exchange_scalar`、多 GPU 把标量铺满
`(n_local, n_sps, 5)` 再交换，两份写法），两个后端同一个类、只差数组模块。

映射对象随块缓存长期存活，做成类而不是闭包，只持有交换器、换序与数组模块（项目规范）。
交换是集体调用：只在块缓存构造全局粗空间 / 耦合图时各 rank 同步调用。
"""

import numpy as np

from autoflowcfd.core.mpi.comm import allgather_array, allreduce_sum
from autoflowcfd.core.time_integration.implicit.coarse import CoarseCommContext


class CompactCellValues:
    """本 rank 逐单元值（主机 numpy）-> 紧凑空间逐单元值（主机 numpy）：经交换器扩到
    local+halo、按 `perm` 换序。CPU 传 `xp=numpy`，多 GPU 传 `xp=cupy`（`perm` 在设备上）。"""

    __slots__ = ("exchange", "perm", "xp")

    def __init__(self, halo_exchange, perm, xp):
        self.exchange, self.perm, self.xp = halo_exchange.exchange, perm, xp

    def __call__(self, values_local):
        out = self.exchange(self.xp.asarray(np.asarray(values_local, dtype=np.float64)))[self.perm]
        return out.get() if hasattr(out, "get") else np.asarray(out)


def coarse_comm_context(partition, compact_cell_values) -> CoarseCommContext:
    """本 rank 的全局粗校正通信上下文（`allgather_array` / 数组 `allreduce_sum` 来自 `comm.py`）。"""
    return CoarseCommContext(rank=int(partition.rank), n_ranks=int(partition.n_ranks), allgather=allgather_array,
                             allreduce_sum=allreduce_sum, compact_cell_values=compact_cell_values)
