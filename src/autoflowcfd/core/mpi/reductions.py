"""AutoFlowCFD V2.0 - 隐式求解的跨 rank 全局归约。

`core/time_integration/implicit/reductions.py::LocalReductions` 的分布式
子类：算法代码（GMRES 内积、残差 RMS、Fréchet 差分步长、物理性限幅的
全局最小 theta、块 Jacobi 的集体调用次数）只调用归约对象，本类把"本进程
的部分结果"经 `comm.py` 的 allreduce 变成全局结果——所有 rank 拿到同一个
标量，于是解的是同一个线性系统、走同一个步长。非 MPI 环境下 `comm.py`
的归约是恒等，本类退化为单进程行为（n_ranks=1 的逐位对照测试靠这一点）。
"""

import numpy as np

from autoflowcfd.core.mpi.comm import allreduce_max, allreduce_min, allreduce_sum
from autoflowcfd.core.time_integration.implicit.reductions import LocalReductions


class MPIReductions(LocalReductions):
    """跨 rank 的全局归约（数组模块可以是 numpy 或 cupy：归约的是主机标量）。"""

    __slots__ = ()

    def __init__(self, xp=np):
        super().__init__(xp)

    def _allreduce_sum(self, value: float) -> float:
        return float(allreduce_sum(float(value)))

    def _allreduce_min(self, value: float) -> float:
        return float(allreduce_min(float(value)))

    def _allreduce_max(self, value: float) -> float:
        return float(allreduce_max(float(value)))
