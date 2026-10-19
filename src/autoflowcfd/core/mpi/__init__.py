"""AutoFlowCFD V2.0 - MPI 域分解并行。

子模块：
- partition/            网格分区（METIS）
- halo.py / compact_halo.py   halo 层与数据交换
- comm.py               MPI 集合 / 点对点通信封装（非 MPI 环境降级为本地操作）
- reductions.py         分布式 Newton-Krylov 的全局归约
- distributed_state.py  分布式状态
- distributed_flat_face/        分布式面几何
- distributed_solver/   分布式 FRSolver（CPU MPI）
- distributed_mesh_loader/      完全分布式网格加载
- distributed_checkpoint/       分布式 checkpoint 读写
- distributed_order_continuation/   分布式 Order Continuation 的换阶与重分发

使用方式：

    mpirun -np N autoflowcfd solve steady <grid_file> --n-ranks N

MPI 进程模型：每个 rank 是一个独立进程，内部用 numba 多线程，总并行度 = rank 数 x 每 rank 线程数。

## 惰性导入（2026-10-19，I-94）

`from mpi4py import MPI` 即调用 `MPI_Init`。此前本包导入时就导入 mpi4py，而单机路径（求解循环、Order
Continuation 等）都要 `from autoflowcfd.core.mpi import is_root`，于是装了 mpi4py 的机器上每次单机运行都初始化
MPI（部分集群在 mpirun 之外调用 `MPI_Init` 会挂起或报错）。现在只在两种情况下导入：检测到 MPI 启动环境
（`_LAUNCH_ENV` 里的环境变量，mpirun / mpiexec / srun 的 PMI 都会设置），或分布式路径显式要求
（`get_mpi(require=True)`）。否则 `get_rank() == 0`、`get_size() == 1`，通信函数降级为本地操作。
"""

import os
import sys

#: MPI 启动器为每个进程设置的环境变量（Open MPI、MPICH / Intel MPI / MS-MPI 的 PMI、PMIx、MVAPICH）
_LAUNCH_ENV = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_RANK", "MV2_COMM_WORLD_SIZE")

_MPI = None
_import_attempted = False


def _launched_under_mpi() -> bool:
    return any(name in os.environ for name in _LAUNCH_ENV) or "mpi4py.MPI" in sys.modules


def get_mpi(require: bool = False):
    """mpi4py.MPI 模块；没有在 MPI 环境下启动且未要求时返回 None（不导入、不初始化 MPI）。

    Args:
        require: 分布式路径显式要求 MPI：未检测到启动环境也导入（mpi4py 没装时仍返回 None，由调用方报告）。
    """
    global _MPI, _import_attempted
    if not _import_attempted and (require or _launched_under_mpi()):
        _import_attempted = True
        try:
            from mpi4py import MPI
            _MPI = MPI
        except ImportError:
            _MPI = None
    return _MPI


def mpi_available(require: bool = False) -> bool:
    """MPI 是否可用（参数同 `get_mpi`）。"""
    return get_mpi(require) is not None


def get_comm():
    """返回 MPI.COMM_WORLD，MPI 不可用时返回 None。"""
    mpi = get_mpi()
    return None if mpi is None else mpi.COMM_WORLD


def get_rank() -> int:
    """当前 rank 编号，非 MPI 环境返回 0。"""
    comm = get_comm()
    return 0 if comm is None else comm.Get_rank()


def get_size() -> int:
    """总 rank 数，非 MPI 环境返回 1。"""
    comm = get_comm()
    return 1 if comm is None else comm.Get_size()


def is_root() -> bool:
    """当前是否为 root rank (rank 0)。"""
    return get_rank() == 0
