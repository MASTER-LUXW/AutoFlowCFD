"""
AutoFlowCFD V2.0 - MPI 域分解并行模块

提供工业级 HPC 多节点并行计算基础设施：
- partition.py: 网格分区（METIS 接口）
- halo.py: halo 层管理与数据交换
- comm.py: MPI 通信封装
- distributed_state.py: 分布式 FRState
- distributed_flat_face.py: 分布式面几何
- distributed_solver.py: 分布式 FRSolver

使用方式:
    mpirun -np N autoflowcfd solve steady <grid_file> --n-ranks N
    或
    python -m mpi4py -m autoflowcfd solve steady <grid_file>

MPI 进程模型:
    每个 MPI rank 是一个独立进程，内部用 Numba 多线程。
    总并行度 = n_ranks * n_threads_per_rank。
    典型配置: 4 nodes * 16 ranks/node * 4 threads = 256 总并行度。
"""

# MPI 是可选依赖——非 MPI 环境（单机开发/测试）下不导入 mpi4py
# 也能正常使用整个求解器。mpi_available 标志供求解器判断是否启用
# 分布式路径。
try:
    from mpi4py import MPI as _MPI
    mpi_available = True
except ImportError:
    _MPI = None
    mpi_available = False


def get_mpi():
    """返回 mpi4py.MPI 模块，不可用时返回 None。"""
    return _MPI


def get_comm():
    """返回 MPI.COMM_WORLD，不可用时返回 None。"""
    if _MPI is not None:
        return _MPI.COMM_WORLD
    return None


def get_rank() -> int:
    """当前 rank 编号，非 MPI 环境返回 0。"""
    comm = get_comm()
    if comm is not None:
        return comm.Get_rank()
    return 0


def get_size() -> int:
    """总 rank 数，非 MPI 环境返回 1。"""
    comm = get_comm()
    if comm is not None:
        return comm.Get_size()
    return 1


def is_root() -> bool:
    """当前是否为 root rank (rank 0)。"""
    return get_rank() == 0
