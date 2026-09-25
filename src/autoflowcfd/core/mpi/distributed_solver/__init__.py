"""
AutoFlowCFD V2.0 - 分布式 FRSolver

将 FRSolver 扩展为 MPI 域分解版本。每个 rank 持有 local cells 的数据，
通过 halo 交换获取邻居信息，独立计算 local cells 的残差。

设计:
- 组合模式：DistributedFRSolver 持有 FRSolver 实例 + 分区/通信基础设施
- 覆盖残差计算方法：先 halo 交换，再调用单机残差（只处理 local cells）
- 全局操作（残差范数、时间步长）通过 MPI Allreduce

使用:
    # 所有 rank 加载同一网格文件
    mesh = load_mesh(grid_file)
    solver = DistributedFRSolver(mesh, ...)
    # 每个 rank 自动获取自己的分区，执行分布式求解
    for step in range(n_steps):
        solver.step(dt)
"""

# 本模块 2026-09-24 按职责拆成子包（项目「单文件不超 500 行」规范）。
# 下面 re-export 全部公开名与测试在用的私有名，所以全仓库
# `from autoflowcfd.core.mpi.distributed_solver import ...` 一个字都不用改。

from .from_package import (  # noqa: F401
    _DistributedFromPackageMixin,
)
from .solve_loop import (  # noqa: F401
    _DistributedSolveMixin,
)
from .step import (  # noqa: F401
    _DistributedStepMixin,
)
from .support import (  # noqa: F401
    _DistributedSupportMixin,
)
from .core import (  # noqa: F401
    DistributedFRSolver,
)

__all__ = [
    "DistributedFRSolver",
]
