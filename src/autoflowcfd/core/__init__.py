"""AutoFlowCFD V2.0 核心模块导出。

本模块统一导出 FR 求解器所需的核心类和函数。
重构后的模块结构：
- fr_solver/: FR 求解器主模块
- fr_residual/: FR 残差计算
- fr_operators/: FR 算子与内核
- turbulence/: 湍流模型
- time_integration/: 时间积分
- utils/: 辅助工具
- backend/: CPU/GPU 后端
- gpu/: GPU 加速
- mpi/: MPI 并行
"""

# FR Solver 模块
from .fr_solver import FRSolver, FRState, SolverResult

# Time Integration 模块
from .time_integration.base import TimeIntegrator, TimeIntegrationScheme

# FR Operators 模块
from .fr_operators.kernels import compute_ausm_up_flux

# Utils 模块
from .utils.wall_distance import compute_wall_distance

# Turbulence 模块
from .turbulence.sst import SSTModelFR
from .turbulence.des import DDESModel, IDDESModel
from .turbulence.wmles import WMLESModel
from .turbulence.sgs import WALEModel, SmagorinskyModel

# Backend 模块（第四次评审：删除了 V1 时代与生产路径完全脱节、且用
# 已被弃用的 numba.cuda 判据的 create_backend/NumbaBackend/CUDABackend/
# BackendBase，见 backend/base.py、backend/__init__.py 模块文档。
# get_available_backends/list_available_backends 现在只有一份实现
# （backend/__init__.py），这里直接复用，不再各自维护一份可能给出不同
# 结论的判据。）
from .backend import get_available_backends, list_available_backends, SolutionVector

# TransientSolver 是 FRSolver 的别名，用于瞬态仿真
TransientSolver = FRSolver


__all__ = [
    'FRState',
    'SolverResult',
    'TimeIntegrator',
    'TimeIntegrationScheme',
    'compute_ausm_up_flux',
    'compute_wall_distance',
    'SSTModelFR',
    'DDESModel',
    'IDDESModel',
    'WMLESModel',
    'WALEModel',
    'SmagorinskyModel',
    'SolutionVector',
    'FRSolver',
    'TransientSolver',
    'get_available_backends',
    'list_available_backends',
]
