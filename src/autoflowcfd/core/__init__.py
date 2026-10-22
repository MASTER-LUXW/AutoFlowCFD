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
from .utils.wall_distance import WallDistanceSource

# Turbulence 模块
from .turbulence.sst import SSTModelFR
from .turbulence.des import DDESModel, IDDESModel
from .turbulence.wall_model import EquilibriumWallModel
from .turbulence.sgs import WALEModel


__all__ = [
    'FRState',
    'SolverResult',
    'TimeIntegrator',
    'TimeIntegrationScheme',
    'compute_ausm_up_flux',
    'WallDistanceSource',
    'SSTModelFR',
    'DDESModel',
    'IDDESModel',
    'EquilibriumWallModel',
    'WALEModel',
    'FRSolver',
]
