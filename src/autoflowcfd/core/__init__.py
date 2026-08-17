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

# Backend 模块
from .backend.base import BackendBase
from .backend.cpu_backend import NumbaBackend
from .backend.gpu_backend import CUDABackend

# TransientSolver 是 FRSolver 的别名，用于瞬态仿真
TransientSolver = FRSolver


def create_backend(backend_type: str = "cpu", **kwargs):
    """创建计算后端实例。
    
    Args:
        backend_type: 后端类型 ('cpu', 'gpu' 或 'auto')
        **kwargs: 后端特定参数
        
    Returns:
        BackendBase: 后端实例
        
    Raises:
        RuntimeError: GPU后端不可用时
        ValueError: 未知的后端类型
    """
    if backend_type.lower() == "cpu":
        return NumbaBackend(**kwargs)
    elif backend_type.lower() == "gpu":
        backend = CUDABackend(**kwargs)
        if not backend.available:
            raise RuntimeError(
                "GPU backend requested but CUDA is not available. "
                "Please install CUDA Toolkit and ensure NVIDIA GPU is present."
            )
        return backend
    elif backend_type.lower() == "auto":
        # 先尝试GPU，失败则回退到CPU
        try:
            gpu_backend = CUDABackend(**kwargs)
            if gpu_backend.available:
                return gpu_backend
        except Exception:
            pass
        
        # 回退到CPU
        return NumbaBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def get_available_backends():
    """获取可用的后端列表。
    
    Returns:
        dict: 后端名称到可用状态的映射字典
    """
    backends = {"cpu": True}
    # GPU 检测统一为 CuPy（替代此前的 numba.cuda）
    try:
        from autoflowcfd.core.gpu import gpu_available
        backends["gpu"] = gpu_available
    except ImportError:
        backends["gpu"] = False
    return backends


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
    'BackendBase',
    'NumbaBackend',
    'CUDABackend',
    'FRSolver',
    'TransientSolver',
    'create_backend',
    'get_available_backends',
]
