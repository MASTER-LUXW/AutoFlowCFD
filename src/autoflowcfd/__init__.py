"""AutoFlowCFD - 面向汽车空气动力学的高性能计算流体力学（CFD）软件。

AutoFlowCFD 是一款开源计算流体力学（CFD）软件，
专注于汽车外流场仿真分析。它提供
高精度、高速度的 CFD 分析，并具备 AI Agent 集成能力。

主要特性：
    - 原生支持 NAS 网格格式（ANSA v22/v23/v24）
    - 混合 CPU/GPU 计算（Numba/CUDA）
    - 高阶通量重构（Flux Reconstruction）求解器
    - 先进的湍流模型（SST k-ω, DES/DDES, LES）
    - 双重接口（CLI + Python API）
    - 模块化且可扩展的架构

示例：
    >>> from autoflowcfd import AutoFlowCFDAPI
    >>> api = AutoFlowCFDAPI()
    >>> grid = api.load_grid("car_model.nas")
    >>> result = api.run_steady(grid, backend="gpu", order=3)
    >>> coeffs = api.calculate_coefficients(result)
    >>> print(f"Drag Coefficient: {coeffs['Cd']:.4f}")
"""

# ============================================================================
# 关键：在导入 NumPy 之前设置 BLAS/线性代数线程数
# 这确保了向量化操作能够最大限度地利用多核性能
# ============================================================================
import os
import multiprocessing

_cpu_count = multiprocessing.cpu_count()
os.environ.setdefault('MKL_NUM_THREADS', str(_cpu_count))
os.environ.setdefault('OPENBLAS_NUM_THREADS', str(_cpu_count))
os.environ.setdefault('NUMEXPR_NUM_THREADS', str(_cpu_count))
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', str(_cpu_count))
os.environ.setdefault('OMP_NUM_THREADS', str(_cpu_count))

__version__ = "0.1.0"
__author__ = "AutoFlowCFD Team"
__maintainer__ = "Mr Lu"
__email__ = "luxw_chd@126.com"
__license__ = "Apache-2.0"
__name__ = "AutoFlowCFD"  # 添加包名称

from typing import Any, Dict

# 导入主 API 类
from .api import AutoFlowCFDAPI

# 模块元数据
__all__ = [
    "__version__",
    "__name__",
    "__author__",
    "__email__",
    "__license__",
    "AutoFlowCFDAPI",
]


def get_version() -> str:
    """获取 AutoFlowCFD 的当前版本。
    
    Returns:
        str: 语义化版本字符串（例如 "0.1.0"）
        
    Example:
        >>> import autoflowcfd
        >>> autoflowcfd.get_version()
        '0.1.0'
    """
    return __version__


def get_info() -> Dict[str, str]:
    """获取 AutoFlowCFD 的系统信息。
    
    Returns:
        Dict: 包含版本、作者等信息的字典
        
    Example:
        >>> import autoflowcfd
        >>> info = autoflowcfd.get_info()
        >>> print(info['version'])
    """
    return {
        "name": __name__,
        "version": __version__,
        "author": __author__,
        "maintainer": __maintainer__,
        "email": __email__,
        "license": __license__,
    }


def create_api(verbose: bool = False) -> AutoFlowCFDAPI:
    """创建 AutoFlowCFD API 实例。
    
    用于创建 API 实例的便捷函数。
    
    Args:
        verbose: 启用详细日志输出
        
    Returns:
        AutoFlowCFDAPI: API 实例
        
    Example:
        >>> api = autoflowcfd.create_api()
        >>> grid = api.load_grid("model.nas")
    """
    return AutoFlowCFDAPI(verbose=verbose)
