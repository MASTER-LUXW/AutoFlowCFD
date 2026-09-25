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
#
# 这里保持 `cpu_count()`（**不要**改成 1），但求解循环开始前会被
# `core/fr_solver/solver.py::_limit_blas_threads` 在运行时压到 1——
# 两段式是 2026-09-13 性能优化实测后**刻意**的安排，改动前请读完：
#
# 为什么求解阶段要把 BLAS 压到 1：本项目的计算热点（无粘/粘性残差的
#   体积项与界面项、梯度、湍流输运）全部是 numba `prange` kernel，由
#   `numba.set_num_threads()` 管理自己的线程池。让 OpenBLAS 同时也开
#   cpu_count 个线程，与 numba 线程池叠加成 2 倍超额订阅、互相抢核。
#   79 万单元真实网格 P1、16 核机器实测（同一 checkpoint，只改
#   OPENBLAS_NUM_THREADS）：BLAS=16 时 numba nt=4/8 分别 51.3/48.2 s/步；
#   BLAS=1 时 45.5/43.8 s/步——每个档位都快 9~11%，没有任何档位变慢。
#
# 为什么**构造阶段**必须保持多线程：网格几何（`inv_jacs` 等度量量由
#   LAPACK 求逆得到）与 FR 算子（Vandermonde 求解）的结果会随 BLAS/
#   LAPACK 线程数不同而在最后一位上变化，而离散 GCL / 自由流场保持性
#   依赖这些度量量之间**近乎精确的抵消**。真实复现（2026-09-13）：把
#   这里直接改成 1 之后，`tests/unit/test_native_tet_inviscid_residual_
#   wiring.py::test_native_mesh_free_stream_preservation` 的棱柱 P2 判据
#   从 8.07e-7 变成 3.35e-6（容差 1e-6，测试真实失败）；逐项回退证明与
#   同批新增的 numba 融合核无关（全部回退后仍是 3.35e-6），单独把 BLAS
#   线程数改回 16 就恢复到 8.0e-7。改为"构造阶段保持多线程、构造完成后
#   再压到 1"后：残差与"全程多线程"**逐位相同**（P1/P2/P3 三阶实测最大
#   绝对差 0.0），同时完整保留上面 9~11% 的求解阶段收益。
#
# 全部用 `setdefault`：用户/CI 显式设过的环境变量一律优先，不覆盖。
# ============================================================================
import os
import multiprocessing

# numba 磁盘缓存按核源码版本隔离——必须先于任何 @njit(cache=True) 模块导入，
# 理由见 _numba_cache.py（被内联的核函数改了、缓存不失效的真实事故）。
from ._numba_cache import configure_numba_cache_dir

configure_numba_cache_dir()

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
