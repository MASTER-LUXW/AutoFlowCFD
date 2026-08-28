"""Backend 可用性检测。

此前本模块还提供 `create_backend`/`NumbaBackend`/`CUDABackend`——V1
时代 Numba CUDA 方案的遗留骨架，与真正的生产求解路径（`FRSolver`/
`GPUFRSolver`，全程 CuPy）完全脱节，且 GPU 可用性检测用的是已被项目
明确弃用的 `numba.cuda.is_available()`（见
`ProjectFiles/V2.0/7_重大问题修复-GPU大规模并行计算.md`），与
`core/gpu/__init__.py::gpu_available`（真正的 CuPy 检测，`solve
--backend gpu` 实际依据的判据）可能给出不一致的结果——`utils
doctor`/`status` 报告的 GPU 可用性可能与 `solve --backend gpu` 实际
能否运行不一致。已确认这些类/函数全仓库零真实调用点，第四次评审时
一并删除（详见 base.py 模块文档），本模块现在只保留真正被使用的
GPU 可用性检测，且与生产路径共享同一个判据（`core.gpu.gpu_available`）。
"""

from typing import Dict, List

from .base import SolutionVector


def get_available_backends() -> Dict[str, bool]:
    """检查当前系统上哪些 backend 可用。

    与 `solve --backend gpu` 实际依据的判据完全一致（`core.gpu.
    gpu_available`，CuPy 检测），不是另一套可能给出不同结论的判据。

    Returns:
        backend 名称到可用状态的映射字典

    Examples:
        >>> backends = get_available_backends()
        >>> print(backends)
        {'cpu': True, 'gpu': True}
    """
    result = {"cpu": True}  # CPU backend 永远可用
    from autoflowcfd.core.gpu import gpu_available
    result["gpu"] = gpu_available
    return result


def list_available_backends() -> List[str]:
    """列出所有可用的 backend 名称。

    Returns:
        可用的 backend 名称列表
    """
    backends = get_available_backends()
    return [name for name, available in backends.items() if available]


__all__ = [
    "get_available_backends",
    "list_available_backends",
    "SolutionVector",
]
