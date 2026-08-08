"""Backend 工厂与初始化。"""

from typing import Optional, Dict, Any
from .base import BackendBase, SolutionVector
from .cpu_backend import NumbaBackend
from .gpu_backend import CUDABackend


def create_backend(
    backend_type: str = "auto",
    **kwargs
) -> BackendBase:
    """按用户偏好和硬件可用性创建对应的计算 backend（CPU 或 GPU）的工厂函数。

    Args:
        backend_type: backend 类型选择
            - "auto": 自动选择最优的可用 backend
            - "cpu": 强制使用 CPU backend (Numba)
            - "gpu": 强制使用 GPU backend (CUDA)
        **kwargs: 其它 backend 专属参数
            - n_threads: CPU 线程数（CPU backend 用）
            - device_id: CUDA 设备 ID（GPU backend 用）

    Returns:
        已初始化的 backend 实例

    Raises:
        ValueError: backend_type 无效时
        RuntimeError: 请求的 backend 不可用时

    Examples:
        >>> # 自动选择最优 backend
        >>> backend = create_backend("auto")

        >>> # 强制 CPU，8 线程
        >>> backend = create_backend("cpu", n_threads=8)

        >>> # 强制 GPU，设备 0
        >>> backend = create_backend("gpu", device_id=0)
    """
    if backend_type == "auto":
        # 先尝试 GPU，不行再退回 CPU
        try:
            gpu_backend = CUDABackend(**kwargs)
            if gpu_backend.available:
                print("[Backend] Using GPU (CUDA) backend")
                return gpu_backend
            else:
                print("[Backend] GPU not available, falling back to CPU")
        except Exception as e:
            print(f"[Backend] GPU initialization failed: {e}, using CPU")

        # 退回 CPU
        n_threads = kwargs.get('n_threads', 4)
        cpu_backend = NumbaBackend(n_threads=n_threads)
        print(f"[Backend] Using CPU (Numba) backend with {n_threads} threads")
        return cpu_backend

    elif backend_type == "cpu":
        n_threads = kwargs.get('n_threads', 4)
        backend = NumbaBackend(n_threads=n_threads)
        print(f"[Backend] Using CPU (Numba) backend with {n_threads} threads")
        return backend

    elif backend_type == "gpu":
        device_id = kwargs.get('device_id', 0)
        backend = CUDABackend(device_id=device_id)

        if not backend.available:
            raise RuntimeError(
                "GPU backend requested but CUDA is not available. "
                "Please install CUDA Toolkit and ensure NVIDIA GPU is present."
            )

        print(f"[Backend] Using GPU (CUDA) backend on device {device_id}")
        return backend

    else:
        raise ValueError(
            f"Invalid backend_type: {backend_type}. "
            "Must be 'auto', 'cpu', or 'gpu'."
        )


def get_available_backends() -> Dict[str, bool]:
    """检查当前系统上哪些 backend 可用。

    Returns:
        backend 名称到可用状态的映射字典

    Examples:
        >>> backends = get_available_backends()
        >>> print(backends)
        {'cpu': True, 'gpu': True}
    """
    result = {"cpu": True}  # CPU backend 永远可用

    # 检查 GPU backend
    try:
        gpu_test = CUDABackend()
        result["gpu"] = gpu_test.available
    except Exception:
        result["gpu"] = False

    return result


__all__ = [
    "create_backend",
    "get_available_backends",
    "BackendBase",
    "SolutionVector",
    "NumbaBackend",
    "CUDABackend"
]
