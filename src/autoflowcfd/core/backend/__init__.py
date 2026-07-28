"""Backend factory and initialization."""

from typing import Optional, Dict, Any
from .base import BackendBase, SolutionVector
from .cpu_backend import NumbaBackend
from .gpu_backend import CUDABackend


def create_backend(
    backend_type: str = "auto",
    **kwargs
) -> BackendBase:
    """Factory function to create computational backend.
    
    This function creates the appropriate backend (CPU or GPU) based on
    user preference and hardware availability.
    
    Args:
        backend_type: Backend type selector
            - "auto": Automatically choose best available backend
            - "cpu": Force CPU backend (Numba)
            - "gpu": Force GPU backend (CUDA)
        **kwargs: Additional backend-specific parameters
            - n_threads: Number of CPU threads (for CPU backend)
            - device_id: CUDA device ID (for GPU backend)
    
    Returns:
        Initialized backend instance
    
    Raises:
        ValueError: If backend_type is invalid
        RuntimeError: If requested backend is not available
    
    Examples:
        >>> # Auto-select best backend
        >>> backend = create_backend("auto")
        
        >>> # Force CPU with 8 threads
        >>> backend = create_backend("cpu", n_threads=8)
        
        >>> # Force GPU on device 0
        >>> backend = create_backend("gpu", device_id=0)
    """
    if backend_type == "auto":
        # Try GPU first, fall back to CPU
        try:
            gpu_backend = CUDABackend(**kwargs)
            if gpu_backend.available:
                print("[Backend] Using GPU (CUDA) backend")
                return gpu_backend
            else:
                print("[Backend] GPU not available, falling back to CPU")
        except Exception as e:
            print(f"[Backend] GPU initialization failed: {e}, using CPU")
        
        # Fall back to CPU
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
    """Check which backends are available on current system.
    
    Returns:
        Dictionary mapping backend names to availability status
    
    Examples:
        >>> backends = get_available_backends()
        >>> print(backends)
        {'cpu': True, 'gpu': True}
    """
    result = {"cpu": True}  # CPU backend always available
    
    # Check GPU backend
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
