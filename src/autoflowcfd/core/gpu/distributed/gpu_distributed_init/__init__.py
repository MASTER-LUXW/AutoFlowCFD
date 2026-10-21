"""MultiGPUDistributedSolver 初始化和 I/O 混入类。

从 gpu_distributed.py 拆出，控制单文件行数。
"""

# 包根 re-export 各子模块的公开名与测试在用的私有名。

from .geometry import (  # noqa: F401
    _GPUDistributedInitMixin,
)

__all__ = [
]
