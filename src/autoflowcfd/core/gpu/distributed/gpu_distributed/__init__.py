"""AutoFlowCFD V2.0 - 多 GPU + MPI 分布式求解器（包）。

子模块：`core`（类与装配顺序）、`setup`（`__init__` 的装配阶段）、`residual`、`timestep`、`stepping`、`compact_view`。
"""

from .compact_view import _CompactMeshDataView  # noqa: F401
from .core import MultiGPUDistributedSolver  # noqa: F401

__all__ = ["MultiGPUDistributedSolver"]
