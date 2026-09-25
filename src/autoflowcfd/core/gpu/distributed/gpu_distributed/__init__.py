"""AutoFlowCFD V2.0 - 多 GPU + MPI 分布式求解器（包）。

从单文件 `core/gpu/distributed/gpu_distributed.py`（1270 行）拆成子包（2026-09-25，
项目「单文件不超 500 行」规范）：`core`（类与装配顺序）、`setup`（`__init__` 的装配
阶段）、`residual`、`timestep`、`stepping`、`compact_view`。
"""

from .compact_view import _CompactMeshDataView  # noqa: F401
from .core import MultiGPUDistributedSolver  # noqa: F401

__all__ = ["MultiGPUDistributedSolver"]
