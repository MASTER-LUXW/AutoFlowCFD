"""
AutoFlowCFD V2.0 - FRSolver 边界条件配置构建 (从 fr_solver.py 拆分)

把网格边界组信息与用户/默认 BC 参数接到 boundary/fr_ghost_state.py 的
BoundaryGhostStateProvider 接口上，供 core/fr_residual/inviscid.py 使用。

## 文件分工

    ghost.py             边界幽灵态 provider（含 SEM 入口 FP 位置）

本 `__init__.py` re-export 各子模块的公开名。
"""

from .ghost import (  # noqa: F401
    BoundaryPhysics,
    _compute_inlet_fp_positions,
    boundary_physics,
    build_boundary_ghost_provider,
)

__all__ = [
    "BoundaryPhysics",
    "boundary_physics",
    "build_boundary_ghost_provider",
]
