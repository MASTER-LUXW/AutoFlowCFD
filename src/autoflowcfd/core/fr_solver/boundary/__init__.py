"""
AutoFlowCFD V2.0 - FRSolver 边界条件配置构建 (从 fr_solver.py 拆分)

把网格边界组信息与用户/默认 BC 参数接到 boundary/fr_ghost_state.py 的
BoundaryGhostStateProvider 接口上，供 core/fr_residual_inviscid.py 使用。


## 文件分工(2026-09-24 拆包, 原 506 行)

    ghost.py             边界幽灵态 provider（含 SEM 入口 FP 位置）
    tables.py            按边界组预建的查表：Dirichlet 值、镜像法向、BJ 判据包络

本 `__init__.py` re-export 全部既有名, 所以全仓库导入一字不改。
"""

from .constants import (  # noqa: F401
    _NO_DIRICHLET,
    _SEM_DEFAULT_NUM_EDDIES,
    _SEM_DEFAULT_TURBULENCE_INTENSITY,
)
from .ghost import (  # noqa: F401
    _compute_inlet_fp_positions,
    build_boundary_ghost_provider,
)
from .tables import (  # noqa: F401
    build_boundary_dirichlet_table,
    build_boundary_mirror_normals,
    make_bj_boundary_tables,
)

__all__ = [
    "build_boundary_dirichlet_table",
    "build_boundary_ghost_provider",
    "build_boundary_mirror_normals",
    "make_bj_boundary_tables",
]
