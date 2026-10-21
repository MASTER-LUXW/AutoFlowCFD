"""边界条件模块（BD-01/BD-02）。

真正参与 FR 残差组装的边界条件实现只有两个文件：

- `fr_ghost_state.py`：弱形式边界条件的幽灵态构造（BD-01）——
  WALL/INLET/OUTLET/FARFIELD/SYMMETRY 各自的 `*_ghost_state` 函数，以及
  统一入口 `BoundaryGhostStateProvider`（按边界组代码分派到对应幽灵态
  构造函数）。由 `core/fr_solver/boundary/ghost.py::build_boundary_ghost_provider`
  构建并接入求解器。
- `synthetic_inlet.py`：合成湍流入口 (SEM，BD-02)——`SyntheticEddyMethod`
  提供满足目标雷诺应力的入口速度脉动，`InletSEMGhostState`
  （定义在 `fr_ghost_state.py`）把它包装成幽灵态提供者可用的形式。
"""

from .fr_ghost_state import (
    BoundaryGhostStateProvider,
    InletSEMGhostState,
    wall_ghost_state,
    farfield_ghost_state,
    inlet_ghost_state,
    outlet_ghost_state,
    symmetry_ghost_state,
)
from .synthetic_inlet import SyntheticEddyMethod

__all__ = [
    "BoundaryGhostStateProvider",
    "InletSEMGhostState",
    "SyntheticEddyMethod",
    "wall_ghost_state",
    "farfield_ghost_state",
    "inlet_ghost_state",
    "outlet_ghost_state",
    "symmetry_ghost_state",
]
