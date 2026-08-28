"""边界条件模块（BD-01/BD-02）。

真正参与 FR 残差组装的边界条件实现只有两个文件：

- `fr_ghost_state.py`：弱形式边界条件的幽灵态构造（BD-01）——
  WALL/INLET/OUTLET/FARFIELD/SYMMETRY 各自的 `*_ghost_state` 函数，以及
  统一入口 `BoundaryGhostStateProvider`（按边界组代码分派到对应幽灵态
  构造函数）。由 `core/fr_solver/boundary.py::build_boundary_ghost_provider`
  构建并接入求解器。
- `synthetic_inlet.py`：合成湍流入口 (SEM，BD-02)——`SyntheticEddyMethod`
  提供满足目标雷诺应力的入口速度脉动，`InletSEMGhostState`
  （定义在 `fr_ghost_state.py`）把它包装成幽灵态提供者可用的形式。

V2.0 专家组盲审发现并清理（2026-08-28）：本模块此前还包含
`BoundaryManager`/`YAMLConfigLoader`/`InletBC`/`OutletBC`/`WallBC` 等一整套
V1（FVM）时代遗留的边界条件管理基础设施（conditions.py/manager.py/
config.py/outlet_bc.py/manager_configure.py/config_validators.py/
conditions_advanced.py，合计约 2560 行），文档字符串曾声称"实际边界值
计算由 core/bc_handler.py 完成"——`bc_handler.py` 在 V2 重构中已被删除，
这套基础设施在真实求解路径上零调用点，与上面两个真正生效的文件完全
无关。已确认零外部引用后整体删除，不再保留"看似完整、实则孤立"的死代码。
二次开发应以 `fr_ghost_state.py`/`synthetic_inlet.py` 为起点。
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
