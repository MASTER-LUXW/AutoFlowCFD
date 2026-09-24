"""
AutoFlowCFD V2.0 - FRSolver 湍流模型管理 (从 fr_solver.py 拆分)

本文件把 FRSolver 里与湍流模型初始化/壁面距离/源项计算/涡粘度耦合相关
的逻辑拆出来，避免 fr_solver.py 单文件过长（>400行需拆分的项目规范）。
函数签名都以 `solver: FRSolver` 为第一参数，FRSolver 里保留同名的薄
委托方法，调用方式不变——与代码库里 solver_helpers.py/order_continuation.py
已经在用的委托模式一致。

## 文件分工（2026-09-24 拆包，原 795 行）

    init.py           湍流模型构造、自由来流/边界值、生产项 ramp
    wall_distance.py  壁面距离场计算与跨阶数重算（纯几何量，见该文件文档）
    source.py         每步的湍流源项与输运求值
    corrections.py    涡粘修正施加与涡粘场取用

本 `__init__.py` re-export 全部既有公开名**以及跨模块在用的私有名**
（`_set_freestream_turbulence` 被 6 个分布式/GPU 文件直接导入），所以
全仓库 `from autoflowcfd.core.fr_solver.turbulence import ...` 一个字
都不用改。
"""

from .init import (  # noqa: F401
    _filter_matrices_are_identity,
    _set_freestream_turbulence,
    _set_turbulence_bounds,
    _update_production_ramp,
    init_turbulence_models,
)
from .wall_distance import (  # noqa: F401
    _map_node_distances_to_points,
    _map_wall_distance_fallback,
    compute_wall_distance_field,
    recompute_wall_distance_for_current_order,
)
from .source import compute_turbulence_source  # noqa: F401
from .corrections import (  # noqa: F401
    apply_turbulence_corrections,
    get_turbulent_viscosity_field,
)

__all__ = [
    "apply_turbulence_corrections",
    "compute_turbulence_source",
    "compute_wall_distance_field",
    "get_turbulent_viscosity_field",
    "init_turbulence_models",
    "recompute_wall_distance_for_current_order",
]
