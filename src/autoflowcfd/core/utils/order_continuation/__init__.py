"""
AutoFlowCFD V2.0 - Order Continuation Utilities

本模块包含 Order Continuation 方法所需的插值工具。


## 文件分工(2026-09-24 拆包, 原 926 行)

    interp.py            阶数切换时解场/湍流场的插值（延拓）算子
    turbulence_reset.py  resume 后湍流场被上界大面积钳制时的重置安全网（全部后端共用的唯一实现）
    p0_reset.py          全新求解器在 Order Continuation 起步时重建到 P0
    run.py               顶层编排：P0 -> P1 -> ... -> 目标阶数

本 `__init__.py` re-export 全部既有名, 所以全仓库导入一字不改。
"""

from .interp import (  # noqa: F401
    _build_linear_interp_matrix_3d,
    _lagrange_basis_matrix_1d,
    interpolate_to_new_order,
    interpolate_to_new_order_checked,
)
from .turbulence_reset import (  # noqa: F401
    _reset_turbulence_if_resumed_field_exploded,
)
from .p0_reset import (  # noqa: F401
    _reset_state_to_p0,
)
from .run import (  # noqa: F401
    _print_pseudo_time_summary,
    run_order_continuation,
)

__all__ = [
    "interpolate_to_new_order",
    "interpolate_to_new_order_checked",
    "run_order_continuation",
]
