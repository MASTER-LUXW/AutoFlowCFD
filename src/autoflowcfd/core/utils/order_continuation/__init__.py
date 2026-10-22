"""
AutoFlowCFD V2.0 - Order Continuation Utilities

本模块包含 Order Continuation 方法所需的插值工具。

## 文件分工

    order_switch.py      换阶中与后端无关的部分（场的延拓 / 来流重建、网格切阶、跨步状态复位），六条路径共用
    interp.py            单机 CPU 升阶
    turbulence_reset.py  续算的湍流场被上界大面积截断时拒绝续算（或按 --reset-turbulence 重置），全部后端共用
    p0_reset.py          全新求解器在 Order Continuation 起步时重建到 P0
    run.py               顶层编排：P0 -> P1 -> ... -> 目标阶数（四个后端共用的唯一循环）
    checkpoint_state.py  随 checkpoint 持久化的阶段状态（单机/分布式写入与恢复共用）
    initial_field.py     从 checkpoint 场起步（--init-from）：低阶 checkpoint 精确延拓到目标阶数、不做阶数爬坡

本 `__init__.py` re-export 各子模块的公开名。
"""

from .interp import (  # noqa: F401
    interpolate_to_new_order,
)
from .turbulence_reset import (  # noqa: F401
    check_resumed_turbulence,
)
from .p0_reset import (  # noqa: F401
    _reset_state_to_p0,
)
from .run import (  # noqa: F401
    run_order_continuation,
)

__all__ = [
    "interpolate_to_new_order",
    "run_order_continuation",
]
