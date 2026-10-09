"""求解命令的结果/checkpoint 持久化（写出、续算重建、恢复到已有求解器）。


## 文件分工(2026-09-24 拆包, 原 614 行)

    write.py             落盘：checkpoint 与最终结果
    rebuild.py           从 checkpoint 重建一个全新 solver（含网格/算子重建）
    restore.py           把 checkpoint 里的场恢复到一个已存在的 solver 上

本 `__init__.py` re-export 全部既有名, 所以全仓库导入一字不改。
"""

from .write import (  # noqa: F401
    periodic_checkpoint_callback,
    save_results,
    write_checkpoint,
    write_single_node_outputs,
)
from .rebuild import (  # noqa: F401
    physics_from_metadata,
    rebuild_solver_from_checkpoint,
)
from .restore import (  # noqa: F401
    restore_solver_state_from_fields,
    restore_state_from_checkpoint,
)

__all__ = [
    "periodic_checkpoint_callback",
    "physics_from_metadata",
    "rebuild_solver_from_checkpoint",
    "restore_solver_state_from_fields",
    "restore_state_from_checkpoint",
    "save_results",
    "write_checkpoint",
    "write_single_node_outputs",
]
