"""求解命令的结果/checkpoint 持久化辅助函数 —— 从 solve_helpers.py 拆出，控制单文件行数。

见 solve_helpers.py 文档说明整体拆分结构。


## 文件分工(2026-09-24 拆包, 原 614 行)

    write.py             落盘：checkpoint 与最终结果
    rebuild.py           从 checkpoint 重建一个全新 solver（含网格/算子重建）
    restore.py           把 checkpoint 里的场恢复到一个已存在的 solver 上

本 `__init__.py` re-export 全部既有名, 所以全仓库导入一字不改。
"""

from .write import (  # noqa: F401
    save_results,
    write_checkpoint,
)
from .rebuild import (  # noqa: F401
    rebuild_solver_from_checkpoint,
)
from .restore import (  # noqa: F401
    restore_solver_state_from_fields,
    restore_state_from_checkpoint,
)

__all__ = [
    "rebuild_solver_from_checkpoint",
    "restore_solver_state_from_fields",
    "restore_state_from_checkpoint",
    "save_results",
    "write_checkpoint",
]
