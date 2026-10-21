"""求解命令的结果/checkpoint 持久化（写出、续算重建、恢复到已有求解器）。


## 文件分工

    write.py             落盘：checkpoint 与最终结果
    rebuild.py           从 checkpoint 重建一个全新 solver（含网格/算子重建）
    restore.py           把 checkpoint 里的场恢复到一个已存在的 solver 上
    initial_field.py     `--init-from`：恢复场并从它起步（稳态 / 瞬态、四个后端共用）

本 `__init__.py` re-export 各子模块的公开名。
"""

from .resume_run import continue_single_node  # noqa: F401
from .write import (  # noqa: F401
    periodic_checkpoint_callback,
    save_results,
    write_checkpoint,
    write_single_node_outputs,
)
from .rebuild import (  # noqa: F401
    physics_from_metadata,
    rebuild_from_checkpoint,
    rebuild_solver_from_checkpoint,
)
from .initial_field import (  # noqa: F401
    start_distributed_from_checkpoint,
    start_single_node_from_checkpoint,
)
from .restore import (  # noqa: F401
    restore_solver_state_from_fields,
    restore_state_from_checkpoint,
)

__all__ = [
    "continue_single_node",
    "periodic_checkpoint_callback",
    "physics_from_metadata",
    "rebuild_from_checkpoint",
    "rebuild_solver_from_checkpoint",
    "restore_solver_state_from_fields",
    "restore_state_from_checkpoint",
    "save_results",
    "start_distributed_from_checkpoint",
    "start_single_node_from_checkpoint",
    "write_checkpoint",
    "write_single_node_outputs",
]
