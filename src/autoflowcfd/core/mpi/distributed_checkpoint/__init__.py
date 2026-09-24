"""
AutoFlowCFD V2.0 - 分布式 Checkpoint 保存/加载 + 结果保存

将单机 checkpoint 和结果保存扩展为分布式版本：
- Root rank 收集所有 rank 的 local cells 数据
- 组装全局状态后保存为单文件（与单机格式兼容）
- 加载时 root rank 读取后分发到各 rank

关键设计:
- 保存格式与单机完全一致（HDF5 checkpoint + pickle 结果），后处理工具无需修改
- 使用 partition.local_cells（全局索引）定位每个 rank 的数据在全局数组中的位置
- 支持变 rank 数恢复（4 ranks 保存 → 8 ranks 恢复）


## 文件分工(2026-09-24 拆包, 原 515 行)

    state.py             局部与全局状态之间的聚集与散射
    save.py              分布式落盘：checkpoint 与最终结果
    load.py              分布式读取与状态恢复

本 `__init__.py` re-export 全部既有名, 所以全仓库导入一字不改。
"""

from .state import (  # noqa: F401
    gather_global_state,
    scatter_local_state,
)
from .save import (  # noqa: F401
    distributed_save_checkpoint,
    distributed_save_results,
)
from .load import (  # noqa: F401
    distributed_load_checkpoint,
    restore_distributed_state_from_checkpoint,
)

__all__ = [
    "distributed_load_checkpoint",
    "distributed_save_checkpoint",
    "distributed_save_results",
    "gather_global_state",
    "restore_distributed_state_from_checkpoint",
    "scatter_local_state",
]
