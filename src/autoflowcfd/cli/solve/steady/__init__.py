"""AutoFlowCFD V2.0 - `solve steady` 命令（包）。

`command`：click 命令本体；按后端执行在 `cli/solve/run.py`（与 `solve transient` 共用）。
"""

from .command import solve_steady  # noqa: F401

__all__ = ["solve_steady"]
