"""AutoFlowCFD V2.0 - `solve steady` 命令（包）。

`command`：click 命令本体与后端分派；`multi_gpu`/`single_gpu`/`cpu_mpi`/`cpu_single`：四个后端分支。
"""

from .command import solve_steady  # noqa: F401

__all__ = ["solve_steady"]
