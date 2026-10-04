"""AutoFlowCFD V2.0 - `solve steady` 命令（包）。

`command`：click 命令本体与后端分派；`multi_gpu`/`cpu_mpi`/`single_node`：三个后端分支（单机分支覆盖 CPU 与单 GPU）。
"""

from .command import solve_steady  # noqa: F401

__all__ = ["solve_steady"]
