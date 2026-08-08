"""AutoFlowCFD 的命令行界面模块。

本模块提供基于 Click 的 CLI 命令，用于运行仿真、
后处理结果以及执行实用函数。
"""

from .main import cli

__all__ = ["cli"]
