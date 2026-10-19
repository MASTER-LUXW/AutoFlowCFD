"""AutoFlowCFD API（V2.0 纯 FR 架构）。

提供 AutoFlowCFD V2.0 的高层接口，支持网格处理、FR求解和后处理。
"""

from .facade import AutoFlowCFDAPI  # noqa: F401

__all__ = ["AutoFlowCFDAPI"]
