"""AutoFlowCFD API（V2.0 纯 FR 架构）。

提供 AutoFlowCFD V2.0 的高层接口，支持网格处理、FR求解和后处理。
"""

# 本模块 2026-09-24 按职责拆成子包（项目「单文件不超 500 行」规范）。
# 下面 re-export 全部公开名与测试在用的私有名，所以全仓库
# `from autoflowcfd.api import ...` 一个字都不用改。

from .helpers import (  # noqa: F401
    _turbulence_model_str,
)
from .solve import (  # noqa: F401
    _APISolveMixin,
)
from .post import (  # noqa: F401
    _APIPostMixin,
)
from .facade import (  # noqa: F401
    AutoFlowCFDAPI,
    create_api,
)

__all__ = [
    "AutoFlowCFDAPI",
    "create_api",
]
