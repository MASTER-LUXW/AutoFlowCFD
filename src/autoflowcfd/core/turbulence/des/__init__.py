"""
AutoFlowCFD V2.0 - DDES/IDDES 混合湍流模型 (T-04)

本模块实现 Delayed Detached Eddy Simulation (DDES) 逻辑，
通过屏蔽函数在边界层内保持 RANS，在分离区切换为 LES。

核心功能:
1. DDES 屏蔽函数 f_d 与混合长度尺度（Gritskevich et al. 2012）
2. IDDES 改进型延迟分离涡模拟（Shur et al. 2008）
3. 与 SST k-omega 模型的集成：长度尺度在 SST 源项求值里按当前状态算出（`DESLengthScale`）


## 文件分工(2026-09-24 拆包, 原 657 行)

    grid_scale.py        DES 网格尺度(h_max / h_wn)
    ddes.py              DDES 模型
    iddes.py             IDDES 模型(Shur et al. 2008)

本 `__init__.py` re-export 全部既有名, 所以全仓库导入一字不改。
"""

from .grid_scale import (  # noqa: F401
    compute_h_max_and_h_wn,
)
from .ddes import (  # noqa: F401
    DDESModel, DESLengthScale, des_length_for,
)
from .iddes import (  # noqa: F401
    IDDESModel,
)

__all__ = [
    "DDESModel",
    "DESLengthScale",
    "des_length_for",
    "IDDESModel",
    "compute_h_max_and_h_wn",
]
