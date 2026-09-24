"""
AutoFlowCFD V2.0 - DDES/IDDES 混合湍流模型 (T-04)

本模块实现 Delayed Detached Eddy Simulation (DDES) 逻辑，
通过屏蔽函数在边界层内保持 RANS，在分离区切换为 LES。

核心功能:
1. DDES 屏蔽函数 F_d 计算
2. 有效长度尺度 l_eff 计算（RANS/LES 切换）
3. IDDES 改进型延迟分离涡模拟
4. 与 SST k-ω 模型的无缝集成


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
    DDESModel,
)
from .iddes import (  # noqa: F401
    IDDESModel,
)

__all__ = [
    "DDESModel",
    "IDDESModel",
    "compute_h_max_and_h_wn",
]
