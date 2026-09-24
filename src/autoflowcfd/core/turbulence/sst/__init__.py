"""
AutoFlowCFD V2.0 - SST k-omega 湍流模型 FR 离散 (T-01, T-02)

核心功能:
1. SST k-ω 模型源项计算（产生项/耗散项/交叉扩散项）
2. 正性保持限制器 (Positivity-preserving Limiter)
3. blending functions F1, F2（Menter 1994 标准公式，V2.0 二次评审修复：
   此前 F1/F2 都误用 Von Karman 常数 kappa 顶替 beta_star，且各丢失了
   标准公式里的一项，已改正并用近壁/远场极限数值验证）

已知局限（V2.0 已修复）：k/omega 输运方程现已包含完整的对流+扩散输运项
（见 core/turbulence_transport.py），k/omega 随流场对流、跨单元扩散，
不再仅是逐点源项 ODE 近似。F1/F2 混合函数、源项量纲、正性限制器等
此前的问题也均已在 V2.0 评审中修复。
"""

# 本模块 2026-09-24 按职责拆成子包（项目「单文件不超 500 行」规范）。
# 下面 re-export 全部公开名与测试在用的私有名，所以全仓库
# `from autoflowcfd.core.turbulence.sst import ...` 一个字都不用改。

from .kernels import (  # noqa: F401
    _strain_vorticity_magnitude_kernel,
    compute_strain_and_vorticity_magnitude,
)
from .blending import (  # noqa: F401
    _SSTBlendingMixin,
)
from .source import (  # noqa: F401
    _SSTSourceMixin,
)
from .update import (  # noqa: F401
    _SSTUpdateMixin,
)
from .model import (  # noqa: F401
    SSTModelFR,
)

__all__ = [
    "compute_strain_and_vorticity_magnitude",
    "SSTModelFR",
]
