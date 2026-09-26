"""AutoFlowCFD V2.0 - k-omega 隐式步的解析单元块 Jacobian（背景与组成见 `assemble.py`）。

    pointwise.py    源项与有效扩散系数的逐点导数（对模型函数差分）
    cell_blocks.py  源项 + 对流/扩散体积项
    faces.py        对流/扩散界面项
    assemble.py     组装、取负除 rho、omega 壁面强约束行、耦合块
    backend.py      单机 / 分布式装配器
"""

from .assemble import TurbulenceLinearization, assemble_turbulence_blocks
from .backend import TurbulenceBlockAssembler, turbulence_linearization

__all__ = ["TurbulenceLinearization", "assemble_turbulence_blocks", "TurbulenceBlockAssembler",
           "turbulence_linearization"]
