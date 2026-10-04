"""AutoFlowCFD V2.0 - SA-neg 湍流模型（Allmaras, Johnson & Spalart 2012）。

    constants.py      模型常数、来流 chi 的反解
    pointwise.py      逐点函数（源项两支、涡粘、扩散系数、梯度项；numpy / cupy 共用的唯一定义）
    model.py          模型对象 `SAModel`（`turbulence/transported.py` 接口）
    rates.py          速率求值（全部后端共用，后端注入标量输运原语）
    linearization.py  解析单元块 Jacobian 的逐点求值器与壁面 Dirichlet 规格

为什么引入它：高阶 DG 下 SST 的 C0 折点（涡粘限制器、k_bar、产生项上限、F1/F2）与 Newton 的
相互作用使 P2/P3 无法快速收敛（ProjectFiles/V2.0/38 第二十四节）；SA-neg 是为高阶离散设计的
负值鲁棒变体，是高阶 DG RANS 的主流选择。
"""

from .model import SAModel
from .rates import evaluate_sa_rates

__all__ = ["SAModel", "evaluate_sa_rates"]
