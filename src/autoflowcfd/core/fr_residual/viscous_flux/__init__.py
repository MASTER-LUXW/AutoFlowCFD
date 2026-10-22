"""
AutoFlowCFD V2.0 - FR 粘性物理通量与界面耦合（S-03）。

牛顿流体应力张量 + 傅里叶热传导的物理通量函数，以及基于真实单元-面连接关系的界面耦合（BR1 格式：界面处的
原始变量与梯度取相邻单元外插值的平均，规范文档 3_系统实现方式-算法流程.md §2.3 允许的做法——"在单元界面处，
Θ̂ 取左右单元的平均值（或加权平均值）"），外加内部面与边界面的 IP 罚项（强制性，见
`core/fr_operators/flux_kernels/viscous.py::viscous_ip_penalty_tilde`）。

正确性判据：均匀常数流场（零梯度）粘性残差严格为零（牛顿粘性应力与热传导对常数场恒为零），见
`tests/unit/test_fr_viscous_flux.py`；离散扩散算子的强制性见 `tests/unit/test_viscous_operator_coercivity.py`。

本函数不对残差做任何异常抑制：退化单元的对策是求解前的网格质量门（`grid/validation/solve_gate.py`）。

## 文件分工
## 文件分工

    pointwise.py         逐点量：温度、粘性物理通量
    overintegration.py   粘性体积项过积分（去混叠）开关与细点路径
    residual.py          粘性残差顶层：体积项 + 界面项 + IP 罚项

本 `__init__.py` re-export 各子模块的公开名。
"""

from .pointwise import (  # noqa: F401
    compute_temperature,
)
from .overintegration import (  # noqa: F401
    viscous_volume_term,
)
from .residual import (  # noqa: F401
    compute_viscous_residual_fr,
)

__all__ = [
    "compute_temperature",
    "compute_viscous_residual_fr",
    "viscous_volume_term",
]
