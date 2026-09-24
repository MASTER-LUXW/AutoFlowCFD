"""
AutoFlowCFD V2.0 - FR 粘性物理通量与界面耦合 (Tier-0 重建版, 对应 S-03)

牛顿流体应力张量 + 傅里叶热传导的物理通量函数，以及基于真实单元-面连接
关系的界面耦合（BR1 格式：界面处的原始变量与梯度取相邻单元外插值的平均，
这是规范文档 3_系统实现方式-算法流程.md §2.3 明确允许的做法——
"在单元界面处，Θ̂ 取左右单元的平均值（或加权平均值）"）。

取代旧版本 fr_residual_viscous.py 中：
1. 从未被满足的 `hasattr(mesh,'face_connectivity')` 分支（死代码，从未执行）
2. 唯一实际执行的 fallback——用**单元内部梯度模长**冒充界面跳跃
   （`jump_estimate = h_local * |grad_u|`），这不是任何邻居信息，纯粹是
   同一个单元自己的局部量，物理上不构成"界面耦合"
3. 体积项用 D_3d 直接当物理导数使用（缺少度量项变换，见
   core/fr_gradients.py 文档），对本代码库的每个曲边/坍缩坐标单元都是
   错误导数

正确性通过「均匀常数流场（零梯度）粘性残差应严格为零」验证——牛顿粘性
应力和热传导对常数场恒为零，这是比自由流场保持性更基础但同样严格的
判据，见 tests/unit/test_fr_residual_viscous.py。

问题单元保护：`compute_physical_gradient` 用 `inv_jac`（近似正比于
adj(J)/det(J)）把参考空间导数转成物理梯度，坍缩坐标退化 SP 处 det(J)
极小，`inv_jac` 对应地极大——梯度本身在这类点先被放大一次，随后
`residual = div_comp/det(J)`（体积项）与 `correction/det(J)`（界面项）
在同一个退化 det(J) 上再放大一次，是比无粘残差更严重的*双重*放大（真实
Couette 合成算例复现：粘性残差 3 步内从 4e-2 量级放大到 1.16e7）。

此前曾仿照无粘那边"先用 det(J)/法向失配几何量预判、按整个单元降阶"的
机制1/2 实现过一版保护，但发现该判据有两个真实缺陷（见
fr_troubled_cell.py 模块文档"机制3"一节）：(1) 绝对 det(J) 阈值是照一个
特定网格的绝对尺度标定的，换个尺度就可能失效（真实复现：det(J) 比阈值
高 828 倍仍被放大到灾难量级）；(2) 按整个单元降阶，会在网格所有单元
恰好同一绝对尺度、以至于机制1对*每个*单元都命中时（合成验证网格常见），
把全网格的粘性物理都拍平成零梯度，等于关掉了粘性扩散本身。

此后改用过机制3（按 (cell,SP,变量) 粒度检测残差量级异常并清零），
**机制3 也已于 2026-09-19 删除** —— 真实网格消融对照证明它触发了但只把
残差轨迹改变 ~1e-10 相对量、且不改变发散这个结局，完整依据见
`fr_residual/inviscid.py` 里那段记录。

所以本函数现在**不对残差做任何异常抑制**：退化单元的对策是网格质量门
（项目记忆 `industry_practice_degenerate_cell_gcl`），不是运行期限制器。


## 文件分工(2026-09-24 拆包, 原 560 行)

    pointwise.py         逐点量：温度、粘性物理通量
    overintegration.py   粘性体积项过积分（去混叠）开关与细点路径
    residual.py          粘性残差顶层：体积项 + 界面项 + IP 罚项

本 `__init__.py` re-export 全部既有名, 所以全仓库导入一字不改。
"""

from .constants import (  # noqa: F401
    GAMMA,
    R_AIR,
)
from .pointwise import (  # noqa: F401
    compute_temperature,
    viscous_physical_flux,
)
from .overintegration import (  # noqa: F401
    _viscous_volume_overintegrated,
    resolve_viscous_overintegration,
)
from .residual import (  # noqa: F401
    compute_viscous_residual_fr,
)

__all__ = [
    "GAMMA",
    "R_AIR",
    "compute_temperature",
    "compute_viscous_residual_fr",
    "resolve_viscous_overintegration",
    "viscous_physical_flux",
]
