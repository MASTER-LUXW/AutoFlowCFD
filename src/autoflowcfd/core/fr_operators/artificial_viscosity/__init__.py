"""
AutoFlowCFD V2.0 - Persson-Peraire 模态传感器 + 局部人工粘性（可选能力）

背景（见 ProjectFiles/V2.0/7_重大问题修复-求解稳定性.md 五、节）：本项目
现有的两层稳定性机制——模态滤波器（fr/modal_filter.py）与残差量级异常
检测（core/fr_operators/troubled_cell.py 机制3）——在 2026-08-29 的
cube_demo 真实网格调查中被证明存在结构性张力：模态滤波器对"中间"模态
的压制是真实的稳定性刚需，放松它会重新引入灾难性失稳；机制3无论是
局部（同单元）还是全局（全网格）参照，都无法可靠区分"合法的强局部
物理"（边界层、驻点、尖角）与"真正的欠分辨率伪影"，因为两者都表现为
"残差量级大"。

行业调研（同一份文档）：开源 FR 参照实现 PyFR 的稳定性工具箱第三层是
基于 Persson & Peraire (2006, "Sub-Cell Shock Capturing for Discontinuous
Galerkin Methods") 传感器的局部人工粘性——这个传感器看的不是残差量级，
而是**解本身的模态谱衰减速率**：把节点值变换到模态系数空间，比较"完整
表示"与"截断掉最高阶模态后的表示"之间的能量占比。光滑函数的最高阶
模态系数按 O(1/N^4) 衰减；如果实测占比明显超过这个理论预期，说明该
单元的解含有这个多项式阶数无法真正解析的高频内容（混叠/欠分辨率的
直接证据），与残差本身是大是小无关——边界层里残差可以很大但完全光滑
（多项式能精确表示），这种单元传感器不会触发；反之一个残差看起来不大
但解本身在振荡的单元，传感器会触发。

本模块是一个独立、默认关闭（opt-in）的新增能力，不修改任何现有稳定性
机制的行为，不影响任何未显式启用它的现有测试/求解路径。启用后，对触发
传感器的单元，把一个局部人工粘性系数叠加进现有粘性残差已经在消费的
`mu_t_field` 通道（core/fr_residual/viscous_flux.py 的 `mu_t_field`
参数）——复用已经过充分验证的 BR1 面耦合粘性通量组装机制，不新建一条
独立的扩散残差路径。

**原先的范围限制已补齐（2026-09-14）**：Persson & Peraire 原始方法对
*全部*守恒变量（含质量/连续性方程）叠加人工扩散项；本实现最初只通过
动量/能量方程既有的粘性应力/热传导通道施加，不直接扩散密度
（`viscous_physical_flux` 的质量分量 G[...,0] 恒为 0），当时把这一点
记作"许多实际 DG/FR 实现采用的简化"。用户明确指出本项目不接受简化，
现已补上缺的那一项：`FRSolver._artificial_mass_diffusion_residual`
把 `+div(epsilon * grad(rho))` 加进连续性方程。

实现选择（为什么不改粘性通量本身）：AV 默认关闭，没有理由为它给所有
运行的 `viscous_physical_flux_batch` 热路径增加参数与分支。而
`div(eps*grad(rho))` 本身就是一个标量扩散算子，直接复用湍流输运已经
验证过的 BR1 面耦合标量扩散装配
（`turbulence/transport.py::compute_scalar_diffusion_residual`，它返回的
就是 `+div(Gamma*grad(phi))`，与 dU/dt 的符号约定一致）。AV 关闭时这段
完全不执行，零开销。

性质：散度形式 -> 严格守恒；均匀流场下 grad(rho)=0 -> 该项恒为 0，
不破坏自由流场保持性。两条都有测试钉住
（tests/unit/test_artificial_viscosity_mass_diffusion.py）。

公式来源：Persson & Peraire (2006) 原始传感器公式 + mirgecom
（Illinois/DOE 现役生产级 DG 代码）文档给出的精确数值实现细节
（s_e=log10(S_e)，分段 sine 过渡，kappa 默认值 1.0）——本模块的具体
数值实现直接对照 mirgecom 文档核实过，不是凭记忆重新推导。


本模块 2026-09-20 从单文件（619 行）拆成子包（项目"单文件不超 500 行"
规范）：

    sensor_operators.py   三条基各自的传感器算子构造 + 缓存 + 设备搬运
    sensor.py             三个 `compute_persson_peraire_sensor*` 求值函数
    viscosity.py          斜坡映射 / troubled-cell 掩码 / 完整流水线

这里只 re-export 公开名，全仓库既有的
`from ...artificial_viscosity import X` 不用改。
"""

from .sensor import (  # noqa: F401
    compute_persson_peraire_sensor,
    compute_persson_peraire_sensor_native_prism,
    compute_persson_peraire_sensor_native_tet,
)
from .sensor_operators import (  # noqa: F401
    DEFAULT_SENSOR_VAR_INDEX,
    SENSOR_KAPPA,
)
from .viscosity import (  # noqa: F401
    compute_artificial_viscosity_ramp,
    compute_persson_peraire_artificial_viscosity,
    compute_troubled_cell_mask,
)

__all__ = [
    "DEFAULT_SENSOR_VAR_INDEX",
    "SENSOR_KAPPA",
    "compute_artificial_viscosity_ramp",
    "compute_persson_peraire_artificial_viscosity",
    "compute_persson_peraire_sensor",
    "compute_persson_peraire_sensor_native_prism",
    "compute_persson_peraire_sensor_native_tet",
    "compute_troubled_cell_mask",
]
