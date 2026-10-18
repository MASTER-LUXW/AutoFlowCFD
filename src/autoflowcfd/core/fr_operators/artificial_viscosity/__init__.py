"""
AutoFlowCFD V2.0 - 问题单元局部人工粘性（可选能力，默认关闭）。

人工粘性系数由**熵残差判据**给出（`entropy_viscosity.py`：单元局部、P1 起可用；为什么不是 Persson-Peraire 模态
衰减判据或 BJ 越界比，见那边模块文档的实测表）。施加形式是全部守恒变量的拉普拉斯 `div(nu grad U_k)`（冻结
`nu`），复用湍流输运已验证的 BR1 面耦合标量扩散装配：散度形式严格守恒，均匀流场下恒为零。

    entropy_viscosity.py  熵残差判据、人工扩散系数与守恒变量拉普拉斯残差

2026-10-18：Persson-Peraire 判据（`sensor.py`、`sensor_operators.py`、`viscosity.py`）随模态滤波子系统一起删除——
人工粘性 2026-10-01 改用熵残差判据后，它只剩滤波门控一个用途。
"""

from .entropy_viscosity import (  # noqa: F401
    ENTROPY_SENSOR_FULL,
    ENTROPY_SENSOR_LOW,
    artificial_diffusion_residual,
    compute_artificial_diffusivity,
    compute_entropy_sensor,
    entropy_sensor_ramp,
)

__all__ = [
    "ENTROPY_SENSOR_FULL",
    "ENTROPY_SENSOR_LOW",
    "artificial_diffusion_residual",
    "compute_artificial_diffusivity",
    "compute_entropy_sensor",
    "entropy_sensor_ramp",
]
