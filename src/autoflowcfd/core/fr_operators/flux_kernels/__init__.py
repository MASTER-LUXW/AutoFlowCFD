"""欧拉 / 粘性物理通量的逐点 numba 核：残差体积项、界面项与隐式 Jacobian 共用（GPU 版
`core/gpu/residual/gpu_flux.py` 公式相同）。

逐点核与测试里独立写出的参考公式逐位对比（`tests/unit/test_fr_flux_kernels_pointwise.py`）。2026-10-22（I-25）
以前 `core/fr_residual/` 另有一份向量化 numpy 实现（`inviscid.py::euler_physical_flux`、
`viscous_flux/pointwise.py::viscous_physical_flux`）作为"公式来源"，src 里已无调用方，删除，参考公式移入该测试。

## 文件分工

    constants.py         粘性 IP 罚项常数
    euler.py             欧拉物理通量（逐点 / 批量）
    viscous.py           粘性物理通量（逐点 / 批量 / 有效涡粘）、光滑非负截断与 IP 罚项
    viscous_bc.py        粘性边界种类与边界"另一侧"梯度镜像
"""

from .constants import (  # noqa: F401
    VISCOUS_IP_C_BASE,
)
from .euler import (  # noqa: F401
    euler_physical_flux_batch,
    euler_physical_flux_point,
)
from .viscous import (  # noqa: F401
    effective_eddy_viscosity,
    effective_eddy_viscosity_array,
    resolve_viscous_ip_constant,
    smooth_nonnegative,
    smooth_nonnegative_array,
    smooth_nonnegative_derivative,
    viscous_physical_flux_point_effective,
    viscous_ip_penalty_tilde,
    viscous_physical_flux_batch,
    viscous_physical_flux_point,
)
from .viscous_bc import (  # noqa: F401
    VBC_DIRICHLET,
    VBC_INTERIOR,
    VBC_MIRROR,
    VBC_NEUMANN,
    VBC_NOSLIP_WALL,
    boundary_other_gradients,
    mirror_normal_component,
    mirror_normal_derivative,
    mirror_velocity_gradient,
)

__all__ = [
    "VBC_DIRICHLET",
    "VBC_INTERIOR",
    "VBC_MIRROR",
    "VBC_NEUMANN",
    "VBC_NOSLIP_WALL",
    "VISCOUS_IP_C_BASE",
    "boundary_other_gradients",
    "euler_physical_flux_batch",
    "euler_physical_flux_point",
    "mirror_normal_component",
    "mirror_normal_derivative",
    "mirror_velocity_gradient",
    "effective_eddy_viscosity",
    "effective_eddy_viscosity_array",
    "resolve_viscous_ip_constant",
    "smooth_nonnegative",
    "smooth_nonnegative_array",
    "smooth_nonnegative_derivative",
    "viscous_physical_flux_point_effective",
    "viscous_ip_penalty_tilde",
    "viscous_physical_flux_batch",
    "viscous_physical_flux_point",
]
