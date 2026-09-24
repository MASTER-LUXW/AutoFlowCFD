"""欧拉/粘性物理通量的逐点标量 numba 版本 (性能优化配套)。

`core/fr_residual_inviscid.py::euler_physical_flux` 和
`core/fr_viscous_flux.py::viscous_physical_flux` 是向量化 numpy 实现
（`np.stack`/`np.zeros`/`np.swapaxes`/`np.eye`/`einsum`），在
`fr_residual_inviscid_kernel.py`/`fr_viscous_flux_kernel.py` 的逐点
numba `@njit` 主循环里会被反复调用（每个 Flux Point 调一次）——numba
nopython 模式不支持 `einsum`/`swapaxes`，所以不能直接复用，必须重新写
逐点标量版。这是整个性能优化里除了 AUSM+up 之外风险最高的新代码
（尤其 `viscous_physical_flux_point` 涉及真实物理：Boussinesq 假设下
`mu_total=mu+mu_t` 统一处理应力张量、`k_cond` 混合分子/湍流普朗特数、
`work=vel·tau` 粘性功）——因此单独在这里、用随机输入与现有向量化实现
逐位对比验证（见 `tests/unit/test_fr_flux_kernels_pointwise.py`），不
把这一步的验证并入端到端残差对比，出问题能立刻定位到这里而不是别处。

两个函数的公式必须与 `fr_residual_inviscid.py::euler_physical_flux`/
`fr_viscous_flux.py::viscous_physical_flux` 严格一致，改动前者时必须
同步检查后者是否也要改。


## 文件分工(2026-09-24 拆包, 原 615 行)

    euler.py             欧拉物理通量与熵稳定体积散度
    viscous.py           粘性物理通量、边界梯度镜像与 IP 罚项

本 `__init__.py` re-export 全部既有名, 所以全仓库导入一字不改。
"""

from .constants import (  # noqa: F401
    CP_AIR,
    GAMMA,
    R_AIR,
    VISCOUS_IP_C_BASE,
)
from .euler import (  # noqa: F401
    _log_mean_point,
    chandrashekar_flux_point,
    entropy_stable_volume_divergence_batch,
    euler_physical_flux_batch,
    euler_physical_flux_point,
)
from .viscous import (  # noqa: F401
    mirror_normal_component,
    resolve_viscous_ip_constant,
    viscous_ip_penalty_tilde,
    viscous_physical_flux_batch,
    viscous_physical_flux_point,
)

__all__ = [
    "CP_AIR",
    "GAMMA",
    "R_AIR",
    "VISCOUS_IP_C_BASE",
    "chandrashekar_flux_point",
    "entropy_stable_volume_divergence_batch",
    "euler_physical_flux_batch",
    "euler_physical_flux_point",
    "mirror_normal_component",
    "resolve_viscous_ip_constant",
    "viscous_ip_penalty_tilde",
    "viscous_physical_flux_batch",
    "viscous_physical_flux_point",
]
