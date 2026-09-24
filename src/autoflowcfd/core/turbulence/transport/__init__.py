"""
AutoFlowCFD V2.0 - 湍流标量输运方程 FR 残差（完整 SST k-omega 输运）

为 SST k-omega 湍流模型补全对流和扩散输运项，使 k/omega 不再仅是逐点
ODE 源项弛豫，而是通过 FR 高阶离散真正参与空间输运。

输运方程:
    d(rho*k)/dt + div(rho*U*k) = S_k + div(Gamma_k * grad(k))
    d(rho*omega)/dt + div(rho*U*omega) = S_omega + div(Gamma_omega * grad(omega))

其中 Gamma_k = mu + sigma_k * rho * nu_t, Gamma_omega = mu + sigma_omega * rho * nu_t。

离散方法:
    - 对流项：FR 体积项（逆变标量通量散度）+ 界面上风通量校正
    - 扩散项：FR 体积项（逆变扩散通量散度）+ BR1 界面平均通量校正
    - 界面校正分配与 fr_residual_inviscid.py 使用相同的 g'/dist 映射

符号约定:
    残差 = 对流残差 + 扩散残差，返回值直接作为 dphi/dt 被调用方相加
    （fr_solver/turbulence.py::update_fields: k += dt*(Sk + transport_k)），
    与平均流残差的 RHS 约定一致（step.py: dU/dt = inv_res + visc_res）:
    - 对流残差 = -div(rho*U*phi)/det(J)（含界面上风校正）
    - 扩散残差 = +div(Gamma*grad(phi))/det(J)（含 BR1 界面校正）——
      与 viscous_flux.py 的"粘性项是 +div(G)"完全同一约定。此前这里误写为
      -div(Gamma*grad(phi))（反扩散），指数放大 2Δx 棋盘模态，把 k/omega 场
      两极分化到正性限制器的上下界（真实复现：cube_demo 全新计算 100 步内
      54% 单元贴下界 1e-12、38% 贴上界，之后冻结、残差停滞），2026-08-25 修复。
    更新: phi += dt * (transport_residual + source/rho)


## 文件分工（2026-09-24 拆包，原 1476 行）

    faces.py        标量场的面外插与面校正分配（对流/扩散共用）
    convection.py   对流残差（含去混叠体积项）与 `AFCFD_TURB_OVERINT` 解析
    diffusion.py    扩散残差（含去混叠体积项）
    omega_wall.py   omega 壁面 Dirichlet 掩码、解析目标值、每步松弛
    residual.py     顶层编排：对流 + 扩散

本 `__init__.py` re-export 全部既有公开名**以及测试在用的几个私有名**，
所以全仓库 `from autoflowcfd.core.turbulence.transport import ...` 一个字
都不用改。拆包判据是 SST 黄金轨迹逐位相同（平均流 + k + omega + nu_t
四个场的哈希），见 `ProjectFiles/V2.0/29`。
"""

from .faces import (  # noqa: F401
    ScalarConvectionGeometry,
    _distribute_correction_to_cells,
    _extrapolate_owner_only_to_faces,
    _extrapolate_scalar_to_faces,
    precompute_scalar_convection_geometry,
)
from .convection import (  # noqa: F401
    _TURB_OVERINT_CHUNK_CELLS,
    _turb_overint_ops,
    _scalar_convection_volume_overintegrated,
    compute_scalar_convection_residual,
    resolve_turb_overintegration,
)
from .diffusion import (  # noqa: F401
    _scalar_diffusion_volume_overintegrated,
    compute_scalar_diffusion_residual,
)
from .omega_wall import (  # noqa: F401
    _OMEGA_WALL_CMU,
    _OMEGA_WALL_KAPPA,
    _OMEGA_WALL_MODES,
    _compute_omega_wall_target,
    _compute_wall_dirichlet_face_mask,
    _omega_wall_formula,
    enforce_omega_wall_relaxation,
    resolve_omega_wall_mode,
)
from .residual import compute_turbulence_transport_residual  # noqa: F401

__all__ = [
    "ScalarConvectionGeometry",
    "compute_scalar_convection_residual",
    "compute_scalar_diffusion_residual",
    "compute_turbulence_transport_residual",
    "enforce_omega_wall_relaxation",
    "precompute_scalar_convection_geometry",
    "resolve_omega_wall_mode",
    "resolve_turb_overintegration",
]
