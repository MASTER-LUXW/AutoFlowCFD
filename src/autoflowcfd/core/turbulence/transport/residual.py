"""AutoFlowCFD V2.0 - 湍流输运残差的顶层编排。

把对流（`convection.py`）与扩散（`diffusion.py`）两条残差按包 `__init__.py`
记录的符号约定相加。第二个方程输运的是 `w = ln(omega)`（`sst/log_omega.py`）：
对 `w` 做对流与扩散（边界取 `ln` 值），再加变换带出的 `Gamma_w |grad w|^2`。
"""

from typing import Tuple

import numpy as np

from autoflowcfd.core.turbulence.nonfinite import replace_nonfinite

from ..sst.bounds import k_wall_zero_mask
from ..sst.log_omega import log_omega, log_omega_gradient_source
from .omega_wall import _compute_omega_wall_target
from .primitives import CpuScalarTransport


def turbulence_diffusivities(turb, rho_nu_t, mu):
    """k / ln(omega) 方程的有效扩散系数 `(Gamma_k, Gamma_w) = mu + sigma(F1) * rho * nu_t`。

    `F1` 与 `rho_nu_t = rho * nu_t` 都取源项求值（`compute_source_terms`，CPU 与
    GPU 同一份）在同一组 `(k, omega)` 上刷新的模型缓存，不在这里重算
    （此前另算一遍 F1，交叉扩散的乘法顺序不同、两份 F1 只在舍入上一致）。调用方须先
    求源项，输运残差（`fr_solver/turbulence/source.py` 等四个后端入口）与湍流解析
    Jacobian 的逐点求值器（`jacobian/pointwise.py`）都是这个顺序。逐点运算，numpy /
    cupy 共用。
    """
    F1 = turb._last_F1
    if F1 is None:
        raise RuntimeError("有效扩散系数需要源项求值缓存的 F1：须先在同一组 (k, omega) 上求源项")
    sigma_k = F1 * turb.sigma_k1 + (1.0 - F1) * turb.sigma_k2
    sigma_w = F1 * turb.sigma_w1 + (1.0 - F1) * turb.sigma_w2
    return mu + sigma_k * rho_nu_t, mu + sigma_w * rho_nu_t


def compute_turbulence_transport_residual(
    solver,
    grad_k: np.ndarray = None,
    grad_log_omega: np.ndarray = None,
    flat_face_override=None,
    conv_geom=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算 k 与 `w = ln(omega)` 的完整输运残差（对流 + 扩散）。

    入口函数：从 solver 获取流场和湍流场信息，分别计算 k 和 w 的对流+扩散残差
    （w 另加 `Gamma_w |grad w|^2`），返回 dk/dt 和 dw/dt 的输运贡献（已除以密度）。
    有效扩散系数读模型上源项求值刷新的 `nu_t` 与 `F1`（`turbulence_diffusivities`），
    调用方须先在同一组 `(k, omega)` 上求源项。

    Args:
        solver: FRSolver 实例（需要已初始化 SST/DDES 湍流模型）
        grad_k, grad_log_omega: 可选，调用方（`fr_solver/turbulence/source.py::
            compute_turbulence_source`）为源项算过的同一份 k、ln(omega) 物理梯度，
            传进来复用（`compute_physical_scalar_gradient` 是 profile 过的热点）。None 时在这里现算。
        flat_face_override: 显式传入时优先使用，透传给内部四次
            `compute_scalar_convection_residual`/`compute_scalar_
            diffusion_residual` 调用（2026-09-02 分布式湍流移植新增，
            见这两个函数同名参数文档）——分布式路径下 `solver` 是
            `DistributedTurbulenceSolverAdapter`（`solver.mesh` 是
            `DistributedMeshAdapter`），必须传入 `dist_fc.base_flat`，
            否则会尝试从压缩索引空间的适配器重新构建全局面几何。

        conv_geom: 可选，`prepare_convection_geometry` 的结果。它只依赖平均流
            （rho、速度），隐式 k-omega 在一个 Newton 步内平均流冻结，由调用方
            在步起点算一次、每次求值传入（每次省约 0.18 s，plate_demo P1）；
            None 时在这里现算（显式路径每步只求值一次）。
    Returns:
        (dk_dt_transport, dw_dt_transport): 各自 (n_cells, n_sps)，
        输运项对 dk/dt 和 d(ln omega)/dt 的贡献
    """
    from ..sst import SSTModelFR

    turb = solver.turb_model
    if not isinstance(turb, SSTModelFR):
        # 2026-10-19（I-62）以前非 SST 模型静默返回全零输运（湍流场只剩逐点源项弛豫，不报错）
        raise TypeError(f"k/ln(omega) 输运残差只对 SST 族模型定义，收到 {type(turb).__name__}")
    mu = solver.mu_molecular
    transport = CpuScalarTransport(solver, conv_geom=conv_geom, flat_face_override=flat_face_override)
    rho = transport.rho

    w_log = log_omega(turb.omega_field, np)
    if grad_k is None:
        grad_k = transport.gradient(turb.k_field)
    if grad_log_omega is None:
        grad_log_omega = transport.gradient(w_log)

    gamma_k, gamma_w = turbulence_diffusivities(turb, rho * turb.nu_t, mu)

    # k 的壁面：resolved 在物理无滑移壁面 Dirichlet 0，auto 为零通量（`sst/bounds.py::k_wall_zero_mask`）。数值作用点
    # 是对流项的上风 ghost（镜像成 -owner 强制壁面 k=0）；扩散项为梯度差形式，奇镜像对梯度对称、掩码不改变它。
    wall_mask_k = k_wall_zero_mask(transport.wall_zero, turb.wall_treatment, np)
    conv_k = transport.convection(turb.k_field, turb.k_inf, wall_zero_face=wall_mask_k)
    diff_k = transport.diffusion(turb.k_field, gamma_k, wall_zero_face=wall_mask_k)
    with np.errstate(over='ignore', invalid='ignore'):
        dk_dt_transport = (conv_k + diff_k) / np.maximum(rho, 1e-10)

    # omega 的壁面：解析壁面值作 Dirichlet 目标（`omega_wall.py::_compute_omega_wall_target`；需要壁距已设定，
    # 没有壁距场时直接报错——不假装能算出解析壁面值），两种近壁处理方式都施加；w = ln(omega) 取它的对数
    omega_wall_value_face, has_omega_wall = _compute_omega_wall_target(
        solver, transport.wall_zero, mu, rho, flat_face_override=flat_face_override,
    )
    wall_w = (log_omega(omega_wall_value_face, np), has_omega_wall)
    conv_w = transport.convection(w_log, float(np.log(turb.omega_inf)), wall_value=wall_w)
    diff_w = transport.diffusion(w_log, gamma_w, wall_value=wall_w)
    with np.errstate(over='ignore', invalid='ignore'):
        dw_dt_transport = (conv_w + diff_w + log_omega_gradient_source(gamma_w, grad_log_omega, np)) \
            / np.maximum(rho, 1e-10)

    # 非有限值隔离：退化网格上梯度/Jacobian 可能产生非有限值，一旦进入场更新就不可恢复——替换为零并计数告警
    # （`turbulence/nonfinite.py`；2026-10-19（I-48）以前静默归零）
    dk_dt_transport = replace_nonfinite(turb, dk_dt_transport, "SST k 输运", np)
    dw_dt_transport = replace_nonfinite(turb, dw_dt_transport, "SST ln(omega) 输运", np)

    return dk_dt_transport, dw_dt_transport
