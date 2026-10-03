"""AutoFlowCFD V2.0 - 湍流输运残差的顶层编排。

把对流（`convection.py`）与扩散（`diffusion.py`）两条残差按包 `__init__.py`
记录的符号约定相加。第二个方程输运的是 `w = ln(omega)`（`sst/log_omega.py`）：
对 `w` 做对流与扩散（边界取 `ln` 值），再加变换带出的 `Gamma_w |grad w|^2`。
"""

import numpy as np
from typing import Tuple


from autoflowcfd.core.fr_operators.gradients import compute_physical_scalar_gradient
from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry

from .faces import precompute_scalar_convection_geometry
from ..sst.bounds import clip_gradient_magnitude
from ..sst.log_omega import log_omega, log_omega_gradient_source
from .convection import compute_scalar_convection_residual
from .diffusion import compute_scalar_diffusion_residual
from .omega_wall import (
    _compute_omega_wall_target,
    _compute_open_boundary_face_mask,
    _compute_wall_dirichlet_face_mask,
)


def turbulence_diffusivities(turb, rho_nu_t, mu):
    """k / ln(omega) 方程的有效扩散系数 `(Gamma_k, Gamma_w) = mu + sigma(F1) * rho * nu_t`。

    `F1` 与 `rho_nu_t = rho * nu_t` 都取源项求值（CPU `compute_source_terms` / GPU
    `compute_source_terms_gpu`）在同一组 `(k, omega)` 上刷新的模型缓存，不在这里重算
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


def prepare_convection_geometry(solver, flat_face_override=None):
    """标量对流的共享几何（只依赖平均流），见 `compute_turbulence_transport_residual`
    的 `conv_geom` 参数。"""
    Q = solver.state.Q
    flat = (flat_face_override if flat_face_override is not None
            else get_flat_face_geometry(solver.mesh, solver.ops))
    return precompute_scalar_convection_geometry(Q[:, :, 0], Q[:, :, 1:4], solver.mesh, solver.ops, flat)


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
            传进来复用（`compute_physical_scalar_gradient` 是 profile 过的热点）；
            须已按 `clip_gradient_magnitude` 裁剪。None 时在这里现算并裁剪。
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
    if solver.turb_model is None or not hasattr(solver.turb_model, 'k_field'):
        n_cells, n_sps = solver.state.U.shape[:2]
        return np.zeros((n_cells, n_sps)), np.zeros((n_cells, n_sps))

    turb = solver.turb_model
    Q = solver.state.Q
    rho = Q[:, :, 0]  # (n_cells, n_sps)
    vel = Q[:, :, 1:4]  # (n_cells, n_sps, 3)

    mu = solver.mu_molecular

    w_log = log_omega(turb.omega_field, np)
    # 梯度模长上限在求梯度处施加一次（调用方传入的梯度已裁剪，见参数文档）
    if grad_k is None:
        grad_k = clip_gradient_magnitude(compute_physical_scalar_gradient(turb.k_field, solver.mesh, solver.ops), np)
    if grad_log_omega is None:
        grad_log_omega = clip_gradient_magnitude(compute_physical_scalar_gradient(w_log, solver.mesh, solver.ops), np)

    gamma_k, gamma_w = turbulence_diffusivities(turb, rho * turb.nu_t, mu)

    # WALL 上 k=0 的 Dirichlet 掩码（真实修复，2026-08-21，见
    # transport_kernel.py::extrapolate_scalar_to_faces_kernel 文档）。
    # 数值作用点：对流项的上风 phi ghost（镜像成 -owner 强制壁面 k=0）；
    # 扩散项自 2026-08-25 校正改梯度差形式后该掩码不再有数值影响（奇镜像
    # 对梯度对称，见 compute_scalar_diffusion_residual 参数文档），仍传入
    # 以保持接口一致。
    wall_mask_k = _compute_wall_dirichlet_face_mask(solver)

    # 共享几何量（性能优化 2026-09-13，见 `ScalarConvectionGeometry` 文档）：
    # k 与 omega 的对流调用此前各自重复算了一遍与标量无关的逆变质量通量
    # 和面上 mass_flux，这里统一算一次传给两者。
    _flat_conv = (flat_face_override if flat_face_override is not None
                  else get_flat_face_geometry(solver.mesh, solver.ops))
    if conv_geom is None:
        conv_geom = prepare_convection_geometry(solver, flat_face_override)

    # 开放边界（流入/流出）掩码：k/omega 的来流条件（见
    # compute_scalar_convection_residual 的 open_boundary_face 参数文档）
    open_mask = _compute_open_boundary_face_mask(solver, _flat_conv)

    # 计算 k 的对流 + 扩散残差
    conv_k = compute_scalar_convection_residual(
        turb.k_field, rho, vel, solver.mesh, solver.ops, wall_dirichlet_zero_face=wall_mask_k,
        flat_face_override=flat_face_override, conv_geom=conv_geom,
        open_boundary_face=open_mask, freestream_value=float(turb.k_inf),
    )
    diff_k = compute_scalar_diffusion_residual(
        turb.k_field, gamma_k, solver.mesh, solver.ops, wall_dirichlet_zero_face=wall_mask_k,
        flat_face_override=flat_face_override,
    )
    with np.errstate(over='ignore', invalid='ignore'):
        dk_dt_transport = (conv_k + diff_k) / np.maximum(rho, 1e-10)

    # WALL 上 omega 解析壁面值的 Dirichlet 目标（真实修复，V2.0 专家组
    # 盲审发现，2026-08-28，见 _compute_omega_wall_target 文档）：此前
    # omega 恒用 Neumann（零梯度）默认，是明确记录过的已知限制——现在
    # 用 Wilcox 解析式 60*nu/(beta1*d1^2) 代替。d1 需要 solver.wall_
    # distance 已经计算好（SST/DDES/WMLES 初始化时必然如此，见
    # fr_solver/turbulence.py），否则 _compute_omega_wall_target 里的
    # np.min(solver.wall_distance[...]) 会直接因 wall_distance 为 None
    # 报错——这是有意的（没有壁面距离场，压根不该假装能算出解析壁面值）。
    omega_wall_value_face, has_omega_wall = _compute_omega_wall_target(
        solver, wall_mask_k, mu, rho, flat_face_override=flat_face_override,
    )
    # w = ln(omega) 的边界值：壁面目标取对数（没有目标的面该值不被读取），来流取 ln(omega_inf）
    log_wall_value_face = log_omega(omega_wall_value_face, np)

    # 计算 w 的对流 + 扩散残差，另加变换带出的 Gamma_w |grad w|^2（sst/log_omega.py）
    conv_w = compute_scalar_convection_residual(
        w_log, rho, vel, solver.mesh, solver.ops,
        wall_dirichlet_value_face=log_wall_value_face, has_wall_dirichlet_value=has_omega_wall,
        flat_face_override=flat_face_override, conv_geom=conv_geom,
        open_boundary_face=open_mask, freestream_value=float(np.log(turb.omega_inf)),
    )
    diff_w = compute_scalar_diffusion_residual(
        w_log, gamma_w, solver.mesh, solver.ops,
        wall_dirichlet_value_face=log_wall_value_face, has_wall_dirichlet_value=has_omega_wall,
        flat_face_override=flat_face_override,
    )
    with np.errstate(over='ignore', invalid='ignore'):
        dw_dt_transport = (conv_w + diff_w + log_omega_gradient_source(gamma_w, grad_log_omega, np)) \
            / np.maximum(rho, 1e-10)

    # 机制3（按 (cell,SP,变量) 粒度检测残差量级异常并清零）**已于
    # 2026-09-19 整体删除**，本函数曾经在这里调用它。删除依据是真实网格
    # （plate_demo_volume_les，179,237 单元，P1 + LES）上的消融对照：
    # 机制3 触发 15 次、清零 729 个槽位，而残差轨迹与关掉它的那条只差
    # ~1e-10 相对量，两条都在 146~148 步发散。完整记录见
    # `fr_residual/inviscid.py` 同一处。
    #
    # 所以本函数现在只保留下面那道 isfinite 归零作为唯一防线 —— 那是
    # 真正必要的（退化网格上梯度/Jacobian 会产生非有限值，非有限值一旦
    # 进入 SST 的场更新就不可恢复），而"明显偏大但还是有限值"这一类
    # 由机制3 处理的情形，实测它处理与不处理没有可观测差别。
    # NaN/Inf 隔离（最后一道防线）：退化网格上梯度/Jacobian 可能产生非
    # 有限值，归零后由 SST.update_fields 的二次防护和 positivity
    # limiter 接管
    dk_dt_transport = np.where(np.isfinite(dk_dt_transport), dk_dt_transport, 0.0)
    dw_dt_transport = np.where(np.isfinite(dw_dt_transport), dw_dt_transport, 0.0)

    return dk_dt_transport, dw_dt_transport
