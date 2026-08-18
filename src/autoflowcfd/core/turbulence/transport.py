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
    残差 = 对流残差 + 扩散残差，其中:
    - 对流残差 ≈ -div(rho*U*phi)/det(J)（含界面上风校正）
    - 扩散残差 ≈ -div(Gamma*grad(phi))/det(J)（含 BR1 界面校正）
    更新: phi += dt * (transport_residual + source/rho)
"""

import os
import numpy as np
from typing import Optional, Tuple

import numba

from autoflowcfd.core.fr_operators.gradients import compute_physical_scalar_gradient
from autoflowcfd.core.fr_operators.volume_contract import contract_shared_operator_2axis
from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.turbulence.transport_kernel import (
    extrapolate_scalar_to_faces_kernel,
    distribute_corrections_to_cells_kernel,
    distribute_corrections_to_cells_kernel_colored,
)


def _extrapolate_scalar_to_faces(scalar_sps, flat, ops, mesh):
    """将 SPs 上的标量场外插到所有面的通量点（numba kernel 版本）。

    使用 turbulence_transport_kernel.py 的 numba 编译函数替代纯 Python 循环。
    owner 侧用 boundary_extrap 矩阵，neighbor 侧用 neighbor_sources 矩阵。

    Returns:
        phi_owner_fp: (n_faces, n_fp)
        phi_neighbor_fp: (n_faces, n_fp)
    """
    return extrapolate_scalar_to_faces_kernel(
        scalar_sps, flat.boundary_extrap,
        flat.neighbor_src0_cell, flat.neighbor_src0_mat,
        flat.neighbor_src1_idx, flat.neighbor_src1_cell, flat.neighbor_src1_mat,
        flat.owner_cell, flat.owner_axis, flat.owner_side,
        flat.n_prism, flat.n_faces, flat.n_fp, flat.n_sps,
    )


def _distribute_correction_to_cells(correction_fp, flat, ops, mesh):
    """将面通量点上的校正量分配回 SPs 残差（numba kernel 版本）。

    使用 turbulence_transport_kernel.py 的 kernel 替代纯 Python for f in range(n_faces) 循环。
    默认使用图着色方案（同色面无冲突，直接写入共享 buffer），
    通过环境变量 AFCFD_USE_COLORING 可回退到 per-thread buffer 方案。

    Returns:
        correction_sps: (n_cells, n_sps)
    """
    n_cells = mesh.n_cells
    n_sps = flat.n_sps
    det_jacs = mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)

    use_coloring = os.environ.get("AFCFD_USE_COLORING", "1") == "1"

    if use_coloring:
        correction_sps = np.zeros((n_cells, n_sps))
        for c in range(flat.n_colors):
            face_indices = flat.color_face_indices[c]
            if len(face_indices) == 0:
                continue
            distribute_corrections_to_cells_kernel_colored(
                correction_fp,
                flat.owner_cell, flat.neighbor_cell,
                flat.owner_axis, flat.owner_side,
                flat.neighbor_axis, flat.neighbor_side,
                det_jacs,
                flat.g_left, flat.g_right,
                flat.dist_fp_of_sp, flat.dist_axis_coord_of_sp,
                n_cells, n_sps,
                face_indices,
                correction_sps,
            )
        return correction_sps
    else:
        n_threads = numba.get_num_threads()
        return distribute_corrections_to_cells_kernel(
            correction_fp,
            flat.owner_cell, flat.neighbor_cell,
            flat.owner_axis, flat.owner_side,
            flat.neighbor_axis, flat.neighbor_side,
            det_jacs,
            flat.g_left, flat.g_right,
            flat.dist_fp_of_sp, flat.dist_axis_coord_of_sp,
            n_cells, n_sps, flat.n_faces,
            n_threads,
        )


def compute_scalar_convection_residual(
    scalar_field: np.ndarray,
    rho: np.ndarray,
    velocity: np.ndarray,
    mesh,
    ops,
) -> np.ndarray:
    """计算标量对流 FR 残差（体积项 + 界面上风校正）。

    对流方程: d(rho*phi)/dt + div(rho*U*phi) = 0
    残差 = -div(rho*U*phi)/det(J) + interface_correction/det(J)

    Args:
        scalar_field: (n_cells, n_sps) 标量场（k 或 omega）
        rho: (n_cells, n_sps) 密度
        velocity: (n_cells, n_sps, 3) 速度
        mesh: HighOrderMesh
        ops: FROperators

    Returns:
        residual: (n_cells, n_sps) 对流残差（dphi/dt 量纲，已除以 rho 前的
                  原始残差，调用方需自行除以 rho）
    """
    n_cells = mesh.n_cells
    n_sps = mesh.n_sps_per_cell
    n_prism = mesh.n_prism_cells

    det_jacs = mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)
    inv_jacs = mesh.jacobians["inv_jacs"].reshape(n_cells, n_sps, 3, 3)
    adj_j = det_jacs[..., None, None] * inv_jacs  # (n_cells, n_sps, 3, 3)

    # === 体积项 ===
    # 标量通量: F_phys[...,i] = rho * u_i * phi
    rho_u_phi = rho[..., None] * velocity * scalar_field[..., None]  # (n_cells, n_sps, 3)
    # 逆变通量: F_tilde[...,m] = adj(J)[m,i] * F_phys[i]
    F_tilde = np.matmul(adj_j, rho_u_phi[..., None]).squeeze(-1)  # (n_cells, n_sps, 3)
    # 散度: div(F_tilde) = sum_m D_3d[m,:] . F_tilde[...,m]
    div_F = np.zeros((n_cells, n_sps))
    if n_prism > 0:
        for m in range(3):
            div_F[:n_prism] += np.tensordot(F_tilde[:n_prism, :, m], ops.D_3d_prism[:, :, m], axes=([1], [1]))
    if n_cells > n_prism:
        for m in range(3):
            div_F[n_prism:] += np.tensordot(F_tilde[n_prism:, :, m], ops.D_3d_tet[:, :, m], axes=([1], [1]))

    residual = -div_F / det_jacs  # 体积项对流残差

    # === 界面项（上风校正）===
    flat = get_flat_face_geometry(mesh, ops)
    n_fp = flat.n_fp

    # 外插 rho, velocity, scalar 到面通量点
    rho_owner_fp, rho_neighbor_fp = _extrapolate_scalar_to_faces(rho, flat, ops, mesh)
    vel_owner_fp = np.zeros((flat.n_faces, n_fp, 3))
    vel_neighbor_fp = np.zeros((flat.n_faces, n_fp, 3))
    for d in range(3):
        vo, vn = _extrapolate_scalar_to_faces(velocity[:, :, d], flat, ops, mesh)
        vel_owner_fp[:, :, d] = vo
        vel_neighbor_fp[:, :, d] = vn
    phi_owner_fp, phi_neighbor_fp = _extrapolate_scalar_to_faces(scalar_field, flat, ops, mesh)

    # 计算每个面通量点的物理质量通量（使用 true_normal）
    # mass_flux_phys[f,fp] = (rho * U) . n̂_true
    rho_u_owner = rho_owner_fp[..., None] * vel_owner_fp  # (n_faces, n_fp, 3)
    mass_flux = np.sum(rho_u_owner * flat.true_normal, axis=-1)  # (n_faces, n_fp)

    # 迎风选择
    phi_upwind = np.where(mass_flux >= 0, phi_owner_fp, phi_neighbor_fp)

    # 通量差（用于校正分配）
    delta_phi = phi_upwind - phi_owner_fp  # (n_faces, n_fp)
    correction_fp = mass_flux * delta_phi  # (n_faces, n_fp)

    # 分配回 SPs
    interface_correction = _distribute_correction_to_cells(correction_fp, flat, ops, mesh)
    residual = residual + interface_correction

    return residual


def compute_scalar_diffusion_residual(
    scalar_field: np.ndarray,
    gamma_field: np.ndarray,
    mesh,
    ops,
) -> np.ndarray:
    """计算标量扩散 FR 残差（体积项 + BR1 界面校正）。

    扩散方程: d(rho*phi)/dt = div(Gamma * grad(phi))
    残差 = -div(Gamma*grad(phi))/det(J) + interface_correction/det(J)
    （注意符号：残差定义为 dphi/dt = -residual，扩散项贡献为正，
     因此残差本身为负散度）

    Args:
        scalar_field: (n_cells, n_sps) 标量场
        gamma_field: (n_cells, n_sps) 有效扩散系数 Gamma
        mesh: HighOrderMesh
        ops: FROperators

    Returns:
        residual: (n_cells, n_sps) 扩散残差
    """
    n_cells = mesh.n_cells
    n_sps = mesh.n_sps_per_cell
    n_prism = mesh.n_prism_cells

    det_jacs = mesh.jacobians["det_jacs"].reshape(n_cells, n_sps)
    inv_jacs = mesh.jacobians["inv_jacs"].reshape(n_cells, n_sps, 3, 3)
    adj_j = det_jacs[..., None, None] * inv_jacs

    # === 体积项 ===
    # 计算标量梯度（度量项一致）
    grad_phi = compute_physical_scalar_gradient(scalar_field, mesh, ops)  # (n_cells, n_sps, 3)
    # 扩散通量: G_phys[...,i] = Gamma * grad(phi)[...,i]
    G_phys = gamma_field[..., None] * grad_phi  # (n_cells, n_sps, 3)
    # 逆变通量
    G_tilde = np.matmul(adj_j, G_phys[..., None]).squeeze(-1)  # (n_cells, n_sps, 3)
    # 散度
    div_G = np.zeros((n_cells, n_sps))
    if n_prism > 0:
        for m in range(3):
            div_G[:n_prism] += np.tensordot(G_tilde[:n_prism, :, m], ops.D_3d_prism[:, :, m], axes=([1], [1]))
    if n_cells > n_prism:
        for m in range(3):
            div_G[n_prism:] += np.tensordot(G_tilde[n_prism:, :, m], ops.D_3d_tet[:, :, m], axes=([1], [1]))

    # 扩散残差 = -div(G)/det(J)（使 update 中 -residual = +div(G)/det(J)）
    residual = -div_G / det_jacs

    # === 界面项（BR1 平均通量校正）===
    flat = get_flat_face_geometry(mesh, ops)
    n_fp = flat.n_fp

    # 外插标量和 Gamma 到面通量点
    phi_owner_fp, phi_neighbor_fp = _extrapolate_scalar_to_faces(scalar_field, flat, ops, mesh)
    gamma_owner_fp, gamma_neighbor_fp = _extrapolate_scalar_to_faces(gamma_field, flat, ops, mesh)

    # 外插标量梯度到面通量点（逐分量）
    grad_owner_fp = np.zeros((flat.n_faces, n_fp, 3))
    for d in range(3):
        go, _ = _extrapolate_scalar_to_faces(grad_phi[:, :, d], flat, ops, mesh)
        grad_owner_fp[:, :, d] = go

    # BR1 平均：phi_avg = 0.5*(phi_o + phi_n), gamma_face = 0.5*(gamma_o + gamma_n)
    phi_avg = 0.5 * (phi_owner_fp + phi_neighbor_fp)
    gamma_face = 0.5 * (gamma_owner_fp + gamma_neighbor_fp)

    # 通量差：BR1 使用平均梯度代替 owner 梯度
    # 简化 BR1：只修正通量平均跳跃（与 fr_viscous_flux.py 策略一致）
    # delta_G_contrav_m = adj(J)[m,:] . (gamma_face * grad_phi_owner)
    #                   - adj(J)[m,:] . (gamma_face * grad_phi_avg)
    #                 = adj(J)[m,:] . gamma_face * (grad_phi_owner - grad_phi_avg)
    #                 = adj(J)[m,:] . gamma_face * 0.5*(grad_phi_owner - grad_phi_neighbor)
    # 但 grad_phi_neighbor 在 neighbor 侧 SPs 上，不能直接外推到面 FPs。
    # 简化：使用 owner 侧梯度，通量差 = gamma_face * (grad_o - 0) 的逆变形式
    # 这实际上等价于：common flux 使用 grad_phi_avg=0（无梯度跳跃时的零梯度），
    # 而 owner flux 使用 grad_phi_owner。
    # 更合理的简化：common flux 使用 (phi_avg - phi_owner) 的等效梯度方向
    # 最终采用：delta_phi = phi_avg - phi_owner = 0.5*(phi_n - phi_o)
    # 通量差 = gamma_face * delta_phi 作为等效扩散通量跳跃的标量度量
    delta_phi = phi_avg - phi_owner_fp  # 0.5*(phi_n - phi_o)

    # 将标量跳跃转为等效扩散通量差（使用 gamma_face 缩放）
    # 在 FR 框架中，扩散校正的严格形式需要梯度跳跃，但简化 BR1 只使用
    # 状态跳跃乘以扩散系数作为代理——这在网格足够细时收敛到正确解。
    with np.errstate(over='ignore', invalid='ignore'):
        correction_fp = gamma_face * delta_phi  # (n_faces, n_fp)

    # 分配回 SPs
    interface_correction = _distribute_correction_to_cells(correction_fp, flat, ops, mesh)
    # 扩散校正符号：与对流相反（扩散是"反梯度"通量，校正应减小残差）
    residual = residual - interface_correction

    return residual


def compute_turbulence_transport_residual(solver) -> Tuple[np.ndarray, np.ndarray]:
    """计算 k/omega 的完整输运残差（对流 + 扩散）。

    入口函数：从 solver 获取流场和湍流场信息，分别计算 k 和 omega 的
    对流+扩散残差，返回 dk/dt 和 domega/dt 的输运贡献（已除以密度）。

    Args:
        solver: FRSolver 实例（需要已初始化 SST/DDES 湍流模型）

    Returns:
        (dk_dt_transport, domega_dt_transport): 各自 (n_cells, n_sps)，
        输运项对 dk/dt 和 domega/dt 的贡献
    """
    if solver.turb_model is None or not hasattr(solver.turb_model, 'k_field'):
        n_cells, n_sps = solver.state.U.shape[:2]
        return np.zeros((n_cells, n_sps)), np.zeros((n_cells, n_sps))

    turb = solver.turb_model
    Q = solver.state.Q
    rho = Q[:, :, 0]  # (n_cells, n_sps)
    vel = Q[:, :, 1:4]  # (n_cells, n_sps, 3)

    mu = solver.mu_molecular
    rho_nu_t = rho * turb.nu_t  # 动力涡粘度 mu_t = rho * nu_t

    # 计算有效扩散系数 Gamma_k, Gamma_omega
    # 需要 F1 blending 来确定 sigma_k, sigma_omega
    grad_U = solver._compute_gradients()
    grad_vel = grad_U[:, :, 1:4, :]
    S_mag = turb.compute_strain_rate_magnitude(grad_vel)
    nu = mu / np.maximum(rho, 1e-10)

    # 交叉扩散项（F1 计算需要）
    grad_k = compute_physical_scalar_gradient(turb.k_field, solver.mesh, solver.ops)
    grad_omega = compute_physical_scalar_gradient(turb.omega_field, solver.mesh, solver.ops)
    grad_dot = np.sum(grad_k * grad_omega, axis=-1)
    omega_safe = np.maximum(turb.omega_field, 1e-10)
    CD_kw = np.maximum(2.0 * rho * turb.sigma_w2 / omega_safe * grad_dot, 1e-10)

    F1 = turb.compute_blending_function_F1(
        turb.k_field, turb.omega_field, solver.wall_distance, nu, S_mag, rho, CD_kw
    )

    sigma_k = F1 * turb.sigma_k1 + (1.0 - F1) * turb.sigma_k2
    sigma_w = F1 * turb.sigma_w1 + (1.0 - F1) * turb.sigma_w2

    gamma_k = mu + sigma_k * rho_nu_t    # (n_cells, n_sps)
    gamma_w = mu + sigma_w * rho_nu_t    # (n_cells, n_sps)

    # 计算 k 的对流 + 扩散残差
    conv_k = compute_scalar_convection_residual(turb.k_field, rho, vel, solver.mesh, solver.ops)
    diff_k = compute_scalar_diffusion_residual(turb.k_field, gamma_k, solver.mesh, solver.ops)
    with np.errstate(over='ignore', invalid='ignore'):
        dk_dt_transport = (conv_k + diff_k) / np.maximum(rho, 1e-10)

    # 计算 omega 的对流 + 扩散残差
    conv_w = compute_scalar_convection_residual(turb.omega_field, rho, vel, solver.mesh, solver.ops)
    diff_w = compute_scalar_diffusion_residual(turb.omega_field, gamma_w, solver.mesh, solver.ops)
    with np.errstate(over='ignore', invalid='ignore'):
        domega_dt_transport = (conv_w + diff_w) / np.maximum(rho, 1e-10)

    # NaN/Inf 隔离：退化网格上梯度/Jacobian 可能产生非有限值，
    # 归零后由 SST.update_fields 的二次防护和 positivity limiter 接管
    dk_dt_transport = np.where(np.isfinite(dk_dt_transport), dk_dt_transport, 0.0)
    domega_dt_transport = np.where(np.isfinite(domega_dt_transport), domega_dt_transport, 0.0)

    return dk_dt_transport, domega_dt_transport
