"""
AutoFlowCFD V2.0 - GPU 版湍流标量（k/omega）输运残差 (#7 第四次评审第四轮)

与 core/turbulence/transport.py + transport_kernel.py 对应的 CuPy 版本，
补齐 GPU SST/DDES/IDDES 长期缺失的输运项（此前 GPUTurbulenceSST 只有
逐点源项 ODE，`update_fields_gpu` 的 `transport_k`/`transport_omega`
参数从未被调用方传入，见 gpu_turbulence_sst.py 模块文档）。

与 CPU 版的一处刻意保留的行为差异（不是遗漏，是逐字复刻 CPU 实际行为）：
CPU 版 `extrapolate_scalar_to_faces_kernel`/`distribute_corrections_to_
cells_kernel[_colored]` 都不做 owner_is_primary/neighbor_is_primary 过滤
——与 `fr_residual/inviscid_kernel.py`/`viscous_flux_kernel.py`（及其
GPU 版 gpu_inviscid.py/gpu_viscous.py）不同，那两处数值上被证实必须过滤
（棱柱四边形侧面拆分面场景，见 gpu_inviscid.py 模块文档）。这里没有新增
一个 CPU 没有的过滤，避免制造 GPU/CPU 分歧；若 CPU 版本身在分裂面场景
下有同一类问题，那是 transport.py/transport_kernel.py 自身的既有行为，
不在本次 GPU 移植范围内。

不需要图着色分组：CPU numba 版本用图着色规避多线程写冲突（per-thread
buffer 的替代方案），但 `cp.scatter_add` 本身就正确处理重复索引累加，
对全部面一次性向量化处理即可，颜色分组对 GPU 版本的正确性和性能都没有
必要（也没有 owner_is_primary 那样的强制要求）。

分配机制复用 `gpu_inviscid_volume.py::distribute_face_correction_to_sps`
（V2.0 专家组盲审第四轮修复的 gather 机制，与 CPU numba kernel
`_distribute_point_scalar` 完全一致），不是矩阵乘法。
"""

from typing import Optional, Tuple

from autoflowcfd.core.gpu import get_cupy
from autoflowcfd.core.gpu.residual.gpu_volume_contract import gpu_contract_shared_operator_2axis
from autoflowcfd.core.gpu.residual.gpu_gradients import compute_physical_scalar_gradient_gpu
from autoflowcfd.core.gpu.residual.gpu_inviscid_volume import distribute_face_correction_to_sps


def _extrapolate_scalar_to_faces_gpu(
    cp, ff, n_prism, scalar_sps,
    wall_dirichlet_zero_face=None,
    wall_dirichlet_value_face=None,
    has_wall_dirichlet_value=None,
):
    """CuPy 版 `extrapolate_scalar_to_faces_kernel`，对全部面一次性向量化
    处理（不分色、不按 owner/neighbor_is_primary 过滤，见模块文档）。

    与 CPU 版逐字对应的三条边界 ghost 规则（真实边界面，即
    neighbor_src0_cell<0 且 neighbor_src1_idx<0）：
    - wall_dirichlet_zero_face: ghost = -owner（k=0 Dirichlet 镜像）
    - has_wall_dirichlet_value: ghost = 2*target - owner（omega 解析壁面值）
    - 都不是：ghost = owner（Neumann 默认，零梯度）
    以及混合拆分面（B-8）配对边界面同一规则的逐 FP 覆盖。

    Args:
        scalar_sps: (n_cells, n_sps) CuPy 数组

    Returns:
        (phi_owner_fp, phi_neighbor_fp) 各 (n_faces, n_fp)
    """
    n_faces = ff.n_faces

    oc = ff.owner_cell
    oax = ff.owner_axis
    oside = ff.owner_side
    oside_idx = cp.where(oside <= 0, 0, 1)
    compact_cell_type = getattr(ff, 'compact_cell_type', None)
    if compact_cell_type is not None:
        celltype_o = compact_cell_type[oc]
    else:
        celltype_o = cp.where(oc < n_prism, 0, 1)
    E_o = ff.boundary_extrap[celltype_o, oax, oside_idx]  # (n_faces, n_fp, n_sps)
    phi_owner = cp.einsum('fps,fs->fp', E_o, scalar_sps[oc])

    c0 = ff.neighbor_src0_cell
    valid0 = c0 >= 0
    c0_safe = cp.maximum(c0, 0)
    phi_neighbor = cp.einsum('fps,fs->fp', ff.neighbor_src0_mat, scalar_sps[c0_safe]) * valid0[:, None]

    idx1 = ff.neighbor_src1_idx
    valid1 = idx1 >= 0
    if bool(cp.any(valid1)):
        idx1_safe = cp.maximum(idx1, 0)
        c1 = ff.neighbor_src1_cell[idx1_safe]
        mat1 = ff.neighbor_src1_mat[idx1_safe]
        phi_neighbor = phi_neighbor + cp.einsum('fps,fs->fp', mat1, scalar_sps[c1]) * valid1[:, None]

    n_fp = phi_owner.shape[1]
    if wall_dirichlet_zero_face is None:
        wall_dirichlet_zero_face = cp.zeros(n_faces, dtype=cp.bool_)
    if has_wall_dirichlet_value is None:
        has_wall_dirichlet_value = cp.zeros(n_faces, dtype=cp.bool_)
    if wall_dirichlet_value_face is None:
        wall_dirichlet_value_face = cp.zeros((n_faces, n_fp), dtype=cp.float64)

    is_true_boundary = (~valid0) & (~valid1)
    dirichlet_zero = is_true_boundary & wall_dirichlet_zero_face
    dirichlet_value = is_true_boundary & has_wall_dirichlet_value & (~wall_dirichlet_zero_face)
    neumann = is_true_boundary & (~wall_dirichlet_zero_face) & (~has_wall_dirichlet_value)

    phi_neighbor = cp.where(dirichlet_zero[:, None], -phi_owner, phi_neighbor)
    phi_neighbor = cp.where(dirichlet_value[:, None], 2.0 * wall_dirichlet_value_face - phi_owner, phi_neighbor)
    phi_neighbor = cp.where(neumann[:, None], phi_owner, phi_neighbor)

    mp = ff.mixed_nb_partner
    has_partner = mp >= 0
    if bool(cp.any(has_partner)):
        mp_safe = cp.maximum(mp, 0)
        partner_dirichlet_zero = wall_dirichlet_zero_face[mp_safe]
        partner_has_value = has_wall_dirichlet_value[mp_safe]
        partner_value = wall_dirichlet_value_face[mp_safe]
        mask = has_partner[:, None] & ff.mixed_nb_mask

        val_dirichlet_zero = -phi_owner
        val_dirichlet_value = 2.0 * partner_value - phi_owner
        val_neumann = phi_owner
        chosen = cp.where(
            partner_dirichlet_zero[:, None], val_dirichlet_zero,
            cp.where(partner_has_value[:, None], val_dirichlet_value, val_neumann),
        )
        phi_neighbor = cp.where(mask, chosen, phi_neighbor)

    return phi_owner, phi_neighbor


def _distribute_scalar_correction_gpu(cp, ff, correction_fp, det_jacs, n_cells, n_sps):
    """CuPy 版 `distribute_corrections_to_cells_kernel[_colored]`，标量版，
    对全部面一次性向量化处理（不分色，见模块文档）。

    correction_fp: (n_faces, n_fp)
    Returns: (n_cells, n_sps)
    """
    correction = cp.zeros((n_cells, n_sps), dtype=cp.float64)

    oc = ff.owner_cell
    oax = ff.owner_axis
    oside = ff.owner_side
    contrib_owner = distribute_face_correction_to_sps(
        cp, correction_fp, oax, oside, ff.dist_fp_of_sp, ff.dist_axis_coord_of_sp,
        ff.g_left, ff.g_right,
    )  # (n_faces, n_sps)
    contrib_owner = contrib_owner / det_jacs[oc]
    cp.scatter_add(correction, (oc, slice(None)), -contrib_owner)

    nc = ff.neighbor_cell
    has_neighbor = nc >= 0
    if bool(cp.any(has_neighbor)):
        sel = cp.where(has_neighbor)[0]
        nc_sel = nc[sel]
        nax_sel = ff.neighbor_axis[sel]
        nside_sel = ff.neighbor_side[sel]
        contrib_neighbor = distribute_face_correction_to_sps(
            cp, correction_fp[sel], nax_sel, nside_sel,
            ff.dist_fp_of_sp, ff.dist_axis_coord_of_sp, ff.g_left, ff.g_right,
        )
        contrib_neighbor = contrib_neighbor / det_jacs[nc_sel]
        cp.scatter_add(correction, (nc_sel, slice(None)), contrib_neighbor)

    return correction


def compute_scalar_convection_residual_gpu(
    scalar_field, rho, velocity, mesh_data, ops_data, ff, n_cells, n_prism, n_sps,
    wall_dirichlet_zero_face=None, wall_dirichlet_value_face=None, has_wall_dirichlet_value=None,
):
    """标量对流 FR 残差（体积项 + 界面上风校正），与 CPU 版
    `compute_scalar_convection_residual` 逐字对应。"""
    cp = get_cupy()
    det_jacs = mesh_data['det_jacs']
    adj_j = mesh_data['adj_j']

    rho_u_phi = rho[..., None] * velocity * scalar_field[..., None]  # (n_cells,n_sps,3)
    F_tilde = cp.matmul(adj_j, rho_u_phi[..., None]).squeeze(-1)  # (n_cells,n_sps,3)

    div_F = cp.zeros((n_cells, n_sps), dtype=cp.float64)
    if n_prism > 0:
        div_F[:n_prism] = gpu_contract_shared_operator_2axis(
            ops_data['D_3d_prism'], F_tilde[:n_prism, :, :, None]
        )[..., 0]
    if n_cells > n_prism:
        div_F[n_prism:] = gpu_contract_shared_operator_2axis(
            ops_data['D_3d_tet'], F_tilde[n_prism:, :, :, None]
        )[..., 0]

    residual = -div_F / det_jacs

    rho_o, rho_n = _extrapolate_scalar_to_faces_gpu(cp, ff, n_prism, rho)
    n_fp = rho_o.shape[1]
    vel_o = cp.zeros((ff.n_faces, n_fp, 3), dtype=cp.float64)
    for d in range(3):
        vo, _ = _extrapolate_scalar_to_faces_gpu(cp, ff, n_prism, velocity[..., d])
        vel_o[..., d] = vo
    phi_o, phi_n = _extrapolate_scalar_to_faces_gpu(
        cp, ff, n_prism, scalar_field,
        wall_dirichlet_zero_face, wall_dirichlet_value_face, has_wall_dirichlet_value,
    )

    mass_flux = cp.sum(rho_o[..., None] * vel_o * ff.true_normal, axis=-1)  # (n_faces,n_fp)
    phi_upwind = cp.where(mass_flux >= 0, phi_o, phi_n)
    delta_phi = phi_upwind - phi_o

    adj_mag = cp.linalg.norm(ff.owner_adj_row_exact, axis=-1)
    correction_fp = adj_mag * mass_flux * delta_phi

    interface_correction = _distribute_scalar_correction_gpu(cp, ff, correction_fp, det_jacs, n_cells, n_sps)
    return residual + interface_correction


def compute_scalar_diffusion_residual_gpu(
    scalar_field, gamma_field, mesh_data, ops_data, ff, n_cells, n_prism, n_sps,
):
    """标量扩散 FR 残差（体积项 + BR1 梯度差界面校正），与 CPU 版
    `compute_scalar_diffusion_residual` 逐字对应（2026-08-25 梯度差
    符号约定修复后的版本，见该函数文档）。"""
    cp = get_cupy()
    det_jacs = mesh_data['det_jacs']
    adj_j = mesh_data['adj_j']

    grad_phi = compute_physical_scalar_gradient_gpu(scalar_field, mesh_data, ops_data)  # (n_cells,n_sps,3)
    G_phys = gamma_field[..., None] * grad_phi
    G_tilde = cp.matmul(adj_j, G_phys[..., None]).squeeze(-1)  # (n_cells,n_sps,3)

    div_G = cp.zeros((n_cells, n_sps), dtype=cp.float64)
    if n_prism > 0:
        div_G[:n_prism] = gpu_contract_shared_operator_2axis(
            ops_data['D_3d_prism'], G_tilde[:n_prism, :, :, None]
        )[..., 0]
    if n_cells > n_prism:
        div_G[n_prism:] = gpu_contract_shared_operator_2axis(
            ops_data['D_3d_tet'], G_tilde[n_prism:, :, :, None]
        )[..., 0]

    residual = div_G / det_jacs

    gamma_o, gamma_n = _extrapolate_scalar_to_faces_gpu(cp, ff, n_prism, gamma_field)
    n_fp = gamma_o.shape[1]
    grad_o = cp.zeros((ff.n_faces, n_fp, 3), dtype=cp.float64)
    grad_n = cp.zeros((ff.n_faces, n_fp, 3), dtype=cp.float64)
    for d in range(3):
        go, gn = _extrapolate_scalar_to_faces_gpu(cp, ff, n_prism, grad_phi[..., d])
        grad_o[..., d] = go
        grad_n[..., d] = gn

    gamma_face = 0.5 * (gamma_o + gamma_n)
    delta_grad = 0.5 * (grad_n - grad_o)
    flux_jump_phys = gamma_face * cp.sum(delta_grad * ff.true_normal, axis=-1)

    adj_mag = cp.linalg.norm(ff.owner_adj_row_exact, axis=-1)
    correction_fp = adj_mag * flux_jump_phys

    interface_correction = _distribute_scalar_correction_gpu(cp, ff, correction_fp, det_jacs, n_cells, n_sps)
    return residual - interface_correction


def compute_omega_wall_target_gpu(cp, ff, wall_mask, wall_distance_gpu, Q_gpu, mu, beta1):
    """CuPy 版 `_compute_omega_wall_target`：omega_wall = 60*nu/(beta1*d1^2)
    （Wilcox 解析式），逐字对应 CPU 版同名函数——全程 GPU 原生实现（不像
    边界幽灵态那样需要 CPU round-trip：wall_distance_gpu/Q_gpu/owner_cell
    都已经常驻显存，没有必要为这个小计算专门下载/上传）。

    Args:
        wall_mask: (n_faces,) CuPy bool，WALL 边界面掩码（由
            `compute_wall_dirichlet_masks_gpu` 一次性算出并缓存）

    Returns:
        (omega_wall_value_face, has_value_face)，与 CPU 版返回语义一致
    """
    n_faces = ff.n_faces
    n_fp = ff.boundary_extrap.shape[-2]
    omega_wall_value_face = cp.zeros((n_faces, n_fp), dtype=cp.float64)
    wall_idx = cp.where(wall_mask)[0]
    if wall_idx.shape[0] > 0:
        owner_cells = ff.owner_cell[wall_idx]
        d1 = cp.min(wall_distance_gpu[owner_cells], axis=1)
        d1 = cp.maximum(d1, 1e-8)
        rho_owner = cp.mean(Q_gpu[owner_cells, :, 0], axis=1)
        nu_owner = mu / cp.maximum(rho_owner, 1e-10)
        omega_wall = 60.0 * nu_owner / (beta1 * d1 ** 2)
        omega_wall_value_face[wall_idx, :] = omega_wall[:, None]
    return omega_wall_value_face, wall_mask


def compute_wall_dirichlet_mask_gpu(mesh, boundary_ghost_provider):
    """CuPy 版 `_compute_wall_dirichlet_face_mask`：纯拓扑查询（哪些面是
    WALL 类型边界组），与 wall_distance/流场状态无关，只依赖网格自身，
    求解过程中不变——调用方（gpu_solver_io.py）应该只在初始化时调用一次
    并缓存结果，不是每步都重新算（CPU 版每步都重算是因为它本身开销可
    忽略；这里直接给出 numpy 版本供调用方自行决定何时上传/缓存）。

    Returns:
        wall_mask: (n_faces,) numpy bool 数组
    """
    import numpy as np
    n_faces = mesh.face_connectivity.n_faces
    group_code = getattr(boundary_ghost_provider, "group_code", None)
    code_to_config = getattr(boundary_ghost_provider, "code_to_config", None)
    if group_code is None or code_to_config is None:
        return np.zeros(n_faces, dtype=np.bool_)
    wall_codes = [code for code, cfg in code_to_config.items() if cfg.get("type") == "WALL"]
    if not wall_codes:
        return np.zeros(n_faces, dtype=np.bool_)
    return np.isin(group_code, wall_codes)


def compute_turbulence_transport_residual_gpu(
    solver, grad_vel=None,
) -> Tuple:
    """GPU 版 k/omega 完整输运残差入口（对流+扩散），与 CPU 版
    `compute_turbulence_transport_residual` 逐字对应。

    真实差异说明（不是简化）：不做 CPU 版末尾的
    `suppress_residual_outliers`（troubled_cell.py 的中位数离群值抑制，
    该函数是纯 numba/numpy 实现，本身就没有 GPU 版本）——只保留 CPU 版
    自己也称为"最后一道防线"的 isfinite 归零。这是本次移植唯一没有
    对应实现的 CPU 侧安全网，明确记录，不是隐藏的简化：GPU 湍流输运
    整体仍然是本项目未在真实硬件验证过的新功能，缺这一层次要防护不会
    掩盖其余数值行为，但退化网格上的离群值抑制确实比 CPU 路径弱。

    Args:
        solver: GPUFRSolver 实例，需要 turb_model_gpu 已初始化
            （SST/DDES/IDDES 均可，都是 GPUTurbulenceSST 实例）
        grad_vel: 可选，调用方已经算好的速度梯度（CuPy (n_cells,n_sps,3,3)），
            复用避免重复计算物理梯度这个真实热点，与 CPU 版同名参数
            同一个性能考量。

    Returns:
        (dk_dt_transport, domega_dt_transport)，各自 (n_cells, n_sps)
    """
    cp = get_cupy()
    turb = solver.turb_model_gpu
    n_cells = solver.mesh.n_cells
    n_sps = solver.mesh.n_sps_per_cell
    if turb is None:
        z = cp.zeros((n_cells, n_sps), dtype=cp.float64)
        return z, z

    Q = solver.Q_gpu
    rho = Q[:, :, 0]
    vel = Q[:, :, 1:4]
    mu = solver.mu_molecular
    rho_nu_t = rho * turb.nu_t

    if grad_vel is None:
        from autoflowcfd.core.gpu.residual.gpu_gradients import compute_physical_gradient_gpu
        grad_U = compute_physical_gradient_gpu(solver.U_gpu[..., :5], solver.mesh_data, solver.ops_data)
        grad_vel = grad_U[..., 1:4, :]

    nu = mu / cp.maximum(rho, 1e-10)

    grad_k = compute_physical_scalar_gradient_gpu(turb.k_field, solver.mesh_data, solver.ops_data)
    grad_omega = compute_physical_scalar_gradient_gpu(turb.omega_field, solver.mesh_data, solver.ops_data)

    max_grad_mag = 1e6
    grad_k_mag = cp.linalg.norm(grad_k, axis=-1)
    grad_omega_mag = cp.linalg.norm(grad_omega, axis=-1)
    scale_k = cp.clip(max_grad_mag / cp.maximum(grad_k_mag, 1e-10), 0, 1)
    grad_k = cp.where((grad_k_mag > max_grad_mag)[..., None], grad_k * scale_k[..., None], grad_k)
    scale_omega = cp.clip(max_grad_mag / cp.maximum(grad_omega_mag, 1e-10), 0, 1)
    grad_omega = cp.where(
        (grad_omega_mag > max_grad_mag)[..., None], grad_omega * scale_omega[..., None], grad_omega
    )

    grad_dot = cp.sum(grad_k * grad_omega, axis=-1)
    omega_safe = cp.maximum(turb.omega_field, 1e-10)
    CD_kw = cp.maximum(2.0 * rho * turb.sigma_w2 / omega_safe * grad_dot, 1e-10)

    d_wall = solver.wall_distance_gpu
    F1 = turb.compute_blending_F1_gpu(turb.k_field, turb.omega_field, d_wall, nu, rho, CD_kw)

    sigma_k = F1 * turb.sigma_k1 + (1.0 - F1) * turb.sigma_k2
    sigma_w = F1 * turb.sigma_w1 + (1.0 - F1) * turb.sigma_w2
    gamma_k = mu + sigma_k * rho_nu_t
    gamma_w = mu + sigma_w * rho_nu_t

    ff = solver.flat_face_gpu
    n_prism = solver.mesh_data.get('n_prism', solver.mesh.n_prism_cells)

    wall_mask_k = solver._wall_mask_k_gpu  # 见 gpu_solver_io.py 缓存点文档

    conv_k = compute_scalar_convection_residual_gpu(
        turb.k_field, rho, vel, solver.mesh_data, solver.ops_data, ff, n_cells, n_prism, n_sps,
        wall_dirichlet_zero_face=wall_mask_k,
    )
    diff_k = compute_scalar_diffusion_residual_gpu(
        turb.k_field, gamma_k, solver.mesh_data, solver.ops_data, ff, n_cells, n_prism, n_sps,
    )
    dk_dt_transport = (conv_k + diff_k) / cp.maximum(rho, 1e-10)

    omega_wall_value_face, has_omega_wall = compute_omega_wall_target_gpu(
        cp, ff, wall_mask_k, d_wall, Q, mu, getattr(turb, "beta1", 0.075)
    )

    conv_w = compute_scalar_convection_residual_gpu(
        turb.omega_field, rho, vel, solver.mesh_data, solver.ops_data, ff, n_cells, n_prism, n_sps,
        wall_dirichlet_value_face=omega_wall_value_face, has_wall_dirichlet_value=has_omega_wall,
    )
    diff_w = compute_scalar_diffusion_residual_gpu(
        turb.omega_field, gamma_w, solver.mesh_data, solver.ops_data, ff, n_cells, n_prism, n_sps,
    )
    domega_dt_transport = (conv_w + diff_w) / cp.maximum(rho, 1e-10)

    dk_dt_transport = cp.where(cp.isfinite(dk_dt_transport), dk_dt_transport, 0.0)
    domega_dt_transport = cp.where(cp.isfinite(domega_dt_transport), domega_dt_transport, 0.0)

    return dk_dt_transport, domega_dt_transport
