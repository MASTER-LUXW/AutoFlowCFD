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
from autoflowcfd.core.fr_operators.troubled_cell import suppress_residual_outliers
from autoflowcfd.core.turbulence.transport_kernel import (
    extrapolate_scalar_to_faces_kernel,
    distribute_corrections_to_cells_kernel,
    distribute_corrections_to_cells_kernel_colored,
)


def _extrapolate_scalar_to_faces(scalar_sps, flat, ops, mesh, wall_dirichlet_zero_face=None):
    """将 SPs 上的标量场外插到所有面的通量点（numba kernel 版本）。

    使用 turbulence_transport_kernel.py 的 numba 编译函数替代纯 Python 循环。
    owner 侧用 boundary_extrap 矩阵，neighbor 侧用 neighbor_sources 矩阵。

    Args:
        wall_dirichlet_zero_face: (n_faces,) bool，可选。对标记为 True 的
            WALL 边界面，ghost 值用 Dirichlet-zero 镜像（ghost=-owner）
            而不是默认的 Neumann（ghost=owner）——只在外插 k 场时由
            调用方传入，其余场（omega/rho/velocity/gamma_field 等）不传，
            退回原有 Neumann 默认，见 extrapolate_scalar_to_faces_kernel
            文档。

    Returns:
        phi_owner_fp: (n_faces, n_fp)
        phi_neighbor_fp: (n_faces, n_fp)
    """
    if wall_dirichlet_zero_face is None:
        wall_dirichlet_zero_face = np.zeros(flat.n_faces, dtype=np.bool_)
    return extrapolate_scalar_to_faces_kernel(
        scalar_sps, flat.boundary_extrap,
        flat.neighbor_src0_cell, flat.neighbor_src0_mat,
        flat.neighbor_src1_idx, flat.neighbor_src1_cell, flat.neighbor_src1_mat,
        flat.owner_cell, flat.owner_axis, flat.owner_side,
        flat.n_prism, flat.n_faces, flat.n_fp, flat.n_sps,
        wall_dirichlet_zero_face,
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
    wall_dirichlet_zero_face: np.ndarray = None,
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
        wall_dirichlet_zero_face: (n_faces,) bool，可选，见
            `_extrapolate_scalar_to_faces` 文档——只影响 scalar_field
            自身外插到面的 ghost 值（决定上风通量的物理量），不影响
            rho/velocity 的外插（无滑移壁面上 u=0 已经由平均流的
            WALL 幽灵态保证，这里不需要重复处理）。

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

    # 真实复现（2026-08-21，79万单元生产网格，Order Continuation P0->P1
    # 切换后）：退化单元（坍缩坐标/BL 挤出导致 det(J) 局部极小，见
    # troubled_cell.py 模块文档——平均流残差有 mechanism-1/2 两道专门
    # 保护，本模块至今没有）上这里会真的溢出到 inf，`np.errstate` 只是
    # 让这个*已知、已经在下游处理*的溢出不再往 stderr 打印 RuntimeWarning
    # 噪音——不改变任何数值结果：`compute_turbulence_transport_residual`
    # 末尾的 `np.where(np.isfinite(...), ..., 0.0)` 本来就会把这类 inf/nan
    # 结果清零，交给 SST.update_fields 的正性限制器接管（见该函数文档
    # "NaN/Inf 隔离"一节），此前只是没有抑制这个警告，容易被误读成新
    # 出现的异常。真正的解析修复（把 troubled-cell 的 mechanism-1/2
    # 保护接入湍流标量输运）是比这更大的独立工作，见
    # transport_kernel.py::extrapolate_scalar_to_faces_kernel 文档同类
    # 说明。
    with np.errstate(over='ignore', invalid='ignore'):
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
    phi_owner_fp, phi_neighbor_fp = _extrapolate_scalar_to_faces(
        scalar_field, flat, ops, mesh, wall_dirichlet_zero_face
    )

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
    with np.errstate(over='ignore', invalid='ignore'):
        residual = residual + interface_correction

    return residual


def compute_scalar_diffusion_residual(
    scalar_field: np.ndarray,
    gamma_field: np.ndarray,
    mesh,
    ops,
    wall_dirichlet_zero_face: np.ndarray = None,
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
        wall_dirichlet_zero_face: (n_faces,) bool，可选，见
            `_extrapolate_scalar_to_faces` 文档——只影响 scalar_field
            自身外插到面的 ghost 值（决定 BR1 平均 phi_avg），不影响
            gamma_field 的外插（扩散系数在壁面用 Neumann 外插即可，
            与是否 Dirichlet 无关）或 grad_phi 的外插（只用 owner 侧，
            见下方，本来就不经过 ghost）。

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

    # 扩散残差 = -div(G)/det(J)（使 update 中 -residual = +div(G)/det(J)）。
    # 退化单元溢出保护：理由/验证方式同 compute_scalar_convection_
    # residual 里对应的 errstate（见该函数文档），同一类已知、已在
    # compute_turbulence_transport_residual 末尾被下游清零处理的溢出。
    with np.errstate(over='ignore', invalid='ignore'):
        residual = -div_G / det_jacs

    # === 界面项（BR1 平均通量校正）===
    flat = get_flat_face_geometry(mesh, ops)
    n_fp = flat.n_fp

    # 外插标量和 Gamma 到面通量点
    phi_owner_fp, phi_neighbor_fp = _extrapolate_scalar_to_faces(
        scalar_field, flat, ops, mesh, wall_dirichlet_zero_face
    )
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
    with np.errstate(over='ignore', invalid='ignore'):
        residual = residual - interface_correction

    return residual


def _compute_wall_dirichlet_face_mask(solver) -> np.ndarray:
    """算出哪些面是 WALL 边界面，供 k 场的 Dirichlet-zero ghost 使用
    （见 extrapolate_scalar_to_faces_kernel 文档）。

    数据来源：`solver.boundary_ghost_provider`——真实求解路径下是
    `boundary.fr_ghost_state.BoundaryGhostStateProvider`，持有
    `group_code`（每个面所属边界组的整数编码，-1 表示内部面/未匹配）和
    `code_to_config`（编码 -> {'type': 'WALL'/...}）。用
    `code_to_config` 里显式标记为 WALL 的编码集合对 `group_code` 做一次
    向量化匹配（`np.isin`），成本是对 187 万面级别网格的一次数组比较，
    不是逐面 Python 循环。

    防御性回退：如果 `boundary_ghost_provider` 不是这个类型（例如某些
    测试用的自定义 ghost provider 只是一个普通 callable，没有
    group_code/code_to_config），拿不到分组信息时返回全 False——退回
    调用方原有的 Neumann 默认，不是新的静默 bug（这是修复前唯一的行为，
    对这些没有分组信息的场景数值结果不变）。
    """
    mesh = solver.mesh
    n_faces = mesh.face_connectivity.n_faces
    provider = getattr(solver, "boundary_ghost_provider", None)
    group_code = getattr(provider, "group_code", None)
    code_to_config = getattr(provider, "code_to_config", None)
    if group_code is None or code_to_config is None:
        return np.zeros(n_faces, dtype=np.bool_)

    wall_codes = [code for code, cfg in code_to_config.items() if cfg.get("type") == "WALL"]
    if not wall_codes:
        return np.zeros(n_faces, dtype=np.bool_)
    return np.isin(group_code, wall_codes)


def compute_turbulence_transport_residual(
    solver,
    grad_vel: np.ndarray = None,
    grad_k: np.ndarray = None,
    grad_omega: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算 k/omega 的完整输运残差（对流 + 扩散）。

    入口函数：从 solver 获取流场和湍流场信息，分别计算 k 和 omega 的
    对流+扩散残差，返回 dk/dt 和 domega/dt 的输运贡献（已除以密度）。

    Args:
        solver: FRSolver 实例（需要已初始化 SST/DDES 湍流模型）
        grad_vel, grad_k, grad_omega: 可选，调用方（`fr_solver_
            turbulence.compute_turbulence_source`）如果已经算过这三个量，
            直接传进来复用，跳过内部重新计算——性能优化：唯一真实调用方
            `compute_turbulence_source` 在调用本函数*之前*就已经为
            `compute_source_terms` 算过完全相同的 grad_vel/grad_k/
            grad_omega（同一个 solver.state.U/turb_model.k_field/
            omega_field，同一套 mesh/ops，数学上是同一个量），此前这里
            总是无条件重新算一遍——`compute_physical_gradient` 是本项目
            profile 过的真实热点（79万单元 P1 阶段单步 7.5s 累计），这里
            的重复调用是三次里的一次，真实测得省下约 1.6s/步。三者任一
            为 None 时退回原来的内部计算（保持本函数可独立调用的公开
            API 行为不变，不依赖调用方一定会传）。

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
    if grad_vel is None:
        grad_U = solver._compute_gradients()
        grad_vel = grad_U[:, :, 1:4, :]
    S_mag = turb.compute_strain_rate_magnitude(grad_vel)
    nu = mu / np.maximum(rho, 1e-10)

    # 交叉扩散项（F1 计算需要）
    if grad_k is None:
        grad_k = compute_physical_scalar_gradient(turb.k_field, solver.mesh, solver.ops)
    if grad_omega is None:
        grad_omega = compute_physical_scalar_gradient(turb.omega_field, solver.mesh, solver.ops)

    # 梯度幅值裁剪（真实 bug，已修复，2026-08-21）：这里的 grad_k/grad_omega
    # 此前完全没有上限保护——`fr_solver/turbulence.py::compute_turbulence_
    # source` 里给 compute_source_terms 用的那一份 grad_k/grad_omega 早就有
    # 同样的 max_grad_mag=1e6 裁剪（见该文件"正性保持检查"注释），但本函数
    # 参数文档明确说明这里*刻意*不复用那份裁剪后的值、自己独立重新计算，
    # 于是这份独立计算的副本一直没有对应的裁剪。真实复现（cube_demo 生产
    # 网格，P1 阶数，DDES）：mesh 在坍缩坐标+度量退化单元（troubled_cell.py
    # 诊断此网格 P1 阶段 95.15% 单元面法向失配>1度）上，对*理论上处处为
    # 常数*的初始 k/omega 场求梯度，参考空间导数本应恰好为 0，但浮点舍入
    # 误差量级的非零值被 adj(J)/det(J) 这个在退化单元上可以任意大的度量
    # 比值放大到 >1e150（np.linalg.norm 内部计算 x*x 时溢出到 inf，py-spy
    # 采样证实的真实复现）——多数为普通浮点噪声，但间或有值落入次正规数
    # （denormal/subnormal）区间，x86 硬件处理这类数值要走慢得多的微码
    # 路径：单次 `np.sum(grad_k*grad_omega, axis=-1)`（下面这一行）在
    # ~19M 元素规模上因此实测卡住数分钟，而不是正常的毫秒级——是一次
    # "看起来像死锁、实际是每个浮点算子被拖慢几十~上百倍"的真实性能故障，
    # py-spy 对卡住进程的调用栈采样直接定位到本行。与 fr_solver/
    # turbulence.py 用完全相同的裁剪公式（不是发明新阈值，是把已经在
    # 别处验证过、这里唯一遗漏的同一道安全网补齐）。
    # np.linalg.norm 内部对每个分量求平方——在同一类退化单元上分量本身
    # 就已经是溢出级别的量，平方会先于这里的裁剪逻辑触发一次 inf；
    # errstate 只是抑制这一步的警告噪音，紧接着的 np.maximum(...,1e-10)/
    # np.clip(...,0,1) 已经能正确处理 inf 输入（inf>max_grad_mag 恒真，
    # scale=max_grad_mag/inf=0，裁剪结果趋于 0，不是 nan），不依赖这个
    # errstate 才能得到正确结果。
    with np.errstate(over='ignore', invalid='ignore'):
        max_grad_mag = 1e6
        grad_k_mag = np.linalg.norm(grad_k, axis=-1)
        grad_omega_mag = np.linalg.norm(grad_omega, axis=-1)
        if np.any(grad_k_mag > max_grad_mag):
            scale_k = max_grad_mag / np.maximum(grad_k_mag, 1e-10)
            grad_k = grad_k * np.clip(scale_k, 0, 1)[..., None]
        if np.any(grad_omega_mag > max_grad_mag):
            scale_omega = max_grad_mag / np.maximum(grad_omega_mag, 1e-10)
            grad_omega = grad_omega * np.clip(scale_omega, 0, 1)[..., None]

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

    # WALL 上 k=0 的 Dirichlet 掩码（真实修复，2026-08-21，见
    # transport_kernel.py::extrapolate_scalar_to_faces_kernel 文档）：
    # 只对 k 场生效，omega 仍用 Neumann 默认（omega 解析壁面值需要额外
    # 的壁面距离数据，留作后续独立工作，见该文档）。
    wall_mask_k = _compute_wall_dirichlet_face_mask(solver)

    # 计算 k 的对流 + 扩散残差
    conv_k = compute_scalar_convection_residual(
        turb.k_field, rho, vel, solver.mesh, solver.ops, wall_dirichlet_zero_face=wall_mask_k
    )
    diff_k = compute_scalar_diffusion_residual(
        turb.k_field, gamma_k, solver.mesh, solver.ops, wall_dirichlet_zero_face=wall_mask_k
    )
    with np.errstate(over='ignore', invalid='ignore'):
        dk_dt_transport = (conv_k + diff_k) / np.maximum(rho, 1e-10)

    # 计算 omega 的对流 + 扩散残差
    conv_w = compute_scalar_convection_residual(turb.omega_field, rho, vel, solver.mesh, solver.ops)
    diff_w = compute_scalar_diffusion_residual(turb.omega_field, gamma_w, solver.mesh, solver.ops)
    with np.errstate(over='ignore', invalid='ignore'):
        domega_dt_transport = (conv_w + diff_w) / np.maximum(rho, 1e-10)

    # 机制3（症状检测，2026-08-22）：退化单元（坍缩坐标/BL 挤出，见
    # fr_operators/troubled_cell.py 模块文档）上本函数算出的残差可能
    # 出现量级异常（真实复现：cube_demo 生产网格 P0->P1 切换后，
    # transport.py 内部多处除以 det(J) 的地方溢出到 inf，见本文件
    # 上方的 errstate 注释）——平均流残差（inviscid.py/viscous_flux.py）
    # 早就用 suppress_residual_outliers 处理同一类问题（"取代此前先
    # 用 det(J)/法向失配几何量预判、按整个单元降阶的机制1/2"，见
    # troubled_cell.py 文档"机制3"一节），本函数此前一直没有接入这套
    # 机制，只在最后做一次朴素的 isfinite 归零——两者不冲突：
    # suppress_residual_outliers 用同单元其余 SP 的残差中位数做参照，
    # 能捕捉"明显偏大但还是有限值"的异常（isfinite 捕捉不到这类），
    # 按 (cell,SP) 粒度清零，不牵连同一单元里其余健康 SP；下面的
    # isfinite 归零保留作最后一道防线（例如整个单元所有 SP 都异常、
    # 中位数参照本身也失真的极端情形）。
    dk_dt_transport = suppress_residual_outliers(
        dk_dt_transport[:, :, None], turb.k_field[:, :, None]
    )[:, :, 0]
    domega_dt_transport = suppress_residual_outliers(
        domega_dt_transport[:, :, None], turb.omega_field[:, :, None]
    )[:, :, 0]

    # NaN/Inf 隔离（最后一道防线）：退化网格上梯度/Jacobian 可能产生非
    # 有限值，归零后由 SST.update_fields 的二次防护和 positivity
    # limiter 接管
    dk_dt_transport = np.where(np.isfinite(dk_dt_transport), dk_dt_transport, 0.0)
    domega_dt_transport = np.where(np.isfinite(domega_dt_transport), domega_dt_transport, 0.0)

    return dk_dt_transport, domega_dt_transport
