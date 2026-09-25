"""AutoFlowCFD V2.0 - 对流/扩散残差与顶层编排(GPU)

从 `src/autoflowcfd/core/gpu/turbulence/gpu_scalar_transport.py`(原 743 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""

from typing import Tuple

from autoflowcfd.core.gpu import get_cupy

from autoflowcfd.core.gpu.residual.gpu_volume_contract import (
    gpu_contract_shared_operator_1axis, gpu_contract_shared_operator_2axis,
)

from autoflowcfd.core.gpu.residual.gpu_gradients import compute_physical_scalar_gradient_gpu


from autoflowcfd.core.turbulence.transport import resolve_turb_overintegration

# 过积分上下文提取到 `core/gpu/gpu_overintegration.py`（2026-09-15，粘性
# 体积项 GPU 侧补齐时共用同一份，避免 residual 模块反向依赖 turbulence
# 模块）。原先那层 `as _turb_overint_segs_gpu` 兼容别名全仓库零引用，
# 2026-09-24 拆包时一并删除，调用点直接用真名。
from autoflowcfd.core.gpu.gpu_overintegration import (
    get_overintegration_segs_gpu,
)
from .faces import _distribute_scalar_correction_gpu, _extrapolate_scalar_to_faces_gpu
from .omega_wall import compute_omega_wall_target_gpu


def _scalar_volume_div_overintegrated_gpu(cp, factors, segs,
                                          n_cells, n_sps):
    """标量体积项 `div(adj(J) * prod(factors))` 的去混叠版（GPU）。

    `factors` 是一串 (n_cells, n_sps, k) 的场，k 为 1 或 3；各自精确插值
    到 FINE 点后**在 FINE 点相乘**（去混叠的全部内容就是"先插值再相乘"），
    再用细点度量算逆变通量、细网格微分矩阵求散度、精确限制回 coarse。

    对流用 (rho, phi, u)、扩散用 (gamma, grad_phi)，与 CPU 端
    `_scalar_convection_volume_overintegrated` /
    `_scalar_diffusion_volume_overintegrated` 逐字对应。
    """
    div = cp.zeros((n_cells, n_sps), dtype=cp.float64)
    # 每段自带自己的 n_fine 与**已切好**的细点度量（2026-09-17，与 CPU 端
    # 同一次改动）：四面体过积分的细网格轴不再填充到棱柱宽度，两段的
    # n_fine 不同，所以不能再用一份共享的 adj_j_fine 按全局 [lo:hi] 切。
    # 本循环一次处理整段，`adj_seg` 正好就对应 [lo:hi]，直接用即可。
    for lo, hi, _n_fine_seg, adj_seg, c2f, D_fine, f2c in segs:
        if hi <= lo:
            continue
        prod = None
        for f in factors:
            ff_ = gpu_contract_shared_operator_1axis(c2f, f[lo:hi])
            prod = ff_ if prod is None else prod * ff_
        F_tilde = cp.matmul(adj_seg, prod[..., None]).squeeze(-1)
        div_f = gpu_contract_shared_operator_2axis(D_fine, F_tilde[..., None])[..., 0]
        div[lo:hi] = gpu_contract_shared_operator_1axis(
            f2c, div_f[..., None])[..., 0]
    return div


def compute_scalar_convection_residual_gpu(
    scalar_field, rho, velocity, mesh_data, ops_data, ff, n_cells, n_prism, n_sps,
    wall_dirichlet_zero_face=None, wall_dirichlet_value_face=None, has_wall_dirichlet_value=None,
    open_boundary_face=None, freestream_value=None,
):
    """标量对流 FR 残差（体积项 + 界面上风校正），与 CPU 版
    `compute_scalar_convection_residual` 逐字对应（含来流条件
    `open_boundary_face/freestream_value`，见 CPU 版参数文档）。
    `open_boundary_face` 是按边界组类型的掩码，这里与面邻居源推出的真
    边界面求与。"""
    cp = get_cupy()
    det_jacs = mesh_data['det_jacs']
    adj_j = mesh_data['adj_j']

    # 去混叠（AFCFD_TURB_OVERINT，默认 on）：与 CPU 端
    # `compute_scalar_convection_residual` 同一个开关、同一条链路。
    # 此前 GPU 侧完全没有这一层，导致同一个环境变量在两个后端意味着
    # 不同的数值方案——本项目不接受这种静默不一致（同一原则见
    # fr_solver/filter.py::resolve_filter_mode）。
    _segs = (get_overintegration_segs_gpu(mesh_data, ops_data, n_cells, n_prism)
             if resolve_turb_overintegration() == "on" else None)
    if _segs is not None:
        div_F = _scalar_volume_div_overintegrated_gpu(
            cp, (rho[..., None], scalar_field[..., None], velocity),
            _segs, n_cells, n_sps)
    else:
        rho_u_phi = rho[..., None] * velocity * scalar_field[..., None]  # (n_cells,n_sps,3)
        F_tilde = cp.matmul(adj_j, rho_u_phi[..., None]).squeeze(-1)  # (n_cells,n_sps,3)

        div_F = cp.zeros((n_cells, n_sps), dtype=cp.float64)
        if n_prism > 0:
            div_F[:n_prism] = gpu_contract_shared_operator_2axis(
                ops_data['D_3d_prism'], F_tilde[:n_prism, :, :, None]
            )[..., 0]
        if n_cells > n_prism:
            # native 四面体 D 矩阵分派（2026-09-03 补齐，见模块文档 /
            # gpu_gradients.py::compute_physical_gradient_gpu 同一处判据）。
            D_tet_op = (
                ops_data['D_native_tet_padded']
                if 'D_native_tet_padded' in ops_data
                else ops_data['D_3d_tet']
            )
            div_F[n_prism:] = gpu_contract_shared_operator_2axis(
                D_tet_op, F_tilde[n_prism:, :, :, None]
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
    if open_boundary_face is not None:
        is_true_boundary = (ff.neighbor_src0_cell < 0) & (ff.neighbor_src1_idx < 0)
        inflow = (open_boundary_face & is_true_boundary)[:, None] & (mass_flux < 0)
        phi_n = cp.where(inflow, freestream_value, phi_n)
    phi_upwind = cp.where(mass_flux >= 0, phi_o, phi_n)
    # 真实 bug 修复（2026-09-12，与 CPU 版 `transport.py::
    # compute_scalar_convection_residual` 同一处修复，完整推导见该函数
    # 模块文档"owner/neighbor 跳变量不对称"一节）：owner/neighbor 两侧
    # 的正确校正必须分别相对各自的面值计算——owner 侧沿用
    # `phi_upwind-phi_o`；neighbor 侧此前错误地复用了同一个 owner 参照
    # 的跳变量，在 owner 恰好是上风侧（mass_flux>=0，phi_upwind==phi_o）
    # 时该跳变量恒为 0，等价于 neighbor（真实网格中占全部内部面一半）
    # 完全收不到这个面本该有的对流稀释/浓缩效果。
    delta_phi_owner = phi_upwind - phi_o
    delta_phi_neighbor = phi_upwind - phi_n

    # 未加权原始跳变量（2026-09-03 修复，见 `_distribute_scalar_correction_
    # gpu` 文档）：不再在这里提前乘 |owner_adj_row_exact|——加权方式（
    # collapsed 用 |adj_row|，native 用 true_area_weight）延后到分配阶段
    # 按面类型分派，与 CPU 版 `raw_jump_fp = mass_flux * delta_phi_owner`
    # 逐字对应。
    raw_jump_fp = mass_flux * delta_phi_owner
    raw_jump_fp_neighbor = mass_flux * delta_phi_neighbor

    interface_correction = _distribute_scalar_correction_gpu(
        cp, ff, raw_jump_fp, det_jacs, n_cells, n_sps,
        raw_jump_fp_neighbor=raw_jump_fp_neighbor,
    )
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

    # 去混叠，与 CPU 端 `_scalar_diffusion_volume_overintegrated` 对应
    # （含那边写明的"Gamma 自身混叠仍在"这条已量化、刻意不实施的局限）。
    _segs = (get_overintegration_segs_gpu(mesh_data, ops_data, n_cells, n_prism)
             if resolve_turb_overintegration() == "on" else None)
    if _segs is not None:
        div_G = _scalar_volume_div_overintegrated_gpu(
            cp, (gamma_field[..., None], grad_phi),
            _segs, n_cells, n_sps)
    else:
        G_phys = gamma_field[..., None] * grad_phi
        G_tilde = cp.matmul(adj_j, G_phys[..., None]).squeeze(-1)  # (n_cells,n_sps,3)

        div_G = cp.zeros((n_cells, n_sps), dtype=cp.float64)
        if n_prism > 0:
            div_G[:n_prism] = gpu_contract_shared_operator_2axis(
                ops_data['D_3d_prism'], G_tilde[:n_prism, :, :, None]
            )[..., 0]
        if n_cells > n_prism:
            # native 四面体 D 矩阵分派（2026-09-03 补齐，同上）。
            D_tet_op = (
                ops_data['D_native_tet_padded']
                if 'D_native_tet_padded' in ops_data
                else ops_data['D_3d_tet']
            )
            div_G[n_prism:] = gpu_contract_shared_operator_2axis(
                D_tet_op, G_tilde[n_prism:, :, :, None]
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

    # 未加权原始跳变量（2026-09-03 修复，同 convection 侧，见
    # `_distribute_scalar_correction_gpu` 文档），与 CPU 版
    # `raw_jump_fp = flux_jump_phys` 逐字对应。
    raw_jump_fp = flux_jump_phys

    interface_correction = _distribute_scalar_correction_gpu(cp, ff, raw_jump_fp, det_jacs, n_cells, n_sps)
    return residual - interface_correction


def compute_turbulence_transport_residual_gpu(
    solver, grad_vel=None,
) -> Tuple:
    """GPU 版 k/omega 完整输运残差入口（对流+扩散），与 CPU 版
    `compute_turbulence_transport_residual` 逐字对应。

    2026-09-02 补齐：此前这里不做 CPU 版末尾的 `suppress_residual_
    outliers`（机制3，中位数离群值抑制）。那条"遗漏"曾于 2026-09-02
    被补齐（照抄 gpu_inviscid.py 的"倒回 CPU 复用 numba 实现"模式），
    **而机制3 已于 2026-09-19 整体删除** —— 真实网格消融对照证明它触发
    了但只把残差轨迹改变 ~1e-10 相对量、不改变发散结局（完整记录见
    `fr_residual/inviscid.py`）。所以这里连带去掉了那次
    `GPU -> CPU -> GPU` 往返，CPU 与 GPU 两侧现在都只有 isfinite 归零，
    对称性由"都没有"保证。

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
        # 真实 bug 修复（2026-09-03，同 fr_solver/turbulence.py::
        # compute_turbulence_source 文档同一处）：不能对*守恒*变量 U_gpu
        # 求梯度再切片动量分量冒充速度梯度——`vel`（上面已从 Q 取出）
        # 本来就是真正的速度，直接对它求梯度。
        from autoflowcfd.core.gpu.residual.gpu_gradients import compute_physical_gradient_gpu
        grad_vel = compute_physical_gradient_gpu(vel, solver.mesh_data, solver.ops_data)

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
    open_mask = solver._open_mask_gpu      # 同上，来流条件（compute_turbulence_face_masks_gpu）

    conv_k = compute_scalar_convection_residual_gpu(
        turb.k_field, rho, vel, solver.mesh_data, solver.ops_data, ff, n_cells, n_prism, n_sps,
        wall_dirichlet_zero_face=wall_mask_k,
        open_boundary_face=open_mask, freestream_value=float(turb.k_inf),
    )
    diff_k = compute_scalar_diffusion_residual_gpu(
        turb.k_field, gamma_k, solver.mesh_data, solver.ops_data, ff, n_cells, n_prism, n_sps,
    )
    dk_dt_transport = (conv_k + diff_k) / cp.maximum(rho, 1e-10)

    omega_wall_value_face, has_omega_wall = compute_omega_wall_target_gpu(
        cp, ff, wall_mask_k, d_wall, Q, mu, getattr(turb, "beta1", 0.075),
        omega_max=getattr(turb, "omega_max", 1e6),
        turb_k_field=getattr(turb, "k_field", None),
    )

    conv_w = compute_scalar_convection_residual_gpu(
        turb.omega_field, rho, vel, solver.mesh_data, solver.ops_data, ff, n_cells, n_prism, n_sps,
        wall_dirichlet_value_face=omega_wall_value_face, has_wall_dirichlet_value=has_omega_wall,
        open_boundary_face=open_mask, freestream_value=float(turb.omega_inf),
    )
    diff_w = compute_scalar_diffusion_residual_gpu(
        turb.omega_field, gamma_w, solver.mesh_data, solver.ops_data, ff, n_cells, n_prism, n_sps,
    )
    domega_dt_transport = (conv_w + diff_w) / cp.maximum(rho, 1e-10)

    # 机制3 已于 2026-09-19 删除（依据见 `fr_residual/inviscid.py`
    # 同一处）。这里同时去掉了那次为复用 CPU numba 实现而做的
    # `GPU -> CPU -> GPU` 往返（k/omega 的残差场与场值各拷一轮）。

    dk_dt_transport = cp.where(cp.isfinite(dk_dt_transport), dk_dt_transport, 0.0)
    domega_dt_transport = cp.where(cp.isfinite(domega_dt_transport), domega_dt_transport, 0.0)

    return dk_dt_transport, domega_dt_transport
