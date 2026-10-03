"""AutoFlowCFD V2.0 - 对流/扩散残差与顶层编排(GPU)

从 `src/autoflowcfd/core/gpu/turbulence/gpu_scalar_transport.py`(原 743 行)拆出(2026-09-24)；界面项 2026-09-26 改为两侧各自坐标系、扩散改为 IIPG 内罚（与 CPU 版同一结构）。
"""

from typing import Tuple

import numpy as np

from autoflowcfd.core.fr_operators.flux_kernels import resolve_viscous_ip_constant
from autoflowcfd.core.gpu import get_cupy

from autoflowcfd.core.gpu.residual.gpu_volume_contract import (
    gpu_contract_shared_operator_1axis, gpu_contract_shared_operator_2axis,
)

from autoflowcfd.core.gpu.residual.gpu_gradients import compute_physical_scalar_gradient_gpu


from autoflowcfd.core.turbulence.transport import resolve_turb_overintegration
from autoflowcfd.core.turbulence.sst.bounds import (
    clip_gradient_magnitude, model_evaluation_fields, omega_realizability_floor,
)
from autoflowcfd.core.turbulence.transport.faces import boundary_diffusion_targets
from autoflowcfd.core.turbulence.sst.log_omega import log_omega, log_omega_gradient_source

# 过积分上下文提取到 `core/gpu/gpu_overintegration.py`（2026-09-15，粘性
# 体积项 GPU 侧补齐时共用同一份，避免 residual 模块反向依赖 turbulence
# 模块）。原先那层 `as _turb_overint_segs_gpu` 兼容别名全仓库零引用，
# 2026-09-24 拆包时一并删除，调用点直接用真名。
from autoflowcfd.core.gpu.gpu_overintegration import (
    get_overintegration_segs_gpu,
)
from .faces import (
    _extrapolate_scalar_pair_gpu,
    _face_mass_flux_gpu,
    _lift_side_jumps_gpu,
)
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


def scalar_convection_volume_divergence_gpu(cp, scalar_field, rho, velocity, mesh_data, ops_data,
                                            n_cells, n_prism, n_sps):
    """对流体积算子 `div_vol(adj(J) rho u phi)`（除 det 之前），与 CPU 版
    `scalar_convection_volume_divergence` 逐字对应；残差（`phi`）与质量通量体积散度
    （`phi = 1`）共用这一个函数。

    去混叠（AFCFD_TURB_OVERINT，默认 on）与 CPU 端同一个开关、同一条链路——同一个
    环境变量在两个后端不允许意味着不同的数值方案（同一原则见
    fr_solver/filter.py::resolve_filter_mode）。
    """
    segs = (get_overintegration_segs_gpu(mesh_data, ops_data, n_cells, n_prism)
            if resolve_turb_overintegration() == "on" else None)
    if segs is not None:
        return _scalar_volume_div_overintegrated_gpu(
            cp, (rho[..., None], scalar_field[..., None], velocity), segs, n_cells, n_sps)
    rho_u_phi = rho[..., None] * velocity * scalar_field[..., None]  # (n_cells,n_sps,3)
    F_tilde = cp.matmul(mesh_data['adj_j'], rho_u_phi[..., None]).squeeze(-1)  # (n_cells,n_sps,3)
    div_F = cp.zeros((n_cells, n_sps), dtype=cp.float64)
    if n_prism > 0:
        div_F[:n_prism] = gpu_contract_shared_operator_2axis(
            ops_data['D_3d_prism'], F_tilde[:n_prism, :, :, None])[..., 0]
    if n_cells > n_prism:
        # native 四面体 D 矩阵分派（gpu_gradients.py::compute_physical_gradient_gpu 同一处判据）
        D_tet_op = ops_data['D_native_tet_padded'] if 'D_native_tet_padded' in ops_data else ops_data['D_3d_tet']
        div_F[n_prism:] = gpu_contract_shared_operator_2axis(D_tet_op, F_tilde[n_prism:, :, :, None])[..., 0]
    return div_F


def compute_scalar_convection_residual_gpu(
    scalar_field, rho, velocity, mesh_data, ops_data, ff, n_cells, n_prism, n_sps, mass_divergence,
    wall_dirichlet_zero_face=None, wall_dirichlet_value_face=None, has_wall_dirichlet_value=None,
    open_boundary_face=None, freestream_value=None,
):
    """标量对流 FR 残差（对流形式体积项 + 界面上风校正），与 CPU 版
    `compute_scalar_convection_residual` 逐字对应（含来流条件
    `open_boundary_face/freestream_value`，见 CPU 版参数文档）。
    `open_boundary_face` 是按边界组类型的掩码，这里与面邻居源推出的真
    边界面求与。`mass_divergence`：`scalar_convection_volume_divergence_gpu` 在
    `phi = 1` 上的值（只依赖冻结平均流，k 与 omega 两次调用共用，CPU 版对应
    `ScalarConvectionGeometry.mass_divergence`）。"""
    cp = get_cupy()
    det_jacs = mesh_data['det_jacs']
    div_F = scalar_convection_volume_divergence_gpu(
        cp, scalar_field, rho, velocity, mesh_data, ops_data, n_cells, n_prism, n_sps)
    div_F -= scalar_field * mass_divergence
    residual = -div_F / det_jacs

    # 界面项：两侧各自坐标系（CPU 版 `compute_scalar_convection_residual` 同一结构）
    masks = (wall_dirichlet_zero_face, wall_dirichlet_value_face, has_wall_dirichlet_value)
    m_o, m_n = _face_mass_flux_gpu(cp, ff, rho[..., None] * velocity)
    phi_o, phi_o_other = _extrapolate_scalar_pair_gpu(cp, ff, scalar_field, "owner", *masks)
    if open_boundary_face is not None:
        is_true_boundary = (ff.neighbor_src0_cell < 0) & (ff.neighbor_src1_idx < 0)
        inflow = (open_boundary_face & is_true_boundary)[:, None] & (m_o < 0)
        phi_o_other = cp.where(inflow, freestream_value, phi_o_other)
    jump_o = m_o * (cp.where(m_o >= 0, phi_o, phi_o_other) - phi_o)
    phi_n, phi_n_other = _extrapolate_scalar_pair_gpu(cp, ff, scalar_field, "neighbor", *masks)
    jump_n = m_n * (cp.where(m_n >= 0, phi_n, phi_n_other) - phi_n)
    return residual + _lift_side_jumps_gpu(cp, ff, jump_o, jump_n, -1.0, det_jacs, n_cells, n_sps)


def compute_scalar_diffusion_residual_gpu(
    scalar_field, gamma_field, mesh_data, ops_data, ff, n_cells, n_prism, n_sps, *, c_ip,
    wall_dirichlet_zero_face=None, wall_dirichlet_value_face=None, has_wall_dirichlet_value=None,
):
    """标量扩散 FR 残差（体积项 + IIPG 内罚界面项），与 CPU 版
    `compute_scalar_diffusion_residual` 逐项对应。`c_ip` 由调用方按阶数取
    `flux_kernels.resolve_viscous_ip_constant`（与平均流粘性项同一处定义）。"""
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

    # 界面项：IIPG + 内罚项，两侧各自坐标系；边界点壁面内罚 Dirichlet、其余齐次
    # Neumann（公式与边界分类见 CPU 版同名函数，分类规则共用 `boundary_diffusion_targets`）
    h_face = ff.ip_length[:, None]
    masks = (wall_dirichlet_zero_face, wall_dirichlet_value_face, has_wall_dirichlet_value)
    jumps = []
    for frame, normal in (("owner", ff.owner_unit_normal), ("neighbor", ff.neighbor_unit_normal)):
        g_self, g_other = _extrapolate_scalar_pair_gpu(cp, ff, gamma_field, frame)
        p_self, p_other = _extrapolate_scalar_pair_gpu(cp, ff, scalar_field, frame)
        gn_self = 0.0
        gn_other = 0.0
        for d in range(3):
            a, b = _extrapolate_scalar_pair_gpu(cp, ff, cp.ascontiguousarray(grad_phi[..., d]), frame)
            gn_self = gn_self + a * normal[..., d]
            gn_other = gn_other + b * normal[..., d]
        eta = c_ip * 0.5 * (g_self + g_other) / h_face
        is_bnd, is_dir, target = boundary_diffusion_targets(cp, ff, frame, *masks)
        j_int = 0.5 * (g_other * gn_other - g_self * gn_self) - eta * (p_self - p_other)
        j_bnd = cp.where(is_dir, -(c_ip * g_self / h_face) * (p_self - target), -g_self * gn_self)
        jumps.append(cp.where(is_bnd, j_bnd, j_int))
    return residual + _lift_side_jumps_gpu(cp, ff, jumps[0], jumps[1], +1.0, det_jacs, n_cells, n_sps)


def turbulence_diffusivities_gpu(cp, turb, k, omega, grad_k, grad_log_omega, rho, rho_nu_t, nu, mu, S_mag, d_wall):
    """GPU 版 k / ln(omega) 有效扩散系数 `mu + sigma(F1) rho nu_t`（CPU 版
    `transport/residual.py::turbulence_diffusivities` 的对应：梯度模长上限只作用在
    `grad k` 与 `grad ln(omega)` 上，交叉扩散用物理梯度 `omega grad ln(omega)`），输运
    残差与湍流解析 Jacobian 的 GPU 逐点求值器共用；两个梯度须已裁剪，同 CPU 版约定）。"""
    grad_dot = omega * cp.sum(grad_k * grad_log_omega, axis=-1)
    # 模型项求值用有效值（与 CPU 版同一处，定义在 `sst/bounds.py`）
    k_eff, omega_safe = model_evaluation_fields(k, omega, omega_realizability_floor(turb, S_mag, cp), cp)
    CD_kw = cp.maximum(2.0 * rho * turb.sigma_w2 / omega_safe * grad_dot, 1e-10)
    F1 = turb.compute_blending_F1_gpu(k_eff, omega_safe, d_wall, nu, rho, CD_kw)
    sigma_k = F1 * turb.sigma_k1 + (1.0 - F1) * turb.sigma_k2
    sigma_w = F1 * turb.sigma_w1 + (1.0 - F1) * turb.sigma_w2
    return mu + sigma_k * rho_nu_t, mu + sigma_w * rho_nu_t


def compute_turbulence_transport_residual_gpu(
    solver, grad_vel=None, grad_k=None, grad_log_omega=None,
) -> Tuple:
    """GPU 版 k 与 `w = ln(omega)` 的完整输运残差入口（对流+扩散，w 另加
    `Gamma_w |grad w|^2`，见 `core/turbulence/sst/log_omega.py`），与 CPU 版
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
        grad_vel, grad_k, grad_log_omega: 可选，调用方已经算好的速度梯度
            （CuPy (n_cells,n_sps,3,3)）与 k、ln(omega) 的物理梯度，复用避免重复
            计算物理梯度这个真实热点，与 CPU 版同名参数同一个性能考量。

    Returns:
        (dk_dt_transport, dw_dt_transport)，各自 (n_cells, n_sps)
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

    w_log = log_omega(turb.omega_field, cp)
    # 梯度模长上限在求梯度处施加一次（调用方传入的梯度已裁剪，同 CPU 版约定）
    if grad_k is None:
        grad_k = clip_gradient_magnitude(
            compute_physical_scalar_gradient_gpu(turb.k_field, solver.mesh_data, solver.ops_data), cp)
    if grad_log_omega is None:
        grad_log_omega = clip_gradient_magnitude(
            compute_physical_scalar_gradient_gpu(w_log, solver.mesh_data, solver.ops_data), cp)

    S_mag = turb.compute_strain_rate_magnitude_gpu(grad_vel)
    d_wall = solver.wall_distance_gpu
    gamma_k, gamma_w = turbulence_diffusivities_gpu(
        cp, turb, turb.k_field, turb.omega_field, grad_k, grad_log_omega, rho, rho_nu_t, nu, mu, S_mag,
        d_wall)

    ff = solver.flat_face_gpu
    n_prism = solver.mesh_data.get('n_prism', solver.mesh.n_prism_cells)

    wall_mask_k = solver._wall_mask_k_gpu  # 见 gpu_solver_io.py 缓存点文档
    open_mask = solver._open_mask_gpu      # 同上，来流条件（compute_turbulence_face_masks_gpu）

    # 对流形式体积项要减去的质量通量体积散度（k 与 omega 共用，CPU 版对应
    # `ScalarConvectionGeometry.mass_divergence`）
    mass_div = scalar_convection_volume_divergence_gpu(
        cp, cp.ones((n_cells, n_sps), dtype=cp.float64), rho, vel, solver.mesh_data, solver.ops_data,
        n_cells, n_prism, n_sps)
    conv_k = compute_scalar_convection_residual_gpu(
        turb.k_field, rho, vel, solver.mesh_data, solver.ops_data, ff, n_cells, n_prism, n_sps, mass_div,
        wall_dirichlet_zero_face=wall_mask_k,
        open_boundary_face=open_mask, freestream_value=float(turb.k_inf),
    )
    c_ip = resolve_viscous_ip_constant(int(solver.mesh.order))
    diff_k = compute_scalar_diffusion_residual_gpu(
        turb.k_field, gamma_k, solver.mesh_data, solver.ops_data, ff, n_cells, n_prism, n_sps,
        c_ip=c_ip, wall_dirichlet_zero_face=wall_mask_k,
    )
    dk_dt_transport = (conv_k + diff_k) / cp.maximum(rho, 1e-10)

    omega_wall_value_face, has_omega_wall = compute_omega_wall_target_gpu(
        cp, ff, wall_mask_k, d_wall, Q, mu, getattr(turb, "beta1", 0.075),
        omega_max=getattr(turb, "omega_max", 1e6),
        turb_k_field=getattr(turb, "k_field", None),
    )

    # w = ln(omega) 的边界值：壁面目标取对数（没有目标的面该值不被读取），来流取 ln(omega_inf)
    log_wall_value_face = log_omega(omega_wall_value_face, cp)
    conv_w = compute_scalar_convection_residual_gpu(
        w_log, rho, vel, solver.mesh_data, solver.ops_data, ff, n_cells, n_prism, n_sps, mass_div,
        wall_dirichlet_value_face=log_wall_value_face, has_wall_dirichlet_value=has_omega_wall,
        open_boundary_face=open_mask, freestream_value=float(np.log(turb.omega_inf)),
    )
    diff_w = compute_scalar_diffusion_residual_gpu(
        w_log, gamma_w, solver.mesh_data, solver.ops_data, ff, n_cells, n_prism, n_sps,
        c_ip=c_ip, wall_dirichlet_value_face=log_wall_value_face, has_wall_dirichlet_value=has_omega_wall,
    )
    dw_dt_transport = ((conv_w + diff_w + log_omega_gradient_source(gamma_w, grad_log_omega, cp))
                       / cp.maximum(rho, 1e-10))

    # 机制3 已于 2026-09-19 删除（依据见 `fr_residual/inviscid.py`
    # 同一处）。这里同时去掉了那次为复用 CPU numba 实现而做的
    # `GPU -> CPU -> GPU` 往返（k/omega 的残差场与场值各拷一轮）。

    dk_dt_transport = cp.where(cp.isfinite(dk_dt_transport), dk_dt_transport, 0.0)
    dw_dt_transport = cp.where(cp.isfinite(dw_dt_transport), dw_dt_transport, 0.0)

    return dk_dt_transport, dw_dt_transport
