"""AutoFlowCFD V2.0 - 单个标量的对流 / 扩散 FR 残差（GPU；与湍流模型无关）

CPU 版 `core/turbulence/transport/convection.py` / `diffusion.py` 的对应。湍流模型经
`primitives.py::GpuScalarTransport` 调用（SST 的 k/ln(omega) 输运见 `residual.py`，SA-neg 见
`core/turbulence/sa/rates.py`），人工粘性的标量扩散直接调用 `compute_scalar_diffusion_residual_gpu`。
"""

from autoflowcfd.core.gpu import get_cupy
from autoflowcfd.core.fr_operators.flux_kernels.viscous import smooth_nonnegative_array
from autoflowcfd.core.gpu.gpu_overintegration import get_overintegration_segs_gpu
from autoflowcfd.core.gpu.residual.gpu_gradients import compute_physical_scalar_gradient_gpu
from autoflowcfd.core.gpu.residual.gpu_volume_contract import (
    gpu_contract_shared_operator_1axis, gpu_contract_shared_operator_2axis,
)
from autoflowcfd.core.turbulence.transport.faces import boundary_diffusion_targets, inflow_points

from .faces import (
    _extrapolate_scalar_pair_gpu,
    _face_mass_flux_gpu,
    _lift_side_jumps_gpu,
)


def _scalar_volume_div_overintegrated_gpu(cp, factors, segs,
                                          n_cells, n_sps, first_band=None):
    """标量体积项 `div(adj(J) * prod(factors))` 的去混叠版（GPU）。

    `factors` 是一串 (n_cells, n_sps, k) 的场，k 为 1 或 3；各自精确插值
    到 FINE 点后**在 FINE 点相乘**（去混叠的全部内容就是"先插值再相乘"），
    再用细点度量算逆变通量、细网格微分矩阵求散度、精确限制回 coarse。

    对流用 (rho, phi, u)、扩散用 (gamma, grad_phi)，与 CPU 端
    `_scalar_convection_volume_overintegrated` /
    `_scalar_diffusion_volume_overintegrated` 逐字对应。`first_band` 不为 None 时第一个因子是扩散系数，
    细点上的插值换成取用值 `smooth_nonnegative_array(., first_band)`（与 CPU 同一处理）。
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
            if prod is None and first_band is not None:
                ff_ = smooth_nonnegative_array(ff_, first_band, cp)
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

    体积项恒走过积分（去混叠），与 CPU 端同一条链路；P0 没有过积分几何时走单点路径。
    """
    segs = (get_overintegration_segs_gpu(mesh_data, ops_data, n_cells, n_prism))
    if segs is not None:
        return _scalar_volume_div_overintegrated_gpu(
            cp, (rho[..., None], scalar_field[..., None], velocity), segs, n_cells, n_sps)
    rho_u_phi = rho[..., None] * velocity * scalar_field[..., None]  # (n_cells,n_sps,3)
    F_tilde = cp.matmul(mesh_data['adj_j'], rho_u_phi[..., None]).squeeze(-1)  # (n_cells,n_sps,3)
    div_F = cp.zeros((n_cells, n_sps), dtype=cp.float64)
    if n_prism > 0:
        div_F[:n_prism] = gpu_contract_shared_operator_2axis(
            ops_data['D_native_prism_padded'], F_tilde[:n_prism, :, :, None])[..., 0]
    if n_cells > n_prism:
        # native 四面体 D 矩阵分派（gpu_gradients.py::compute_physical_gradient_gpu 同一处判据）
        D_tet_op = ops_data['D_native_tet_padded']
        div_F[n_prism:] = gpu_contract_shared_operator_2axis(D_tet_op, F_tilde[n_prism:, :, :, None])[..., 0]
    return div_F


def compute_scalar_convection_residual_gpu(
    scalar_field, rho, velocity, mesh_data, ops_data, ff, n_cells, n_prism, n_sps, mass_divergence,
    wall_dirichlet_zero_face=None, wall_dirichlet_value_face=None, has_wall_dirichlet_value=None,
    open_boundary_face=None, freestream_value=None,
):
    """标量对流 FR 残差（对流形式体积项 + 界面上风校正），与 CPU 版
    `compute_scalar_convection_residual` 逐字对应（含来流条件
    `open_boundary_face/freestream_value`，见 CPU 版参数文档；来流点与 CPU 共用
    `turbulence/transport/faces.py::inflow_points`）。`mass_divergence`：`scalar_convection_volume_divergence_gpu` 在
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
        phi_o_other = cp.where(inflow_points(cp, ff, "owner", open_boundary_face, m_o), freestream_value, phi_o_other)
    jump_o = m_o * (cp.where(m_o >= 0, phi_o, phi_o_other) - phi_o)
    phi_n, phi_n_other = _extrapolate_scalar_pair_gpu(cp, ff, scalar_field, "neighbor", *masks)
    if open_boundary_face is not None:
        phi_n_other = cp.where(inflow_points(cp, ff, "neighbor", open_boundary_face, m_n), freestream_value,
                               phi_n_other)
    jump_n = m_n * (cp.where(m_n >= 0, phi_n, phi_n_other) - phi_n)
    return residual + _lift_side_jumps_gpu(cp, ff, jump_o, jump_n, -1.0, det_jacs, n_cells, n_sps)


def compute_scalar_diffusion_residual_gpu(
    scalar_field, gamma_field, mesh_data, ops_data, ff, n_cells, n_prism, n_sps, *, c_ip, gamma_band,
    wall_dirichlet_zero_face=None, wall_dirichlet_value_face=None, has_wall_dirichlet_value=None, grad_phi=None,
):
    """标量扩散 FR 残差（体积项 + IIPG 内罚界面项），与 CPU 版
    `compute_scalar_diffusion_residual` 逐项对应。`c_ip` 由调用方按阶数取
    `flux_kernels.resolve_viscous_ip_constant`（与平均流粘性项同一处定义）；`gamma_band` 是非解点处扩散
    系数取用值的过渡带宽（含义与取值同 CPU 版）。`grad_phi`：调用方已为源项算过的同一份物理梯度，None 时在这里算
    （与 CPU 版同名参数一致，I-63）。"""
    cp = get_cupy()
    det_jacs = mesh_data['det_jacs']
    adj_j = mesh_data['adj_j']

    if grad_phi is None:
        grad_phi = compute_physical_scalar_gradient_gpu(scalar_field, mesh_data, ops_data)  # (n_cells,n_sps,3)

    # 去混叠，与 CPU 端 `_scalar_diffusion_volume_overintegrated` 对应
    # （含那边写明的"Gamma 自身混叠仍在"这条已量化、刻意不实施的局限）。
    _segs = (get_overintegration_segs_gpu(mesh_data, ops_data, n_cells, n_prism))
    if _segs is not None:
        div_G = _scalar_volume_div_overintegrated_gpu(
            cp, (gamma_field[..., None], grad_phi),
            _segs, n_cells, n_sps, first_band=float(gamma_band))
    else:
        G_phys = gamma_field[..., None] * grad_phi
        G_tilde = cp.matmul(adj_j, G_phys[..., None]).squeeze(-1)  # (n_cells,n_sps,3)

        div_G = cp.zeros((n_cells, n_sps), dtype=cp.float64)
        if n_prism > 0:
            div_G[:n_prism] = gpu_contract_shared_operator_2axis(
                ops_data['D_native_prism_padded'], G_tilde[:n_prism, :, :, None]
            )[..., 0]
        if n_cells > n_prism:
            # native 四面体 D 矩阵分派（2026-09-03 补齐，同上）。
            D_tet_op = ops_data['D_native_tet_padded']
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
        # 面上外插的扩散系数换成取用值（与 CPU `face_frames.py::diffusion_jump_point` 同一处理）
        g_self = smooth_nonnegative_array(g_self, float(gamma_band), cp)
        g_other = smooth_nonnegative_array(g_other, float(gamma_band), cp)
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
