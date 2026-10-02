"""AutoFlowCFD V2.0 - k-omega 隐式步的解析单元块 Jacobian。

冻结平均流下的湍流残差（`fr_solver/turbulence/implicit.py::TurbulenceResidual`）

    R_t,v = -[S_v + conv_v + diff_v] / rho          v = k, w = ln omega

对单元自身真实自由度的导数块（布局与 `CellBlockJacobian` 相同：逐单元
`(n_real*2, n_real*2)`，下标 `s*2 + v`）与面邻居耦合块。组成：

    pointwise.py    S、Gamma 对 (k, w, grad k, grad w) 的逐点导数（对模型函数差分）
    cell_blocks.py  源项 + 对流/扩散体积项（含过积分）
    faces.py        对流/扩散界面项（逐点跳变量与残差核共用 `face_frames.py` 的点函数）
    scalar_blocks.py 上面两者的装配与收尾（与模型无关，人工粘性的标量扩散 Jacobian 共用）

平均流量（密度、速度、质量通量、壁面目标值）在整个湍流 Newton 步内冻结，由后端在
步起点准备好传入（`TurbulenceLinearization`），与残差同一份。
"""

from dataclasses import dataclass

import numpy as np

from autoflowcfd.core.fr_operators.gradients import compute_physical_scalar_gradient
from autoflowcfd.core.turbulence.transport.faces import boundary_diffusion_targets

from .pointwise import cpu_turbulence_pointwise, turbulence_pointwise_partials
from .scalar_blocks import assemble_scalar_pair_blocks, convection_ghost_affine
from autoflowcfd.core.turbulence.sst.bounds import turbulence_scales
from autoflowcfd.core.turbulence.sst.log_omega import log_omega


@dataclass
class TurbulenceLinearization:
    """冻结的平均流输入与边界数据（与湍流残差同一份，见模块文档）。"""
    mesh: object
    ops: object
    flat: object
    turb: object
    Q: np.ndarray                 # (n_cells, n_sps, 5) 原始变量
    grad_vel: np.ndarray          # (n_cells, n_sps, 3, 3)
    d_wall: np.ndarray            # (n_cells, n_sps)
    mu: float
    conv_geom: object             # transport/faces.py::ScalarConvectionGeometry
    wall_zero_face: np.ndarray    # (n_faces,) k 的壁面 Dirichlet 0
    omega_wall_face: np.ndarray   # (n_faces, n_fp) omega 壁面目标值（物理量，装配时取 ln）
    has_omega_wall: np.ndarray    # (n_faces,)
    open_face: np.ndarray         # (n_faces,) 开放边界（来流条件）
    pointwise: object = None      # 逐点 (S, Gamma) 求值器；None -> cpu_turbulence_pointwise


def assemble_turbulence_blocks(ctx: TurbulenceLinearization, kw_flat, want_coupling: bool = False):
    """返回 `(blocks_prism, blocks_tet[, coupling])`（float32，`CellBlockJacobian` 布局，`n_var=2`）。"""
    mesh, flat, turb = ctx.mesh, ctx.flat, ctx.turb
    n_cells, n_sps = int(mesh.n_cells), int(mesh.n_sps_per_cell)
    kw = np.asarray(kw_flat, dtype=np.float64).reshape(n_cells, n_sps, 2)
    k = np.ascontiguousarray(kw[..., 0])
    w = np.ascontiguousarray(kw[..., 1])
    grad_k = compute_physical_scalar_gradient(k, mesh, ctx.ops)
    grad_w = compute_physical_scalar_gradient(w, mesh, ctx.ops)
    evaluate = ctx.pointwise if ctx.pointwise is not None else cpu_turbulence_pointwise(
        turb, ctx.Q, ctx.grad_vel, ctx.d_wall, ctx.mu)
    _, gam, dS, dG = turbulence_pointwise_partials(evaluate, k, w, grad_k, grad_w, turbulence_scales(turb))
    gam = np.ascontiguousarray(gam)
    dG = np.ascontiguousarray(dG)
    dS = np.ascontiguousarray(dS)
    rho = ctx.Q[:, :, 0]
    m_o = np.ascontiguousarray(ctx.conv_geom.mass_flux)
    m_n = np.ascontiguousarray(ctx.conv_geom.mass_flux_neighbor)
    wz = np.asarray(ctx.wall_zero_face, dtype=bool)
    hv = np.asarray(ctx.has_omega_wall, dtype=bool)
    ghost_o, a_o = convection_ghost_affine(flat, "owner", m_o, wz, hv, np.asarray(ctx.open_face, dtype=bool))
    ghost_n, a_n = convection_ghost_affine(flat, "neighbor", m_n, wz, hv, np.asarray(ctx.open_face, dtype=bool))
    diff = {}
    # w = ln(omega) 的壁面 Dirichlet 值（与残差 transport/residual.py 同一换算）
    log_wall_face = log_omega(np.asarray(ctx.omega_wall_face, dtype=np.float64), np)
    for frame in ("owner", "neighbor"):
        bk, dk, tk = boundary_diffusion_targets(np, flat, frame, wz, None, None)
        bw, dw, tw = boundary_diffusion_targets(np, flat, frame, None, log_wall_face, hv)
        diff[frame] = (np.ascontiguousarray(bk | bw), np.ascontiguousarray(np.stack([dk, dw])),
                       np.ascontiguousarray(np.stack([tk, tw])))
    return assemble_scalar_pair_blocks(
        mesh, ctx.ops, flat, kw, gam, dS, dG, rho, ctx.Q[:, :, 1:4], ctx.conv_geom.rho_u_tilde, m_o, m_n,
        (ghost_o, np.ascontiguousarray(a_o), ghost_n, np.ascontiguousarray(a_n)), diff, want_coupling)
