"""AutoFlowCFD V2.0 - 湍流输运方程隐式步的解析单元块 Jacobian。

冻结平均流下的湍流残差（紧耦合 Newton 残差的湍流子系统，`time_integration/implicit/coupled_step.py`）

    R_t,v = -[S_v + conv_v + diff_v] / rho          v = 0 .. nv-1（模型的 Newton 未知量）

对单元自身真实自由度的导数块（布局与 `CellBlockJacobian` 相同：逐单元
`(n_real*nv, n_real*nv)`，下标 `s*nv + v`）与面邻居耦合块。组成：

    pointwise.py    S、Gamma 对 (phi, grad phi) 的逐点导数（对模型函数差分）
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
from .scalar_blocks import assemble_scalar_blocks, convection_ghost_affine


@dataclass
class TurbulenceLinearization:
    """冻结的平均流输入与边界数据（与湍流残差同一份，见模块文档）。

    壁面 Dirichlet 条件逐个 Newton 未知量给出（`dirichlet_faces[v]` 为面掩码，
    `dirichlet_values[v]` 为未知量空间的逐通量点目标值，None 表示目标 0）。
    """
    mesh: object
    ops: object
    flat: object
    turb: object
    Q: np.ndarray                 # (n_cells, n_sps, 5) 原始变量
    grad_vel: np.ndarray          # (n_cells, n_sps, 3, 3)
    d_wall: np.ndarray            # (n_cells, n_sps)
    mu: float
    conv_geom: object             # transport/faces.py::ScalarConvectionGeometry
    dirichlet_faces: tuple        # 逐未知量 (n_faces,) 壁面 Dirichlet 面掩码
    dirichlet_values: tuple       # 逐未知量 (n_faces, n_fp) 目标值或 None（目标 0）
    open_face: np.ndarray         # (n_faces,) 开放边界（来流条件）
    pointwise: object = None      # 逐点 (S, Gamma) 求值器；None -> cpu_turbulence_pointwise（SST）


def assemble_turbulence_blocks(ctx: TurbulenceLinearization, u_flat, want_coupling: bool = False):
    """返回 `(blocks_prism, blocks_tet[, coupling])`（float32，`CellBlockJacobian` 布局，`n_var = nv`）。"""
    mesh, flat, turb = ctx.mesh, ctx.flat, ctx.turb
    n_cells, n_sps = int(mesh.n_cells), int(mesh.n_sps_per_cell)
    nv = int(turb.n_transported)
    u = np.asarray(u_flat, dtype=np.float64).reshape(n_cells, n_sps, nv)
    g = np.stack([compute_physical_scalar_gradient(np.ascontiguousarray(u[..., v]), mesh, ctx.ops)
                  for v in range(nv)], axis=-2)
    evaluate = ctx.pointwise if ctx.pointwise is not None else cpu_turbulence_pointwise(
        turb, ctx.Q, ctx.grad_vel, ctx.d_wall, ctx.mu)
    _, gam, dS, dG = turbulence_pointwise_partials(evaluate, u, g, turb.unknown_scales())
    gam = np.ascontiguousarray(gam)
    dG = np.ascontiguousarray(dG)
    dS = np.ascontiguousarray(dS)
    rho = ctx.Q[:, :, 0]
    m_o = np.ascontiguousarray(ctx.conv_geom.mass_flux)
    m_n = np.ascontiguousarray(ctx.conv_geom.mass_flux_neighbor)
    faces = tuple(np.asarray(f, dtype=bool) for f in ctx.dirichlet_faces)
    open_face = np.asarray(ctx.open_face, dtype=bool)
    ghost_o, a_o = convection_ghost_affine(flat, "owner", m_o, faces, open_face)
    ghost_n, a_n = convection_ghost_affine(flat, "neighbor", m_n, faces, open_face)
    diff = {}
    for frame in ("owner", "neighbor"):
        is_bnd, is_dir, target = None, [], []
        for face_mask, value in zip(faces, ctx.dirichlet_values):
            b, d, t = boundary_diffusion_targets(
                np, flat, frame, None, None if value is None else np.asarray(value, dtype=np.float64), face_mask)
            is_bnd = b if is_bnd is None else is_bnd | b
            is_dir.append(d)
            target.append(t)
        diff[frame] = (np.ascontiguousarray(is_bnd), np.ascontiguousarray(np.stack(is_dir)),
                       np.ascontiguousarray(np.stack(target)))
    return assemble_scalar_blocks(
        mesh, ctx.ops, flat, np.ascontiguousarray(u), gam, dS, dG, rho, ctx.Q[:, :, 1:4],
        ctx.conv_geom.rho_u_tilde, m_o, m_n,
        (ghost_o, np.ascontiguousarray(a_o), ghost_n, np.ascontiguousarray(a_n)), diff, want_coupling)
