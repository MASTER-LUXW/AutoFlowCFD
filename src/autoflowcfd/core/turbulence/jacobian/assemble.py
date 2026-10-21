"""AutoFlowCFD V2.0 - 湍流输运方程隐式步的解析单元块 Jacobian。

冻结平均流下的湍流残差（紧耦合 Newton 残差的湍流子系统，`time_integration/implicit/newton/coupled_step.py`）

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
    `dirichlet_values[v]` 为未知量空间的逐通量点目标值，None 表示目标 0）。`strong_rows`
    是解点上的强 Dirichlet 行（`(n_cells, n_sps, nv)` 布尔，残差在那里恒为零，块的对应行
    置零；SA-neg 的壁面解点，见 `sa/model.py` 模块文档），None 表示没有。
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
    strong_rows: np.ndarray = None  # (n_cells, n_sps, nv) 强 Dirichlet 行
    des: object = None            # SST 的 DES 长度尺度求值器（与残差同一个）；None 为 RANS
    wall_target_k_slope: np.ndarray = None  # (n_faces,) 第 1 个未知量的壁面目标值对 owner 单元第 0 个未知量
    #                                         真实自由度均值的导数（SST auto 档 d ln omega_w / d k），None 表示没有


def assemble_turbulence_blocks(ctx: TurbulenceLinearization, u_flat, want_coupling: bool = False):
    """返回 `(blocks_prism, blocks_tet[, coupling])`（float32，`CellBlockJacobian` 布局，`n_var = nv`）。"""
    mesh, flat, turb = ctx.mesh, ctx.flat, ctx.turb
    n_cells, n_sps = int(mesh.n_cells), int(mesh.n_sps_per_cell)
    nv = int(turb.n_transported)
    u = np.asarray(u_flat, dtype=np.float64).reshape(n_cells, n_sps, nv)
    g = np.stack([compute_physical_scalar_gradient(np.ascontiguousarray(u[..., v]), mesh, ctx.ops)
                  for v in range(nv)], axis=-2)
    evaluate = ctx.pointwise if ctx.pointwise is not None else cpu_turbulence_pointwise(
        turb, ctx.Q, ctx.grad_vel, ctx.d_wall, ctx.mu, ctx.des)
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
    out = assemble_scalar_blocks(
        mesh, ctx.ops, flat, np.ascontiguousarray(u), gam, dS, dG, rho, ctx.Q[:, :, 1:4],
        ctx.conv_geom.rho_u_tilde, m_o, m_n,
        (ghost_o, np.ascontiguousarray(a_o), ghost_n, np.ascontiguousarray(a_n)), diff, want_coupling,
        gamma_band=float(ctx.mu))
    if ctx.wall_target_k_slope is not None:
        add_wall_target_k_coupling(ctx, out, gam)
    if ctx.strong_rows is not None:
        zero_strong_rows(out, np.asarray(ctx.strong_rows, dtype=bool), int(mesh.n_prism_cells), nv)
    return out


def add_wall_target_k_coupling(ctx: TurbulenceLinearization, out, gam) -> None:
    """壁面目标值随状态变化带来的耦合（原地加到对角块）：SST auto 档的 omega 壁面目标值
    `w_wall = ln omega_wall(k_owner)`，`k_owner` 为 owner 单元 k 的真实自由度均值（`transport/omega_wall.py`）。

    w 的对流与扩散残差对边界目标值是线性齐次的（对流取镜像幽灵态 `2 g - w`，扩散的罚项与边界通量都对 g 线性，
    质量通量与扩散系数冻结），所以 `sum_f dT_w/dg_f * dg_f/dk_owner` 等于"w 取 0、目标值取 slope_f、来流值取 0"
    时的输运残差本身——与残差同一组函数一次求值得到，没有差分截断。由此

        d R_w[c, s] / d k[c, t] += -T_probe[c, s] / rho[c, s] / n_real(c)   （t 为真实解点）

    边界面只进它的 owner 单元（混合拆分面的配对边界面 owner 是同一单元），所以只有对角块。
    """
    from autoflowcfd.core.turbulence.transport import (
        compute_scalar_convection_residual, compute_scalar_diffusion_residual,
    )
    from autoflowcfd.fr.native_padding import real_sps_per_cell

    slope = np.asarray(ctx.wall_target_k_slope, dtype=np.float64)
    if not np.any(slope):
        return
    mesh, flat = ctx.mesh, ctx.flat
    n_cells, n_sps = int(mesh.n_cells), int(mesh.n_sps_per_cell)
    has = np.asarray(ctx.dirichlet_faces[1], dtype=bool)
    probe = np.ascontiguousarray(np.broadcast_to(slope[:, None], (slope.size, int(flat.n_fp))))
    zero = np.zeros((n_cells, n_sps))
    rho = np.ascontiguousarray(ctx.Q[:, :, 0])
    T = compute_scalar_convection_residual(
        zero, rho, ctx.Q[:, :, 1:4], mesh, ctx.ops, wall_dirichlet_value_face=probe, has_wall_dirichlet_value=has,
        flat_face_override=flat, conv_geom=ctx.conv_geom, open_boundary_face=np.asarray(ctx.open_face, dtype=bool),
        freestream_value=0.0)
    T = T + compute_scalar_diffusion_residual(
        zero, np.ascontiguousarray(gam[..., 1]), mesh, ctx.ops, wall_dirichlet_value_face=probe,
        has_wall_dirichlet_value=has, flat_face_override=flat, gamma_band=float(ctx.mu))
    col = -T / np.maximum(rho, 1e-10)
    nv = int(ctx.turb.n_transported)
    n_prism = int(mesh.n_prism_cells)
    npr, nte = real_sps_per_cell(int(mesh.order))
    for blocks, lo, n in ((out[0], 0, npr), (out[1], n_prism, nte)):
        if blocks.shape[0] == 0:
            continue
        b5 = blocks.reshape(blocks.shape[0], n, nv, n, nv)
        add = (col[lo:lo + blocks.shape[0], :n] / n).astype(blocks.dtype)       # (cells, n)
        b5[:, :, 1, :, 0] += add[:, :, None]


def zero_strong_rows(out, strong, n_prism: int, nv: int) -> None:
    """把强 Dirichlet 行（`strong (n_cells, n_sps, nv)`）在单元块与面耦合块里置零（原地）。
    块的行下标是 `s*nv + v`（`s` 为真实解点序号），与 `CellBlockJacobian` 布局一致。"""
    blocks_prism, blocks_tet = out[0], out[1]
    for blocks, cells in ((blocks_prism, np.arange(n_prism)),
                          (blocks_tet, np.arange(n_prism, n_prism + blocks_tet.shape[0]))):
        if blocks.shape[0] == 0:
            continue
        n_rows = blocks.shape[1]
        rows = strong[cells, : n_rows // nv, :].reshape(cells.size, n_rows)
        blocks[rows] = 0.0
    if len(out) > 2 and out[2] is not None:
        for g in out[2].groups:
            n_rows = g.blocks.shape[1]
            rows = strong[g.rows, : n_rows // nv, :].reshape(g.rows.size, n_rows)
            g.blocks[rows] = 0.0
