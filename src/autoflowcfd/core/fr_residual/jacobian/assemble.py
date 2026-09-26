"""AutoFlowCFD V2.0 - 平均流残差的解析单元对角块（P>=1）。

`R(U) = Gamma(U) * [-(无粘 + 粘性)](U)`（启用低马赫预处理时带 `Gamma`，否则
`Gamma = I`）对单元自身真实自由度的导数块 `J_cc = dR_c/dU_c`，布局与
`time_integration/implicit/block_jacobi.py::CellBlockJacobian` 完全相同
（逐单元 `(n_real*5, n_real*5)`，行列下标 `s*5 + v`，float32）。

## 为什么要解析装配

着色有限差分装配要 `色数 x 真实解点数 x 5` 次整场残差求值：plate_demo P2 是
450 次（单次 5.5 s，约 41 分钟）、P3 约 1000 次。非线性只出现在逐点函数上，
其余是线性算子——`J_cc` 等于"逐点导数"与"参考算子"的收缩，代价只相当于几次
残差求值，而且得到的是同一个矩阵（单元测试逐块对照有限差分装配，相对误差
`1e-6` 量级，即差分截断误差本身）。

## 组成

    体积项（无粘 + 粘性，含过积分）    volume.py
    界面项（本侧迹与本侧梯度迹）        faces.py，逐点跳变量与残差核共用
                                        fr_residual/face_point_jumps.py
    dQ/dU（列）、Gamma（行）            本文件

全部在原始变量空间累加（`K[c, s, a, t, b] = d(dU/dt)_{s,a} / dQ_{t,b}` 乘 det），
最后逐单元除 `det_s`、取负号（`R = -dU/dt`）、右乘 `dQ/dU`、左乘 `Gamma`，并加上
`Gamma` 自身随状态变化的那一项 `d(Gamma R)/dU |_{R 固定}`（逐解点、只进对角）。

`mu_t` 按平均流残差的约定整步冻结，不对它求导。人工粘性质量扩散通道与 WMLES
壁面应力修正不在稳态隐式路径里（前者默认关闭，后者只用于瞬态），熵稳定两点
通量体积项由调用方改用差分装配（见 `supports`）。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry
from autoflowcfd.core.fr_operators.flux_kernels import resolve_viscous_ip_constant
from autoflowcfd.core.fr_operators.gradients import compute_physical_gradient
from autoflowcfd.core.fr_operators.volume_contract import compute_adj_j, get_overintegration_context
from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
from autoflowcfd.core.fr_residual.inviscid_kernel import compute_boundary_ghost_states
from autoflowcfd.core.fr_residual.viscous_flux import compute_temperature, resolve_viscous_overintegration
from autoflowcfd.core.fr_residual.viscous_flux.constants import PRANDTL, PRANDTL_TURBULENT
from autoflowcfd.fr.native_padding import real_sps_per_cell

from .coupling import cross_layout, finalize_coupling
from .faces import add_face_blocks_color
from .pointwise import _SQRT_EPS, primitive_jacobian, temperature_gradient_row
from .transform import finalize_diag_kernel
from .volume import VolumeSegment, add_volume_blocks


@dataclass
class MeanFlowLinearization:
    """平均流残差的离散参数（与残差调用同一组值）。"""
    mesh: object
    ops: object
    ghost_provider: object
    mu: float
    mach_ref: float
    precond_mode: int
    low_mach: bool
    mu_t: Optional[np.ndarray] = None        # (n_cells, n_sps) 冻结的动力涡粘
    flat: object = None                      # None -> get_flat_face_geometry(mesh, ops)
    Pr: float = PRANDTL
    Pr_t: float = PRANDTL_TURBULENT

    def flat_geometry(self):
        return self.flat if self.flat is not None else get_flat_face_geometry(self.mesh, self.ops)


def _segments(ctx, n_real_prism, n_real_tet):
    mesh, ops = ctx.mesh, ctx.ops
    oi = get_overintegration_context(mesh, ops)
    if oi is None:
        raise ValueError("解析 Jacobian 只用于 P>=1（P0 没有过积分几何，走差分装配）")
    visc_overint = resolve_viscous_overintegration() == "on"
    D_sp = {0: np.asarray(ops.D_3d_prism), 1: np.asarray(ops.D_native_tet_padded)}
    out = []
    for kind, (lo, hi, nf, det_f, inv_f, c2f, D_f, f2c) in enumerate(oi["segs"]):
        n = n_real_prism if kind == 0 else n_real_tet
        c2f_r = np.ascontiguousarray(np.asarray(c2f)[:, :n])
        W = np.einsum("sq,qrm->srm", np.asarray(f2c)[:n], np.asarray(D_f), optimize=True)
        Dn = np.ascontiguousarray(D_sp[kind][:n, :n])
        if visc_overint:
            c2f_v, W_v = c2f_r, W
        else:
            c2f_v, W_v = np.eye(n), Dn
        out.append((kind, VolumeSegment(lo, hi, n, c2f_r, W, c2f_v, W_v, Dn), det_f, inv_f, visc_overint))
    return out


def _ghost_perturbations(flat, Q, adj_j, provider):
    """边界幽灵态及其在"整场原始变量逐分量均匀平移"下的值（faces.py 复合差分用）。

    返回 `(Qg0, QgP, HG, row)`：`Qg0` 与残差同形 `(n_faces, n_fp, 5)`；`QgP[v]` 只存
    活跃边界面（紧凑行，`row[f]` 为其下标，非活跃面为 -1）。
    """
    Qg0 = compute_boundary_ghost_states(flat, Q, adj_j, provider)
    active = np.asarray(flat.is_boundary) & (np.asarray(flat.owner_is_primary) | np.asarray(flat.mixed_bnd_face))
    faces = np.nonzero(active)[0]
    row = -np.ones(flat.n_faces, dtype=np.int64)
    row[faces] = np.arange(faces.size)
    rho, vel, p = Q[..., 0], Q[..., 1:4], Q[..., 4]
    c = np.sqrt(np.maximum(1.4 * p / np.maximum(rho, 1e-300), 0.0))
    vref = float((np.linalg.norm(vel, axis=-1) + c).max())
    HG = _SQRT_EPS * np.array([np.abs(rho).max(), vref, vref, vref, np.abs(p).max()])
    QgP = np.empty((5, max(faces.size, 1), Qg0.shape[1], 5))
    for v in range(5):
        Qp = Q.copy()
        Qp[..., v] += HG[v]
        QgP[v, :faces.size] = compute_boundary_ghost_states(flat, Qp, adj_j, provider)[faces]
    return Qg0, QgP, HG, row


def assemble_mean_flow_blocks(ctx: MeanFlowLinearization, U, residual=None, want_coupling: bool = False):
    """返回 `(blocks_prism, blocks_tet)`（float32，`CellBlockJacobian` 布局）；
    `want_coupling` 时再返回面邻居耦合块 `CouplingBlocks`（`coupling.py`，块 ILU 用）。

    Args:
        U: `(n_cells, n_sps, >=5)` 守恒变量（本 rank 的全部单元，含分布式 halo）。
        residual: `(n_cells, n_sps, 5)` Newton 实际在解的残差（启用低马赫预处理时
            是 `Gamma R_raw`，用于 `d(Gamma R_raw)/dU|_{R_raw}`；`Gamma` 可逆
            （行列式 beta^2 > 0），`R_raw` 由它逐解点解出，不再多求一次残差）。
            未启用低马赫预处理时忽略。
    """
    mesh = ctx.mesh
    n_cells, n_sps = int(mesh.n_cells), int(mesh.n_sps_per_cell)
    n_prism = int(mesh.n_prism_cells)
    npr, nte = real_sps_per_cell(int(mesh.order))
    U5 = np.ascontiguousarray(U[:n_cells, :, :5], dtype=np.float64)
    Q = conserved_to_primitive(U5)
    T = compute_temperature(Q)
    gradQ = compute_physical_gradient(Q, mesh, ctx.ops)
    gv_sp = np.ascontiguousarray(gradQ[:, :, 1:4, :])
    del gradQ
    gT_sp = np.ascontiguousarray(compute_physical_gradient(T[:, :, None], mesh, ctx.ops)[:, :, 0, :])
    dTdQ = temperature_gradient_row(Q.reshape(-1, 5)).reshape(n_cells, n_sps, 5)
    mut = (np.zeros((n_cells, n_sps)) if ctx.mu_t is None
           else np.ascontiguousarray(np.asarray(ctx.mu_t, dtype=np.float64)[:n_cells]))
    det = np.asarray(mesh.jacobians["det_jacs"]).reshape(n_cells, n_sps)
    inv_sp = np.ascontiguousarray(np.asarray(mesh.jacobians["inv_jacs"]).reshape(n_cells, n_sps, 3, 3))

    K_prism = np.zeros((n_prism, npr, 5, npr, 5), dtype=np.float32)
    K_tet = np.zeros((n_cells - n_prism, nte, 5, nte, 5), dtype=np.float32)
    slot = np.concatenate([np.arange(n_prism), np.arange(n_cells - n_prism)]).astype(np.int64)

    # ---- 体积项 ----
    for kind, seg, det_f, inv_f, visc_overint in _segments(ctx, npr, nte):
        if seg.hi <= seg.lo:
            continue
        K = K_prism if kind == 0 else K_tet
        n = seg.n
        chunk = seg.chunk_cells()
        for c0 in range(seg.lo, seg.hi, chunk):
            c1 = min(c0 + chunk, seg.hi)
            i0, i1 = c0 - seg.lo, c1 - seg.lo
            adj_f = np.asarray(det_f[i0:i1])[..., None, None] * np.asarray(inv_f[i0:i1])
            if visc_overint:
                adj_v = adj_f
                gv_v = np.einsum("rt,ctab->crab", seg.c2f_visc, gv_sp[c0:c1, :n], optimize=True)
                gT_v = np.einsum("rt,ctb->crb", seg.c2f_visc, gT_sp[c0:c1, :n], optimize=True)
                mut_v = mut[c0:c1, :n] @ seg.c2f_visc.T
            else:
                adj_v = det[c0:c1, :n, None, None] * inv_sp[c0:c1, :n]
                gv_v, gT_v, mut_v = gv_sp[c0:c1, :n], gT_sp[c0:c1, :n], mut[c0:c1, :n]
            add_volume_blocks(K, slot[c0], seg, np.ascontiguousarray(Q[c0:c1, :n]),
                              dTdQ[c0:c1, :n], inv_sp[c0:c1, :n], adj_f, adj_v,
                              mut_v, ctx.mu, ctx.Pr, ctx.Pr_t, gv_v, gT_v)

    # ---- 界面项 ----
    flat = ctx.flat_geometry()
    adj_j = compute_adj_j(det, inv_sp)
    Qg0, QgP, HG, row = _ghost_perturbations(flat, Q, adj_j, ctx.ghost_provider)
    from autoflowcfd.boundary.fr_ghost_state import build_boundary_adiabatic_mask
    adiabatic = np.asarray(build_boundary_adiabatic_mask(flat.n_faces, flat.is_boundary, ctx.ghost_provider),
                           dtype=np.bool_)
    c_ip = resolve_viscous_ip_constant(int(mesh.order))
    if want_coupling:
        cross_offset, expected_col, cross_size, slots = cross_layout(flat, n_prism, npr, nte)
        cross_data = np.zeros(cross_size, dtype=np.float32)
        cross_col = -np.ones_like(cross_offset)
    else:
        cross_offset = np.zeros((0, 2, 3), dtype=np.int64)
        cross_data = np.zeros(0, dtype=np.float32)
        cross_col = cross_offset
    D_prism = np.ascontiguousarray(np.asarray(ctx.ops.D_3d_prism)[:npr, :npr])
    D_tet = np.ascontiguousarray(np.asarray(ctx.ops.D_native_tet_padded)[:nte, :nte])
    for color in range(flat.n_colors):
        faces = flat.color_face_indices[color]
        if len(faces) == 0:
            continue
        add_face_blocks_color(
            faces, K_prism, K_tet, slot, n_prism, npr, nte,
            Q, gv_sp, gT_sp, mut, dTdQ, inv_sp, D_prism, D_tet,
            flat.owner_cell, flat.neighbor_cell, flat.is_boundary,
            flat.owner_is_primary, flat.neighbor_is_primary,
            flat.owner_adj_row_exact, flat.neighbor_adj_row_exact,
            flat.neighbor_src0_cell, flat.neighbor_src0_mat, flat.neighbor_src1_idx,
            flat.neighbor_src1_cell, flat.neighbor_src1_mat,
            flat.owner_src0_cell, flat.owner_src0_mat, flat.owner_src1_idx,
            flat.owner_src1_cell, flat.owner_src1_mat,
            flat.mixed_nb_partner, flat.mixed_nb_mask, flat.mixed_ow_partner, flat.mixed_ow_mask,
            Qg0, QgP, row, HG, adiabatic,
            flat.owner_cube_face, flat.neighbor_cube_face, flat.ref_area_weight,
            flat.boundary_extrap_native, flat.lift_native, flat.ip_length,
            float(ctx.mu), float(ctx.Pr), float(ctx.Pr_t), float(c_ip),
            float(ctx.mach_ref), int(ctx.precond_mode), cross_offset, cross_data, cross_col)
    del QgP, Qg0
    if want_coupling:
        if not np.array_equal(cross_col, expected_col):
            raise RuntimeError("耦合块槽位布局与界面核实际来源不一致（槽位缺写或列单元不同）")

    # ---- 除 det、取负、右乘 dQ/dU、左乘 Gamma ----
    TQ = primitive_jacobian(U5.reshape(-1, 5)).reshape(n_cells, n_sps, 5, 5)
    if ctx.low_mach:
        if residual is None:
            raise ValueError("低马赫预处理启用时必须传入 residual（d(Gamma R)/dU 需要 R）")
        gamma, dgamma = _gamma_matrices(Q, np.asarray(residual, dtype=np.float64)[:n_cells, :, :5],
                                        TQ, ctx.mach_ref)
    else:
        gamma = dgamma = np.zeros((1, n_sps, 5, 5))
    det_c = np.ascontiguousarray(det, dtype=np.float64)
    for K, lo, n in ((K_prism, 0, npr), (K_tet, n_prism, nte)):
        if K.shape[0]:
            finalize_diag_kernel(K, np.arange(lo, lo + K.shape[0], dtype=np.int64), n, det_c, TQ,
                                 gamma, dgamma, bool(ctx.low_mach))
    blocks = (K_prism.reshape(n_prism, npr * 5, npr * 5), K_tet.reshape(n_cells - n_prism, nte * 5, nte * 5))
    if not want_coupling:
        return blocks
    coupling = finalize_coupling(cross_data, slots, det_c, TQ, gamma, bool(ctx.low_mach), npr, nte)
    return blocks + (coupling,)


def _gamma_matrices(Q, gamma_R, TQ, mach_ref):
    """逐解点 `Gamma` 矩阵与 `d(Gamma R_raw)/dU|_{R_raw}`（均为 `(n_cells, n_sps, 5, 5)`）。

    `Gamma` 对残差线性：对单位残差作用即得各列；`R_raw = Gamma^{-1} (Gamma R_raw)`
    逐解点解出；状态导数对原始变量做前向差分再乘 `dQ/dU`
    （`apply_low_mach_preconditioner` 就是残差用的那一个函数；`beta^2` 的上下限
    钳制是拐点，前向差分与块 Jacobi 差分装配取同一侧导数）。
    """
    from autoflowcfd.core.utils.preconditioning import apply_low_mach_preconditioner

    shape = Q.shape[:2]
    gamma = np.empty(shape + (5, 5))
    for v in range(5):
        e = np.zeros(shape + (5,))
        e[..., v] = 1.0
        gamma[..., :, v] = apply_low_mach_preconditioner(e, Q, mach_ref)
    R_raw = np.linalg.solve(gamma, gamma_R[..., None])[..., 0]
    dq = np.empty(shape + (5, 5))
    rho, vel, p = Q[..., 0], Q[..., 1:4], Q[..., 4]
    c = np.sqrt(np.maximum(1.4 * p / np.maximum(rho, 1e-300), 0.0))
    ref = np.stack([np.abs(rho), *(np.linalg.norm(vel, axis=-1) + c,) * 3, np.abs(p)], axis=-1)
    base = apply_low_mach_preconditioner(R_raw, Q, mach_ref)
    for v in range(5):
        h = _SQRT_EPS * (np.abs(Q[..., v]) + ref[..., v]) if v in (1, 2, 3) else _SQRT_EPS * ref[..., v]
        Qp = Q.copy()
        Qp[..., v] += h
        dq[..., :, v] = (apply_low_mach_preconditioner(R_raw, Qp, mach_ref) - base) / h[..., None]
    return gamma, np.einsum("csav,csvb->csab", dq, TQ)
