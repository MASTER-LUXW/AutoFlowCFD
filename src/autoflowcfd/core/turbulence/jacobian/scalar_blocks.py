"""AutoFlowCFD V2.0 - 一对冻结系数标量输运方程的解析单元块 Jacobian（湍流 k-omega 与人工粘性共用）。

    R_v = -[S_v + conv_v + diff_v] / rho          v = 0, 1

`conv_v` 是冻结质量通量下的迎风对流、`diff_v = div(Gamma_v grad phi_v)` 是 IIPG 内罚扩散
（`core/turbulence/transport` 的同一个离散）。调用方给出逐点源项与扩散系数的导数
（`dS`、`dG`，`(n_cells, n_sps, 2, 8)`）、冻结平均流量与边界数据：

* 湍流（`assemble.py`）：`phi = (k, ln omega)`，源项与 `Gamma` 随 `phi` 变化；
* 人工粘性（`core/fr_residual/jacobian/artificial_viscosity.py`）：`dS = dG = 0`、质量通量为零、
  边界全为齐次 Neumann、`rho = 1`，得到冻结 `nu` 的标量扩散算子 `-L_nu` 本身。

布局与 `CellBlockJacobian` 相同：逐单元 `(n_real*2, n_real*2)`，下标 `s*2 + v`；
`want_coupling` 时再给出面邻居耦合块。
"""

import numpy as np
from numba import njit, prange

from autoflowcfd.core.fr_operators.flux_kernels import resolve_viscous_ip_constant
from autoflowcfd.core.fr_operators.volume_contract import get_overintegration_context
from autoflowcfd.core.fr_residual.jacobian.coupling import CouplingBlocks, CouplingGroup, cross_layout
from autoflowcfd.core.turbulence.transport.convection import resolve_turb_overintegration
from autoflowcfd.fr.native_padding import real_sps_per_cell

from .cell_blocks import turbulence_cell_blocks
from .faces import add_turbulence_face_blocks_color

#: 体积项分块的单元数。
_CHUNK = 4096


def convection_ghost_affine(flat, frame, m_side, wall_zero, has_value, open_face):
    """某一侧坐标系下"另一侧是本侧迹的仿射函数"的点与系数 `(ghost (n_faces,n_fp), a (2,n_faces,n_fp))`，
    规则与 `face_frames.extrapolate_scalar_pair_kernel` + 对流来流覆盖一致。"""
    n_faces, n_fp = flat.owner_adj_row_exact.shape[:2]
    if frame == "owner":
        true_bnd = (np.asarray(flat.neighbor_src0_cell) < 0) & (np.asarray(flat.neighbor_src1_idx) < 0)
        partner, mask = np.asarray(flat.mixed_nb_partner), np.asarray(flat.mixed_nb_mask)
    else:
        true_bnd = np.zeros(n_faces, dtype=bool)
        partner, mask = np.asarray(flat.mixed_ow_partner), np.asarray(flat.mixed_ow_mask)
    mp = np.maximum(partner, 0)
    in_mixed = (partner >= 0)[:, None] & mask
    ghost = true_bnd[:, None] | in_mixed
    a = np.ones((2, n_faces, n_fp))
    for v, dirichlet in ((0, wall_zero), (1, has_value)):
        d = np.where(in_mixed, dirichlet[mp][:, None], dirichlet[:, None])
        a[v] = np.where(ghost & d, -1.0, 1.0)
    if frame == "owner":
        inflow = true_bnd[:, None] & open_face[:, None] & (m_side < 0)
        a[:, inflow] = 0.0
    return ghost, a


def _segments(mesh, ops, npr, nte):
    oi = get_overintegration_context(mesh, ops)
    overint = oi is not None and resolve_turb_overintegration() == "on"
    D_sp = {0: np.asarray(ops.D_3d_prism), 1: np.asarray(ops.D_native_tet_padded)}
    n_cells = int(mesh.n_cells)
    n_prism = int(mesh.n_prism_cells)
    for kind, (lo, hi) in enumerate(((0, n_prism), (n_prism, n_cells))):
        n = npr if kind == 0 else nte
        Dn = np.ascontiguousarray(D_sp[kind][:n, :n])
        if overint:
            _, _, _, det_f, inv_f, c2f, D_f, f2c = oi["segs"][kind]
            c2f_r = np.ascontiguousarray(np.asarray(c2f)[:, :n])
            W = np.ascontiguousarray(np.einsum("sq,qrm->srm", np.asarray(f2c)[:n], np.asarray(D_f), optimize=True))
            yield kind, lo, hi, n, Dn, c2f_r, W, det_f, inv_f
        else:
            yield kind, lo, hi, n, Dn, np.eye(n), np.ascontiguousarray(Dn), None, None


def assemble_scalar_pair_blocks(mesh, ops, flat, kw, gam, dS, dG, rho, vel, rho_u_tilde, mass_flux,
                                mass_flux_neighbor, ghost, diff, want_coupling: bool = False):
    """返回 `(blocks_prism, blocks_tet[, coupling])`（float32，`n_var=2`，见模块文档）。

    Args:
        kw: `(n_cells, n_sps, 2)` 两个标量；gam/dS/dG: 逐点扩散系数与导数。
        rho, vel: `(n_cells, n_sps)` / `(n_cells, n_sps, 3)` 冻结密度与速度（过积分对流体积项）。
        rho_u_tilde: `(n_cells, n_sps, 3)` 解点上的逆变质量通量（不过积分时的对流体积项）。
        mass_flux, mass_flux_neighbor: 两侧坐标系下的面质量通量 `(n_faces, n_fp)`。
        ghost: `(ghost_o, a_o, ghost_n, a_n)` 对流 ghost 的仿射规则（`convection_ghost_affine`）。
        diff: `{"owner"|"neighbor": (is_bnd, is_dir (2,...), target (2,...))}` 扩散边界数据。
    """
    n_cells, n_sps = int(mesh.n_cells), int(mesh.n_sps_per_cell)
    n_prism = int(mesh.n_prism_cells)
    npr, nte = real_sps_per_cell(int(mesh.order))
    det = np.asarray(mesh.jacobians["det_jacs"]).reshape(n_cells, n_sps)
    inv_sp = np.ascontiguousarray(np.asarray(mesh.jacobians["inv_jacs"]).reshape(n_cells, n_sps, 3, 3))

    acc = [np.zeros((n_prism, npr, 2, npr, 2), dtype=np.float32),
           np.zeros((n_cells - n_prism, nte, 2, nte, 2), dtype=np.float32)]
    src = [np.zeros_like(acc[0]), np.zeros_like(acc[1])]
    slot = np.concatenate([np.arange(n_prism), np.arange(n_cells - n_prism)]).astype(np.int64)
    # ---- 单元内项 ----
    for kind, lo, hi, n, Dn, c2f, W, det_f, inv_f in _segments(mesh, ops, npr, nte):
        for c0 in range(lo, hi, _CHUNK):
            c1 = min(c0 + _CHUNK, hi)
            if det_f is not None:
                i0, i1 = c0 - lo, c1 - lo
                adj = np.asarray(det_f[i0:i1])[..., None, None] * np.asarray(inv_f[i0:i1])
                rho_f = np.einsum("rt,ct->cr", c2f, rho[c0:c1, :n], optimize=True)
                u_f = np.einsum("rt,ctd->crd", c2f, vel[c0:c1, :n], optimize=True)
                rut = np.einsum("crmi,cri->crm", adj, rho_f[..., None] * u_f, optimize=True)
            else:
                adj = det[c0:c1, :n, None, None] * inv_sp[c0:c1, :n]
                rut = np.asarray(rho_u_tilde)[c0:c1, :n]
            turbulence_cell_blocks(acc[kind], src[kind], int(slot[c0]),
                                   np.ascontiguousarray(kw[c0:c1, :n]), gam[c0:c1, :n], dS[c0:c1, :n],
                                   dG[c0:c1, :n], inv_sp[c0:c1, :n], Dn, np.ascontiguousarray(c2f), W,
                                   np.ascontiguousarray(adj), np.ascontiguousarray(rut))

    # ---- 界面项 ----
    m_o = np.ascontiguousarray(mass_flux)
    m_n = np.ascontiguousarray(mass_flux_neighbor)
    ghost_o, a_o, ghost_n, a_n = ghost
    if want_coupling:
        cross_offset, expected_col, cross_size, slots = cross_layout(flat, n_prism, npr, nte, n_var=2,
                                                                      include_partner=False)
        cross_data = np.zeros(cross_size, dtype=np.float32)
        cross_col = -np.ones_like(cross_offset)
    else:
        cross_offset = np.zeros((0, 2, 3), dtype=np.int64)
        cross_data = np.zeros(0, dtype=np.float32)
        cross_col = cross_offset
    c_ip = resolve_viscous_ip_constant(int(mesh.order))
    D_prism = np.ascontiguousarray(np.asarray(ops.D_3d_prism)[:npr, :npr])
    D_tet = np.ascontiguousarray(np.asarray(ops.D_native_tet_padded)[:nte, :nte])
    n_own, n_nei = flat.owner_unit_normal, flat.neighbor_unit_normal
    for color in range(flat.n_colors):
        faces = flat.color_face_indices[color]
        if len(faces) == 0:
            continue
        add_turbulence_face_blocks_color(
            faces, acc[0], acc[1], slot, n_prism, npr, nte, kw, gam, dG, inv_sp, D_prism, D_tet,
            flat.owner_cell, flat.neighbor_cell, flat.is_boundary, flat.owner_is_primary,
            flat.neighbor_is_primary, flat.owner_face_op, flat.neighbor_face_op, n_own, n_nei,
            flat.owner_adj_row_exact, flat.neighbor_adj_row_exact, flat.ref_area_weight,
            flat.boundary_extrap_native, flat.lift_native, flat.ip_length, float(c_ip),
            flat.neighbor_src0_cell, flat.neighbor_src0_tpl, flat.neighbor_src0_tid, flat.neighbor_src1_idx,
            flat.neighbor_src1_cell, flat.neighbor_src1_mat, flat.owner_src0_cell, flat.owner_src0_tpl, flat.owner_src0_tid,
            flat.owner_src1_idx, flat.owner_src1_cell, flat.owner_src1_mat,
            m_o, m_n, ghost_o, np.ascontiguousarray(a_o), ghost_n, np.ascontiguousarray(a_n),
            *diff["owner"], *diff["neighbor"], cross_offset, cross_data, cross_col)
    if want_coupling and not np.array_equal(cross_col, expected_col):
        raise RuntimeError("湍流耦合块槽位布局与界面核实际来源不一致")

    # ---- 合并、取负、除 rho ----
    inv_rho = 1.0 / np.maximum(rho, 1e-10)
    for kind, lo, n in ((0, 0, npr), (1, n_prism, nte)):
        if acc[kind].shape[0]:
            _finalize_diag(acc[kind], src[kind], lo, n, np.ascontiguousarray(det), np.ascontiguousarray(inv_rho))
    blocks = (acc[0].reshape(n_prism, npr * 2, npr * 2), acc[1].reshape(n_cells - n_prism, nte * 2, nte * 2))
    if not want_coupling:
        return blocks
    return blocks + (_finalize_coupling(cross_data, slots, np.ascontiguousarray(det),
                                        np.ascontiguousarray(inv_rho), npr, nte),)


@njit(cache=True, parallel=True)
def _finalize_diag(acc, src, lo, n, det, inv_rho):
    for k in prange(acc.shape[0]):
        c = lo + k
        for s in range(n):
            scale = -inv_rho[c, s]
            inv_det = 1.0 / det[c, s]
            for v in range(2):
                for t in range(n):
                    for u in range(2):
                        acc[k, s, v, t, u] = scale * (acc[k, s, v, t, u] * inv_det + src[k, s, v, t, u])


def _finalize_coupling(data, slots, det, inv_rho, npr, nte):
    rows, cols, offs, bounds = slots
    groups = []
    for g in range(4):
        lo, hi = int(bounds[g]), int(bounds[g + 1])
        if hi <= lo:
            continue
        row_p, col_p = g < 2, g % 2 == 0
        nr = npr if row_p else nte
        ny = npr if col_p else nte
        r, c = rows[lo:hi], cols[lo:hi]
        new_pair = np.ones(hi - lo, dtype=bool)
        new_pair[1:] = (r[1:] != r[:-1]) | (c[1:] != c[:-1])
        first = np.nonzero(new_pair)[0].astype(np.int64)
        out = np.empty((first.size, nr, 2, ny, 2), dtype=np.float32)
        _finalize_pairs(data, int(offs[lo]), nr, ny, first, hi - lo, r, det, inv_rho, out)
        groups.append(CouplingGroup(row_is_prism=row_p, col_is_prism=col_p, rows=r[first], cols=c[first],
                                    blocks=out.reshape(first.size, nr * 2, ny * 2)))
    return CouplingBlocks(groups=groups)


@njit(cache=True, parallel=True)
def _finalize_pairs(data, base, nr, ny, first, n_slots, rows, det, inv_rho, out):
    blk = 4 * nr * ny
    n_u = first.shape[0]
    for u_ in prange(n_u):
        k0 = first[u_]
        k1 = first[u_ + 1] if u_ + 1 < n_u else n_slots
        r = rows[k0]
        X = np.zeros((nr, 2, ny, 2))
        for kk in range(k0, k1):
            p = base + kk * blk
            for s in range(nr):
                for v in range(2):
                    for t in range(ny):
                        for u in range(2):
                            X[s, v, t, u] += data[p]
                            p += 1
        for s in range(nr):
            scale = -inv_rho[r, s] / det[r, s]
            for v in range(2):
                for t in range(ny):
                    for u in range(2):
                        out[u_, s, v, t, u] = scale * X[s, v, t, u]
