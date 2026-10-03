"""AutoFlowCFD V2.0 - 标量场的面外插、质量通量与面校正提升（GPU）。

与 CPU 版 `core/turbulence/transport/face_frames.py` + `faces.py` 逐项对应：两侧
各在自己的通量点顺序里构造跳变量、按统一符号约定提升（2026-09-26 真实缺陷
修复，理由与实测见 `face_frames.py` 模块文档）。全部面一次性向量化，不分色。
"""


def _src_gather(cp, field, src0_cell, src0_tpl, src0_tid, src1_idx, src1_cell, src1_mat):
    """`src0 @ field[src0] (+ src1_mat @ field[src1])`，返回 `(values, has_source)`；第 f 个面的
    src0 矩阵是 `src0_tpl[src0_tid[f]]`（模板表，`fr/face_flux_points/templates.py`）。"""
    valid0 = src0_cell >= 0
    out = cp.einsum('fps,fs->fp', src0_tpl[src0_tid], field[cp.maximum(src0_cell, 0)]) * valid0[:, None]
    valid1 = src1_idx >= 0
    if bool(cp.any(valid1)):
        sel = cp.where(valid1)[0]
        idx1 = src1_idx[sel]
        out[sel] = out[sel] + cp.einsum('fps,fs->fp', src1_mat[idx1], field[src1_cell[idx1]])
    return out, valid0 | valid1


def _self_extrap(cp, ff, field, self_cell, self_face_op):
    """自身外插 `boundary_extrap_native[op] @ field[self]`；`self_cell<0` 的面为 0。"""
    valid = self_cell >= 0
    E = ff.boundary_extrap_native[cp.where(valid, self_face_op, 0)]
    return cp.einsum('fps,fs->fp', E, field[cp.maximum(self_cell, 0)]) * valid[:, None]


def _frame_args(ff, frame):
    if frame == "owner":
        return (ff.owner_cell, ff.owner_face_op, ff.neighbor_src0_cell, ff.neighbor_src0_tpl, ff.neighbor_src0_tid,
                ff.neighbor_src1_idx, ff.neighbor_src1_cell, ff.neighbor_src1_mat,
                ff.mixed_nb_partner, ff.mixed_nb_mask)
    return (ff.neighbor_cell, ff.neighbor_face_op, ff.owner_src0_cell, ff.owner_src0_tpl, ff.owner_src0_tid,
            ff.owner_src1_idx, ff.owner_src1_cell, ff.owner_src1_mat,
            ff.mixed_ow_partner, ff.mixed_ow_mask)


def _extrapolate_scalar_pair_gpu(cp, ff, scalar_sps, frame,
                                 wall_dirichlet_zero_face=None,
                                 wall_dirichlet_value_face=None,
                                 has_wall_dirichlet_value=None):
    """某一侧坐标系下的 `(phi_self, phi_other)`，CPU 版
    `face_frames.extrapolate_scalar_pair_kernel` 的向量化对应（ghost 规则相同：
    真边界面只在 owner 坐标系出现；混合拆分面按配对边界面规则逐通量点覆盖）。"""
    n_faces, n_fp = ff.n_faces, ff.boundary_extrap_native.shape[1]
    if wall_dirichlet_zero_face is None:
        wall_dirichlet_zero_face = cp.zeros(n_faces, dtype=cp.bool_)
    if has_wall_dirichlet_value is None:
        has_wall_dirichlet_value = cp.zeros(n_faces, dtype=cp.bool_)
    if wall_dirichlet_value_face is None:
        wall_dirichlet_value_face = cp.zeros((n_faces, n_fp), dtype=cp.float64)
    self_cell, self_code, s0c, s0t, s0i, s1i, s1c, s1m, mixed_partner, mixed_mask = _frame_args(ff, frame)
    phi_self = _self_extrap(cp, ff, scalar_sps, self_cell, self_code)
    phi_other, has_src = _src_gather(cp, scalar_sps, s0c, s0t, s0i, s1i, s1c, s1m)
    phi_other = phi_other * (self_cell >= 0)[:, None]

    def _ghost(zero, has_value, value):
        return cp.where(zero[:, None], -phi_self,
                        cp.where(has_value[:, None], 2.0 * value - phi_self, phi_self))

    if frame == "owner":
        bnd = ~has_src
        phi_other = cp.where(bnd[:, None], _ghost(wall_dirichlet_zero_face, has_wall_dirichlet_value,
                                                  wall_dirichlet_value_face), phi_other)
    has_partner = mixed_partner >= 0
    if bool(cp.any(has_partner)):
        mp = cp.maximum(mixed_partner, 0)
        chosen = _ghost(wall_dirichlet_zero_face[mp], has_wall_dirichlet_value[mp], wall_dirichlet_value_face[mp])
        phi_other = cp.where(has_partner[:, None] & mixed_mask, chosen, phi_other)
    return phi_self, phi_other


def _extrapolate_scalar_to_faces_gpu(cp, ff, n_prism, scalar_sps,
                                     wall_dirichlet_zero_face=None,
                                     wall_dirichlet_value_face=None,
                                     has_wall_dirichlet_value=None):
    """owner 坐标系下的 `(phi_owner_fp, phi_neighbor_fp)`（CPU 版
    `faces._extrapolate_scalar_to_faces`）。`n_prism` 保留以兼容既有调用签名。"""
    return _extrapolate_scalar_pair_gpu(cp, ff, scalar_sps, "owner", wall_dirichlet_zero_face,
                                        wall_dirichlet_value_face, has_wall_dirichlet_value)


def _face_mass_flux_gpu(cp, ff, rho_u):
    """两侧坐标系下的质量通量 `(m_owner, m_neighbor)`，都取 owner 的迹（CPU 版
    `face_frames.face_mass_flux_kernel`）。"""
    n_o, n_n = ff.owner_unit_normal, ff.neighbor_unit_normal
    has_nb = ff.neighbor_cell >= 0
    mp = ff.mixed_ow_partner
    mixed = (mp >= 0)[:, None] & ff.mixed_ow_mask
    m_o = 0.0
    m_n = 0.0
    for d in range(3):
        comp = cp.ascontiguousarray(rho_u[..., d])
        m_o = m_o + _self_extrap(cp, ff, comp, ff.owner_cell, ff.owner_face_op) * n_o[..., d]
        at_n, _ = _src_gather(cp, comp, ff.owner_src0_cell, ff.owner_src0_tpl, ff.owner_src0_tid,
                              ff.owner_src1_idx, ff.owner_src1_cell, ff.owner_src1_mat)
        own_n = _self_extrap(cp, ff, comp, ff.neighbor_cell, ff.neighbor_face_op)
        m_n = m_n + cp.where(mixed, own_n, at_n) * n_n[..., d]
    return m_o, m_n * has_nb[:, None]


def _lift_side_jumps_gpu(cp, ff, jump_owner, jump_neighbor, sign, det_jacs, n_cells, n_sps):
    """两侧物理跳变量提升回解点（CPU 版 `faces._lift_side_jumps`）：
    `corr[cell] += sign * lift_native[op] @ (ref_area_weight*|adj_row|*J) / det`。跳变
    `(n_faces, n_fp)` 返回 `(n_cells, n_sps)`，多分量 `(n_faces, n_fp, n_comp)` 返回
    `(n_cells, n_sps, n_comp)`（同一条路径，单分量是 n_comp = 1）。"""
    scalar = jump_owner.ndim == 2
    if scalar:
        jump_owner, jump_neighbor = jump_owner[..., None], jump_neighbor[..., None]
    out = cp.zeros((n_cells, n_sps, jump_owner.shape[2]), dtype=cp.float64)
    for cells, ops_, adj, jump, keep in (
            (ff.owner_cell, ff.owner_face_op, ff.owner_adj_row_exact, jump_owner, ff.owner_is_primary),
            (ff.neighbor_cell, ff.neighbor_face_op, ff.neighbor_adj_row_exact, jump_neighbor,
             (ff.neighbor_cell >= 0) & ff.neighbor_is_primary)):
        sel = cp.where(keep)[0]
        if not bool(sel.shape[0] > 0):
            continue
        c = cells[sel]
        w = ff.ref_area_weight[None, :] * cp.sqrt(cp.sum(adj[sel] * adj[sel], axis=-1))
        contrib = cp.einsum('nsf,nfm->nsm', ff.lift_native[ops_[sel]], w[..., None] * jump[sel])
        contrib = contrib / det_jacs[c][..., None]
        cp.scatter_add(out, (c, slice(None)), sign * contrib)
    return out[..., 0] if scalar else out
