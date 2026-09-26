"""AutoFlowCFD V2.0 - k-omega 输运界面项对单元块的贡献（numba）。

与残差（`transport/convection.py`、`transport/diffusion.py`、`transport/face_frames.py`）
逐项对应：每个 primary 面侧、每个通量点、每个变量 `v`，

    对流跳变量  Jc = convection_jump_point(m, ps, po)              提升符号 -1
    扩散跳变量  Jd = diffusion_jump_point(ps, gs, dns, po, go, dno)   提升符号 +1

`d(rho phi_v)/dt[s] += (1/det_s) lift[s,i] w_i (Jd - Jc)`，`w_i = ref_area_weight_i |adj_row_i|`。
两个点函数与残差核共用（`face_frames.py`），导数对它们差分（双线性，差分在舍入内精确）。

迹对单元解点的线性链（`M` 为本侧外插或另一侧 sources 插值矩阵）：

    phi 迹   p_v   = M phi_v
    Gamma 迹 g_v   = M Gamma_v，Gamma_v 依赖两个未知量及其梯度（`pointwise.py`）
    法向梯度 dn_v  = M (Gop phi_v) . n

边界与混合拆分面边界半区上，另一侧状态是本侧迹的仿射函数 `po = a ps + b`
（壁面 k=0 镜像 `a=-1`、omega 壁面解析值 `a=-1`、零梯度 `a=1`、来流 `a=0`），由调用方按
残差同一套规则给出（`assemble.py::_convection_ghost_affine`）；扩散在这些点上是内罚
Dirichlet 或齐次 Neumann，只依赖本侧迹。混合面配对边界面的 owner 就是本侧单元
（`face_frames.extrapolate_scalar_pair_kernel` 文档），不产生额外耦合。
"""

import numpy as np
from numba import njit, prange

from autoflowcfd.core.turbulence.transport.face_frames import convection_jump_point, diffusion_jump_point
from .cell_blocks import _gradient_operator, _pointwise_chain

_SQRT_EPS = float(np.sqrt(np.finfo(np.float64).eps))


@njit(cache=True, inline='always')
def _diffusion_partials(ps, gs, dns, po, go, dno, h, c_ip, is_bnd, is_dir, target):
    """`dJd/d(ps, gs, dns, po, go, dno)`（前向差分，双线性函数上精确到舍入）。"""
    x = np.array([ps, gs, dns, po, go, dno])
    j0 = diffusion_jump_point(ps, gs, dns, po, go, dno, h, c_ip, is_bnd, is_dir, target)
    out = np.zeros(6)
    n_in = 3 if is_bnd else 6
    for q in range(n_in):
        hq = _SQRT_EPS * (abs(x[q]) + 1.0)
        y = x.copy()
        y[q] += hq
        out[q] = (diffusion_jump_point(y[0], y[1], y[2], y[3], y[4], y[5], h, c_ip, is_bnd, is_dir,
                                       target) - j0) / hq
    return out


@njit(cache=True)
def _trace_ops(M, phi, gam, dG, inv_c, D, cell, n, nrm):
    """一侧的迹与迹导数算子：返回 `(p (fp,2), g (fp,2), dn (fp,2), DGt (fp,2,n,2), Nt (fp,n))`，
    `DGt[i,v,t,u] = d g_v(i) / d phi_u(t)`，`Nt[i,t] = d dn_v(i) / d phi_v(t)`（与变量无关）。"""
    n_fp = M.shape[0]
    Gop = _gradient_operator(inv_c, D, n)
    DG = _pointwise_chain(dG[cell], Gop, n)
    p = np.zeros((n_fp, 2))
    g = np.zeros((n_fp, 2))
    dn = np.zeros((n_fp, 2))
    DGt = np.zeros((n_fp, 2, n, 2))
    Nt = np.zeros((n_fp, n))
    for i in range(n_fp):
        for sp in range(n):
            e = M[i, sp]
            if e == 0.0:
                continue
            for v in range(2):
                p[i, v] += e * phi[cell, sp, v]
                g[i, v] += e * gam[cell, sp, v]
            for t in range(n):
                nt = nrm[i, 0] * Gop[sp, 0, t] + nrm[i, 1] * Gop[sp, 1, t] + nrm[i, 2] * Gop[sp, 2, t]
                Nt[i, t] += e * nt
                for v in range(2):
                    for u in range(2):
                        DGt[i, v, t, u] += e * DG[sp, v, t, u]
        for v in range(2):
            acc = 0.0
            for t in range(n):
                acc += Nt[i, t] * phi[cell, t, v]
            dn[i, v] = acc
    return p, g, dn, DGt, Nt


@njit(cache=True)
def _side_blocks(c, n, E, L, w, nrm, m_side, h, c_ip, ghost_row, conv_a, is_bnd_row, is_dir_row, target_row,
                 src_cells, src_mats, phi, gam, dG, inv_sp, D_prism, D_tet, n_prism, n_real_prism, n_real_tet,
                 want_cross):
    n_fp = E.shape[0]
    D = D_prism if c < n_prism else D_tet
    ps, gs, dns, DGs, Ns = _trace_ops(E, phi, gam, dG, inv_sp[c], D, c, n, nrm)
    # 另一侧（sources 插值，两个来源累加）
    po = np.zeros((n_fp, 2))
    go = np.zeros((n_fp, 2))
    dno = np.zeros((n_fp, 2))
    for k in range(2):
        y = src_cells[k]
        if y < 0:
            continue
        n_y = n_real_prism if y < n_prism else n_real_tet
        Dy = D_prism if y < n_prism else D_tet
        py, gy, dy, _, _ = _trace_ops(src_mats[k][:, :n_y], phi, gam, dG, inv_sp[y], Dy, y, n_y, nrm)
        po += py
        go += gy
        dno += dy
    # 逐点逐变量的跳变量导数（本侧 / 另一侧），写成 J = Jd - Jc
    dS_p = np.zeros((n_fp, 2))
    dS_g = np.zeros((n_fp, 2))
    dS_n = np.zeros((n_fp, 2))
    dO_p = np.zeros((n_fp, 2))
    dO_g = np.zeros((n_fp, 2))
    dO_n = np.zeros((n_fp, 2))
    for i in range(n_fp):
        m = m_side[i]
        for v in range(2):
            if ghost_row[i]:
                a = conv_a[v, i]
                p_other = a * ps[i, v]       # 常数部分不影响导数
            else:
                a = 0.0
                p_other = po[i, v]
            # 对流（m<0 时 Jc = m (po - ps)），经 J = -Jc
            dc_s = 0.0
            dc_o = 0.0
            if m < 0.0:
                j0 = convection_jump_point(m, ps[i, v], p_other)
                hq = _SQRT_EPS * (abs(ps[i, v]) + 1.0)
                dc_s = (convection_jump_point(m, ps[i, v] + hq, p_other) - j0) / hq
                hq = _SQRT_EPS * (abs(p_other) + 1.0)
                dc_o = (convection_jump_point(m, ps[i, v], p_other + hq) - j0) / hq
            dd = _diffusion_partials(ps[i, v], gs[i, v], dns[i, v], po[i, v], go[i, v], dno[i, v], h, c_ip,
                                     is_bnd_row[i], is_dir_row[v, i], target_row[v, i])
            if ghost_row[i]:
                dS_p[i, v] = dd[0] - (dc_s + a * dc_o)
            else:
                dS_p[i, v] = dd[0] - dc_s
                dO_p[i, v] = dd[3] - dc_o
                dO_g[i, v] = dd[4]
                dO_n[i, v] = dd[5]
            dS_g[i, v] = dd[1]
            dS_n[i, v] = dd[2]
    self_block = np.zeros((n, 2, n, 2))
    for i in range(n_fp):
        for v in range(2):
            coef_p = dS_p[i, v]
            coef_g = dS_g[i, v]
            coef_n = dS_n[i, v]
            for s in range(n):
                lw = L[s, i] * w[i]
                if lw == 0.0:
                    continue
                for t in range(n):
                    self_block[s, v, t, v] += lw * (coef_p * E[i, t] + coef_n * Ns[i, t])
                    for u in range(2):
                        self_block[s, v, t, u] += lw * coef_g * DGs[i, v, t, u]
    nmax = max(n_real_prism, n_real_tet)
    cross = np.zeros((2, n, 2, nmax, 2))
    if want_cross:
        for k in range(2):
            y = src_cells[k]
            if y < 0:
                continue
            n_y = n_real_prism if y < n_prism else n_real_tet
            Dy = D_prism if y < n_prism else D_tet
            My = src_mats[k][:, :n_y]
            _, _, _, DGy, Ny = _trace_ops(My, phi, gam, dG, inv_sp[y], Dy, y, n_y, nrm)
            for i in range(n_fp):
                for v in range(2):
                    for s in range(n):
                        lw = L[s, i] * w[i]
                        if lw == 0.0:
                            continue
                        for t in range(n_y):
                            cross[k, s, v, t, v] += lw * (dO_p[i, v] * My[i, t] + dO_n[i, v] * Ny[i, t])
                            for u in range(2):
                                cross[k, s, v, t, u] += lw * dO_g[i, v] * DGy[i, v, t, u]
    return self_block, cross


@njit(cache=True, parallel=True)
def add_turbulence_face_blocks_color(face_indices, acc_prism, acc_tet, slot, n_prism, n_real_prism, n_real_tet,
                                     phi, gam, dG, inv_sp, D_prism, D_tet,
                                     owner_cell, neighbor_cell, is_boundary, owner_is_primary, neighbor_is_primary,
                                     owner_cube_face, neighbor_cube_face, normal_owner, normal_neighbor,
                                     adj_owner, adj_neighbor, ref_area_weight, E_nat, lift_nat, ip_length, c_ip,
                                     neighbor_src0_cell, neighbor_src0_mat, neighbor_src1_idx, neighbor_src1_cell,
                                     neighbor_src1_mat, owner_src0_cell, owner_src0_mat, owner_src1_idx,
                                     owner_src1_cell, owner_src1_mat,
                                     m_owner, m_neighbor, ghost_o, a_o, ghost_n, a_n,
                                     bnd_o, dir_o, tgt_o, bnd_n, dir_n, tgt_n,
                                     cross_offset, cross_data, cross_col):
    """一个颜色组内全部面的块贡献（乘 det 的量）：对角部分累加到 `acc_prism/acc_tet`，
    耦合块写入 `cross_data`（`cross_offset (n_faces, 2, 3)` 为空数组时不输出）。

    `ghost_*/a_*`：两个坐标系下"另一侧是本侧仿射函数"的点与系数 `a (2, n_faces, n_fp)`；
    `bnd_*/dir_*/tgt_*`：扩散边界点分类（`transport/faces.py::boundary_diffusion_targets`），
    `dir/tgt` 按变量 `(2, n_faces, n_fp)`。
    """
    want_cross = cross_offset.shape[0] > 0
    n_fp = E_nat.shape[1]
    for fi in prange(face_indices.shape[0]):
        f = face_indices[fi]
        for side in range(2):
            src_cells = -np.ones(2, dtype=np.int64)
            if side == 0:
                if not owner_is_primary[f]:
                    continue
                c = owner_cell[f]
                code = owner_cube_face[f]
                nrm = normal_owner[f]
                adj = adj_owner[f]
                m_side = m_owner[f]
                ghost_row, conv_a = ghost_o[f], a_o[:, f]
                bnd_row, dir_row, tgt_row = bnd_o[f], dir_o[:, f], tgt_o[:, f]
                if not is_boundary[f]:
                    src_cells[0] = neighbor_src0_cell[f]
                i1 = neighbor_src1_idx[f]
                m0 = neighbor_src0_mat[f]
                if i1 >= 0:
                    src_cells[1] = neighbor_src1_cell[i1]
                    m1 = neighbor_src1_mat[i1]
                else:
                    m1 = m0
            else:
                if is_boundary[f] or not neighbor_is_primary[f]:
                    continue
                c = neighbor_cell[f]
                code = neighbor_cube_face[f]
                nrm = normal_neighbor[f]
                adj = adj_neighbor[f]
                m_side = m_neighbor[f]
                ghost_row, conv_a = ghost_n[f], a_n[:, f]
                bnd_row, dir_row, tgt_row = bnd_n[f], dir_n[:, f], tgt_n[:, f]
                src_cells[0] = owner_src0_cell[f]
                i1 = owner_src1_idx[f]
                m0 = owner_src0_mat[f]
                if i1 >= 0:
                    src_cells[1] = owner_src1_cell[i1]
                    m1 = owner_src1_mat[i1]
                else:
                    m1 = m0
            is_p = c < n_prism
            n = n_real_prism if is_p else n_real_tet
            E = E_nat[code - 6][:, :n]
            L = lift_nat[code - 6][:n, :]
            w = np.empty(n_fp)
            for i in range(n_fp):
                w[i] = ref_area_weight[i] * np.sqrt(adj[i, 0] ** 2 + adj[i, 1] ** 2 + adj[i, 2] ** 2)
            blk, cross = _side_blocks(c, n, E, L, w, nrm, m_side, ip_length[f], c_ip, ghost_row, conv_a,
                                      bnd_row, dir_row, tgt_row, src_cells, (m0, m1), phi, gam, dG, inv_sp,
                                      D_prism, D_tet, n_prism, n_real_prism, n_real_tet, want_cross)
            k = slot[c]
            acc = acc_prism if is_p else acc_tet
            for s in range(n):
                for v in range(2):
                    for t in range(n):
                        for u in range(2):
                            acc[k, s, v, t, u] += blk[s, v, t, u]
            if want_cross:
                for src in range(2):
                    off = cross_offset[f, side, src]
                    if off < 0:
                        continue
                    y = src_cells[src]
                    cross_col[f, side, src] = y
                    n_y = n_real_prism if y < n_prism else n_real_tet
                    pos = off
                    for s in range(n):
                        for v in range(2):
                            for t in range(n_y):
                                for u in range(2):
                                    cross_data[pos] = cross[src, s, v, t, u]
                                    pos += 1
