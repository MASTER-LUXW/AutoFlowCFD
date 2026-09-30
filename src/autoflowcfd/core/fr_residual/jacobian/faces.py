"""AutoFlowCFD V2.0 - 界面项对单元块的贡献（numba，原始变量空间）。

与残差核（`inviscid_kernel_colored.py` / `viscous_flux_kernel_colored.py`）逐项
对应：每个 primary 面侧、每个通量点算跳变量 `J_i`（`face_point_jumps.py` 的同一
个函数），`dU/dt` 增量是 `(1/det_s) sum_i lift[s,i] w_i J_i`，其中

    J_i = viscous_jump_point(...) - inviscid_jump_point(...)

（无粘项在残差里带负号，粘性项带正号，见两个核的符号约定）。`J_i` 依赖两侧的
迹：状态迹 `Q = M Q_cell`、梯度迹 `grad = M Gop Q_cell`（速度分量与温度，温度再
乘 `dT/dQ`），`M` 是本侧的外插矩阵或另一侧的 sources 插值矩阵。逐点导数
`dJ_i/d(迹)` 对跳变量函数做前向差分，再经上述线性链落到单元解点上：

* 本侧迹 -> 对角块 `J_cc`；
* 另一侧迹 -> 与 sources 单元的耦合块 `J_cy`（`want_cross` 时输出，块 ILU 用）。

## 边界点：复合差分

边界面上另一侧状态是本侧迹的函数（幽灵态、边界梯度规则）。在分段光滑的通量
（AUSM+up 的马赫数分裂、壁面法向速度过零处）上，把"对本侧偏导 + 对另一侧偏导
x 幽灵态导数"分开差分再用链式法则拼起来，在拐点上与真实方向导数不等（实测
棱柱壁面单元 `dR_rho_u/drho` 差 7.6%）。所以边界点对**复合函数**整体差分：
本侧原始变量平移 `HG[v]`、幽灵态取同一平移下预先算好的 `QgP[v, ghost_row[f]]`
（见 `assemble.py`：整场均匀平移后调用残差用的同一个幽灵态函数；外插保常数，
迹也恰好平移 `HG[v]`）；梯度输入平移时另一侧梯度按
`boundary_other_gradients`（同一粘性边界种类）重新构造。

混合拆分面的边界半区（`mixed_*_mask`）同样按边界规则取另一侧梯度；另一侧状态
是配对边界面 `mp` 的幽灵态，它随 `mp` 的 owner 单元 `p` 变化，经
`dQ_ghost/dQ_trace`（同样由 `QgP` 差分得到）与 `mp` 的外插矩阵链入 `J_cp`
（`p` 就是本单元时进对角块）。

着色与残差核相同（`flat.color_face_indices`：同色面两侧单元互不相同），同色面
并行、直接累加到共享对角块；耦合块按 (面, 侧, 来源) 各占一个槽位，不冲突。
"""

import numpy as np
from numba import njit, prange

from autoflowcfd.core.fr_operators.flux_kernels import VBC_INTERIOR, boundary_other_gradients
from autoflowcfd.core.fr_residual.face_point_jumps import inviscid_jump_point, viscous_jump_point
from .pointwise import N_VISC_INPUTS, _SQRT_EPS, primitive_step

#: 每个面侧的耦合块来源：src0 单元、src1 单元、混合拆分面配对边界面的 owner。
N_CROSS_SOURCES = 3


@njit(cache=True, inline='always')
def _total_jump(Qs, gvs, gTs, mus, Qx, gvx, gTx, mux, adjrow, h_ip, bkind, mu, Pr, Pr_t, c_ip,
                mach_ref, precond_mode):
    J = inviscid_jump_point(Qs, Qx, adjrow, mach_ref, precond_mode)
    Jv = viscous_jump_point(Qs, gvs, gTs, mus, Qx, gvx, gTx, mux, adjrow, h_ip, bkind,
                            mu, Pr, Pr_t, c_ip)
    for v in range(5):
        J[v] = Jv[v] - J[v]
    return J


@njit(cache=True, inline='always')
def _gradient_step(x, scale):
    if scale > 0.0:
        return _SQRT_EPS * (abs(x) + scale)
    return _SQRT_EPS


@njit(cache=True)
def _gradient_trace_operator(M, inv_y, D, n_y):
    """`GM[i,b,t] = sum_s M[i,s] sum_m inv_y[s,m,b] D[s,t,m]`：解点值 -> 迹点物理梯度。"""
    n_fp = M.shape[0]
    GM = np.zeros((n_fp, 3, n_y))
    for s in range(n_y):
        for t in range(n_y):
            g0 = 0.0
            g1 = 0.0
            g2 = 0.0
            for m in range(3):
                d = D[s, t, m]
                g0 += inv_y[s, m, 0] * d
                g1 += inv_y[s, m, 1] * d
                g2 += inv_y[s, m, 2] * d
            for i in range(n_fp):
                e = M[i, s]
                if e != 0.0:
                    GM[i, 0, t] += e * g0
                    GM[i, 1, t] += e * g1
                    GM[i, 2, t] += e * g2
    return GM


@njit(cache=True)
def _chain(dJ, M, GM, dTdQ_y, L, w, n, n_y, with_grad):
    """`out[s,o,t,b] = sum_i L[s,i] w_i dJ_i/dQ_{t,b}`（经状态迹与梯度迹两条线性链）。"""
    n_fp = M.shape[0]
    out = np.zeros((n, 5, n_y, 5))
    Dt = np.zeros((n_y, 5, 5))
    for i in range(n_fp):
        for t in range(n_y):
            e = M[i, t]
            for o in range(5):
                for b in range(5):
                    Dt[t, o, b] = dJ[i, o, b] * e
            if with_grad:
                for o in range(5):
                    for k in range(3):
                        acc = 0.0
                        for b in range(3):
                            acc += dJ[i, o, 5 + 3 * k + b] * GM[i, b, t]
                        Dt[t, o, 1 + k] += acc
                    accT = 0.0
                    for b in range(3):
                        accT += dJ[i, o, 14 + b] * GM[i, b, t]
                    for b in range(5):
                        Dt[t, o, b] += accT * dTdQ_y[t, b]
        for s in range(n):
            lw = L[s, i] * w[i]
            if lw != 0.0:
                for t in range(n_y):
                    for o in range(5):
                        for b in range(5):
                            out[s, o, t, b] += lw * Dt[t, o, b]
    return out


@njit(cache=True)
def _trace(M, field, c, n):
    n_fp = M.shape[0]
    tail = field.shape[2:]
    out = np.zeros((n_fp,) + tail)
    for i in range(n_fp):
        for t in range(n):
            e = M[i, t]
            if e != 0.0:
                out[i] += e * field[c, t]
    return out


@njit(cache=True)
def _side_blocks(c, n, E, L, w, adjrow, f, is_bnd_face, masked_row, mp, src_cells, src_mats,
                 Q, gv_sp, gT_sp, mut, dTdQ, inv_sp, D_prism, D_tet, n_prism, n_real_prism, n_real_tet,
                 Qg0, QgP, ghost_row, HG, vbc_kind, owner_cell, owner_cube_face, E_nat,
                 h_ip, mu, Pr, Pr_t, c_ip, mach_ref, precond_mode, want_cross):
    """一个 primary 面侧的块贡献（未除 det，原始变量空间）。

    返回 `(self_block (n,5,n,5), cross (3, n, 5, nmax, 5), cross_cells (3,))`；
    `cross_cells[k] < 0` 表示该来源不存在。
    """
    n_fp = E.shape[0]
    D = D_prism if c < n_prism else D_tet
    Qs = _trace(E, Q, c, n)
    gvs = _trace(E, gv_sp, c, n)
    gTs = _trace(E, gT_sp, c, n)
    mus = np.zeros(n_fp)
    for i in range(n_fp):
        for t in range(n):
            mus[i] += E[i, t] * mut[c, t]
    GE = _gradient_trace_operator(E, inv_sp[c], D, n)

    dS = np.zeros((n_fp, 5, N_VISC_INPUTS))     # dJ/d(本侧输入)
    dO = np.zeros((n_fp, 5, N_VISC_INPUTS))     # dJ/d(另一侧输入)（sources 点）
    dP = np.zeros((n_fp, 5, 5))                 # dJ/dQ_本侧迹经配对幽灵态（混合半区点）
    any_masked = False
    for i in range(n_fp):
        masked = masked_row[i] if mp >= 0 else False
        bnd_i = is_bnd_face or masked
        bk = VBC_INTERIOR
        if is_bnd_face:
            Qx = Qg0[f, i].copy()
        elif masked:
            Qx = Qg0[mp, i].copy()
        else:
            Qx = np.zeros(5)
        gvx = np.zeros((3, 3))
        gTx = np.zeros(3)
        mux = 0.0
        if bnd_i:
            bk = vbc_kind[f] if is_bnd_face else vbc_kind[mp]
            gvx, gTx = boundary_other_gradients(gvs[i], gTs[i], adjrow[i], bk)
            mux = mus[i]
        else:
            for k in range(2):
                cc = src_cells[k]
                if cc < 0:
                    continue
                Mk = src_mats[k]
                for t in range(Q.shape[1]):
                    wgt = Mk[i, t]
                    if wgt != 0.0:
                        Qx += wgt * Q[cc, t]
                        gvx += wgt * gv_sp[cc, t]
                        gTx += wgt * gT_sp[cc, t]
                        mux += wgt * mut[cc, t]
        J0 = _total_jump(Qs[i], gvs[i], gTs[i], mus[i], Qx, gvx, gTx, mux, adjrow[i], h_ip, bk,
                         mu, Pr, Pr_t, c_ip, mach_ref, precond_mode)
        # 梯度输入只进粘性跳变量：对它们差分时不必重算 AUSM+up
        Jv0 = viscous_jump_point(Qs[i], gvs[i], gTs[i], mus[i], Qx, gvx, gTx, mux, adjrow[i], h_ip, bk,
                                 mu, Pr, Pr_t, c_ip)
        # ---- 本侧原始变量（边界面：复合差分）----
        for v in range(5):
            q = Qs[i].copy()
            if is_bnd_face:
                h = HG[v]
                q[v] += h
                qx = QgP[v, ghost_row[f], i].copy()
            else:
                h = primitive_step(Qs[i], v)
                q[v] += h
                qx = Qx
            J1 = _total_jump(q, gvs[i], gTs[i], mus[i], qx, gvx, gTx, mux, adjrow[i], h_ip, bk,
                             mu, Pr, Pr_t, c_ip, mach_ref, precond_mode)
            for o in range(5):
                dS[i, o, v] = (J1[o] - J0[o]) / h
        # ---- 本侧梯度（边界点另一侧梯度随之重构）----
        sv = np.sqrt((gvs[i] * gvs[i]).sum())
        st = np.sqrt((gTs[i] * gTs[i]).sum())
        for j in range(5, N_VISC_INPUTS):
            g = gvs[i].copy()
            tt = gTs[i].copy()
            if j < 14:
                a = (j - 5) // 3
                b = (j - 5) % 3
                h = _gradient_step(g[a, b], sv)
                g[a, b] += h
            else:
                h = _gradient_step(tt[j - 14], st)
                tt[j - 14] += h
            if bnd_i:
                g_x, t_x = boundary_other_gradients(g, tt, adjrow[i], bk)
            else:
                g_x = gvx
                t_x = gTx
            J1 = viscous_jump_point(Qs[i], g, tt, mus[i], Qx, g_x, t_x, mux, adjrow[i], h_ip, bk,
                                    mu, Pr, Pr_t, c_ip)
            for o in range(5):
                dS[i, o, j] = (J1[o] - Jv0[o]) / h
        # ---- 另一侧状态：混合半区经配对幽灵态；sources 点经插值 ----
        if masked:
            any_masked = True
            dX = np.zeros((5, 5))
            for v in range(5):
                h = primitive_step(Qx, v)
                qx = Qx.copy()
                qx[v] += h
                J1 = _total_jump(Qs[i], gvs[i], gTs[i], mus[i], qx, gvx, gTx, mux, adjrow[i], h_ip,
                                 bk, mu, Pr, Pr_t, c_ip, mach_ref, precond_mode)
                for o in range(5):
                    dX[o, v] = (J1[o] - J0[o]) / h
            r = ghost_row[mp]
            for o in range(5):
                for b in range(5):
                    acc = 0.0
                    for v in range(5):
                        acc += dX[o, v] * (QgP[b, r, i, v] - Qg0[mp, i, v]) / HG[b]
                    dP[i, o, b] = acc
        elif want_cross and not is_bnd_face:
            sgv = np.sqrt((gvx * gvx).sum())
            sgt = np.sqrt((gTx * gTx).sum())
            for j in range(N_VISC_INPUTS):
                qx = Qx.copy()
                g = gvx.copy()
                tt = gTx.copy()
                if j < 5:
                    h = primitive_step(Qx, j)
                    qx[j] += h
                elif j < 14:
                    a = (j - 5) // 3
                    b = (j - 5) % 3
                    h = _gradient_step(g[a, b], sgv)
                    g[a, b] += h
                else:
                    h = _gradient_step(tt[j - 14], sgt)
                    tt[j - 14] += h
                if j < 5:
                    J1 = _total_jump(Qs[i], gvs[i], gTs[i], mus[i], qx, g, tt, mux, adjrow[i], h_ip, VBC_INTERIOR,
                                     mu, Pr, Pr_t, c_ip, mach_ref, precond_mode)
                    for o in range(5):
                        dO[i, o, j] = (J1[o] - J0[o]) / h
                else:
                    J1 = viscous_jump_point(Qs[i], gvs[i], gTs[i], mus[i], qx, g, tt, mux, adjrow[i], h_ip,
                                            VBC_INTERIOR, mu, Pr, Pr_t, c_ip)
                    for o in range(5):
                        dO[i, o, j] = (J1[o] - Jv0[o]) / h

    self_block = _chain(dS, E, GE, dTdQ[c], L, w, n, n, True)
    nmax = max(n_real_prism, n_real_tet)
    cross = np.zeros((N_CROSS_SOURCES, n, 5, nmax, 5))
    cross_cells = -np.ones(N_CROSS_SOURCES, dtype=np.int64)
    empty_grad = np.zeros((n_fp, 3, 1))
    if any_masked:
        p = owner_cell[mp]
        n_p = n_real_prism if p < n_prism else n_real_tet
        Ep = E_nat[owner_cube_face[mp] - 6][:, :n_p]
        blk = _chain(dP, Ep, empty_grad, dTdQ[p], L, w, n, n_p, False)
        if p == c:
            self_block += blk
        elif want_cross:
            cross[2, :, :, :n_p, :] = blk
            cross_cells[2] = p
    if want_cross and not is_bnd_face:
        for k in range(2):
            y = src_cells[k]
            if y < 0:
                continue
            n_y = n_real_prism if y < n_prism else n_real_tet
            Dy = D_prism if y < n_prism else D_tet
            My = src_mats[k][:, :n_y]
            GMy = _gradient_trace_operator(My, inv_sp[y], Dy, n_y)
            cross[k, :, :, :n_y, :] = _chain(dO, My, GMy, dTdQ[y], L, w, n, n_y, True)
            cross_cells[k] = y
    return self_block, cross, cross_cells


@njit(cache=True, parallel=True)
def add_face_blocks_color(face_indices, K_prism, K_tet, slot, n_prism, n_real_prism, n_real_tet,
                          Q, gv_sp, gT_sp, mut, dTdQ, inv_sp, D_prism, D_tet,
                          owner_cell, neighbor_cell, is_boundary, owner_is_primary, neighbor_is_primary,
                          owner_adj_row_exact, neighbor_adj_row_exact,
                          neighbor_src0_cell, neighbor_src0_tpl, neighbor_src0_tid, neighbor_src1_idx, neighbor_src1_cell,
                          neighbor_src1_mat, owner_src0_cell, owner_src0_tpl, owner_src0_tid, owner_src1_idx,
                          owner_src1_cell, owner_src1_mat, mixed_nb_partner, mixed_nb_mask,
                          mixed_ow_partner, mixed_ow_mask, Qg0, QgP, ghost_row, HG, vbc_kind,
                          owner_cube_face, neighbor_cube_face, ref_area_weight, E_nat, lift_nat,
                          ip_length, mu, Pr, Pr_t, c_ip, mach_ref, precond_mode,
                          cross_offset, cross_data, cross_col):
    """一个颜色组内全部面的块贡献：对角部分累加到 `K_prism` / `K_tet`（float32）；
    `cross_offset` 非空（形状 `(n_faces, 2, 3)`）时耦合块写入 `cross_data`
    （扁平 float32，槽位偏移见 `assemble.py::_cross_layout`），列单元写入 `cross_col`。
    """
    want_cross = cross_offset.shape[0] > 0
    for fi in prange(face_indices.shape[0]):
        f = face_indices[fi]
        for side in range(2):
            src_cells = -np.ones(2, dtype=np.int64)
            if side == 0:
                if not owner_is_primary[f]:
                    continue
                c = owner_cell[f]
                code = owner_cube_face[f]
                adjrow = owner_adj_row_exact[f]
                is_bnd_face = is_boundary[f]
                mp = mixed_nb_partner[f]
                masked_row = mixed_nb_mask[f]
                src_cells[0] = neighbor_src0_cell[f]
                m0 = neighbor_src0_tpl[neighbor_src0_tid[f]]
                i1 = neighbor_src1_idx[f]
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
                adjrow = neighbor_adj_row_exact[f]
                is_bnd_face = False
                mp = mixed_ow_partner[f]
                masked_row = mixed_ow_mask[f]
                src_cells[0] = owner_src0_cell[f]
                m0 = owner_src0_tpl[owner_src0_tid[f]]
                i1 = owner_src1_idx[f]
                if i1 >= 0:
                    src_cells[1] = owner_src1_cell[i1]
                    m1 = owner_src1_mat[i1]
                else:
                    m1 = m0
            if is_bnd_face:
                src_cells[0] = -1
            src_mats = (m0, m1)
            is_p = c < n_prism
            n = n_real_prism if is_p else n_real_tet
            E = E_nat[code - 6][:, :n]
            L = lift_nat[code - 6][:n, :]
            blk, cross, cross_cells = _side_blocks(
                c, n, E, L, ref_area_weight, adjrow, f, is_bnd_face, masked_row, mp, src_cells, src_mats,
                Q, gv_sp, gT_sp, mut, dTdQ, inv_sp, D_prism, D_tet, n_prism, n_real_prism, n_real_tet,
                Qg0, QgP, ghost_row, HG, vbc_kind, owner_cell, owner_cube_face, E_nat,
                ip_length[f], mu, Pr, Pr_t, c_ip, mach_ref, precond_mode, want_cross)
            k = slot[c]
            K = K_prism if is_p else K_tet
            for s in range(n):
                for o in range(5):
                    for t in range(n):
                        for b in range(5):
                            K[k, s, o, t, b] += blk[s, o, t, b]
            if want_cross:
                for src in range(N_CROSS_SOURCES):
                    off = cross_offset[f, side, src]
                    if off < 0:
                        continue
                    y = cross_cells[src]
                    cross_col[f, side, src] = y
                    n_y = n_real_prism if y < n_prism else n_real_tet
                    pos = off
                    for s in range(n):
                        for o in range(5):
                            for t in range(n_y):
                                for b in range(5):
                                    cross_data[pos] = cross[src, s, o, t, b]
                                    pos += 1
