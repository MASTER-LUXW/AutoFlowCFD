"""AutoFlowCFD V2.0 - 湍流标量输运的面数据：按"本侧坐标系"外插与提升（numba）。

## 每一侧都在自己的通量点顺序里算（2026-09-26，真实缺陷修复）

同一个物理面上，owner 与 neighbor 各自的面局部参数化给出的通量点**顺序
不同**（plate_demo 真实网格：53.3% 的内部面两侧顺序不一致——三角面的
点可以旋转/镜像）。DG 提升算子 `lift_native[code-6]` 按**该侧自己的**顺序
消费跳变量，所以两侧的跳变量必须分别在各自的顺序里构造：

    owner 侧      self  = E_owner    @ phi[owner]             （owner 顺序）
                  other = neighbor_src @ phi[neighbor 来源]    （同上）
    neighbor 侧   self  = E_neighbor @ phi[neighbor]          （neighbor 顺序）
                  other = owner_src    @ phi[owner 来源]       （同上）

这与平均流无粘/粘性 kernel（`fr_residual/inviscid_kernel_colored.py` 等）的
neighbor-primary 分支是同一个结构。此前本模块把 owner 顺序下算出的跳变量
直接喂给 neighbor 的提升算子：对流算子在均匀来流下线性不稳定（CFL 0.1 的
SSP-RK3 下扰动 200 步 rms 从 8e-3 涨到 4e-2 并扩散到 13% 的单元，修正后
单调衰减），这是 P1 湍流场单元内锯齿、贴下限、逐单元松弛冻结的根因。

## 提升的符号约定只有一个

两侧都按 `J_side = (F* - F_side) . n_side`（`n_side` 为该侧外法向）构造跳变量，
提升统一为 `corr[cell] += sign * lift_side @ (w_side * J_side) / det`：对流
（`dphi/dt = -div F`）`sign=-1`，扩散（`dphi/dt = +div G`）`sign=+1`。面积权重
`w_side = ref_area_weight * |adj_row_side|`（物理面积权重，跳变量是物理通量
密度差；owner 侧它逐位等于既有的 `true_area_weight`）。此前 owner "-=" /
neighbor "+=" 的非对称约定只对反对称跳变量成立，扩散（BR1 跳变量两侧对称）
因此在 neighbor 侧是**反扩散**。
"""

import numpy as np
from numba import get_thread_id, njit, prange

from autoflowcfd.core.fr_operators.small_dense import matvec_small


@njit(cache=True, parallel=True)
def extrapolate_scalar_pair_kernel(
    scalar_sps,
    self_cell, self_cube_face, boundary_extrap_native,
    other_src0_cell, other_src0_mat,
    other_src1_idx, other_src1_cell, other_src1_mat,
    apply_boundary_ghost,
    wall_dirichlet_zero_face, has_wall_dirichlet_value, wall_dirichlet_value_face,
    mixed_partner, mixed_mask,
):
    """某一侧坐标系下的 `(phi_self, phi_other)`，各 `(n_faces, n_fp)`。

    `self_cell[f] < 0` 的面（neighbor 坐标系下的边界面）两者都留 0，调用方不消费。

    ghost 规则（`apply_boundary_ghost` 只在 owner 坐标系为真——真边界面只有 owner
    一侧）：`other` 没有任何来源的面，Dirichlet-zero 壁面取镜像 `-self`（k=0），
    非零 Dirichlet 取 `2*target - self`（omega 壁面解析值），其余取零梯度 `self`。
    混合拆分面（B-8，`fr/face_flux_points/merge.py`）的边界半区按配对边界面同一
    规则逐通量点覆盖；配对边界面的 owner 就是本侧单元，所以它的逐通量点目标值
    与本侧同一顺序。
    """
    n_faces = self_cell.shape[0]
    n_fp = boundary_extrap_native.shape[1]
    n_sps = scalar_sps.shape[1]
    phi_self = np.zeros((n_faces, n_fp))
    phi_other = np.zeros((n_faces, n_fp))
    for f in prange(n_faces):
        sc = self_cell[f]
        if sc < 0:
            continue
        E = boundary_extrap_native[self_cube_face[f] - 6]
        for i in range(n_fp):
            v = 0.0
            for s in range(n_sps):
                v += E[i, s] * scalar_sps[sc, s]
            phi_self[f, i] = v
        c0 = other_src0_cell[f]
        idx1 = other_src1_idx[f]
        if c0 >= 0:
            m0 = other_src0_mat[f]
            for i in range(n_fp):
                v = 0.0
                for s in range(n_sps):
                    v += m0[i, s] * scalar_sps[c0, s]
                phi_other[f, i] = v
        if idx1 >= 0:
            c1 = other_src1_cell[idx1]
            m1 = other_src1_mat[idx1]
            for i in range(n_fp):
                v = 0.0
                for s in range(n_sps):
                    v += m1[i, s] * scalar_sps[c1, s]
                phi_other[f, i] += v
        if apply_boundary_ghost and c0 < 0 and idx1 < 0:
            for i in range(n_fp):
                if wall_dirichlet_zero_face[f]:
                    phi_other[f, i] = -phi_self[f, i]
                elif has_wall_dirichlet_value[f]:
                    phi_other[f, i] = 2.0 * wall_dirichlet_value_face[f, i] - phi_self[f, i]
                else:
                    phi_other[f, i] = phi_self[f, i]
        mp = mixed_partner[f]
        if mp >= 0:
            for i in range(n_fp):
                if mixed_mask[f, i]:
                    if wall_dirichlet_zero_face[mp]:
                        phi_other[f, i] = -phi_self[f, i]
                    elif has_wall_dirichlet_value[mp]:
                        phi_other[f, i] = 2.0 * wall_dirichlet_value_face[mp, i] - phi_self[f, i]
                    else:
                        phi_other[f, i] = phi_self[f, i]
    return phi_self, phi_other


@njit(cache=True, parallel=True)
def face_mass_flux_kernel(
    rho_u, owner_cell, owner_cube_face, neighbor_cell, neighbor_cube_face, boundary_extrap_native,
    owner_src0_cell, owner_src0_mat, owner_src1_idx, owner_src1_cell, owner_src1_mat,
    mixed_ow_partner, mixed_ow_mask, normal_owner, normal_neighbor,
):
    """两侧坐标系下通量点上的质量通量 `(m_owner, m_neighbor)`，各 `(n_faces, n_fp)`。

    都取 **owner 的迹** `(rho*u)|_owner`：owner 坐标系点乘 owner 外法向，neighbor
    坐标系（经 `owner_src` 取到 neighbor 通量点上）点乘 neighbor 外法向，于是同一
    物理点上 `m_neighbor = -m_owner`——公共通量单值（守恒）、上风选择两侧一致。
    混合拆分面的边界半区没有 owner 覆盖，取 neighbor 自身的迹（与 owner 坐标系在
    真边界面上用自身迹同一原则）。
    """
    n_faces = owner_cell.shape[0]
    n_fp = boundary_extrap_native.shape[1]
    n_sps = rho_u.shape[1]
    m_o = np.zeros((n_faces, n_fp))
    m_n = np.zeros((n_faces, n_fp))
    for f in prange(n_faces):
        oc = owner_cell[f]
        E = boundary_extrap_native[owner_cube_face[f] - 6]
        for i in range(n_fp):
            acc = 0.0
            for d in range(3):
                t = 0.0
                for s in range(n_sps):
                    t += E[i, s] * rho_u[oc, s, d]
                acc += t * normal_owner[f, i, d]
            m_o[f, i] = acc
        nc = neighbor_cell[f]
        if nc < 0:
            continue
        c0 = owner_src0_cell[f]
        idx1 = owner_src1_idx[f]
        mp = mixed_ow_partner[f]
        En = boundary_extrap_native[neighbor_cube_face[f] - 6]
        for i in range(n_fp):
            acc = 0.0
            for d in range(3):
                t = 0.0
                if mp >= 0 and mixed_ow_mask[f, i]:
                    for s in range(n_sps):
                        t += En[i, s] * rho_u[nc, s, d]
                else:
                    if c0 >= 0:
                        for s in range(n_sps):
                            t += owner_src0_mat[f, i, s] * rho_u[c0, s, d]
                    if idx1 >= 0:
                        c1 = owner_src1_cell[idx1]
                        for s in range(n_sps):
                            t += owner_src1_mat[idx1, i, s] * rho_u[c1, s, d]
                acc += t * normal_neighbor[f, i, d]
            m_n[f, i] = acc
    return m_o, m_n


@njit(cache=True, parallel=True)
def diffusion_face_jumps_kernel(
    phi, gamma, grad_phi,
    self_cell, self_cube_face, boundary_extrap_native,
    other_src0_cell, other_src0_mat, other_src1_idx, other_src1_cell, other_src1_mat,
    normal, h_face, c_ip, is_boundary, is_dirichlet, target,
):
    """某一侧坐标系下扩散界面项的跳变量 `J = (G* - G_self) . n_self`，`(n_faces, n_fp)`。

    与 `diffusion.py` 文档的公式逐项对应，一趟算完（自身/对侧的 phi、Gamma、
    法向梯度各自外插再组合，不物化十份中间数组）：

        内部点   J = 1/2 (Gamma_o dphi_o/dn - Gamma_s dphi_s/dn) - eta (phi_s - phi_o)
                 eta = c_ip * (Gamma_s + Gamma_o) / 2 / h_face
        壁面点   J = -(c_ip Gamma_s / h_face) (phi_s - target)       （内罚 Dirichlet）
        其余边界 J = -Gamma_s dphi_s/dn                               （齐次 Neumann）

    边界点分类由 `faces.boundary_diffusion_targets` 给出（CPU/GPU 共用那份规则）。
    """
    n_faces = self_cell.shape[0]
    n_fp = boundary_extrap_native.shape[1]
    n_sps = phi.shape[1]
    J = np.zeros((n_faces, n_fp))
    for f in prange(n_faces):
        sc = self_cell[f]
        if sc < 0:
            continue
        E = boundary_extrap_native[self_cube_face[f] - 6]
        c0 = other_src0_cell[f]
        idx1 = other_src1_idx[f]
        h = h_face[f]
        for i in range(n_fp):
            n0 = normal[f, i, 0]
            n1 = normal[f, i, 1]
            n2 = normal[f, i, 2]
            ps = 0.0
            gs = 0.0
            dns = 0.0
            for s in range(n_sps):
                w = E[i, s]
                ps += w * phi[sc, s]
                gs += w * gamma[sc, s]
                dns += w * (grad_phi[sc, s, 0] * n0 + grad_phi[sc, s, 1] * n1 + grad_phi[sc, s, 2] * n2)
            if is_boundary[f, i]:
                if is_dirichlet[f, i]:
                    J[f, i] = -(c_ip * gs / h) * (ps - target[f, i])
                else:
                    J[f, i] = -gs * dns
                continue
            po = 0.0
            go = 0.0
            dno = 0.0
            if c0 >= 0:
                for s in range(n_sps):
                    w = other_src0_mat[f, i, s]
                    po += w * phi[c0, s]
                    go += w * gamma[c0, s]
                    dno += w * (grad_phi[c0, s, 0] * n0 + grad_phi[c0, s, 1] * n1 + grad_phi[c0, s, 2] * n2)
            if idx1 >= 0:
                c1 = other_src1_cell[idx1]
                for s in range(n_sps):
                    w = other_src1_mat[idx1, i, s]
                    po += w * phi[c1, s]
                    go += w * gamma[c1, s]
                    dno += w * (grad_phi[c1, s, 0] * n0 + grad_phi[c1, s, 1] * n1 + grad_phi[c1, s, 2] * n2)
            eta = c_ip * 0.5 * (gs + go) / h
            J[f, i] = 0.5 * (go * dno - gs * dns) - eta * (ps - po)
    return J


@njit(cache=True, inline='always')
def _lift_one_side(jump_f, adj_row_f, ref_area_weight, lift, n_fp):
    """`lift @ (ref_area_weight * |adj_row| * jump)`，返回 `(n_sps,)`。"""
    weighted = np.empty(n_fp)
    for i in range(n_fp):
        a0 = adj_row_f[i, 0]
        a1 = adj_row_f[i, 1]
        a2 = adj_row_f[i, 2]
        weighted[i] = ref_area_weight[i] * np.sqrt(a0 * a0 + a1 * a1 + a2 * a2) * jump_f[i]
    return matvec_small(lift, weighted)


@njit(cache=True, parallel=True)
def lift_side_jumps_kernel_colored(
    jump_owner, jump_neighbor, sign,
    owner_cell, neighbor_cell, owner_cube_face, neighbor_cube_face,
    owner_adj_row_exact, neighbor_adj_row_exact, ref_area_weight, lift_native,
    owner_is_primary, neighbor_is_primary, det_jacs,
    face_indices, out,
):
    """两侧跳变量提升回解点（图着色版：同色面不共享单元，直接写 `out`）。"""
    n_fp = jump_owner.shape[1]
    n_sps = out.shape[1]
    for fi in prange(face_indices.shape[0]):
        f = face_indices[fi]
        if owner_is_primary[f]:
            oc = owner_cell[f]
            c = _lift_one_side(jump_owner[f], owner_adj_row_exact[f], ref_area_weight,
                               lift_native[owner_cube_face[f] - 6], n_fp)
            for s in range(n_sps):
                out[oc, s] += sign * c[s] / det_jacs[oc, s]
        nc = neighbor_cell[f]
        if nc >= 0 and neighbor_is_primary[f]:
            c = _lift_one_side(jump_neighbor[f], neighbor_adj_row_exact[f], ref_area_weight,
                               lift_native[neighbor_cube_face[f] - 6], n_fp)
            for s in range(n_sps):
                out[nc, s] += sign * c[s] / det_jacs[nc, s]


@njit(cache=True, parallel=True)
def lift_side_jumps_kernel(
    jump_owner, jump_neighbor, sign,
    owner_cell, neighbor_cell, owner_cube_face, neighbor_cube_face,
    owner_adj_row_exact, neighbor_adj_row_exact, ref_area_weight, lift_native,
    owner_is_primary, neighbor_is_primary, det_jacs, n_threads,
):
    """同 `lift_side_jumps_kernel_colored`，逐线程私有缓冲版（`AFCFD_USE_COLORING=0`）。"""
    n_faces, n_fp = jump_owner.shape
    n_cells, n_sps = det_jacs.shape
    buf = np.zeros((n_threads, n_cells, n_sps))
    for f in prange(n_faces):
        tid = get_thread_id()
        if owner_is_primary[f]:
            oc = owner_cell[f]
            c = _lift_one_side(jump_owner[f], owner_adj_row_exact[f], ref_area_weight,
                               lift_native[owner_cube_face[f] - 6], n_fp)
            for s in range(n_sps):
                buf[tid, oc, s] += sign * c[s] / det_jacs[oc, s]
        nc = neighbor_cell[f]
        if nc >= 0 and neighbor_is_primary[f]:
            c = _lift_one_side(jump_neighbor[f], neighbor_adj_row_exact[f], ref_area_weight,
                               lift_native[neighbor_cube_face[f] - 6], n_fp)
            for s in range(n_sps):
                buf[tid, nc, s] += sign * c[s] / det_jacs[nc, s]
    return buf.sum(axis=0)
