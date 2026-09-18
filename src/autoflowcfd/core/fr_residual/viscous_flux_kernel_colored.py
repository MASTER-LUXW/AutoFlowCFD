"""粘性残差界面项 —— numba 图着色版本 kernel（从 viscous_flux_kernel.py
拆出，控制单文件行数；与该文件此前的拆分理由完全相同，见
fr_residual_inviscid_kernel.py / inviscid_kernel_colored.py 的先例）。

与 compute_viscous_interface_correction_kernel（非 colored 版本，见
viscous_flux_kernel.py）逐字同一套逻辑，唯一区别是按颜色分组避免
scatter-add 冲突、直接写入共享 buffer 而不是 per-thread buffer + 归约。
两处必须同步修改，见各自文档。
"""

import numpy as np
from numba import njit, prange

from autoflowcfd.core.fr_operators.flux_kernels import (
    viscous_physical_flux_point, viscous_boundary_penalty_tilde, mirror_normal_component,
)
from autoflowcfd.core.fr_residual.inviscid_kernel import _extrap_matmul, _distribute_point
from autoflowcfd.core.fr_residual.viscous_flux_kernel import _extrap_matrix3x3, _VISCOUS_BOUNDARY_IP_C


@njit(cache=True, parallel=True)
def compute_viscous_interface_correction_kernel_colored(
    Q: np.ndarray, grad_vel: np.ndarray, grad_T: np.ndarray, mu_t_field: np.ndarray,
    det_jacs: np.ndarray, mu: float, Pr: float, Pr_t: float,
    owner_cell: np.ndarray, neighbor_cell: np.ndarray, is_boundary: np.ndarray,
    owner_axis: np.ndarray, owner_side: np.ndarray,
    neighbor_axis: np.ndarray, neighbor_side: np.ndarray,
    owner_is_primary: np.ndarray, neighbor_is_primary: np.ndarray,
    owner_adj_row_exact: np.ndarray, neighbor_adj_row_exact: np.ndarray,
    neighbor_src0_cell: np.ndarray, neighbor_src0_mat: np.ndarray,
    neighbor_src1_idx: np.ndarray, neighbor_src1_cell: np.ndarray, neighbor_src1_mat: np.ndarray,
    owner_src0_cell: np.ndarray, owner_src0_mat: np.ndarray,
    owner_src1_idx: np.ndarray, owner_src1_cell: np.ndarray, owner_src1_mat: np.ndarray,
    mixed_nb_partner: np.ndarray, mixed_nb_mask: np.ndarray,
    mixed_ow_partner: np.ndarray, mixed_ow_mask: np.ndarray,
    boundary_extrap: np.ndarray,
    g_left: np.ndarray, g_right: np.ndarray,
    Q_ghost: np.ndarray, bnd_adiabatic: np.ndarray,
    dist_fp_of_sp: np.ndarray, dist_axis_coord_of_sp: np.ndarray,
    n_prism: int,
    face_indices: np.ndarray,  # 当前颜色组的面索引
    correction: np.ndarray,    # 共享输出 buffer（同色面无冲突，直接写入）
    owner_cube_face: np.ndarray, neighbor_cube_face: np.ndarray,
    ref_area_weight: np.ndarray,
    boundary_extrap_native: np.ndarray, lift_native: np.ndarray,
) -> None:
    """图着色版本的粘性界面 kernel。

    与 compute_viscous_interface_correction_kernel 相同逻辑，但：
    1. 只处理 face_indices 指定的面（当前颜色组）
    2. 直接写入共享 correction buffer（同色面无 owner_cell 冲突）
    3. 无需 per-thread buffer 和 sum 归约

    调用方按颜色循环调用此函数，每种颜色处理约 n_faces/n_colors 个面。
    内存从 O(n_threads * n_cells * n_sps * 5) 降至 O(n_cells * n_sps * 5)。

    真实 bug 修复（2026-08-23）：`owner_adj_row_exact`/
    `neighbor_adj_row_exact` 取代 `adj_j` 外插，理由与
    compute_viscous_interface_correction_kernel（非 colored 版本）
    完全相同，两处必须同步修改。`adj_j` 参数已从签名中移除。

    native 四面体（路径C）支持：与非 colored 版本
    （viscous_flux_kernel.py）同一套 native 分支，两处必须同步修改，
    见该文件模块文档。
    """
    n_cells = Q.shape[0]
    n_sps = Q.shape[1]
    n_fp = Q_ghost.shape[1]
    n_faces_in_color = face_indices.shape[0]

    for fi in prange(n_faces_in_color):
        f = face_indices[fi]
        oc = owner_cell[f]
        oc_code = owner_cube_face[f]
        o_is_native = oc_code >= 6
        oax = owner_axis[f]
        oside = owner_side[f]
        oside_idx = 0 if oside < 0 else 1
        celltype_o = 0 if oc < n_prism else 1

        if owner_is_primary[f]:
            if o_is_native:
                E_o = boundary_extrap_native[oc_code - 6]
            else:
                E_o = boundary_extrap[celltype_o, oax, oside_idx]

            Q_o = _extrap_matmul(Q[oc], E_o)
            gv_o = _extrap_matrix3x3(grad_vel[oc], E_o)
            gT_o = _extrap_matmul(grad_T[oc], E_o)
            mut_o = E_o @ mu_t_field[oc]
            adjrow_o = owner_adj_row_exact[f]  # (n_fp,3)，逐 FP 精确值，见函数文档

            vol_o = 0.0
            for s in range(n_sps):
                vol_o += det_jacs[oc, s]
            vol_o /= n_sps

            jump_owner = np.zeros((n_fp, 5))
            for i in range(n_fp):
                # 混合拆分面（B-8，与非着色版同步，见 compute_viscous_interface_correction_kernel 同名注释）。
                mp = mixed_nb_partner[f]
                is_bnd_i = is_boundary[f] or (mp >= 0 and mixed_nb_mask[f, i])
                if is_boundary[f]:
                    # 边界温度梯度按热边界类型分派，见非着色版模块文档
                    # "边界温度梯度"一节（两处必须同步）。
                    Q_n = Q_ghost[f, i]
                    gv_n = gv_o[i]
                    if bnd_adiabatic[f]:
                        gT_n = mirror_normal_component(gT_o[i], adjrow_o[i])
                    else:
                        gT_n = gT_o[i].copy()
                    mut_n = mut_o[i]
                else:
                    Q_n = np.zeros(5)
                    gv_n = np.zeros((3, 3))
                    gT_n = np.zeros(3)
                    mut_n = 0.0
                    c0 = neighbor_src0_cell[f]
                    if c0 >= 0:
                        mat0 = neighbor_src0_mat[f]
                        for s in range(n_sps):
                            w = mat0[i, s]
                            if w != 0.0:
                                for v in range(5):
                                    Q_n[v] += w * Q[c0, s, v]
                                for a in range(3):
                                    for b in range(3):
                                        gv_n[a, b] += w * grad_vel[c0, s, a, b]
                                    gT_n[a] += w * grad_T[c0, s, a]
                                mut_n += w * mu_t_field[c0, s]
                    idx1 = neighbor_src1_idx[f]
                    if idx1 >= 0:
                        c1 = neighbor_src1_cell[idx1]
                        mat1 = neighbor_src1_mat[idx1]
                        for s in range(n_sps):
                            w = mat1[i, s]
                            if w != 0.0:
                                for v in range(5):
                                    Q_n[v] += w * Q[c1, s, v]
                                for a in range(3):
                                    for b in range(3):
                                        gv_n[a, b] += w * grad_vel[c1, s, a, b]
                                    gT_n[a] += w * grad_T[c1, s, a]
                                mut_n += w * mu_t_field[c1, s]
                    # 混合拆分面边界半区（B-8）：状态取配对面幽灵态（逐元素拷贝，
                    # 避免 numba C/A 布局赋值冲突），梯度镜像内部值——与真边界面同规则。
                    if mp >= 0 and mixed_nb_mask[f, i]:
                        for v in range(5):
                            Q_n[v] = Q_ghost[mp, i, v]
                        if bnd_adiabatic[mp]:
                            gT_bnd = mirror_normal_component(gT_o[i], adjrow_o[i])
                        else:
                            gT_bnd = gT_o[i]
                        for a in range(3):
                            for b in range(3):
                                gv_n[a, b] = gv_o[i, a, b]
                            gT_n[a] = gT_bnd[a]
                        mut_n = mut_o[i]

                Q_avg = np.empty(5)
                for v in range(5):
                    Q_avg[v] = 0.5 * (Q_o[i, v] + Q_n[v])
                gv_avg = np.empty((3, 3))
                for a in range(3):
                    for b in range(3):
                        gv_avg[a, b] = 0.5 * (gv_o[i, a, b] + gv_n[a, b])
                gT_avg = np.empty(3)
                for a in range(3):
                    gT_avg[a] = 0.5 * (gT_o[i, a] + gT_n[a])
                mut_avg = 0.5 * (mut_o[i] + mut_n)

                G_common = viscous_physical_flux_point(Q_avg, gv_avg, gT_avg, mu, Pr, mut_avg, Pr_t)
                a0 = adjrow_o[i, 0]
                a1 = adjrow_o[i, 1]
                a2 = adjrow_o[i, 2]
                G_tilde_common = np.empty(5)
                for v in range(5):
                    G_tilde_common[v] = a0 * G_common[0, v] + a1 * G_common[1, v] + a2 * G_common[2, v]

                G_phys_o = viscous_physical_flux_point(Q_o[i], gv_o[i], gT_o[i], mu, Pr, mut_o[i], Pr_t)
                G_tilde_own = np.empty(5)
                for v in range(5):
                    G_tilde_own[v] = a0 * G_phys_o[0, v] + a1 * G_phys_o[1, v] + a2 * G_phys_o[2, v]

                for v in range(5):
                    jump_owner[i, v] = G_tilde_common[v] - G_tilde_own[v]

                if is_bnd_i:
                    # 边界 IP 罚项，见 compute_viscous_interface_correction_kernel
                    # 同名分支的文档（图着色版本，逻辑必须逐字保持一致；含混合面边界半区）。
                    adj_mag_o = np.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
                    pen = viscous_boundary_penalty_tilde(
                        Q_o[i], Q_n, mu + mut_o[i], vol_o, adj_mag_o, oside, _VISCOUS_BOUNDARY_IP_C,
                    )
                    for v in range(1, 4):
                        jump_owner[i, v] += pen[v]

            if o_is_native:
                weighted_jump_o = np.empty((n_fp, 5))
                for i in range(n_fp):
                    w_area = ref_area_weight[i]
                    for v in range(5):
                        weighted_jump_o[i, v] = w_area * jump_owner[i, v]
                contrib_owner = lift_native[oc_code - 6] @ weighted_jump_o
            else:
                g_prime_owner = g_left if oside < 0 else g_right
                contrib_owner = _distribute_point(
                    jump_owner, dist_fp_of_sp[oax], dist_axis_coord_of_sp[oax], g_prime_owner
                )
            for s in range(n_sps):
                dj = det_jacs[oc, s]
                for v in range(5):
                    correction[oc, s, v] += contrib_owner[s, v] / dj

        if (not is_boundary[f]) and neighbor_is_primary[f]:
            nc = neighbor_cell[f]
            nc_code = neighbor_cube_face[f]
            n_is_native = nc_code >= 6
            nax = neighbor_axis[f]
            nside = neighbor_side[f]
            nside_idx = 0 if nside < 0 else 1
            celltype_n = 0 if nc < n_prism else 1

            if n_is_native:
                E_n = boundary_extrap_native[nc_code - 6]
            else:
                E_n = boundary_extrap[celltype_n, nax, nside_idx]

            Q_n_native = _extrap_matmul(Q[nc], E_n)
            gv_n_native = _extrap_matrix3x3(grad_vel[nc], E_n)
            gT_n_native = _extrap_matmul(grad_T[nc], E_n)
            mut_n_native = E_n @ mu_t_field[nc]
            adjrow_n_native = neighbor_adj_row_exact[f]  # (n_fp,3)，逐 FP 精确值，见函数文档

            # neighbor 侧单元体积代理（混合面边界半区 IP 罚项用，B-8），算法同 vol_o。
            vol_n = 0.0
            for s in range(n_sps):
                vol_n += det_jacs[nc, s]
            vol_n /= n_sps

            jump_neighbor = np.zeros((n_fp, 5))
            for i in range(n_fp):
                Q_o_at_n = np.zeros(5)
                gv_o_at_n = np.zeros((3, 3))
                gT_o_at_n = np.zeros(3)
                mut_o_at_n = 0.0
                c0 = owner_src0_cell[f]
                if c0 >= 0:
                    mat0 = owner_src0_mat[f]
                    for s in range(n_sps):
                        w = mat0[i, s]
                        if w != 0.0:
                            for v in range(5):
                                Q_o_at_n[v] += w * Q[c0, s, v]
                            for a in range(3):
                                for b in range(3):
                                    gv_o_at_n[a, b] += w * grad_vel[c0, s, a, b]
                                gT_o_at_n[a] += w * grad_T[c0, s, a]
                            mut_o_at_n += w * mu_t_field[c0, s]
                idx1 = owner_src1_idx[f]
                if idx1 >= 0:
                    c1 = owner_src1_cell[idx1]
                    mat1 = owner_src1_mat[idx1]
                    for s in range(n_sps):
                        w = mat1[i, s]
                        if w != 0.0:
                            for v in range(5):
                                Q_o_at_n[v] += w * Q[c1, s, v]
                            for a in range(3):
                                for b in range(3):
                                    gv_o_at_n[a, b] += w * grad_vel[c1, s, a, b]
                                gT_o_at_n[a] += w * grad_T[c1, s, a]
                            mut_o_at_n += w * mu_t_field[c1, s]
                # 混合拆分面边界半区（B-8）：neighbor 侧对称处理——对侧状态取配对面幽灵态（逐元素拷贝），
                # 梯度镜像本单元内部值，与边界面同规则。
                mp_o = mixed_ow_partner[f]
                if mp_o >= 0 and mixed_ow_mask[f, i]:
                    for v in range(5):
                        Q_o_at_n[v] = Q_ghost[mp_o, i, v]
                    if bnd_adiabatic[mp_o]:
                        gT_bnd_n = mirror_normal_component(gT_n_native[i], adjrow_n_native[i])
                    else:
                        gT_bnd_n = gT_n_native[i]
                    for a in range(3):
                        for b in range(3):
                            gv_o_at_n[a, b] = gv_n_native[i, a, b]
                        gT_o_at_n[a] = gT_bnd_n[a]
                    mut_o_at_n = mut_n_native[i]

                Q_avg_n = np.empty(5)
                for v in range(5):
                    Q_avg_n[v] = 0.5 * (Q_n_native[i, v] + Q_o_at_n[v])
                gv_avg_n = np.empty((3, 3))
                for a in range(3):
                    for b in range(3):
                        gv_avg_n[a, b] = 0.5 * (gv_n_native[i, a, b] + gv_o_at_n[a, b])
                gT_avg_n = np.empty(3)
                for a in range(3):
                    gT_avg_n[a] = 0.5 * (gT_n_native[i, a] + gT_o_at_n[a])
                mut_avg_n = 0.5 * (mut_n_native[i] + mut_o_at_n)

                G_common_native = viscous_physical_flux_point(Q_avg_n, gv_avg_n, gT_avg_n, mu, Pr, mut_avg_n, Pr_t)
                a0 = adjrow_n_native[i, 0]
                a1 = adjrow_n_native[i, 1]
                a2 = adjrow_n_native[i, 2]
                G_tilde_common_n = np.empty(5)
                for v in range(5):
                    G_tilde_common_n[v] = a0 * G_common_native[0, v] + a1 * G_common_native[1, v] + a2 * G_common_native[2, v]

                G_phys_n = viscous_physical_flux_point(Q_n_native[i], gv_n_native[i], gT_n_native[i], mu, Pr, mut_n_native[i], Pr_t)
                G_tilde_own_n = np.empty(5)
                for v in range(5):
                    G_tilde_own_n[v] = a0 * G_phys_n[0, v] + a1 * G_phys_n[1, v] + a2 * G_phys_n[2, v]

                for v in range(5):
                    jump_neighbor[i, v] = G_tilde_common_n[v] - G_tilde_own_n[v]

                # 混合拆分面边界半区（B-8）：neighbor 侧同样需要边界 IP 罚项（罚项的“内部态”
                # 是本单元外插值 Q_n_native、“对侧”是幽灵态），与 owner 侧一致。
                if mp_o >= 0 and mixed_ow_mask[f, i]:
                    adj_mag_n = np.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
                    pen_n = viscous_boundary_penalty_tilde(
                        Q_n_native[i], Q_o_at_n, mu + mut_n_native[i], vol_n, adj_mag_n, nside, _VISCOUS_BOUNDARY_IP_C,
                    )
                    for v in range(1, 4):
                        jump_neighbor[i, v] += pen_n[v]

            if n_is_native:
                weighted_jump_n = np.empty((n_fp, 5))
                for i in range(n_fp):
                    w_area = ref_area_weight[i]
                    for v in range(5):
                        weighted_jump_n[i, v] = w_area * jump_neighbor[i, v]
                contrib_neighbor = lift_native[nc_code - 6] @ weighted_jump_n
            else:
                g_prime_neighbor = g_left if nside < 0 else g_right
                contrib_neighbor = _distribute_point(
                    jump_neighbor, dist_fp_of_sp[nax], dist_axis_coord_of_sp[nax], g_prime_neighbor
                )
            for s in range(n_sps):
                dj = det_jacs[nc, s]
                for v in range(5):
                    correction[nc, s, v] += contrib_neighbor[s, v] / dj
