"""
AutoFlowCFD V2.0 - P0 专用粘性界面校正 numba kernel

从 viscous_flux_kernel.py 拆出。当 n_sps=1（P0 order continuation 阶段）
时使用本 kernel 替代通用 kernel，消除所有 SP 循环（编译期常量 n_sps=1），
将外插矩阵乘法简化为标量乘法。

性能收益（791K 单元 / 188 万面网格，P0 阶段）：
- 消除 for s in range(n_sps) 循环（n_sps=1 时仍有一次迭代开销）
- 外插 E@field 简化为 E[i,0]*field[0]（标量乘，避免矩阵乘法开销）
- 输出 correction 形状 (n_cells, 1, 5) 而非 (n_cells, n_sps, 5)

算法与通用 viscous_flux_kernel.py 完全一致（n_sps=1 的特化），
数学等价，仅浮点重排顺序不同。
"""

import numpy as np
from numba import njit, prange, get_thread_id

from autoflowcfd.core.fr_operators.flux_kernels import viscous_physical_flux_point


@njit(cache=True, parallel=True)
def compute_viscous_interface_correction_p0_kernel(
    Q: np.ndarray,             # (n_cells, 1, 5)
    grad_vel: np.ndarray,      # (n_cells, 1, 3, 3)
    grad_T: np.ndarray,        # (n_cells, 1, 3)
    mu_t_field: np.ndarray,    # (n_cells, 1)
    adj_j: np.ndarray,         # (n_cells, 1, 3, 3)
    det_jacs: np.ndarray,      # (n_cells, 1)
    mu: float, Pr: float, Pr_t: float,
    owner_cell: np.ndarray, neighbor_cell: np.ndarray, is_boundary: np.ndarray,
    owner_axis: np.ndarray, owner_side: np.ndarray,
    neighbor_axis: np.ndarray, neighbor_side: np.ndarray,
    owner_is_primary: np.ndarray, neighbor_is_primary: np.ndarray,
    neighbor_src0_cell: np.ndarray, neighbor_src0_mat: np.ndarray,
    neighbor_src1_idx: np.ndarray, neighbor_src1_cell: np.ndarray, neighbor_src1_mat: np.ndarray,
    owner_src0_cell: np.ndarray, owner_src0_mat: np.ndarray,
    owner_src1_idx: np.ndarray, owner_src1_cell: np.ndarray, owner_src1_mat: np.ndarray,
    boundary_extrap: np.ndarray,  # (2, 3, 2, n_fp, 1)
    g_left: np.ndarray, g_right: np.ndarray,
    Q_ghost: np.ndarray,          # (n_boundary_faces, n_fp, 5)
    dist_fp_of_sp: np.ndarray,    # (3, 1)
    dist_axis_coord_of_sp: np.ndarray,  # (3, 1)
    n_prism: int,
    n_threads: int,
) -> np.ndarray:
    """P0 专用粘性界面校正 kernel。

    与通用 compute_viscous_interface_correction_kernel 数学等价，
    但 n_sps=1 时：
    - 外插简化为标量乘：E[i,0]*field[0,...]
    - 分布简化为单 SP：out[0,v] = g * fp_data[fp_i, v]
    - correction 形状 (n_cells, 1, 5)
    """
    n_cells = Q.shape[0]
    n_faces = owner_cell.shape[0]
    n_fp = Q_ghost.shape[1]

    correction_per_thread = np.zeros((n_threads, n_cells, 1, 5))

    for f in prange(n_faces):
        tid = get_thread_id()
        oc = owner_cell[f]
        oax = owner_axis[f]
        oside = owner_side[f]
        oside_idx = 0 if oside < 0 else 1
        celltype_o = 0 if oc < n_prism else 1

        if owner_is_primary[f]:
            E_o = boundary_extrap[celltype_o, oax, oside_idx]  # (n_fp, 1)

            # P0 简化外插：E (n_fp,1) @ field (1,k) -> E[i,0]*field[0,...]
            Q_o_s0 = Q[oc, 0]  # (5,)
            gv_o_s0 = grad_vel[oc, 0]  # (3,3)
            gT_o_s0 = grad_T[oc, 0]  # (3,)
            mut_o_s0 = mu_t_field[oc, 0]  # scalar
            adj_o_s0 = adj_j[oc, 0, oax]  # (3,)

            jump_owner = np.zeros((n_fp, 5))
            for i in range(n_fp):
                e_i = E_o[i, 0]  # 标量

                # 外插到 FP i（P0 简化：标量乘）
                Q_o_i = np.empty(5)
                for v in range(5):
                    Q_o_i[v] = e_i * Q_o_s0[v]
                gv_o_i = np.empty((3, 3))
                for a in range(3):
                    for b in range(3):
                        gv_o_i[a, b] = e_i * gv_o_s0[a, b]
                gT_o_i = np.empty(3)
                for a in range(3):
                    gT_o_i[a] = e_i * gT_o_s0[a]
                mut_o_i = e_i * mut_o_s0

                # 邻居状态（源矩阵插值，n_sps=1 简化）
                if is_boundary[f]:
                    Q_n = Q_ghost[f, i]
                    gv_n = gv_o_i.copy()
                    gT_n = gT_o_i.copy()
                    mut_n = mut_o_i
                else:
                    Q_n = np.zeros(5)
                    gv_n = np.zeros((3, 3))
                    gT_n = np.zeros(3)
                    mut_n = 0.0
                    c0 = neighbor_src0_cell[f]
                    if c0 >= 0:
                        mat0 = neighbor_src0_mat[f]
                        w = mat0[i, 0]  # n_sps=1，只有 s=0
                        if w != 0.0:
                            for v in range(5):
                                Q_n[v] += w * Q[c0, 0, v]
                            for a in range(3):
                                for b in range(3):
                                    gv_n[a, b] += w * grad_vel[c0, 0, a, b]
                                gT_n[a] += w * grad_T[c0, 0, a]
                            mut_n += w * mu_t_field[c0, 0]
                    idx1 = neighbor_src1_idx[f]
                    if idx1 >= 0:
                        c1 = neighbor_src1_cell[idx1]
                        mat1 = neighbor_src1_mat[idx1]
                        w = mat1[i, 0]
                        if w != 0.0:
                            for v in range(5):
                                Q_n[v] += w * Q[c1, 0, v]
                            for a in range(3):
                                for b in range(3):
                                    gv_n[a, b] += w * grad_vel[c1, 0, a, b]
                                gT_n[a] += w * grad_T[c1, 0, a]
                            mut_n += w * mu_t_field[c1, 0]

                # 算术平均
                Q_avg = np.empty(5)
                for v in range(5):
                    Q_avg[v] = 0.5 * (Q_o_i[v] + Q_n[v])
                gv_avg = np.empty((3, 3))
                for a in range(3):
                    for b in range(3):
                        gv_avg[a, b] = 0.5 * (gv_o_i[a, b] + gv_n[a, b])
                gT_avg = np.empty(3)
                for a in range(3):
                    gT_avg[a] = 0.5 * (gT_o_i[a] + gT_n[a])
                mut_avg = 0.5 * (mut_o_i + mut_n)

                # 粘性通量
                G_common = viscous_physical_flux_point(Q_avg, gv_avg, gT_avg, mu, Pr, mut_avg, Pr_t)
                a0 = adj_o_s0[0]
                a1 = adj_o_s0[1]
                a2 = adj_o_s0[2]
                G_tilde_common = np.empty(5)
                for v in range(5):
                    G_tilde_common[v] = a0 * G_common[0, v] + a1 * G_common[1, v] + a2 * G_common[2, v]

                G_phys_o = viscous_physical_flux_point(Q_o_i, gv_o_i, gT_o_i, mu, Pr, mut_o_i, Pr_t)
                G_tilde_own = np.empty(5)
                for v in range(5):
                    G_tilde_own[v] = a0 * G_phys_o[0, v] + a1 * G_phys_o[1, v] + a2 * G_phys_o[2, v]

                for v in range(5):
                    jump_owner[i, v] = G_tilde_common[v] - G_tilde_own[v]

            # P0 简化分布：n_sps=1，只有 s=0
            g_prime_owner = g_left if oside < 0 else g_right
            fp_i = dist_fp_of_sp[oax, 0]
            g_val = g_prime_owner[dist_axis_coord_of_sp[oax, 0]]
            dj = det_jacs[oc, 0]
            for v in range(5):
                correction_per_thread[tid, oc, 0, v] += g_val * jump_owner[fp_i, v] / dj

        # Neighbor 侧（与通用 kernel 相同逻辑，n_sps=1 简化）
        if (not is_boundary[f]) and neighbor_is_primary[f]:
            nc = neighbor_cell[f]
            nax = neighbor_axis[f]
            nside = neighbor_side[f]
            nside_idx = 0 if nside < 0 else 1
            celltype_n = 0 if nc < n_prism else 1

            E_n = boundary_extrap[celltype_n, nax, nside_idx]

            Q_n_s0 = Q[nc, 0]
            gv_n_s0 = grad_vel[nc, 0]
            gT_n_s0 = grad_T[nc, 0]
            mut_n_s0 = mu_t_field[nc, 0]
            adj_n_s0 = adj_j[nc, 0, nax]

            jump_neighbor = np.zeros((n_fp, 5))
            for i in range(n_fp):
                e_i = E_n[i, 0]

                Q_n_i = np.empty(5)
                for v in range(5):
                    Q_n_i[v] = e_i * Q_n_s0[v]
                gv_n_i = np.empty((3, 3))
                for a in range(3):
                    for b in range(3):
                        gv_n_i[a, b] = e_i * gv_n_s0[a, b]
                gT_n_i = np.empty(3)
                for a in range(3):
                    gT_n_i[a] = e_i * gT_n_s0[a]
                mut_n_i = e_i * mut_n_s0

                # Owner 侧插值（n_sps=1 简化）
                Q_o_at_n = np.zeros(5)
                gv_o_at_n = np.zeros((3, 3))
                gT_o_at_n = np.zeros(3)
                mut_o_at_n = 0.0
                c0 = owner_src0_cell[f]
                if c0 >= 0:
                    mat0 = owner_src0_mat[f]
                    w = mat0[i, 0]
                    if w != 0.0:
                        for v in range(5):
                            Q_o_at_n[v] += w * Q[c0, 0, v]
                        for a in range(3):
                            for b in range(3):
                                gv_o_at_n[a, b] += w * grad_vel[c0, 0, a, b]
                            gT_o_at_n[a] += w * grad_T[c0, 0, a]
                        mut_o_at_n += w * mu_t_field[c0, 0]
                idx1 = owner_src1_idx[f]
                if idx1 >= 0:
                    c1 = owner_src1_cell[idx1]
                    mat1 = owner_src1_mat[idx1]
                    w = mat1[i, 0]
                    if w != 0.0:
                        for v in range(5):
                            Q_o_at_n[v] += w * Q[c1, 0, v]
                        for a in range(3):
                            for b in range(3):
                                gv_o_at_n[a, b] += w * grad_vel[c1, 0, a, b]
                            gT_o_at_n[a] += w * grad_T[c1, 0, a]
                        mut_o_at_n += w * mu_t_field[c1, 0]

                Q_avg_n = np.empty(5)
                for v in range(5):
                    Q_avg_n[v] = 0.5 * (Q_n_i[v] + Q_o_at_n[v])
                gv_avg_n = np.empty((3, 3))
                for a in range(3):
                    for b in range(3):
                        gv_avg_n[a, b] = 0.5 * (gv_n_i[a, b] + gv_o_at_n[a, b])
                gT_avg_n = np.empty(3)
                for a in range(3):
                    gT_avg_n[a] = 0.5 * (gT_n_i[a] + gT_o_at_n[a])
                mut_avg_n = 0.5 * (mut_n_i + mut_o_at_n)

                G_common_native = viscous_physical_flux_point(Q_avg_n, gv_avg_n, gT_avg_n, mu, Pr, mut_avg_n, Pr_t)
                a0 = adj_n_s0[0]
                a1 = adj_n_s0[1]
                a2 = adj_n_s0[2]
                G_tilde_common_n = np.empty(5)
                for v in range(5):
                    G_tilde_common_n[v] = a0 * G_common_native[0, v] + a1 * G_common_native[1, v] + a2 * G_common_native[2, v]

                G_phys_n = viscous_physical_flux_point(Q_n_i, gv_n_i, gT_n_i, mu, Pr, mut_n_i, Pr_t)
                G_tilde_own_n = np.empty(5)
                for v in range(5):
                    G_tilde_own_n[v] = a0 * G_phys_n[0, v] + a1 * G_phys_n[1, v] + a2 * G_phys_n[2, v]

                for v in range(5):
                    jump_neighbor[i, v] = G_tilde_common_n[v] - G_tilde_own_n[v]

            # P0 简化分布
            g_prime_neighbor = g_left if nside < 0 else g_right
            fp_i = dist_fp_of_sp[nax, 0]
            g_val = g_prime_neighbor[dist_axis_coord_of_sp[nax, 0]]
            dj = det_jacs[nc, 0]
            for v in range(5):
                correction_per_thread[tid, nc, 0, v] += g_val * jump_neighbor[fp_i, v] / dj

    return correction_per_thread.sum(axis=0)
