"""无粘残差界面项 —— 图着色版本 kernel。

从 inviscid_kernel.py 拆出，控制单文件行数。

图着色版本的无粘界面 kernel，与主 kernel
(compute_inviscid_interface_correction_kernel) 逻辑相同，但：
1. 只处理 face_indices 指定的面（当前颜色组）
2. 直接写入共享 correction buffer（同色面无 owner_cell 冲突）
3. 无需 per-thread buffer 和 sum 归约

调用方按颜色循环调用此函数，每种颜色处理约 n_faces/n_colors 个面。
内存从 O(n_threads * n_cells * n_sps * 5) 降至 O(n_cells * n_sps * 5)。
"""

import numpy as np
from numba import njit, prange

from autoflowcfd.core.fr_operators.kernels import compute_ausm_up_flux
from autoflowcfd.core.fr_operators.flux_kernels import euler_physical_flux_point


@njit(cache=True)
def _extrap_matmul(field_cell: np.ndarray, E: np.ndarray) -> np.ndarray:
    """外插矩阵乘法：field_cell (n_sps, k), E (n_fp, n_sps) -> (n_fp, k)。"""
    return E @ field_cell


@njit(cache=True)
def _distribute_point(fp_data: np.ndarray, fp_of_sp_axis: np.ndarray,
                       axis_coord_of_sp_axis: np.ndarray, g_prime: np.ndarray) -> np.ndarray:
    """`_distribute_from_face` 的逐点等价形式。"""
    n_sps = fp_of_sp_axis.shape[0]
    n_vars = fp_data.shape[1]
    out = np.zeros((n_sps, n_vars))
    for s in range(n_sps):
        fp_i = fp_of_sp_axis[s]
        g = g_prime[axis_coord_of_sp_axis[s]]
        for v in range(n_vars):
            out[s, v] = g * fp_data[fp_i, v]
    return out


@njit(cache=True, parallel=True)
def compute_inviscid_interface_correction_kernel_colored(
    Q: np.ndarray, adj_j: np.ndarray, det_jacs: np.ndarray,
    owner_cell: np.ndarray, neighbor_cell: np.ndarray, is_boundary: np.ndarray,
    owner_axis: np.ndarray, owner_side: np.ndarray,
    neighbor_axis: np.ndarray, neighbor_side: np.ndarray,
    owner_is_primary: np.ndarray, neighbor_is_primary: np.ndarray,
    true_normal: np.ndarray,
    neighbor_src0_cell: np.ndarray, neighbor_src0_mat: np.ndarray,
    neighbor_src1_idx: np.ndarray, neighbor_src1_cell: np.ndarray, neighbor_src1_mat: np.ndarray,
    owner_src0_cell: np.ndarray, owner_src0_mat: np.ndarray,
    owner_src1_idx: np.ndarray, owner_src1_cell: np.ndarray, owner_src1_mat: np.ndarray,
    boundary_extrap: np.ndarray,
    g_left: np.ndarray, g_right: np.ndarray,
    Q_ghost: np.ndarray,
    dist_fp_of_sp: np.ndarray, dist_axis_coord_of_sp: np.ndarray,
    n_prism: int,
    face_indices: np.ndarray,  # 当前颜色组的面索引
    correction: np.ndarray,    # 共享输出 buffer（同色面无冲突，直接写入）
) -> None:
    """图着色版本的无粘界面 kernel。

    与 compute_inviscid_interface_correction_kernel 相同逻辑，但：
    1. 只处理 face_indices 指定的面（当前颜色组）
    2. 直接写入共享 correction buffer（同色面无 owner_cell 冲突）
    3. 无需 per-thread buffer 和 sum 归约

    调用方按颜色循环调用此函数，每种颜色处理约 n_faces/n_colors 个面。
    内存从 O(n_threads * n_cells * n_sps * 5) 降至 O(n_cells * n_sps * 5)。
    """
    n_cells = Q.shape[0]
    n_sps = Q.shape[1]
    n_fp = true_normal.shape[1]
    n_faces_in_color = face_indices.shape[0]

    for fi in prange(n_faces_in_color):
        f = face_indices[fi]
        oc = owner_cell[f]
        oax = owner_axis[f]
        oside = owner_side[f]
        oside_idx = 0 if oside < 0 else 1
        celltype_o = 0 if oc < n_prism else 1

        if owner_is_primary[f]:
            E_o = boundary_extrap[celltype_o, oax, oside_idx]

            Q_o = _extrap_matmul(Q[oc], E_o)
            adjrow_o = _extrap_matmul(np.ascontiguousarray(adj_j[oc, :, oax, :]), E_o)

            jump_owner = np.zeros((n_fp, 5))
            for i in range(n_fp):
                a0 = adjrow_o[i, 0]
                a1 = adjrow_o[i, 1]
                a2 = adjrow_o[i, 2]
                adj_mag = np.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
                if adj_mag < 1e-300:
                    adj_mag = 1e-300
                dirx = a0 / adj_mag * oside
                diry = a1 / adj_mag * oside
                dirz = a2 / adj_mag * oside

                alignment = dirx * true_normal[f, i, 0] + diry * true_normal[f, i, 1] + dirz * true_normal[f, i, 2]
                if alignment < 0.5:
                    dirx = true_normal[f, i, 0]
                    diry = true_normal[f, i, 1]
                    dirz = true_normal[f, i, 2]

                if is_boundary[f]:
                    Q_n = Q_ghost[f, i]
                else:
                    Q_n = np.zeros(5)
                    c0 = neighbor_src0_cell[f]
                    if c0 >= 0:
                        mat0 = neighbor_src0_mat[f]
                        for s in range(n_sps):
                            w = mat0[i, s]
                            if w != 0.0:
                                for v in range(5):
                                    Q_n[v] += w * Q[c0, s, v]
                    idx1 = neighbor_src1_idx[f]
                    if idx1 >= 0:
                        c1 = neighbor_src1_cell[idx1]
                        mat1 = neighbor_src1_mat[idx1]
                        for s in range(n_sps):
                            w = mat1[i, s]
                            if w != 0.0:
                                for v in range(5):
                                    Q_n[v] += w * Q[c1, s, v]

                normal = np.empty(3)
                normal[0] = dirx
                normal[1] = diry
                normal[2] = dirz
                F_common_n = compute_ausm_up_flux(Q_o[i], Q_n, normal)

                F_tilde_common = np.empty(5)
                for v in range(5):
                    F_tilde_common[v] = F_common_n[v] * adj_mag * oside

                F_phys_o = euler_physical_flux_point(Q_o[i])
                F_tilde_own = np.zeros(5)
                for v in range(5):
                    F_tilde_own[v] = a0 * F_phys_o[0, v] + a1 * F_phys_o[1, v] + a2 * F_phys_o[2, v]

                for v in range(5):
                    jump_owner[i, v] = F_tilde_common[v] - F_tilde_own[v]

            g_prime_owner = g_left if oside < 0 else g_right
            contrib_owner = _distribute_point(
                jump_owner, dist_fp_of_sp[oax], dist_axis_coord_of_sp[oax], g_prime_owner
            )
            for s in range(n_sps):
                dj = det_jacs[oc, s]
                for v in range(5):
                    correction[oc, s, v] += -contrib_owner[s, v] / dj

        if (not is_boundary[f]) and neighbor_is_primary[f]:
            nc = neighbor_cell[f]
            nax = neighbor_axis[f]
            nside = neighbor_side[f]
            nside_idx = 0 if nside < 0 else 1
            celltype_n = 0 if nc < n_prism else 1

            E_n = boundary_extrap[celltype_n, nax, nside_idx]

            Q_n_native = _extrap_matmul(Q[nc], E_n)
            adjrow_n_native = _extrap_matmul(np.ascontiguousarray(adj_j[nc, :, nax, :]), E_n)

            jump_neighbor = np.zeros((n_fp, 5))
            for i in range(n_fp):
                a0 = adjrow_n_native[i, 0]
                a1 = adjrow_n_native[i, 1]
                a2 = adjrow_n_native[i, 2]
                adj_mag = np.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
                if adj_mag < 1e-300:
                    adj_mag = 1e-300
                dirx = a0 / adj_mag * nside
                diry = a1 / adj_mag * nside
                dirz = a2 / adj_mag * nside

                ntnx = -true_normal[f, i, 0]
                ntny = -true_normal[f, i, 1]
                ntnz = -true_normal[f, i, 2]
                alignment_n = dirx * ntnx + diry * ntny + dirz * ntnz
                if alignment_n < 0.5:
                    dirx = ntnx
                    diry = ntny
                    dirz = ntnz

                Q_o_at_n = np.zeros(5)
                c0 = owner_src0_cell[f]
                if c0 >= 0:
                    mat0 = owner_src0_mat[f]
                    for s in range(n_sps):
                        w = mat0[i, s]
                        if w != 0.0:
                            for v in range(5):
                                Q_o_at_n[v] += w * Q[c0, s, v]
                idx1 = owner_src1_idx[f]
                if idx1 >= 0:
                    c1 = owner_src1_cell[idx1]
                    mat1 = owner_src1_mat[idx1]
                    for s in range(n_sps):
                        w = mat1[i, s]
                        if w != 0.0:
                            for v in range(5):
                                Q_o_at_n[v] += w * Q[c1, s, v]

                normal = np.empty(3)
                normal[0] = dirx
                normal[1] = diry
                normal[2] = dirz
                F_common_n_native = compute_ausm_up_flux(Q_n_native[i], Q_o_at_n, normal)

                F_tilde_common_n = np.empty(5)
                for v in range(5):
                    F_tilde_common_n[v] = F_common_n_native[v] * adj_mag * nside

                F_phys_n = euler_physical_flux_point(Q_n_native[i])
                F_tilde_own_n = np.zeros(5)
                for v in range(5):
                    F_tilde_own_n[v] = a0 * F_phys_n[0, v] + a1 * F_phys_n[1, v] + a2 * F_phys_n[2, v]

                for v in range(5):
                    jump_neighbor[i, v] = F_tilde_common_n[v] - F_tilde_own_n[v]

            g_prime_neighbor = g_left if nside < 0 else g_right
            contrib_neighbor = _distribute_point(
                jump_neighbor, dist_fp_of_sp[nax], dist_axis_coord_of_sp[nax], g_prime_neighbor
            )
            for s in range(n_sps):
                dj = det_jacs[nc, s]
                for v in range(5):
                    correction[nc, s, v] += -contrib_neighbor[s, v] / dj
