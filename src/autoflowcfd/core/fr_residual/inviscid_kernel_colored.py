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

from autoflowcfd.core.fr_operators.small_dense import matmul_small
from autoflowcfd.core.fr_operators.kernels import compute_ausm_up_flux
from autoflowcfd.core.fr_operators.flux_kernels import euler_physical_flux_point
from autoflowcfd.core.fr_residual.inviscid_kernel import _extrap_matmul


@njit(cache=True, parallel=True)
def compute_inviscid_interface_correction_kernel_colored(
    Q: np.ndarray, det_jacs: np.ndarray,
    owner_cell: np.ndarray, neighbor_cell: np.ndarray, is_boundary: np.ndarray,
    owner_is_primary: np.ndarray, neighbor_is_primary: np.ndarray,
    owner_adj_row_exact: np.ndarray, neighbor_adj_row_exact: np.ndarray,
    neighbor_src0_cell: np.ndarray, neighbor_src0_mat: np.ndarray,
    neighbor_src1_idx: np.ndarray, neighbor_src1_cell: np.ndarray, neighbor_src1_mat: np.ndarray,
    owner_src0_cell: np.ndarray, owner_src0_mat: np.ndarray,
    owner_src1_idx: np.ndarray, owner_src1_cell: np.ndarray, owner_src1_mat: np.ndarray,
    mixed_nb_partner: np.ndarray, mixed_nb_mask: np.ndarray,
    mixed_ow_partner: np.ndarray, mixed_ow_mask: np.ndarray,
    Q_ghost: np.ndarray,
    face_indices: np.ndarray,  # 当前颜色组的面索引
    correction: np.ndarray,    # 共享输出 buffer（同色面无冲突，直接写入）
    mach_ref: float,
    precond_mode: int,   # AUSM+up 预处理声速作用域，见 kernels.py::
                         # compute_ausm_up_flux 文档；必须由纯 Python 层
                         # 用 resolve_ausm_precond_mode() 解析后传入
    owner_cube_face: np.ndarray, neighbor_cube_face: np.ndarray,
    ref_area_weight: np.ndarray,
    boundary_extrap_native: np.ndarray, lift_native: np.ndarray,
) -> None:
    """图着色版本的无粘界面 kernel。

    与 compute_inviscid_interface_correction_kernel 相同逻辑，但：
    1. 只处理 face_indices 指定的面（当前颜色组）
    2. 直接写入共享 correction buffer（同色面无 owner_cell 冲突）
    3. 无需 per-thread buffer 和 sum 归约

    调用方按颜色循环调用此函数，每种颜色处理约 n_faces/n_colors 个面。
    内存从 O(n_threads * n_cells * n_sps * 5) 降至 O(n_cells * n_sps * 5)。

    真实 bug 修复（2026-08-23，见 fr/face_flux_points/exact_normal.py
    模块文档）：`owner_adj_row_exact`/`neighbor_adj_row_exact` 取代了
    此前这里对 `adj_j` 做 Lagrange 外插得到"自洽方向"的做法，理由与
    compute_inviscid_interface_correction_kernel（非 colored 版本）
    完全相同，两处必须同步修改。`adj_j` 参数因此从签名中移除。

原生基（四面体 + 棱柱）：与
    `compute_inviscid_interface_correction_kernel`（非 colored 版本）
    完全相同的做法，两处必须同步修改——见该函数文档完整说明
    （`owner_cube_face`/`neighbor_cube_face` 减 6 索引原生算子表、side
    因子固定 +1、DG 提升算子 `lift_native`）。坍缩坐标那条并行路径已于
    2026-09-23 在两处同步删除（生产不可达，见该函数文档）。
    """
    n_cells = Q.shape[0]
    n_sps = Q.shape[1]
    n_fp = owner_adj_row_exact.shape[1]
    n_faces_in_color = face_indices.shape[0]

    for fi in prange(n_faces_in_color):
        f = face_indices[fi]
        oc = owner_cell[f]
        oc_code = owner_cube_face[f]

        if owner_is_primary[f]:
            E_o = boundary_extrap_native[oc_code - 6]

            Q_o = _extrap_matmul(Q[oc], E_o)
            adjrow_o = owner_adj_row_exact[f]  # (n_fp, 3)，逐 FP 精确值，见函数文档

            jump_owner = np.zeros((n_fp, 5))
            for i in range(n_fp):
                a0 = adjrow_o[i, 0]
                a1 = adjrow_o[i, 1]
                a2 = adjrow_o[i, 2]
                adj_mag = np.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
                if adj_mag < 1e-300:
                    adj_mag = 1e-300
                dirx = a0 / adj_mag
                diry = a1 / adj_mag
                dirz = a2 / adj_mag


                # 法向恒用本侧**精确度量行**的方向（2026-09-24 删除了此前的
                # `alignment < 0.5` 兜底——它在夹角过大时把方向换成
                # true_normal、但拥有侧投影仍用 adj_row，于是对均匀流
                # 跳跃量 = |adj| F.(n_ref - dir(adj)) != 0，凭空注入压力
                # 量级的源项。完整依据见本函数文档"法向一律取自本侧度量"。）

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
                    # 混合拆分面（B-8，与主 kernel 同步修改，见 inviscid_kernel.py 同名注释）
                    mp = mixed_nb_partner[f]
                    if mp >= 0 and mixed_nb_mask[f, i]:
                        Q_n = Q_ghost[mp, i]

                normal = np.empty(3)
                normal[0] = dirx
                normal[1] = diry
                normal[2] = dirz
                F_common_n = compute_ausm_up_flux(Q_o[i], Q_n, normal, mach_ref, precond_mode)

                F_tilde_common = np.empty(5)
                for v in range(5):
                    F_tilde_common[v] = F_common_n[v] * adj_mag

                F_phys_o = euler_physical_flux_point(Q_o[i])
                F_tilde_own = np.zeros(5)
                for v in range(5):
                    F_tilde_own[v] = a0 * F_phys_o[0, v] + a1 * F_phys_o[1, v] + a2 * F_phys_o[2, v]

                for v in range(5):
                    jump_owner[i, v] = F_tilde_common[v] - F_tilde_own[v]

            weighted_jump_o = np.empty((n_fp, 5))
            for i in range(n_fp):
                w_area = ref_area_weight[i]
                for v in range(5):
                    weighted_jump_o[i, v] = w_area * jump_owner[i, v]
            contrib_owner = matmul_small(lift_native[oc_code - 6], weighted_jump_o)
            for s in range(n_sps):
                dj = det_jacs[oc, s]
                for v in range(5):
                    correction[oc, s, v] += -contrib_owner[s, v] / dj

        if (not is_boundary[f]) and neighbor_is_primary[f]:
            nc = neighbor_cell[f]
            nc_code = neighbor_cube_face[f]
            E_n = boundary_extrap_native[nc_code - 6]

            Q_n_native = _extrap_matmul(Q[nc], E_n)
            adjrow_n_native = neighbor_adj_row_exact[f]  # (n_fp,3)，逐 FP 精确值，见函数文档

            jump_neighbor = np.zeros((n_fp, 5))
            for i in range(n_fp):
                a0 = adjrow_n_native[i, 0]
                a1 = adjrow_n_native[i, 1]
                a2 = adjrow_n_native[i, 2]
                adj_mag = np.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
                if adj_mag < 1e-300:
                    adj_mag = 1e-300
                dirx = a0 / adj_mag
                diry = a1 / adj_mag
                dirz = a2 / adj_mag


                # 法向恒用本侧**精确度量行**的方向（2026-09-24 删除了此前的
                # `alignment < 0.5` 兜底——它在夹角过大时把方向换成
                # true_normal、但本侧投影仍用 adj_row，于是对均匀流
                # 跳跃量 = |adj| F.(n_ref - dir(adj)) != 0，凭空注入压力
                # 量级的源项。完整依据见本函数文档"法向一律取自本侧度量"。）

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
                # 混合拆分面（B-8，与主 kernel 同步；逐元素拷贝避免 numba C/A 布局赋值冲突）
                mp_o = mixed_ow_partner[f]
                if mp_o >= 0 and mixed_ow_mask[f, i]:
                    for v in range(5):
                        Q_o_at_n[v] = Q_ghost[mp_o, i, v]

                normal = np.empty(3)
                normal[0] = dirx
                normal[1] = diry
                normal[2] = dirz
                F_common_n_native = compute_ausm_up_flux(Q_n_native[i], Q_o_at_n, normal, mach_ref, precond_mode)

                F_tilde_common_n = np.empty(5)
                for v in range(5):
                    F_tilde_common_n[v] = F_common_n_native[v] * adj_mag

                F_phys_n = euler_physical_flux_point(Q_n_native[i])
                F_tilde_own_n = np.zeros(5)
                for v in range(5):
                    F_tilde_own_n[v] = a0 * F_phys_n[0, v] + a1 * F_phys_n[1, v] + a2 * F_phys_n[2, v]

                for v in range(5):
                    jump_neighbor[i, v] = F_tilde_common_n[v] - F_tilde_own_n[v]

            weighted_jump_n = np.empty((n_fp, 5))
            for i in range(n_fp):
                w_area = ref_area_weight[i]
                for v in range(5):
                    weighted_jump_n[i, v] = w_area * jump_neighbor[i, v]
            contrib_neighbor = matmul_small(lift_native[nc_code - 6], weighted_jump_n)
            for s in range(n_sps):
                dj = det_jacs[nc, s]
                for v in range(5):
                    correction[nc, s, v] += -contrib_neighbor[s, v] / dj
