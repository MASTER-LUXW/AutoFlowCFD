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
from autoflowcfd.core.fr_residual.face_point_jumps import inviscid_common_flux_point
from autoflowcfd.core.fr_residual.inviscid_kernel import _extrap_matmul


@njit(cache=True, parallel=True)
def compute_inviscid_interface_correction_kernel_colored(
    Q: np.ndarray, det_jacs: np.ndarray,
    owner_cell: np.ndarray, neighbor_cell: np.ndarray, is_boundary: np.ndarray,
    owner_is_primary: np.ndarray, neighbor_is_primary: np.ndarray,
    owner_adj_row_exact: np.ndarray, neighbor_adj_row_exact: np.ndarray,
    neighbor_src0_cell: np.ndarray, neighbor_src0_tpl: np.ndarray, neighbor_src0_tid: np.ndarray,
    neighbor_src1_idx: np.ndarray, neighbor_src1_cell: np.ndarray, neighbor_src1_mat: np.ndarray,
    owner_src0_cell: np.ndarray, owner_src0_tpl: np.ndarray, owner_src0_tid: np.ndarray,
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
    owner_face_op: np.ndarray, neighbor_face_op: np.ndarray,
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
    （`owner_face_op`/`neighbor_face_op` 索引原生面算子整表、side
    因子固定 +1、DG 提升算子 `lift_native`）。坍缩坐标那条并行路径已于
    2026-09-23 在两处同步删除（生产不可达，见该函数文档）。
    """
    n_sps = Q.shape[1]
    n_fp = owner_adj_row_exact.shape[1]
    n_faces_in_color = face_indices.shape[0]

    for fi in prange(n_faces_in_color):
        f = face_indices[fi]
        oc = owner_cell[f]
        oc_op = owner_face_op[f]

        if owner_is_primary[f]:
            E_o = boundary_extrap_native[oc_op]

            Q_o = _extrap_matmul(Q[oc], E_o)
            adjrow_o = owner_adj_row_exact[f]  # (n_fp, 3)，逐 FP 精确值，见函数文档

            flux_owner = np.empty((n_fp, 5))
            for i in range(n_fp):
                # 另一侧状态（幽灵态 / sources 插值 / 混合拆分面配对幽灵态）
                if is_boundary[f]:
                    Q_n = Q_ghost[f, i]
                else:
                    Q_n = np.zeros(5)
                    c0 = neighbor_src0_cell[f]
                    if c0 >= 0:
                        mat0 = neighbor_src0_tpl[neighbor_src0_tid[f]]
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
                flux_owner[i] = inviscid_common_flux_point(Q_o[i], Q_n, adjrow_o[i], mach_ref, precond_mode)

            weighted_flux_o = np.empty((n_fp, 5))
            for i in range(n_fp):
                w_area = ref_area_weight[i]
                for v in range(5):
                    weighted_flux_o[i, v] = w_area * flux_owner[i, v]
            contrib_owner = matmul_small(lift_native[oc_op], weighted_flux_o)
            for s in range(n_sps):
                dj = det_jacs[oc, s]
                for v in range(5):
                    correction[oc, s, v] += -contrib_owner[s, v] / dj

        if (not is_boundary[f]) and neighbor_is_primary[f]:
            nc = neighbor_cell[f]
            nc_op = neighbor_face_op[f]
            E_n = boundary_extrap_native[nc_op]

            Q_n_native = _extrap_matmul(Q[nc], E_n)
            adjrow_n_native = neighbor_adj_row_exact[f]  # (n_fp,3)，逐 FP 精确值，见函数文档

            flux_neighbor = np.empty((n_fp, 5))
            for i in range(n_fp):
                # 另一侧状态（幽灵态 / sources 插值 / 混合拆分面配对幽灵态）
                Q_o_at_n = np.zeros(5)
                c0 = owner_src0_cell[f]
                if c0 >= 0:
                    mat0 = owner_src0_tpl[owner_src0_tid[f]]
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
                flux_neighbor[i] = inviscid_common_flux_point(Q_n_native[i], Q_o_at_n, adjrow_n_native[i], mach_ref, precond_mode)

            weighted_flux_n = np.empty((n_fp, 5))
            for i in range(n_fp):
                w_area = ref_area_weight[i]
                for v in range(5):
                    weighted_flux_n[i, v] = w_area * flux_neighbor[i, v]
            contrib_neighbor = matmul_small(lift_native[nc_op], weighted_flux_n)
            for s in range(n_sps):
                dj = det_jacs[nc, s]
                for v in range(5):
                    correction[nc, s, v] += -contrib_neighbor[s, v] / dj
