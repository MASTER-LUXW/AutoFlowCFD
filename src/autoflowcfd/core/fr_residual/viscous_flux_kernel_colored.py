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

from autoflowcfd.core.fr_operators.small_dense import extrap_tensor3x3, matmul_small, matvec_small
from autoflowcfd.core.fr_operators.flux_kernels import (
    VBC_INTERIOR, boundary_other_gradients,
)
from autoflowcfd.core.fr_residual.face_point_jumps import viscous_jump_point
from autoflowcfd.core.fr_residual.inviscid_kernel import _extrap_matmul


@njit(cache=True, parallel=True)
def compute_viscous_interface_correction_kernel_colored(
    Q: np.ndarray, grad_vel: np.ndarray, grad_T: np.ndarray, mu_t_field: np.ndarray,
    det_jacs: np.ndarray, mu: float, Pr: float, Pr_t: float,
    owner_cell: np.ndarray, neighbor_cell: np.ndarray, is_boundary: np.ndarray,
    owner_is_primary: np.ndarray, neighbor_is_primary: np.ndarray,
    owner_adj_row_exact: np.ndarray, neighbor_adj_row_exact: np.ndarray,
    neighbor_src0_cell: np.ndarray, neighbor_src0_tpl: np.ndarray, neighbor_src0_tid: np.ndarray,
    neighbor_src1_idx: np.ndarray, neighbor_src1_cell: np.ndarray, neighbor_src1_mat: np.ndarray,
    owner_src0_cell: np.ndarray, owner_src0_tpl: np.ndarray, owner_src0_tid: np.ndarray,
    owner_src1_idx: np.ndarray, owner_src1_cell: np.ndarray, owner_src1_mat: np.ndarray,
    mixed_nb_partner: np.ndarray, mixed_nb_mask: np.ndarray,
    mixed_ow_partner: np.ndarray, mixed_ow_mask: np.ndarray,
    Q_ghost: np.ndarray, vbc_kind: np.ndarray,
    face_indices: np.ndarray,  # 当前颜色组的面索引
    correction: np.ndarray,    # 共享输出 buffer（同色面无冲突，直接写入）
    owner_cube_face: np.ndarray, neighbor_cube_face: np.ndarray,
    ref_area_weight: np.ndarray,
    boundary_extrap_native: np.ndarray, lift_native: np.ndarray,
    # IP 罚项的长度尺度 `h_f`（`FlatFaceGeometry.ip_length`：面法向的单元
    # 厚度，两侧单值取较薄一侧，湍流扩散读同一个）。见 `flux_kernels.viscous_ip_penalty_tilde`
    # 的"长度尺度"一节：此前用 `mean(det_jacs)**(1/3)`，两处都错 ——
    # `mean(det_jacs)` 不是体积而是体积/参考体积（参考体积基相关：
    # 坍缩 8 / 原生棱柱 4 / 原生四面体 4/3），且几何平均在各向异性
    # 贴壁单元上比壁法向厚度大 2.48 倍（实测平板算例）。
    ip_length: np.ndarray,
    # IP 罚项常数，按阶数解析（见 `flux_kernels.resolve_viscous_ip_constant`）。
    # 做成形参而不是模块全局：njit 把全局当编译期常量、且
    # `cache=True` 的磁盘缓存**不因全局值变化而失效**，那样
    # 扫参数必须隔离缓存目录重编译，极易误读成"改了没生效"。
    c_ip: float,
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
        # 罚项 side 因子恒为 +1（与非 colored 版本同一条，见那边的完整
        # 说明：原生面的 adj_row 已 outward 定向，此前误乘 `owner_side`
        # 让一半的面把耗散变成注入）。

        if owner_is_primary[f]:
            E_o = boundary_extrap_native[oc_code - 6]

            Q_o = _extrap_matmul(Q[oc], E_o)
            gv_o = extrap_tensor3x3(grad_vel[oc], E_o)
            gT_o = _extrap_matmul(grad_T[oc], E_o)
            mut_o = matvec_small(E_o, mu_t_field[oc])
            adjrow_o = owner_adj_row_exact[f]  # (n_fp,3)，逐 FP 精确值，见函数文档


            jump_owner = np.zeros((n_fp, 5))
            for i in range(n_fp):
                # 混合拆分面（B-8，与非着色版同步，见 compute_viscous_interface_correction_kernel 同名注释）。
                mp = mixed_nb_partner[f]
                bk = VBC_INTERIOR
                if is_boundary[f]:
                    # 边界温度梯度按热边界类型分派，见非着色版模块文档
                    # "边界温度梯度"一节（两处必须同步）。
                    Q_n = Q_ghost[f, i]
                    bk = vbc_kind[f]
                    gv_n, gT_n = boundary_other_gradients(gv_o[i], gT_o[i], adjrow_o[i], bk)
                    mut_n = mut_o[i]
                else:
                    Q_n = np.zeros(5)
                    gv_n = np.zeros((3, 3))
                    gT_n = np.zeros(3)
                    mut_n = 0.0
                    c0 = neighbor_src0_cell[f]
                    if c0 >= 0:
                        mat0 = neighbor_src0_tpl[neighbor_src0_tid[f]]
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
                        bk = vbc_kind[mp]
                        gv_bnd, gT_bnd = boundary_other_gradients(gv_o[i], gT_o[i], adjrow_o[i], bk)
                        for a in range(3):
                            for b in range(3):
                                gv_n[a, b] = gv_bnd[a, b]
                            gT_n[a] = gT_bnd[a]
                        mut_n = mut_o[i]

                jump_owner[i] = viscous_jump_point(
                    Q_o[i], gv_o[i], gT_o[i], mut_o[i], Q_n, gv_n, gT_n, mut_n,
                    adjrow_o[i], ip_length[f], bk, mu, Pr, Pr_t, c_ip)

            weighted_jump_o = np.empty((n_fp, 5))
            for i in range(n_fp):
                w_area = ref_area_weight[i]
                for v in range(5):
                    weighted_jump_o[i, v] = w_area * jump_owner[i, v]
            contrib_owner = matmul_small(lift_native[oc_code - 6], weighted_jump_o)
            for s in range(n_sps):
                dj = det_jacs[oc, s]
                for v in range(5):
                    correction[oc, s, v] += contrib_owner[s, v] / dj

        if (not is_boundary[f]) and neighbor_is_primary[f]:
            nc = neighbor_cell[f]
            nc_code = neighbor_cube_face[f]
            E_n = boundary_extrap_native[nc_code - 6]

            Q_n_native = _extrap_matmul(Q[nc], E_n)
            gv_n_native = extrap_tensor3x3(grad_vel[nc], E_n)
            gT_n_native = _extrap_matmul(grad_T[nc], E_n)
            mut_n_native = matvec_small(E_n, mu_t_field[nc])
            adjrow_n_native = neighbor_adj_row_exact[f]  # (n_fp,3)，逐 FP 精确值，见函数文档


            jump_neighbor = np.zeros((n_fp, 5))
            for i in range(n_fp):
                Q_o_at_n = np.zeros(5)
                gv_o_at_n = np.zeros((3, 3))
                gT_o_at_n = np.zeros(3)
                mut_o_at_n = 0.0
                c0 = owner_src0_cell[f]
                if c0 >= 0:
                    mat0 = owner_src0_tpl[owner_src0_tid[f]]
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
                bk_n = VBC_INTERIOR
                if mp_o >= 0 and mixed_ow_mask[f, i]:
                    for v in range(5):
                        Q_o_at_n[v] = Q_ghost[mp_o, i, v]
                    bk_n = vbc_kind[mp_o]
                    gv_bnd_n, gT_bnd_n = boundary_other_gradients(
                        gv_n_native[i], gT_n_native[i], adjrow_n_native[i], bk_n)
                    for a in range(3):
                        for b in range(3):
                            gv_o_at_n[a, b] = gv_bnd_n[a, b]
                        gT_o_at_n[a] = gT_bnd_n[a]
                    mut_o_at_n = mut_n_native[i]

                jump_neighbor[i] = viscous_jump_point(
                    Q_n_native[i], gv_n_native[i], gT_n_native[i], mut_n_native[i],
                    Q_o_at_n, gv_o_at_n, gT_o_at_n, mut_o_at_n, adjrow_n_native[i], ip_length[f],
                    bk_n, mu, Pr, Pr_t, c_ip)

            weighted_jump_n = np.empty((n_fp, 5))
            for i in range(n_fp):
                w_area = ref_area_weight[i]
                for v in range(5):
                    weighted_jump_n[i, v] = w_area * jump_neighbor[i, v]
            contrib_neighbor = matmul_small(lift_native[nc_code - 6], weighted_jump_n)
            for s in range(n_sps):
                dj = det_jacs[nc, s]
                for v in range(5):
                    correction[nc, s, v] += contrib_neighbor[s, v] / dj
