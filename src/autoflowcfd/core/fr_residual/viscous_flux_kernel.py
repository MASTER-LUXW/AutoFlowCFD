"""粘性残差界面项 —— numba 逐点标量 kernel（`core/fr_residual/viscous_flux/residual.py` 调用）。

与无粘界面核（`inviscid_kernel.py`）共用展平几何缓存（`core/fr_operators/face_kernels.py`）。控制流：BR1 两侧
原始变量 + 梯度取算术平均、owner_is_primary/neighbor_is_primary 分组去重、边界面按粘性边界种类构造另一侧。

**符号约定（不能弄反）**：粘性残差是 `+div(G)`，界面提升贡献不带负号——与无粘残差的 `-div(F)` 相反。

**边界面的公共粘性通量按粘性边界种类分派**：状态 Q_n 取
幽灵态，"另一侧"梯度由本侧按种类构造（`flux_kernels/viscous_bc.py::
boundary_other_gradients`），种类数组 `vbc_kind` 由
`boundary/fr_ghost_state.py::build_viscous_boundary_kind` 按边界组 BC 类型给出。
出口/远场/入口的温度、出口与对称面切向的速度都要在扩散算子里有边界条件，否则离散扩散算子失去强制性，完整
论证与实测见 `viscous_bc.py` 模块文档。三份 kernel（本文件、着色版、P0 版）与解析 Jacobian 都走
`face_point_jumps.viscous_common_flux_point` 这一份逐点函数。

**多核并行**：与无粘核同一套按面着色（`inviscid_kernel.py` 模块文档"多核并行"）。**符号约定不能弄反**——
粘性项 `correction[...] += contrib/dj`（不带负号），与无粘残差的负号相反。
"""

import numpy as np
from numba import njit, prange

from autoflowcfd.core.fr_operators.small_dense import extrap_tensor3x3, matvec_small
from autoflowcfd.core.fr_operators.flux_kernels import (
    VBC_INTERIOR, boundary_other_gradients,
)
from autoflowcfd.core.fr_residual.face_point_jumps import lift_into_cell, viscous_common_flux_point
from autoflowcfd.core.fr_residual.inviscid_kernel import _extrap_matmul


@njit(cache=True, parallel=True)
def compute_viscous_interface_correction_kernel(
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
    owner_face_op: np.ndarray, neighbor_face_op: np.ndarray,
    ref_area_weight: np.ndarray,
    boundary_extrap_native: np.ndarray, lift_native: np.ndarray,
    # IP 罚项的长度尺度 `h_f`（`FlatFaceGeometry.ip_length`：面法向的单元
    # 厚度，两侧单值取较薄一侧，湍流扩散读同一个）。见 `flux_kernels.viscous_ip_penalty_tilde`
    # 的"长度尺度"一节。
    ip_length: np.ndarray,
    # IP 罚项常数，按阶数解析（见 `flux_kernels.resolve_viscous_ip_constant`）。
    # 做成形参而不是模块全局：njit 把全局当编译期常量、且
    # `cache=True` 的磁盘缓存**不因全局值变化而失效**，那样
    # 扫参数必须隔离缓存目录重编译，极易误读成"改了没生效"。
    c_ip: float,
) -> None:
    """把颜色组 `face_indices` 里各面的粘性界面修正项累加进 `correction`（(n_cells, n_sps, 5)，就地）。

    **按面着色并行**：调用方按颜色循环，每次只处理一个颜色组 `face_indices`（同色面的 owner/neighbor 单元
    互不重复，构造见 `core/utils/face_coloring.py`），所以可以对组内的面 `prange`、直接累加进共享的
    `correction`，没有写冲突、不需要每线程缓冲与归约（内存 O(n_cells*n_sps*5)，与线程数无关）。2026-10-18 以前
    另有一份每线程缓冲版（环境变量切换），两份逐字同步维护，随开关删除。

    真实 bug 修复（2026-08-23，见 fr/face_flux_points/exact_normal.py
    模块文档）：`owner_adj_row_exact`/`neighbor_adj_row_exact` 取代了
    此前这里对 `adj_j` 做 Lagrange 外插得到自洽方向的做法，理由与
    inviscid_kernel.py 完全相同——本函数没有 inviscid_kernel.py 那样的
    `true_normal` 对齐安全阀，自洽方向是粘性通量法向*唯一*的输入，
    外插截断误差此前没有任何兜底，本次修复对粘性残差的精度改善因此
    更直接。`adj_j` 参数已从签名中移除。

    原生基（四面体 + 棱柱）：`owner_face_op`/`neighbor_face_op`
    索引原生面算子整表（`fr/operators/face_ops.py`）。
    与无粘 kernel 不同，这里**不需要** side_factor/`true_normal` 对齐
    （见上方"真实 bug 修复"一节：`owner_adj_row_exact` 是唯一来源，
    owner/neighbor 两侧各自读同一份精确 adj row，本就衔接，不需要额外
    定向）。只涉及两件事：(a) 自身面外插矩阵用
    `boundary_extrap_native[op]`；(b) 面修正项用 DG 提升算子
    `lift_native[op] @ (ref_area_weight⊙jump)`，理由同
    inviscid_kernel.py / native_tet/basis.py::build_native_tet_lift 文档。

    """
    n_sps = Q.shape[1]
    n_fp = Q_ghost.shape[1]
    n_faces_in_color = face_indices.shape[0]

    for fi in prange(n_faces_in_color):
        f = face_indices[fi]
        oc = owner_cell[f]
        oc_op = owner_face_op[f]

        if owner_is_primary[f]:
            E_o = boundary_extrap_native[oc_op]

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

                jump_owner[i] = viscous_common_flux_point(
                    Q_o[i], gv_o[i], gT_o[i], mut_o[i], Q_n, gv_n, gT_n, mut_n,
                    adjrow_o[i], ip_length[f], bk, mu, Pr, Pr_t, c_ip)

            lift_into_cell(correction, oc, lift_native[oc_op], ref_area_weight, jump_owner, det_jacs, 1.0)

        if (not is_boundary[f]) and neighbor_is_primary[f]:
            nc = neighbor_cell[f]
            nc_op = neighbor_face_op[f]
            E_n = boundary_extrap_native[nc_op]

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

                jump_neighbor[i] = viscous_common_flux_point(
                    Q_n_native[i], gv_n_native[i], gT_n_native[i], mut_n_native[i],
                    Q_o_at_n, gv_o_at_n, gT_o_at_n, mut_o_at_n, adjrow_n_native[i], ip_length[f],
                    bk_n, mu, Pr, Pr_t, c_ip)

            lift_into_cell(correction, nc, lift_native[nc_op], ref_area_weight, jump_neighbor, det_jacs, 1.0)
