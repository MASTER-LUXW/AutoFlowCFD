"""粘性残差界面项 —— numba 逐点标量 kernel (性能优化，替代
fr_viscous_flux.py 里原来的纯 Python `for f in range(fc.n_faces)` 循环)。

与 fr_residual_inviscid_kernel.py 是同一套性能优化、同一套展平几何
缓存（core/fr_face_kernels_flat.py），逐字复刻原循环体的控制流（BR1
两侧原始变量+梯度取算术平均、owner_is_primary/neighbor_is_primary
分组去重、边界面梯度镜像内部值/状态反映真实边界条件），只是把执行
方式从"Python 解释器 + 逐次小 numpy 调用"换成 numba 编译的逐点标量
代码。数学公式与 fr_viscous_flux.py 的向量化版本必须保持完全一致，
改动一处必须同步检查另一处。

**符号约定（不能弄反）**：粘性残差是 `+div(G)`，界面校正项
`contrib = _distribute_from_face(jump, ...)`（不带负号）——与无粘残差
的 `-div(F)`/`contrib = -_distribute_from_face(jump, ...)` 符号相反，
见 fr_viscous_flux.py 模块文档"注意：粘性项是 +div(G)"。

**边界面梯度处理（不能改成用 sources 或幽灵态）**：边界面的状态 Q_n
反映真实边界条件（幽灵态提供者），但梯度 gv_n/gT_n/mut_n 恒等于内部
值本身的镜像（gv_n=gv_o 等）——这是本项目对"没有独立 BR1/IP 罚项方程
处理梯度边界值"这一已知局限的数学自洽选择，见 fr_viscous_flux.py 里
`compute_viscous_residual_fr` 对应分支的详细文档，不要"顺手"改成边界
梯度也走 ghost_provider 或 sources。

**多核并行（阶段二）**：与 `fr_residual_inviscid_kernel.py` 同一套
scatter-add 处理方式（每线程私有累加缓冲区 `correction_per_thread[tid,
...]` + 循环结束后 `sum(axis=0)` 归约）、同一套 `n_threads` 参数化约束
（调用方紧邻调用前取 `numba.get_num_threads()` 传入，不在本函数内部
查询——否则破坏磁盘缓存；`numba.set_num_threads()` 只在求解器启动时
调用一次）、同一套并行化后累加顺序变化的验证方法论，理由与实验依据
完全相同，见该文件模块文档"多核并行"一节，这里不重复。**唯一要
额外核对的是符号约定不能弄反**——粘性项 `correction_per_thread[...]
+= contrib/dj`（不带负号），与无粘残差的负号相反（见本文件模块文档
"符号约定"一节），并行化改造只应该改数组名和循环头，不应该动到任何
公式字符。
"""

import numpy as np
from numba import njit, prange, get_thread_id

from autoflowcfd.core.fr_operators.flux_kernels import viscous_physical_flux_point
from autoflowcfd.core.fr_residual.inviscid_kernel import _extrap_matmul, _distribute_point


@njit(cache=True, inline='always')
def _extrap_matrix3x3(field_cell: np.ndarray, E: np.ndarray) -> np.ndarray:
    """(n_sps,3,3) 场外插到 (n_fp,3,3)。numba 不支持任意维 reshape，
    显式按 9 个分量分别做矩阵乘法。"""
    n_fp = E.shape[0]
    out = np.zeros((n_fp, 3, 3))
    for a in range(3):
        for b in range(3):
            out[:, a, b] = E @ field_cell[:, a, b]
    return out


@njit(cache=True, parallel=True)
def compute_viscous_interface_correction_kernel(
    Q: np.ndarray, grad_vel: np.ndarray, grad_T: np.ndarray, mu_t_field: np.ndarray,
    adj_j: np.ndarray, det_jacs: np.ndarray, mu: float, Pr: float, Pr_t: float,
    owner_cell: np.ndarray, neighbor_cell: np.ndarray, is_boundary: np.ndarray,
    owner_axis: np.ndarray, owner_side: np.ndarray,
    neighbor_axis: np.ndarray, neighbor_side: np.ndarray,
    owner_is_primary: np.ndarray, neighbor_is_primary: np.ndarray,
    neighbor_src0_cell: np.ndarray, neighbor_src0_mat: np.ndarray,
    neighbor_src1_idx: np.ndarray, neighbor_src1_cell: np.ndarray, neighbor_src1_mat: np.ndarray,
    owner_src0_cell: np.ndarray, owner_src0_mat: np.ndarray,
    owner_src1_idx: np.ndarray, owner_src1_cell: np.ndarray, owner_src1_mat: np.ndarray,
    boundary_extrap: np.ndarray,
    g_left: np.ndarray, g_right: np.ndarray,
    Q_ghost: np.ndarray,
    dist_fp_of_sp: np.ndarray, dist_axis_coord_of_sp: np.ndarray,
    n_prism: int,
    n_threads: int,
) -> np.ndarray:
    """返回 correction，形状 (n_cells, n_sps, 5)，与
    fr_viscous_flux.py::compute_viscous_residual_fr 里逐面循环算出的
    correction 逐位对应。`n_threads` 必须是调用方紧邻本次调用之前取的
    `numba.get_num_threads()`，理由见模块文档"多核并行"一节。多线程下
    累加顺序不再是严格的 `range(n_faces)` 顺序，验证判据分层，同
    fr_residual_inviscid_kernel.py。
    """
    n_cells = Q.shape[0]
    n_sps = Q.shape[1]
    n_faces = owner_cell.shape[0]
    n_fp = Q_ghost.shape[1]

    correction_per_thread = np.zeros((n_threads, n_cells, n_sps, 5))

    for f in prange(n_faces):
        tid = get_thread_id()
        oc = owner_cell[f]
        oax = owner_axis[f]
        oside = owner_side[f]
        oside_idx = 0 if oside < 0 else 1
        celltype_o = 0 if oc < n_prism else 1

        if owner_is_primary[f]:
            E_o = boundary_extrap[celltype_o, oax, oside_idx]  # (n_fp,n_sps)

            Q_o = _extrap_matmul(Q[oc], E_o)  # (n_fp,5)
            gv_o = _extrap_matrix3x3(grad_vel[oc], E_o)  # (n_fp,3,3)
            gT_o = _extrap_matmul(grad_T[oc], E_o)  # (n_fp,3)
            mut_o = E_o @ mu_t_field[oc]  # (n_fp,)
            adjrow_o = _extrap_matmul(np.ascontiguousarray(adj_j[oc, :, oax, :]), E_o)  # (n_fp,3)

            jump_owner = np.zeros((n_fp, 5))
            for i in range(n_fp):
                if is_boundary[f]:
                    # 边界面：状态反映真实边界条件，梯度镜像内部值本身
                    # （见模块文档"边界面梯度处理"一节，不能改成 sources/幽灵态）。
                    Q_n = Q_ghost[f, i]
                    gv_n = gv_o[i]
                    gT_n = gT_o[i]
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

                G_common = viscous_physical_flux_point(Q_avg, gv_avg, gT_avg, mu, Pr, mut_avg, Pr_t)  # (3,5)
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

            g_prime_owner = g_left if oside < 0 else g_right
            contrib_owner = _distribute_point(
                jump_owner, dist_fp_of_sp[oax], dist_axis_coord_of_sp[oax], g_prime_owner
            )  # (n_sps,5)，注意：粘性项没有负号（见模块文档符号约定）
            for s in range(n_sps):
                dj = det_jacs[oc, s]
                for v in range(5):
                    correction_per_thread[tid, oc, s, v] += contrib_owner[s, v] / dj

        if (not is_boundary[f]) and neighbor_is_primary[f]:
            nc = neighbor_cell[f]
            nax = neighbor_axis[f]
            nside = neighbor_side[f]
            nside_idx = 0 if nside < 0 else 1
            celltype_n = 0 if nc < n_prism else 1

            E_n = boundary_extrap[celltype_n, nax, nside_idx]

            Q_n_native = _extrap_matmul(Q[nc], E_n)  # (n_fp,5)
            gv_n_native = _extrap_matrix3x3(grad_vel[nc], E_n)  # (n_fp,3,3)
            gT_n_native = _extrap_matmul(grad_T[nc], E_n)  # (n_fp,3)
            mut_n_native = E_n @ mu_t_field[nc]  # (n_fp,)
            adjrow_n_native = _extrap_matmul(np.ascontiguousarray(adj_j[nc, :, nax, :]), E_n)  # (n_fp,3)

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

            g_prime_neighbor = g_left if nside < 0 else g_right
            contrib_neighbor = _distribute_point(
                jump_neighbor, dist_fp_of_sp[nax], dist_axis_coord_of_sp[nax], g_prime_neighbor
            )
            for s in range(n_sps):
                dj = det_jacs[nc, s]
                for v in range(5):
                    correction_per_thread[tid, nc, s, v] += contrib_neighbor[s, v] / dj

    return correction_per_thread.sum(axis=0)


@njit(cache=True, parallel=True)
def compute_viscous_interface_correction_kernel_colored(
    Q: np.ndarray, grad_vel: np.ndarray, grad_T: np.ndarray, mu_t_field: np.ndarray,
    adj_j: np.ndarray, det_jacs: np.ndarray, mu: float, Pr: float, Pr_t: float,
    owner_cell: np.ndarray, neighbor_cell: np.ndarray, is_boundary: np.ndarray,
    owner_axis: np.ndarray, owner_side: np.ndarray,
    neighbor_axis: np.ndarray, neighbor_side: np.ndarray,
    owner_is_primary: np.ndarray, neighbor_is_primary: np.ndarray,
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
    """图着色版本的粘性界面 kernel。

    与 compute_viscous_interface_correction_kernel 相同逻辑，但：
    1. 只处理 face_indices 指定的面（当前颜色组）
    2. 直接写入共享 correction buffer（同色面无 owner_cell 冲突）
    3. 无需 per-thread buffer 和 sum 归约

    调用方按颜色循环调用此函数，每种颜色处理约 n_faces/n_colors 个面。
    内存从 O(n_threads * n_cells * n_sps * 5) 降至 O(n_cells * n_sps * 5)。
    """
    n_cells = Q.shape[0]
    n_sps = Q.shape[1]
    n_fp = Q_ghost.shape[1]
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
            gv_o = _extrap_matrix3x3(grad_vel[oc], E_o)
            gT_o = _extrap_matmul(grad_T[oc], E_o)
            mut_o = E_o @ mu_t_field[oc]
            adjrow_o = _extrap_matmul(np.ascontiguousarray(adj_j[oc, :, oax, :]), E_o)

            jump_owner = np.zeros((n_fp, 5))
            for i in range(n_fp):
                if is_boundary[f]:
                    Q_n = Q_ghost[f, i]
                    gv_n = gv_o[i]
                    gT_n = gT_o[i]
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
            nax = neighbor_axis[f]
            nside = neighbor_side[f]
            nside_idx = 0 if nside < 0 else 1
            celltype_n = 0 if nc < n_prism else 1

            E_n = boundary_extrap[celltype_n, nax, nside_idx]

            Q_n_native = _extrap_matmul(Q[nc], E_n)
            gv_n_native = _extrap_matrix3x3(grad_vel[nc], E_n)
            gT_n_native = _extrap_matmul(grad_T[nc], E_n)
            mut_n_native = E_n @ mu_t_field[nc]
            adjrow_n_native = _extrap_matmul(np.ascontiguousarray(adj_j[nc, :, nax, :]), E_n)

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

            g_prime_neighbor = g_left if nside < 0 else g_right
            contrib_neighbor = _distribute_point(
                jump_neighbor, dist_fp_of_sp[nax], dist_axis_coord_of_sp[nax], g_prime_neighbor
            )
            for s in range(n_sps):
                dj = det_jacs[nc, s]
                for v in range(5):
                    correction[nc, s, v] += contrib_neighbor[s, v] / dj
