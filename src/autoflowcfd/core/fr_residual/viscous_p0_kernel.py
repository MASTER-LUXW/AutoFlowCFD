"""
AutoFlowCFD V2.0 - P0 粘性界面校正 numba kernel（n_sps = 1，Order Continuation 的 P0 阶段）

与通用核（viscous_flux_kernel.py）共用逐点公共通量、边界"另一侧"梯度与 DG 提升（`face_point_jumps.py`），
外插写成标量乘（n_sps = 1 时 `E (n_fp,1) @ field (1,k)` 就是 `E[i,0] * field[0]`，同一个乘法）。不并入通用核
（2026-10-22 I-11 核实）：P0 每面只有一个通量点，两处语义与 P>=1 不同——
  * 本侧法向通量在这里直接减（P>=1 并进体积算子 K，`fr/face_flux_trace.py`）；
  * 混合拆分面与真多源面（棱柱四边形侧面两个半区各挨不同单元）按面积占比 `p0_second_half_frac` 加权两个
    半区的公共通量（P>=1 按逐通量点掩码与源插值矩阵取各点所在半区）。
合并后这两处都要按阶数分支，并不能少写什么。

非 primary 记录（棱柱四边形侧面三角化拆出的重复记录）由 `owner_is_primary` / `neighbor_is_primary` 门控，贡献为零。
"""

import numpy as np
from numba import njit, prange

from autoflowcfd.core.fr_operators.flux_kernels import (
    VBC_INTERIOR, boundary_other_gradients,
)
from autoflowcfd.core.fr_residual.face_point_jumps import (
    lift_into_cell, viscous_common_flux_point, viscous_self_normal_flux_point,
)


@njit(cache=True, parallel=True)
def compute_viscous_interface_correction_p0_kernel(
    Q: np.ndarray,             # (n_cells, 1, 5)
    grad_vel: np.ndarray,      # (n_cells, 1, 3, 3)
    grad_T: np.ndarray,        # (n_cells, 1, 3)
    mu_t_field: np.ndarray,    # (n_cells, 1)
    det_jacs: np.ndarray,      # (n_cells, 1)
    mu: float, Pr: float, Pr_t: float,
    owner_cell: np.ndarray, neighbor_cell: np.ndarray, is_boundary: np.ndarray,
    owner_is_primary: np.ndarray, neighbor_is_primary: np.ndarray,
    # 逐 FP 精确度量伴随行（与 P>=1 kernel 同一个来源，见函数文档
    # "2026-09-23 修复的真实缺陷"）。
    owner_adj_row_exact: np.ndarray, neighbor_adj_row_exact: np.ndarray,
    # 对侧来源单元：P0 每单元一个常数态，直接取单元值，不需要源插值矩阵
    neighbor_src0_cell: np.ndarray, neighbor_src1_idx: np.ndarray, neighbor_src1_cell: np.ndarray,
    owner_src0_cell: np.ndarray, owner_src1_idx: np.ndarray, owner_src1_cell: np.ndarray,
    mixed_nb_partner: np.ndarray, mixed_ow_partner: np.ndarray,
    # 四边形面第二个半区（混合面的边界半区、真多源面的次源半区）的面积占比 b
    # （`FlatFaceGeometry.p0_second_half_frac`）。P0 每面一个通量点，提升算子作用在棱柱的整张四边形面上：棱柱
    # 一侧的跳量取 (1-b) J_第一半区 + b J_第二半区。2026-10-17 以前按这一个通量点落在哪个半区整张面二选一，而
    # 另一侧单元照样施加自己那一半的贡献，不守恒（混合面 I-7，真多源面 I-182）
    p0_second_half_frac: np.ndarray,
    Q_ghost: np.ndarray,          # (n_boundary_faces, n_fp, 5)
    vbc_kind: np.ndarray,         # (n_faces,) 粘性边界种类，见 flux_kernels/viscous_bc.py
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
    face_indices: np.ndarray,     # int64，本颜色组的面
    correction: np.ndarray,       # (n_cells, 1, 5)，就地累加
) -> None:
    """P0 专用粘性界面校正 kernel（颜色组 `face_indices` 的贡献累加进 `correction`）。

    按面着色：调用方逐颜色调用，每次只处理一个颜色组 `face_indices`（同色面的 owner/neighbor 单元互不重复，
    `fr_operators/face_kernels.py` 的面着色），组内 `prange` 直接累加进共享输出，没有写冲突、不需要每线程
    缓冲与归约。与 P>=1 界面核同一套颜色分组。

    与通用 compute_viscous_interface_correction_kernel 数学等价，
    但 n_sps=1 时：
    - 外插写成标量乘：E[i,0]*field[0,...]（n_sps=1 恒等式，非近似）
    - 分布特化为单 SP：out[0,v] = g * fp_data[fp_i, v]
    - correction 形状 (n_cells, 1, 5)

    原生基（四面体 + 棱柱）：与通用 kernel（viscous_flux_kernel.py）
    同一套做法 —— `owner_face_op`/`neighbor_face_op` 索引原生
    算子表（四面体 [6,10)、棱柱 [10,15) 在同一张表里），自身面外插用
    `boundary_extrap_native[op]`，面修正项用
    `lift_native[op] @ (ref_area_weight⊙jump)`。order=0 时原生真实
    自由度数恰好也是 1（受限 PKD 模态数 `(0+1)(0+2)(0+3)/6 = 1`），
    `lift_native[op]` 形状 `(1, n_fp)`，矩阵乘本身就是这里 P0 特化
    要的标量化形式。

    度量伴随行逐 FP 读精确值 `owner_adj_row_exact`/`neighbor_adj_row_exact`（与 P>=1 核同一个来源），不从解点
    度量里取某个轴行。
    """
    n_fp = Q_ghost.shape[1]

    for fi in prange(face_indices.shape[0]):
        f = face_indices[fi]
        oc = owner_cell[f]
        oc_op = owner_face_op[f]
        # 原生面的 `owner_adj_row_exact` 已按 outward 定向（`native_prism/face.py::native_prism_face_adj_rows`、
        # `exact_normal.py`），罚项不再乘任何定向因子。

        if owner_is_primary[f]:
            E_o = boundary_extrap_native[oc_op]  # (n_fp, 1)

            # P0 外插：n_sps=1 时 E (n_fp,1) @ field (1,k) 与
            # E[i,0]*field[0,...] 是**同一个矩阵乘**，不是近似。
            # 而且 P0 的唯一基函数是常数 1，正确的插值算子在这里恒有
            # E[i,0]=1（order=0 下全部面算子整表 `face_extrap_by_op` 的有效行
            # 全为 1.0），所以外插结果就等于
            # 单元自身的值——这正是 P0 常数重构应有的行为。
            Q_o_s0 = Q[oc, 0]  # (5,)
            gv_o_s0 = grad_vel[oc, 0]  # (3,3)
            gT_o_s0 = grad_T[oc, 0]  # (3,)
            mut_o_s0 = mu_t_field[oc, 0]  # scalar
            adjrow_o = owner_adj_row_exact[f]  # (n_fp,3)，逐 FP 精确值

            jump_owner = np.zeros((n_fp, 5))
            for i in range(n_fp):
                # 混合拆分面（B-8，与通用 kernel 同步，见 viscous_flux_kernel.py 同名注释）。
                mp = mixed_nb_partner[f]
                sc = -1                  # 真多源面的次源单元
                e_i = E_o[i, 0]  # 标量
                adj_o_i = adjrow_o[i]  # (3,)，本 FP 的精确度量伴随行

                # 外插到 FP i：n_sps=1 下标量乘 == 矩阵乘（见上方说明，
                # 不是简化）
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

                # 邻居状态（源矩阵插值；n_sps=1 时同样是标量乘 == 矩阵乘
                # 的恒等式，不是简化）。边界面按粘性边界种类构造"另一侧"梯度，
                # 规则与通用 kernel 同一份（`flux_kernels/viscous_bc.py`）。
                bk = VBC_INTERIOR
                if is_boundary[f]:
                    Q_n = Q_ghost[f, i]
                    bk = vbc_kind[f]
                    gv_n, gT_n = boundary_other_gradients(gv_o_i, gT_o_i, adj_o_i, bk)
                    mut_n = mut_o_i
                elif mp >= 0:
                    # 混合拆分面：内部半区的对侧就是唯一的内部邻居单元，P0 取其常数值。不能用源矩阵插值：唯一的
                    # 通量点落在边界半区时那里没有内部来源（权重为 0，对侧成了全零态，2026-10-17 I-181 查出）
                    nc0 = neighbor_cell[f]
                    Q_n = Q[nc0, 0].copy()
                    gv_n = grad_vel[nc0, 0].copy()
                    gT_n = grad_T[nc0, 0].copy()
                    mut_n = mu_t_field[nc0, 0]
                else:
                    # 内部面：P0 每单元一个常数态，对侧直接取来源单元的值（单源面上源矩阵在唯一通量点的权重恒为
                    # 1，二者相同）。真多源面（棱柱四边形侧面的两个半区各挨一个四面体）另取次源单元，下方按面积
                    # 占比加权两半；此前只按源矩阵在唯一通量点上的权重取了一个单元，另一半被忽略（I-182）
                    c0 = neighbor_src0_cell[f]
                    Q_n = Q[c0, 0].copy()
                    gv_n = grad_vel[c0, 0].copy()
                    gT_n = grad_T[c0, 0].copy()
                    mut_n = mu_t_field[c0, 0]
                    idx1 = neighbor_src1_idx[f]
                    if idx1 >= 0:
                        sc = neighbor_src1_cell[idx1]
                # P0 没有体积算子可以并入本侧通量（P>=1 见 fr/face_flux_trace.py），这里直接减
                self_flux = viscous_self_normal_flux_point(Q_o_i, gv_o_i, gT_o_i, mut_o_i, adj_o_i, mu, Pr, Pr_t)
                common = viscous_common_flux_point(
                    Q_o_i, gv_o_i, gT_o_i, mut_o_i, Q_n, gv_n, gT_n, mut_n,
                    adj_o_i, ip_length[f], bk, mu, Pr, Pr_t, c_ip)
                # 混合拆分面（棱柱是 owner）：边界半区的公共通量取配对面幽灵态与该边界种类的梯度，按面积占比加权
                if (not is_boundary[f]) and mp >= 0:
                    bk_b = vbc_kind[mp]
                    gv_bnd, gT_bnd = boundary_other_gradients(gv_o_i, gT_o_i, adj_o_i, bk_b)
                    common_b = viscous_common_flux_point(
                        Q_o_i, gv_o_i, gT_o_i, mut_o_i, Q_ghost[mp, i], gv_bnd, gT_bnd, mut_o_i,
                        adj_o_i, ip_length[f], bk_b, mu, Pr, Pr_t, c_ip)
                    bfrac = p0_second_half_frac[f]
                    for v in range(5):
                        common[v] = (1.0 - bfrac) * common[v] + bfrac * common_b[v]
                elif sc >= 0:
                    common_2 = viscous_common_flux_point(
                        Q_o_i, gv_o_i, gT_o_i, mut_o_i, Q[sc, 0], grad_vel[sc, 0], grad_T[sc, 0], mu_t_field[sc, 0],
                        adj_o_i, ip_length[f], VBC_INTERIOR, mu, Pr, Pr_t, c_ip)
                    bfrac = p0_second_half_frac[f]
                    for v in range(5):
                        common[v] = (1.0 - bfrac) * common[v] + bfrac * common_2[v]
                jump_owner[i] = common - self_flux

            lift_into_cell(correction, oc, lift_native[oc_op], ref_area_weight, jump_owner, det_jacs, 1.0)

        # Neighbor 侧（与通用 kernel 相同逻辑，n_sps=1 特化）
        if (not is_boundary[f]) and neighbor_is_primary[f]:
            nc = neighbor_cell[f]
            nc_op = neighbor_face_op[f]
            E_n = boundary_extrap_native[nc_op]

            Q_n_s0 = Q[nc, 0]
            gv_n_s0 = grad_vel[nc, 0]
            gT_n_s0 = grad_T[nc, 0]
            mut_n_s0 = mu_t_field[nc, 0]
            adjrow_n = neighbor_adj_row_exact[f]  # (n_fp,3)，逐 FP 精确值

            jump_neighbor = np.zeros((n_fp, 5))
            for i in range(n_fp):
                e_i = E_n[i, 0]
                adj_n_i = adjrow_n[i]  # (3,)，本 FP 的精确度量伴随行

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

                # Owner 侧（n_sps=1）：对侧直接取来源单元的常数值；混合拆分面（棱柱是 neighbor）内部半区的对侧是唯一
                # 的 owner 单元；真多源面另取次源单元、下方按面积占比加权（理由同上方 owner 侧）
                mp_o = mixed_ow_partner[f]
                c0 = oc if mp_o >= 0 else owner_src0_cell[f]
                Q_o_at_n = Q[c0, 0].copy()
                gv_o_at_n = grad_vel[c0, 0].copy()
                gT_o_at_n = grad_T[c0, 0].copy()
                mut_o_at_n = mu_t_field[c0, 0]
                sc_n = -1
                if mp_o < 0 and owner_src1_idx[f] >= 0:
                    sc_n = owner_src1_cell[owner_src1_idx[f]]
                self_flux_n = viscous_self_normal_flux_point(
                    Q_n_i, gv_n_i, gT_n_i, mut_n_i, adj_n_i, mu, Pr, Pr_t)
                common_n = viscous_common_flux_point(
                    Q_n_i, gv_n_i, gT_n_i, mut_n_i, Q_o_at_n, gv_o_at_n, gT_o_at_n, mut_o_at_n,
                    adj_n_i, ip_length[f], VBC_INTERIOR, mu, Pr, Pr_t, c_ip)
                # 混合拆分面（棱柱是 neighbor）：对称处理，按面积占比加权
                if mp_o >= 0:
                    bk_n = vbc_kind[mp_o]
                    gv_bnd_n, gT_bnd_n = boundary_other_gradients(gv_n_i, gT_n_i, adj_n_i, bk_n)
                    common_nb = viscous_common_flux_point(
                        Q_n_i, gv_n_i, gT_n_i, mut_n_i, Q_ghost[mp_o, i], gv_bnd_n, gT_bnd_n, mut_n_i,
                        adj_n_i, ip_length[f], bk_n, mu, Pr, Pr_t, c_ip)
                    bfrac = p0_second_half_frac[f]
                    for v in range(5):
                        common_n[v] = (1.0 - bfrac) * common_n[v] + bfrac * common_nb[v]
                elif sc_n >= 0:
                    common_n2 = viscous_common_flux_point(
                        Q_n_i, gv_n_i, gT_n_i, mut_n_i, Q[sc_n, 0], grad_vel[sc_n, 0], grad_T[sc_n, 0],
                        mu_t_field[sc_n, 0], adj_n_i, ip_length[f], VBC_INTERIOR, mu, Pr, Pr_t, c_ip)
                    bfrac = p0_second_half_frac[f]
                    for v in range(5):
                        common_n[v] = (1.0 - bfrac) * common_n[v] + bfrac * common_n2[v]
                jump_neighbor[i] = common_n - self_flux_n

            lift_into_cell(correction, nc, lift_native[nc_op], ref_area_weight, jump_neighbor, det_jacs, 1.0)
