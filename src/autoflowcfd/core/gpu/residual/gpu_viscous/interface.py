"""AutoFlowCFD V2.0 - 粘性界面校正(GPU)

从 `src/autoflowcfd/core/gpu/residual/gpu_viscous.py`(原 723 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""


from autoflowcfd.core.gpu import get_cupy


from autoflowcfd.core.gpu.residual.gpu_inviscid import _lift_native_contrib

from autoflowcfd.core.fr_operators.flux_kernels import CP_AIR, R_AIR
from .extrap import _extrap_side, _self_extrap_side, _viscous_tilde_flux_pair


def _compute_viscous_interface_correction_gpu(
    Q_gpu, grad_vel_gpu, grad_T_gpu, mu_t_gpu,
    det_jacs, mu, Pr, Pr_t,
    flat_face_gpu, Q_ghost_gpu, bnd_adiabatic_gpu,
    n_cells, n_sps, n_prism, device_id, c_ip_visc,
):
    """GPU 粘性界面校正（BR1 平均 + 边界/内部面 IP 罚项），按图着色逐色处理。

    `c_ip_visc` 是按阶数解析的 IP 罚项常数（见
    `flux_kernels.resolve_viscous_ip_constant`）；由调用方算好传进来，
    与 CPU 侧三个 kernel 把它做成形参是同一个理由。

    与 core/fr_residual/viscous_flux_kernel.py::
    compute_viscous_interface_correction_kernel 逐字对应的数学移植，
    架构（分色、src0/src1 批量外插、scatter-add）仿照
    gpu_inviscid.py::_compute_interface_correction_gpu 已验证过的模式。

    与无粘 GPU 界面校正的两处刻意不同（都是有意的正确性选择，不是
    疏漏）：
    1. 保留 owner_is_primary/neighbor_is_primary 过滤——CPU 端
       viscous_flux_kernel.py 对此有明确、不可协商的约定（模块文档：
       "owner_is_primary/neighbor_is_primary 分组去重"），本函数照做；
       无粘 GPU 版本没有这层过滤，是否因此有对应的重复计数问题不在
       本次修复范围内，未去验证/修改那个文件。
    2. tilde 投影用 owner_adj_row_exact/neighbor_adj_row_exact（未归一化
       原始 adj(J) 行），不是无粘版本用的单位 true_normal——粘性 kernel
       没有 true_normal 对齐安全阀，这是自洽方向*唯一*的输入，理由见
       viscous_flux_kernel.py 模块文档。

    owner-侧和 neighbor-侧分开两个独立代码块（不能合并成无粘版本那种
    "算一次通量分配两侧"的写法）：两侧的 G_tilde_own 分别用各自侧的
    adjrow 和各自的"自身原始态"算出来，是两个不同的物理量，不是同一个
    共享数值通量的两次分配。

    真实 bug 修复（问题清单 #5 排查附带发现，2026-09-02）："自身原始态"
    （`Q_o`/`gv_o`/`gT_o`/`mut_o`，`Q_n_native`/`gv_n_native`/
    `gT_n_native`/`mut_n_native`）现在用 `_self_extrap_side`（自身面
    boundary_extrap/native 查表外插，与 CPU 版一致）计算，不再错误地
    复用 `_extrap_side`（src0/src1 跨单元交叉引用机制，专门给"对侧"
    数据用）——完整推导见 `_self_extrap_side` 文档。新增 `n_prism`
    形参就是给这个自身外插用的（`compact_cell_type` 不存在时的分类
    阈值，与 `gpu_inviscid.py` 同一约定）。
    """
    cp = get_cupy()
    correction = cp.zeros((n_cells, n_sps, 5), dtype=cp.float64)

    from autoflowcfd.core.gpu.residual.gpu_inviscid import _scatter_add_to_correction

    ff = flat_face_gpu

    for c in range(ff.n_colors):
        face_idx = ff.color_face_indices[c]
        if face_idx.shape[0] == 0:
            continue

        is_bnd = ff.is_boundary[face_idx]
        owner_primary = ff.owner_is_primary[face_idx]
        neighbor_primary = ff.neighbor_is_primary[face_idx]

        # ── owner-primary 贡献块 ──
        mask_o = owner_primary
        if bool(cp.any(mask_o)):
            idx_o = face_idx[mask_o]
            oc = ff.owner_cell[idx_o]
            is_bnd_o = ff.is_boundary[idx_o]

            # 自身原始态：与 CPU 版 `E_o=boundary_extrap_native[code-6]`
            # 对应，不能用 `_extrap_side`（那是"对侧交叉引用"机制，见
            # `_self_extrap_side` 文档"真实 bug 修复"一节）。
            oc_code_o = ff.owner_cube_face[idx_o]
            Q_o, gv_o, gT_o, mut_o = _self_extrap_side(
                cp, oc, oc_code_o, ff.boundary_extrap_native,
                Q_gpu, grad_vel_gpu, grad_T_gpu, mu_t_gpu,
            )
            Q_n, gv_n, gT_n, mut_n = _extrap_side(
                cp, idx_o, ff.neighbor_src0_cell, ff.neighbor_src0_mat,
                ff.neighbor_src1_idx, ff.neighbor_src1_cell, ff.neighbor_src1_mat,
                Q_gpu, grad_vel_gpu, grad_T_gpu, mu_t_gpu,
            )
            adjrow_o = ff.owner_adj_row_exact[idx_o]

            # 边界面：状态用幽灵态，速度梯度/涡粘镜像内部值本身（不能改成
            # 用 sources/幽灵态梯度，见 viscous_flux_kernel.py 模块文档
            # "边界面梯度处理"一节）；温度梯度按热边界类型分派（同文档
            # "边界温度梯度"一节）。
            bmask3 = is_bnd_o[:, None, None]
            Q_ghost_sub = Q_ghost_gpu[idx_o]
            Q_n = cp.where(bmask3, Q_ghost_sub, Q_n)
            gv_n = cp.where(is_bnd_o[:, None, None, None], gv_o, gv_n)
            # ∇T 法向分量镜像：gT - 2*((gT·a)/|a|^2)*a（|a|=0 的退化面原样返回，
            # 那种面的通量投影本来就是零）。用逆变行 adjrow_o 而非
            # true_normal，理由见 flux_kernels.py::mirror_normal_component。
            a2_o = cp.sum(adjrow_o * adjrow_o, axis=-1, keepdims=True)
            d_o = cp.sum(gT_o * adjrow_o, axis=-1, keepdims=True) / cp.where(a2_o > 0.0, a2_o, 1.0)
            gT_mirror_o = cp.where(a2_o > 0.0, gT_o - 2.0 * d_o * adjrow_o, gT_o)
            adiab_o = bnd_adiabatic_gpu[idx_o][:, None, None]
            gT_n = cp.where(bmask3, cp.where(adiab_o, gT_mirror_o, gT_o), gT_n)
            mut_n = cp.where(is_bnd_o[:, None], mut_o, mut_n)

            # 混合拆分面（B-8，镜像 CPU viscous_flux_kernel.py 同名分支）：边界半区用配对面幽灵态，
            # 梯度镜像内部值——与真边界面同规则，逐 FP 生效。
            mp_o = ff.mixed_nb_partner[idx_o]
            mixed_sel_o = (mp_o[:, None] >= 0) & ff.mixed_nb_mask[idx_o]  # (nO, n_fp)
            mixed3_o = mixed_sel_o[..., None]
            # Q_ghost_gpu 形状 (n_faces, n_fp, 5)，与 Q_n 形状一致，逐 FP 直接替换。
            Q_ghost_partner_o = Q_ghost_gpu[cp.maximum(mp_o, 0)]  # (nO, n_fp, 5)
            Q_n = cp.where(mixed3_o, Q_ghost_partner_o, Q_n)
            gv_n = cp.where(mixed_sel_o[:, :, None, None], gv_o, gv_n)
            adiab_mp_o = bnd_adiabatic_gpu[cp.maximum(mp_o, 0)][:, None, None]
            gT_n = cp.where(mixed3_o, cp.where(adiab_mp_o, gT_mirror_o, gT_o), gT_n)
            mut_n = cp.where(mixed_sel_o, mut_o, mut_n)
            # 逐 FP 的"边界半区"标记（真边界面全 FP 生效 + 混合面仅掩码 FP 生效），下方 IP 罚项共用。
            is_bnd_i_o = is_bnd_o[:, None] | mixed_sel_o

            Q_avg = 0.5 * (Q_o + Q_n)
            gv_avg = 0.5 * (gv_o + gv_n)
            gT_avg = 0.5 * (gT_o + gT_n)
            mut_avg = 0.5 * (mut_o + mut_n)

            G_tilde_common, G_tilde_own = _viscous_tilde_flux_pair(
                Q_avg, gv_avg, gT_avg, mut_avg, Q_o, gv_o, gT_o, mut_o,
                adjrow_o, mu, Pr, Pr_t,
            )
            jump_owner = G_tilde_common - G_tilde_own

            # IP 罚项：**边界面与内部面都要加**（2026-09-23）。逐字对应 CPU 侧
            # `viscous_flux_kernel.py` 的两条分支，只是这里向量化、用
            # `cp.where(is_bnd_i_o, 边界档, 内部档)` 逐 FP 选系数：
            #   边界档：mu 取本侧、`k_total = 0`、不含做功项；
            #   内部档：mu/k 取面平均、含动量罚项做的功。
            # 内部面为什么必须加、以及长度尺度为什么是 `cell_volume/face_area`
            # 而不是 `mean(det_jacs)**(1/3)`，见
            # `flux_kernels.viscous_ip_penalty_tilde` 的两节文档。
            a0 = adjrow_o[..., 0]
            a1 = adjrow_o[..., 1]
            a2 = adjrow_o[..., 2]
            adj_mag_o = cp.sqrt(a0 * a0 + a1 * a1 + a2 * a2)  # (nO,n_fp)
            h_ip_o = cp.maximum(ff.cell_volume[oc] / ff.face_area[idx_o], 1e-300)
            # 罚项 side 因子恒为 +1（原生面的 adj 行已 outward 定向）：
            # 见 CPU 侧 `viscous_flux_kernel.py` 同一处的完整说明
            # （2026-09-22 修复的真实缺陷，此前误乘 `owner_side`）。
            base_o = c_ip_visc * adj_mag_o / h_ip_o[:, None]  # (nO,n_fp)
            eta_v_o = base_o * cp.where(is_bnd_i_o, mu + mut_o, mu + mut_avg)
            eta_T_o = base_o * cp.where(
                is_bnd_i_o, 0.0, mu * CP_AIR / Pr + mut_avg * CP_AIR / Pr_t)
            du_o = Q_o[..., 1:4] - Q_n[..., 1:4]             # (nO,n_fp,3)
            um_o = 0.5 * (Q_o[..., 1:4] + Q_n[..., 1:4])
            work_o = cp.where(is_bnd_i_o, 0.0, cp.sum(um_o * du_o, axis=-1))
            T_o_p = Q_o[..., 4] / (Q_o[..., 0] * R_AIR)
            T_n_p = Q_n[..., 4] / (Q_n[..., 0] * R_AIR)
            pen_full = cp.zeros_like(jump_owner)
            pen_full[..., 1:4] = -eta_v_o[..., None] * du_o
            pen_full[..., 4] = -eta_v_o * work_o - eta_T_o * (T_o_p - T_n_p)
            jump_owner = jump_owner + pen_full

            # 面校正分配：DG 提升算子
            # `lift_native[code-6] @ (ref_area_weight ⊙ jump)`，与 CPU 版
            # viscous_flux_kernel.py 逐字对应 —— 粘性 kernel 不需要像无粘
            # 那样处理 true_normal 对齐安全阀（viscous_flux_kernel.py 模块
            # 文档：只用 owner/neighbor_adj_row_exact 就足够做线性收缩）。
            contrib_o = _lift_native_contrib(
                cp, oc_code_o, ff.lift_native, ff.ref_area_weight, jump_owner,
            )
            contrib_o = contrib_o / det_jacs[oc][..., None]
            _scatter_add_to_correction(correction, contrib_o, oc, n_cells, n_sps)

        # ── neighbor-primary 贡献块（仅内部面）──
        mask_n = neighbor_primary & (~is_bnd)
        if bool(cp.any(mask_n)):
            idx_n = face_idx[mask_n]
            nc = ff.neighbor_cell[idx_n]

            # 自身原始态：同上方 owner-primary 块同名注释，同一处修复
            # （`_extrap_side`+`neighbor_src0_*` 是"对侧交叉引用"机制，
            # 不是"自身外插"）。
            nc_code_n = ff.neighbor_cube_face[idx_n]
            Q_n_native, gv_n_native, gT_n_native, mut_n_native = _self_extrap_side(
                cp, nc, nc_code_n, ff.boundary_extrap_native,
                Q_gpu, grad_vel_gpu, grad_T_gpu, mu_t_gpu,
            )
            Q_o_at_n, gv_o_at_n, gT_o_at_n, mut_o_at_n = _extrap_side(
                cp, idx_n, ff.owner_src0_cell, ff.owner_src0_mat,
                ff.owner_src1_idx, ff.owner_src1_cell, ff.owner_src1_mat,
                Q_gpu, grad_vel_gpu, grad_T_gpu, mu_t_gpu,
            )
            adjrow_n = ff.neighbor_adj_row_exact[idx_n]

            # 混合拆分面（B-8）：neighbor 侧对称处理——边界半区对侧状态取配对面幽灵态，梯度镜像。
            mp_n = ff.mixed_ow_partner[idx_n]
            mixed_sel_n = (mp_n[:, None] >= 0) & ff.mixed_ow_mask[idx_n]  # (nN, n_fp)
            mixed3_n = mixed_sel_n[..., None]
            Q_ghost_partner_n = Q_ghost_gpu[cp.maximum(mp_n, 0)]  # (nN, n_fp, 5)
            Q_o_at_n = cp.where(mixed3_n, Q_ghost_partner_n, Q_o_at_n)
            gv_o_at_n = cp.where(mixed_sel_n[:, :, None, None], gv_n_native, gv_o_at_n)
            a2_n = cp.sum(adjrow_n * adjrow_n, axis=-1, keepdims=True)
            d_n = cp.sum(gT_n_native * adjrow_n, axis=-1, keepdims=True) / cp.where(a2_n > 0.0, a2_n, 1.0)
            gT_mirror_n = cp.where(a2_n > 0.0, gT_n_native - 2.0 * d_n * adjrow_n, gT_n_native)
            adiab_mp_n = bnd_adiabatic_gpu[cp.maximum(mp_n, 0)][:, None, None]
            gT_o_at_n = cp.where(mixed3_n, cp.where(adiab_mp_n, gT_mirror_n, gT_n_native), gT_o_at_n)
            mut_o_at_n = cp.where(mixed_sel_n, mut_n_native, mut_o_at_n)

            Q_avg_n = 0.5 * (Q_n_native + Q_o_at_n)
            gv_avg_n = 0.5 * (gv_n_native + gv_o_at_n)
            gT_avg_n = 0.5 * (gT_n_native + gT_o_at_n)
            mut_avg_n = 0.5 * (mut_n_native + mut_o_at_n)

            G_tilde_common_n, G_tilde_own_n = _viscous_tilde_flux_pair(
                Q_avg_n, gv_avg_n, gT_avg_n, mut_avg_n,
                Q_n_native, gv_n_native, gT_n_native, mut_n_native,
                adjrow_n, mu, Pr, Pr_t,
            )
            jump_neighbor = G_tilde_common_n - G_tilde_own_n

            # neighbor 侧 IP 罚项：与 owner 侧同一套（边界档 = 混合拆分面的
            # 边界半区 `mixed_sel_n`，其余 FP 走内部档）。
            a0n = adjrow_n[..., 0]
            a1n = adjrow_n[..., 1]
            a2n = adjrow_n[..., 2]
            adj_mag_n = cp.sqrt(a0n * a0n + a1n * a1n + a2n * a2n)
            h_ip_n = cp.maximum(ff.cell_volume[nc] / ff.face_area[idx_n], 1e-300)
            base_n = c_ip_visc * adj_mag_n / h_ip_n[:, None]
            mut_avg_n = 0.5 * (mut_n_native + mut_o_at_n)
            eta_v_n = base_n * cp.where(
                mixed_sel_n, mu + mut_n_native, mu + mut_avg_n)
            eta_T_n = base_n * cp.where(
                mixed_sel_n, 0.0, mu * CP_AIR / Pr + mut_avg_n * CP_AIR / Pr_t)
            du_n = Q_n_native[..., 1:4] - Q_o_at_n[..., 1:4]
            um_n = 0.5 * (Q_n_native[..., 1:4] + Q_o_at_n[..., 1:4])
            work_n = cp.where(mixed_sel_n, 0.0, cp.sum(um_n * du_n, axis=-1))
            T_n_p2 = Q_n_native[..., 4] / (Q_n_native[..., 0] * R_AIR)
            T_o_p2 = Q_o_at_n[..., 4] / (Q_o_at_n[..., 0] * R_AIR)
            pen_full_n = cp.zeros_like(jump_neighbor)
            pen_full_n[..., 1:4] = -eta_v_n[..., None] * du_n
            pen_full_n[..., 4] = -eta_v_n * work_n - eta_T_n * (T_n_p2 - T_o_p2)
            jump_neighbor = jump_neighbor + pen_full_n

            contrib_n = _lift_native_contrib(
                cp, nc_code_n, ff.lift_native, ff.ref_area_weight,
                jump_neighbor,
            )
            contrib_n = contrib_n / det_jacs[nc][..., None]
            _scatter_add_to_correction(correction, contrib_n, nc, n_cells, n_sps)

    return correction
