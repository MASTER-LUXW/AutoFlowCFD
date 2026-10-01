"""AutoFlowCFD V2.0 - 粘性界面校正(GPU)

从 `src/autoflowcfd/core/gpu/residual/gpu_viscous.py`(原 723 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""


from autoflowcfd.core.gpu import get_cupy


from autoflowcfd.core.gpu.residual.gpu_inviscid import _lift_native_contrib

from autoflowcfd.core.fr_operators.flux_kernels import (
    CP_AIR, R_AIR, VBC_DIRICHLET, VBC_INLET, VBC_INTERIOR, VBC_MIRROR, VBC_NEUMANN, VBC_NOSLIP_WALL,
)
from .extrap import _extrap_side, _self_extrap_side, _viscous_tilde_flux


def _mirror_normal_gpu(cp, g, adjrow):
    """`flux_kernels.mirror_normal_component` 的向量化版：`g - 2 (g·a)/|a|^2 a`，
    `|a| = 0` 的退化面原样返回。"""
    a2 = cp.sum(adjrow * adjrow, axis=-1, keepdims=True)
    d = cp.sum(g * adjrow, axis=-1, keepdims=True) / cp.where(a2 > 0.0, a2, 1.0)
    return cp.where(a2 > 0.0, g - 2.0 * d * adjrow, g)


def _mirror_velocity_gradient_gpu(cp, gv, adjrow):
    """`flux_kernels.mirror_velocity_gradient` 的向量化版：`R gv R`，`R = I - 2nn^T`。"""
    a2 = cp.sum(adjrow * adjrow, axis=-1, keepdims=True)
    n = adjrow / cp.sqrt(cp.where(a2 > 0.0, a2, 1.0))
    nTg = cp.einsum("...a,...ab->...b", n, gv)
    gn = cp.einsum("...ab,...b->...a", gv, n)
    ngn = cp.sum(n * gn, axis=-1)
    out = (gv - 2.0 * n[..., :, None] * nTg[..., None, :] - 2.0 * gn[..., :, None] * n[..., None, :]
           + 4.0 * ngn[..., None, None] * n[..., :, None] * n[..., None, :])
    return cp.where(a2[..., None] > 0.0, out, gv)


def _boundary_other_side_gpu(cp, K, Q_s, gv_s, gT_s, mut_s, Q_x, gv_x, gT_x, mut_x, adjrow):
    """边界点（`K != VBC_INTERIOR`）的"另一侧"梯度与涡粘，并把入口落到 Dirichlet/Neumann。

    逐点对应 CPU `flux_kernels/viscous_bc.py::boundary_other_gradients` 与
    `resolve_point_kind`：无滑移壁速度梯度取本侧、∇T 法向镜像；镜像类两者都取
    镜像场梯度；Neumann 点（出口、入口回流点）速度梯度取 `gv·R`、∇T 法向镜像
    （充分发展条件，与 CPU `viscous_jump_point` 同一处理）；Dirichlet 取本侧。
    `Q_x` 在边界点上已经是幽灵态。返回解析后的 `K`。
    """
    inflow = cp.sum(Q_s[..., 1:4] * adjrow, axis=-1) < 0.0
    K = cp.where(K == VBC_INLET, cp.where(inflow, VBC_DIRICHLET, VBC_NEUMANN), K)
    bnd = K != VBC_INTERIOR
    mirror = K == VBC_MIRROR
    neumann = K == VBC_NEUMANN
    gv_b = cp.where(mirror[..., None, None], _mirror_velocity_gradient_gpu(cp, gv_s, adjrow),
                    cp.where(neumann[..., None, None],
                             _mirror_normal_gpu(cp, gv_s, adjrow[..., None, :]), gv_s))
    gT_b = cp.where((mirror | neumann | (K == VBC_NOSLIP_WALL))[..., None],
                    _mirror_normal_gpu(cp, gT_s, adjrow), gT_s)
    gv_x = cp.where(bnd[..., None, None], gv_b, gv_x)
    gT_x = cp.where(bnd[..., None], gT_b, gT_x)
    mut_x = cp.where(bnd, mut_s, mut_x)
    return K, gv_x, gT_x, mut_x


def _viscous_jump_gpu(cp, K, Q_s, gv_s, gT_s, mut_s, Q_x, gv_x, gT_x, mut_x, adjrow, h_ip,
                      mu, Pr, Pr_t, c_ip_visc, subtract_self: bool):
    """逐点对应 CPU `face_point_jumps.viscous_common_flux_point`（`K` 已解析）。

    `a·G(平均态)` 加罚项：内部点涡粘与热传导率取面平均，边界点取本侧，热传导率只在
    Dirichlet 点给（无滑移壁/镜像类/Neumann 为 0；Neumann 点幽灵态速度即本侧速度，
    速度罚项为零）。罚项做功项内部与边界同一形式。

    `subtract_self`：P0 再减本侧 `a·G(本侧)`（P0 没有体积算子可并入，对应 CPU 的
    `viscous_self_normal_flux_point`）；P>=1 的本侧通量迹在体积算子 K 里
    （`fr/face_flux_trace.py`）。
    """
    Q_avg = 0.5 * (Q_s + Q_x)
    gv_avg = 0.5 * (gv_s + gv_x)
    gT_avg = 0.5 * (gT_s + gT_x)
    mut_avg = 0.5 * (mut_s + mut_x)
    jump = _viscous_tilde_flux(Q_avg, gv_avg, gT_avg, mut_avg, adjrow, mu, Pr, Pr_t)
    if subtract_self:
        jump = jump - _viscous_tilde_flux(Q_s, gv_s, gT_s, mut_s, adjrow, mu, Pr, Pr_t)
    interior = K == VBC_INTERIOR

    adj_mag = cp.sqrt(cp.sum(adjrow * adjrow, axis=-1))
    # 罚项 side 因子恒为 +1（原生面的 adj 行已 outward 定向，见 CPU 侧
    # `viscous_flux_kernel.py` 同一处说明，2026-09-22 修复的真实缺陷）。
    base = c_ip_visc * adj_mag / h_ip[:, None]
    eta_v = base * cp.where(interior, mu + mut_avg, mu + mut_s)
    k_avg = mu * CP_AIR / Pr + mut_avg * CP_AIR / Pr_t
    k_self = mu * CP_AIR / Pr + mut_s * CP_AIR / Pr_t
    eta_T = base * cp.where(interior, k_avg, cp.where(K == VBC_DIRICHLET, k_self, 0.0))
    du = Q_s[..., 1:4] - Q_x[..., 1:4]
    um = 0.5 * (Q_s[..., 1:4] + Q_x[..., 1:4])
    work = cp.sum(um * du, axis=-1)
    T_s = Q_s[..., 4] / (Q_s[..., 0] * R_AIR)
    T_x = Q_x[..., 4] / (Q_x[..., 0] * R_AIR)
    pen = cp.zeros_like(jump)
    pen[..., 1:4] = -eta_v[..., None] * du
    pen[..., 4] = -eta_v * work - eta_T * (T_s - T_x)
    return jump + pen


def _compute_viscous_interface_correction_gpu(
    Q_gpu, grad_vel_gpu, grad_T_gpu, mu_t_gpu,
    det_jacs, mu, Pr, Pr_t,
    flat_face_gpu, Q_ghost_gpu, vbc_kind_gpu,
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
                cp, idx_o, ff.neighbor_src0_cell, ff.neighbor_src0_tpl, ff.neighbor_src0_tid,
                ff.neighbor_src1_idx, ff.neighbor_src1_cell, ff.neighbor_src1_mat,
                Q_gpu, grad_vel_gpu, grad_T_gpu, mu_t_gpu,
            )
            adjrow_o = ff.owner_adj_row_exact[idx_o]

            # 边界面与混合拆分面边界半区（B-8）：状态取幽灵态（混合面取配对面的），
            # "另一侧"梯度与罚项按粘性边界种类（CPU `flux_kernels/viscous_bc.py`）。
            mp_o = ff.mixed_nb_partner[idx_o]
            mixed_sel_o = (mp_o[:, None] >= 0) & ff.mixed_nb_mask[idx_o]  # (nO, n_fp)
            Q_n = cp.where(is_bnd_o[:, None, None], Q_ghost_gpu[idx_o], Q_n)
            Q_n = cp.where(mixed_sel_o[..., None], Q_ghost_gpu[cp.maximum(mp_o, 0)], Q_n)
            K_o = cp.where(is_bnd_o[:, None], vbc_kind_gpu[idx_o][:, None],
                           cp.where(mixed_sel_o, vbc_kind_gpu[cp.maximum(mp_o, 0)][:, None], VBC_INTERIOR))
            K_o, gv_n, gT_n, mut_n = _boundary_other_side_gpu(
                cp, K_o, Q_o, gv_o, gT_o, mut_o, Q_n, gv_n, gT_n, mut_n, adjrow_o)
            jump_owner = _viscous_jump_gpu(
                cp, K_o, Q_o, gv_o, gT_o, mut_o, Q_n, gv_n, gT_n, mut_n, adjrow_o,
                ff.ip_length[idx_o], mu, Pr, Pr_t, c_ip_visc, n_sps == 1)

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
                cp, idx_n, ff.owner_src0_cell, ff.owner_src0_tpl, ff.owner_src0_tid,
                ff.owner_src1_idx, ff.owner_src1_cell, ff.owner_src1_mat,
                Q_gpu, grad_vel_gpu, grad_T_gpu, mu_t_gpu,
            )
            adjrow_n = ff.neighbor_adj_row_exact[idx_n]

            # 混合拆分面（B-8）：neighbor 侧对称处理——边界半区对侧状态取配对面幽灵态，
            # 梯度与罚项按该配对面的粘性边界种类。
            mp_n = ff.mixed_ow_partner[idx_n]
            mixed_sel_n = (mp_n[:, None] >= 0) & ff.mixed_ow_mask[idx_n]  # (nN, n_fp)
            Q_o_at_n = cp.where(mixed_sel_n[..., None], Q_ghost_gpu[cp.maximum(mp_n, 0)], Q_o_at_n)
            K_n = cp.where(mixed_sel_n, vbc_kind_gpu[cp.maximum(mp_n, 0)][:, None], VBC_INTERIOR)
            K_n, gv_o_at_n, gT_o_at_n, mut_o_at_n = _boundary_other_side_gpu(
                cp, K_n, Q_n_native, gv_n_native, gT_n_native, mut_n_native,
                Q_o_at_n, gv_o_at_n, gT_o_at_n, mut_o_at_n, adjrow_n)
            jump_neighbor = _viscous_jump_gpu(
                cp, K_n, Q_n_native, gv_n_native, gT_n_native, mut_n_native,
                Q_o_at_n, gv_o_at_n, gT_o_at_n, mut_o_at_n, adjrow_n,
                ff.ip_length[idx_n], mu, Pr, Pr_t, c_ip_visc, n_sps == 1)

            contrib_n = _lift_native_contrib(
                cp, nc_code_n, ff.lift_native, ff.ref_area_weight,
                jump_neighbor,
            )
            contrib_n = contrib_n / det_jacs[nc][..., None]
            _scatter_add_to_correction(correction, contrib_n, nc, n_cells, n_sps)

    return correction
