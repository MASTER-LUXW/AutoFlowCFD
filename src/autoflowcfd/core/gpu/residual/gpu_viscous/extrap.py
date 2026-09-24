"""AutoFlowCFD V2.0 - 面通量点外插与 tilde 通量配对(GPU 粘性)

从 `src/autoflowcfd/core/gpu/residual/gpu_viscous.py`(原 723 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""


from autoflowcfd.core.gpu import get_cupy


from autoflowcfd.core.gpu.residual.gpu_flux import viscous_physical_flux_gpu


def _extrap_to_fp(cp, mat, src_cell, field):
    """把按 src0/src1 机制索引到的 cell 场外插到面 FP 网格。

    mat: (nF, n_fp, n_sps)，src_cell: (nF,)，
    field: (n_cells, n_sps, *rest) -> 返回 (nF, n_fp, *rest)。
    与 gpu_inviscid.py::_compute_interface_correction_gpu 里
    `cp.matmul(owner_src0_mat, Q_owner)` 同一机制的通用化版本（那里
    只对 Q 这一个 5 分量场手写了一次，这里额外要对 grad_vel(3,3)/
    grad_T(3)/mu_t(1) 复用，做成通用 helper 避免抄 4 遍同样的 reshape）。
    """
    sub = field[src_cell]
    shape = sub.shape
    flat = sub.reshape(shape[0], shape[1], -1)
    out = cp.matmul(mat, flat)
    return out.reshape(mat.shape[0], mat.shape[1], *shape[2:])


def _add_src1_to_fp(cp, out, src1_idx, src1_cell, src1_mat, field):
    """叠加稀疏第二来源（分裂面场景），对应 CPU kernel 里的 `idx1 >= 0`
    分支——大多数面 idx1 全为 -1，这里用布尔掩码只处理真正需要的那一小
    撮面，不对整批面做无意义的零矩阵乘法。"""
    has1 = src1_idx >= 0
    if not bool(cp.any(has1)):
        return out
    sel = cp.where(has1)[0]
    idx1 = src1_idx[sel]
    c1 = src1_cell[idx1]
    m1 = src1_mat[idx1]
    sub = field[c1]
    shape = sub.shape
    flat = sub.reshape(shape[0], shape[1], -1)
    add = cp.matmul(m1, flat).reshape(sel.shape[0], m1.shape[1], *shape[2:])
    out[sel] = out[sel] + add
    return out


def _extrap_side(cp, idx, src0_cell_all, src0_mat_all, src1_idx_all, src1_cell_all, src1_mat_all,
                  Q_gpu, grad_vel_gpu, grad_T_gpu, mu_t_gpu):
    """按 src0(+src1) 机制把某一侧（owner 或 neighbor）的 Q/grad_vel/
    grad_T/mu_t 外插到面 idx 对应的 FP 网格。"""
    E_cell = src0_cell_all[idx]
    E_mat = src0_mat_all[idx]
    s1_idx = src1_idx_all[idx]

    Q_fp = _extrap_to_fp(cp, E_mat, E_cell, Q_gpu)
    Q_fp = _add_src1_to_fp(cp, Q_fp, s1_idx, src1_cell_all, src1_mat_all, Q_gpu)
    gv_fp = _extrap_to_fp(cp, E_mat, E_cell, grad_vel_gpu)
    gv_fp = _add_src1_to_fp(cp, gv_fp, s1_idx, src1_cell_all, src1_mat_all, grad_vel_gpu)
    gT_fp = _extrap_to_fp(cp, E_mat, E_cell, grad_T_gpu)
    gT_fp = _add_src1_to_fp(cp, gT_fp, s1_idx, src1_cell_all, src1_mat_all, grad_T_gpu)
    mut_fp = _extrap_to_fp(cp, E_mat, E_cell, mu_t_gpu)
    mut_fp = _add_src1_to_fp(cp, mut_fp, s1_idx, src1_cell_all, src1_mat_all, mu_t_gpu)
    return Q_fp, gv_fp, gT_fp, mut_fp[..., 0]


def _self_extrap_side(cp, cell_idx, cube_face_code, boundary_extrap_native,
                       Q_gpu, grad_vel_gpu, grad_T_gpu, mu_t_gpu):
    """自身面外插——某一侧单元用自己的场值按自身面几何外插到 FP
    （owner-primary 块的 `Q_o`/`gv_o`/`gT_o`/`mut_o`，或 neighbor-primary
    块的 `Q_n_native`/`gv_n_native`/`gT_n_native`/`mut_n_native`）。

    与 CPU 版 `viscous_flux_kernel.py` 的
    `E_o = boundary_extrap_native[oc_code - 6]` /
    `Q_o = _extrap_matmul(Q[oc], E_o)` 逐字对应，复用与
    `gpu_inviscid.py::_compute_interface_correction_gpu` 完全同一个
    `_native_self_extrap` helper。

    真实 bug 修复（问题清单 #5 排查附带发现，2026-09-02，均匀自由流场
    残差本应精确为零，实测非零且量级与真实物理量相当才发现）：此前
    `_compute_viscous_interface_correction_gpu` 对"自身"状态错误地
    复用了 `_extrap_side`（src0/src1 跨单元交叉引用机制），owner-primary
    块传的是 `owner_src0_cell`/`owner_src0_mat`，neighbor-primary 块传的
    是 `neighbor_src0_cell`/`neighbor_src0_mat`——但这两组数组的真实
    语义是"对侧记录用来查询*另一侧*单元值的交叉引用表"（`owner_src0_*`
    是给 neighbor-primary 块查 owner 值用的，`neighbor_src0_*` 是给
    owner-primary 块查 neighbor 值用的，见 `gpu_inviscid.py::_compute_
    interface_correction_gpu` 里 `Q_n=_extrap_q_to_fp(...,ff.neighbor_
    src0_mat[idx_o],...)`/`Q_o_at_n=_extrap_q_to_fp(...,ff.owner_src0_
    mat[idx_n],...)` 的对称用法），不是"本单元查自己"——对 owner-primary
    面自身而言 `owner_src0_mat[idx_o]` 恒为零/未设置（那一行本来就不是
    为这个查询设计的），导致 `Q_o`/`gv_o`/`gT_o`/`mut_o` 恒为零而不是
    真实自身状态：BR1 平均态 `Q_avg=0.5*(Q_o+Q_n)` 与边界 IP 罚项
    `pen=-scale*(Q_o[1:4]-Q_n[1:4])` 全部从错误的"自身值"算起，均匀
    自由流场下 `Q_o=0 != Q_n=真实自由流值`，IP 罚项产出物理量级的
    虚假非零残差（实测 ~0.43，CPU 参考给出 ~2e-12）。

    Args:
        cell_idx: (n,) 本侧单元全局索引（owner 块传 `oc`，neighbor 块
            传 `nc`）
        cube_face_code: (n,) 本侧 owner_cube_face/neighbor_cube_face
        boundary_extrap_native: (9,n_fp,n_sps) 原生自身外插表
            （四面体 [6,10)、棱柱 [10,15) 同一张表，按 `code-6` 索引）

    Returns:
        (Q, gv, gT, mut)：形状分别为 (n,n_fp,5)/(n,n_fp,3,3)/(n,n_fp,3)/
        (n,n_fp)
    """
    from autoflowcfd.core.gpu.residual.gpu_inviscid import _native_self_extrap

    E = _native_self_extrap(cp, cube_face_code, boundary_extrap_native)

    Q = _extrap_to_fp(cp, E, cell_idx, Q_gpu)
    gv = _extrap_to_fp(cp, E, cell_idx, grad_vel_gpu)
    gT = _extrap_to_fp(cp, E, cell_idx, grad_T_gpu)
    mut = _extrap_to_fp(cp, E, cell_idx, mu_t_gpu)[..., 0]
    return Q, gv, gT, mut


def _viscous_tilde_flux_pair(Q_common, gv_common, gT_common, mut_common,
                              Q_own, gv_own, gT_own, mut_own,
                              adjrow, mu, Pr, Pr_t):
    """算 (G_tilde_common, G_tilde_own) 一对张量，两者之差就是 jump。
    对应 CPU 端每个 FP 上 `viscous_physical_flux_point` 调用两次
    （一次给 BR1 平均态，一次给自身原始态）再各自投影到 tilde 方向的
    那一段——这里把它向量化到 (n_faces*n_fp,) 展平批量。"""
    cp = get_cupy()
    n, n_fp = Q_common.shape[0], Q_common.shape[1]

    G_common = viscous_physical_flux_gpu(
        Q_common.reshape(n * n_fp, 5), gv_common.reshape(n * n_fp, 3, 3),
        gT_common.reshape(n * n_fp, 3), mu, Pr, mu_t=mut_common.reshape(n * n_fp), Pr_t=Pr_t,
    ).reshape(n, n_fp, 3, 5)
    G_own = viscous_physical_flux_gpu(
        Q_own.reshape(n * n_fp, 5), gv_own.reshape(n * n_fp, 3, 3),
        gT_own.reshape(n * n_fp, 3), mu, Pr, mu_t=mut_own.reshape(n * n_fp), Pr_t=Pr_t,
    ).reshape(n, n_fp, 3, 5)

    a0 = adjrow[..., 0:1]
    a1 = adjrow[..., 1:2]
    a2 = adjrow[..., 2:3]
    G_tilde_common = a0 * G_common[..., 0, :] + a1 * G_common[..., 1, :] + a2 * G_common[..., 2, :]
    G_tilde_own = a0 * G_own[..., 0, :] + a1 * G_own[..., 1, :] + a2 * G_own[..., 2, :]
    return G_tilde_common, G_tilde_own
