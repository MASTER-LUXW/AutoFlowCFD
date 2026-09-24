"""AutoFlowCFD V2.0 - 无粘界面校正与其 gather/scatter 辅助(GPU)

从 `src/autoflowcfd/core/gpu/residual/gpu_inviscid.py`(原 710 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""


from autoflowcfd.core.fr_operators.kernels import resolve_ausm_precond_mode

from autoflowcfd.core.gpu import get_cupy

from autoflowcfd.core.gpu.residual.gpu_flux import euler_physical_flux_gpu

from .flux import _ausm_up_flux_batch_gpu


def _extrap_q_to_fp(cp, mat, src_cell, Q_gpu):
    """(nF,n_fp,n_sps) @ Q_gpu[src_cell] (nF,n_sps,5) -> (nF,n_fp,5)。"""
    return cp.matmul(mat, Q_gpu[src_cell])


def _add_q_src1_to_fp(cp, out, src1_idx, src1_cell, src1_mat, Q_gpu):
    """叠加稀疏第二来源（分裂面场景），Q 专用版本，见
    gpu_viscous.py::_add_src1_to_fp 同名通用版本的文档（这里内联一份
    Q-only 版本，避免 gpu_inviscid.py<->gpu_viscous.py 产生循环 import：
    两者都从 gpu_inviscid_volume.py 导入 prepare_mesh_data/prepare_ops_data，
    不再互相 import）。"""
    has1 = src1_idx >= 0
    if not bool(cp.any(has1)):
        return out
    sel = cp.where(has1)[0]
    idx1 = src1_idx[sel]
    c1 = src1_cell[idx1]
    m1 = src1_mat[idx1]
    out[sel] = out[sel] + cp.matmul(m1, Q_gpu[c1])
    return out


def _ausm_direction(cp, adjrow):
    """AUSM+up 用的法向：本侧**精确度量行**的单位方向（与 CPU 版
    `inviscid_kernel.py` 逐字对应）。

    此前名为 `_ausm_direction_with_fallback`，在 `dir(adjrow).n_ref < 0.5`
    时改用 `n_ref`（owner 侧 true_normal / neighbor 侧 -true_normal）——
    2026-09-24 删除，理由见 CPU 版函数文档"法向一律取自本侧精确度量行"：
    换了方向之后拥有侧投影仍用 adjrow，对均匀流跳跃量不为零，凭空注入
    压力量级的源项。

    **没有 side 参数**：原生面的 `adj_row` 已是 outward 定向（见
    `fr/face_flux_points/exact_normal.py`），方向系数恒为 +1。

    Args:
        adjrow: (n, n_fp, 3) 未归一化 adj(J) 行

    Returns:
        (direction, adj_mag)：direction (n,n_fp,3)，adj_mag (n,n_fp)
    """
    a0 = adjrow[..., 0]
    a1 = adjrow[..., 1]
    a2 = adjrow[..., 2]
    adj_mag = cp.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
    adj_mag_safe = cp.maximum(adj_mag, 1e-300)
    direction = cp.stack([a0 / adj_mag_safe, a1 / adj_mag_safe,
                          a2 / adj_mag_safe], axis=-1)
    return direction, adj_mag


def _native_self_extrap(cp, cube_face_code, boundary_extrap_native):
    """自身面外插矩阵：按 `code - 6` 逐面 gather 原生表（四面体 [6,10)、
    棱柱 [10,15) 在同一张表里）——与 CPU 版
    `E_o = boundary_extrap_native[oc_code - 6]` 逐字对应。

    Args:
        cube_face_code: (n,) 原始 cube face 编码
        boundary_extrap_native: (9, n_fp, n_sps)

    Returns:
        E: (n, n_fp, n_sps)
    """
    return boundary_extrap_native[cube_face_code - 6]


def _lift_native_contrib(cp, cube_face_code, lift_native, ref_area_weight, jump):
    """面校正分配到体积节点：DG 提升算子
    `lift_native[code-6] @ (ref_area_weight ⊙ jump)`——与 CPU 版
    `contrib_owner = lift_native[oc_code-6] @ weighted_jump_o` 逐字对应。

    **权重是参考求积权重、不是物理面积权重**（2026-09-18 修掉的真实缺陷，
    完整记录见 `core/fr_operators/face_kernels.py::FlatFaceGeometry.
    ref_area_weight` 字段文档）：这一路的 `jump` 是 `adj_row . (F*-F_own)`，
    已经是参考空间的法向通量差，再乘物理面积权重会多乘一个 `|adj_row|
    ~ h^2`，界面项从 `~1/h` 变成 `~h`。
    （注意 `gpu_scalar_transport.py` 里同一位置**用物理面积权重是对的**
    —— 它的 `jump` 是物理通量密度差，两路的 `jump` 不在同一个空间。）

    Args:
        cube_face_code: (n,)
        lift_native: (9, n_sps, n_fp)
        ref_area_weight: (n_fp,) 参考面求积权重（逐面相同）
        jump: (n, n_fp, 5)

    Returns:
        contrib: (n, n_sps, 5)
    """
    lift = lift_native[cube_face_code - 6]              # (n, n_sps, n_fp)
    weighted_jump = ref_area_weight[None, :, None] * jump   # (n, n_fp, 5)
    return cp.matmul(lift, weighted_jump)               # (n, n_sps, 5)


def _compute_interface_correction_gpu(
    Q_gpu, adj_j, det_jacs, flat_face_gpu, Q_ghost_gpu,
    n_cells, n_sps, device_id, mach_ref, precond_mode=None,
):
    """GPU 界面校正计算（按图着色逐色处理）。

    真实 bug 修复（2026-08-23，本次移植 GPU 粘性界面项时顺带发现并
    修复，用户明确要求本轮一并处理）：此前的实现有两个复合缺陷：

    1. **无 owner_is_primary/neighbor_is_primary 过滤**：`owner_src0`/
       `neighbor_src0` 字段的真实语义是"跨单元交叉引用数据，只在对应
       角色 primary 时才被写入"（见 face_flux_points/merge.py
       "kernel 只在 owner_primary[f] 为真...时写入 _nb_interp[f]，其余
       情形保持全零"的说明）——此前这里把它们当"这条记录自己的原生值"
       无条件读取。对约 5% 因棱柱四边形侧面拆分成 2 条记录的面：非
       owner_primary 记录的 `neighbor_src0` 全零/cell=-1，AUSM+up 会用
       一个全零假邻居态算出真实错误的通量（不只是重复计数），owner 侧
       的真实贡献还会被两条记录各计入一次。
    2. **owner/neighbor 两侧共享同一个 AUSM+up 通量分别分配**：
       inviscid_kernel.py 模块文档"关键正确性约束"明确记录过这个
       "优化"在真实网格上被验证证伪——自由流场残差从 9e-5 恶化到
       3.1e7，owner/neighbor 两侧必须用各自的度量法向、各自原生 FP
       位置插值出的状态独立各调用一次 AUSM+up，不能合并复用。
    3. **符号约定**：CPU 版每一侧的贡献是 `correction[...] += -contrib
       /dj`（带负号，配合无粘残差 `-div(F)` 的整体约定），此前这里
       scatter 的是不带负号的 `contrib_o`/`contrib_n`。

    修复：逐字仿照 inviscid_kernel.py 的两段独立结构（owner-primary
    过滤块 + neighbor-primary 过滤块），owner/neighbor 各自的"自身
    原生值"改用 `boundary_extrap` 查表外插（与 primary 状态无关、对
    任何记录都有效，不依赖 src0），"跨单元值"才用 src0/src1（只在
    对应角色 primary 时读取有意义）；两侧各自独立调用一次批量 AUSM+up；
    accumulate 时统一带负号。

    本机没有 CUDA/cupy，这处修复无法在本地实际执行验证（全仓库此前
    也没有任何 `compute_inviscid_residual_fr_gpu` 的 crosscheck 测试，
    见 test_gpu_p1_inviscid_interface_crosscheck.py 模块文档——这是
    这次连带发现的又一个"从未被验证过"的 GPU 生产路径），只能靠对照
    已经过详细验证的 CPU kernel 逐字核对；请在有真实 GPU 的环境上跑
    该新增测试文件做最终确认。
    """
    if precond_mode is None:
        precond_mode = resolve_ausm_precond_mode()
    precond_mode = int(precond_mode)

    cp = get_cupy()
    ff = flat_face_gpu
    correction = cp.zeros((n_cells, n_sps, 5), dtype=cp.float64)

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

            # 自身面外插按 `code - 6` gather 原生表（四面体 [6,10)、
            # 棱柱 [10,15)），与 CPU 版 inviscid_kernel.py 逐字对应。
            oc_code_o = ff.owner_cube_face[idx_o]

            E_o = _native_self_extrap(cp, oc_code_o, ff.boundary_extrap_native)
            Q_o = cp.matmul(E_o, Q_gpu[oc])  # (nO,n_fp,5)

            Q_n = _extrap_q_to_fp(cp, ff.neighbor_src0_mat[idx_o], ff.neighbor_src0_cell[idx_o], Q_gpu)
            Q_n = _add_q_src1_to_fp(
                cp, Q_n, ff.neighbor_src1_idx[idx_o], ff.neighbor_src1_cell, ff.neighbor_src1_mat, Q_gpu,
            )

            n_fp = Q_o.shape[1]
            # Q_ghost_gpu 现在是逐 FP 幽灵态 (n_faces,n_fp,5)（见
            # _compute_boundary_ghost_states_gpu 文档），直接按 FP 对齐
            # 使用，不再对该面全部 FP 广播同一个值。
            Q_ghost_face = Q_ghost_gpu[idx_o]  # (nO, n_fp, 5)
            Q_n = cp.where(
                is_bnd_o[:, None, None],
                Q_ghost_face,
                Q_n,
            )

            # 混合拆分面（B-8，镜像 CPU inviscid_kernel.py 同名分支）：混合配对的内部面在边界半区
            # 逐 FP 取配对边界面的幽灵态。
            mp_o = ff.mixed_nb_partner[idx_o]
            mixed_sel_o = (mp_o[:, None] >= 0) & ff.mixed_nb_mask[idx_o]  # (nO, n_fp)
            Q_ghost_partner_o = Q_ghost_gpu[cp.maximum(mp_o, 0)]  # (nO, n_fp, 5)
            Q_n = cp.where(
                mixed_sel_o[..., None],
                Q_ghost_partner_o,
                Q_n,
            )

            # 原生面的 adj 行已是 outward 定向，方向系数恒为 +1（见
            # `_native_self_extrap` 与 CPU 版同一处说明）。
            adjrow_o = ff.owner_adj_row_exact[idx_o]
            direction_o, adj_mag_o = _ausm_direction(cp, adjrow_o)

            nO = Q_o.shape[0]
            flux_o = _ausm_up_flux_batch_gpu(
                Q_o.reshape(nO, n_fp, 5), Q_n.reshape(nO, n_fp, 5), direction_o, mach_ref,
                precond_mode,
            )
            F_tilde_common_o = flux_o * adj_mag_o[..., None]

            a0 = adjrow_o[..., 0]
            a1 = adjrow_o[..., 1]
            a2 = adjrow_o[..., 2]
            F_phys_o = euler_physical_flux_gpu(Q_o.reshape(nO * n_fp, 5)).reshape(nO, n_fp, 3, 5)
            F_tilde_own_o = (
                a0[..., None] * F_phys_o[..., 0, :]
                + a1[..., None] * F_phys_o[..., 1, :]
                + a2[..., None] * F_phys_o[..., 2, :]
            )

            jump_owner = F_tilde_common_o - F_tilde_own_o

            contrib_o = _lift_native_contrib(
                cp, oc_code_o, ff.lift_native, ff.ref_area_weight, jump_owner,
            )
            contrib_o = contrib_o / det_jacs[oc][..., None]
            _scatter_add_to_correction(correction, -contrib_o, oc, n_cells, n_sps)

        # ── neighbor-primary 贡献块（仅内部面）──
        mask_n = neighbor_primary & (~is_bnd)
        if bool(cp.any(mask_n)):
            idx_n = face_idx[mask_n]
            nc = ff.neighbor_cell[idx_n]

            # native 四面体（路径C）GPU 移植（2026-09-02）：见上方
            # owner-primary 块同名注释，同一处修复。
            nc_code_n = ff.neighbor_cube_face[idx_n]
            E_n = _native_self_extrap(cp, nc_code_n, ff.boundary_extrap_native)
            Q_n_native = cp.matmul(E_n, Q_gpu[nc])  # (nN,n_fp,5)

            Q_o_at_n = _extrap_q_to_fp(cp, ff.owner_src0_mat[idx_n], ff.owner_src0_cell[idx_n], Q_gpu)
            Q_o_at_n = _add_q_src1_to_fp(
                cp, Q_o_at_n, ff.owner_src1_idx[idx_n], ff.owner_src1_cell, ff.owner_src1_mat, Q_gpu,
            )

            # 混合拆分面（B-8）：neighbor 侧对称处理——边界半区对侧状态
            # 逐 FP 取配对面幽灵态（Q_ghost_gpu 为逐 FP (n_faces,n_fp,5)）。
            mp_n = ff.mixed_ow_partner[idx_n]
            mixed_sel_n = (mp_n[:, None] >= 0) & ff.mixed_ow_mask[idx_n]
            Q_ghost_partner_n = Q_ghost_gpu[cp.maximum(mp_n, 0)]  # (nN, n_fp, 5)
            Q_o_at_n = cp.where(
                mixed_sel_n[..., None],
                Q_ghost_partner_n,
                Q_o_at_n,
            )

            n_fp = Q_n_native.shape[1]
            nN = Q_n_native.shape[0]

            # neighbor 视角外法向恒为 -true_normal（见 CPU kernel 同名注释）
            adjrow_n = ff.neighbor_adj_row_exact[idx_n]
            direction_n, adj_mag_n = _ausm_direction(cp, adjrow_n)

            flux_n = _ausm_up_flux_batch_gpu(
                Q_n_native.reshape(nN, n_fp, 5), Q_o_at_n.reshape(nN, n_fp, 5), direction_n, mach_ref,
                precond_mode,
            )
            F_tilde_common_n = flux_n * adj_mag_n[..., None]

            a0n = adjrow_n[..., 0]
            a1n = adjrow_n[..., 1]
            a2n = adjrow_n[..., 2]
            F_phys_n = euler_physical_flux_gpu(Q_n_native.reshape(nN * n_fp, 5)).reshape(nN, n_fp, 3, 5)
            F_tilde_own_n = (
                a0n[..., None] * F_phys_n[..., 0, :]
                + a1n[..., None] * F_phys_n[..., 1, :]
                + a2n[..., None] * F_phys_n[..., 2, :]
            )

            jump_neighbor = F_tilde_common_n - F_tilde_own_n

            contrib_n = _lift_native_contrib(
                cp, nc_code_n, ff.lift_native, ff.ref_area_weight, jump_neighbor,
            )
            contrib_n = contrib_n / det_jacs[nc][..., None]
            _scatter_add_to_correction(correction, -contrib_n, nc, n_cells, n_sps)

    return correction


def _scatter_add_to_correction(correction, contrib, cell_indices, n_cells, n_sps):
    """将面的校正贡献写入全局 correction 数组。

    同色面无 owner_cell 冲突，但不同面可能写同一个 cell（虽然同色面之间不会），
    所以这里使用 CuPy 的 scatter add 模式。

    由于同色面保证无冲突，可以直接用索引赋值。
    """
    cp = get_cupy()
    # 同色面无冲突，直接用 advanced indexing 写入
    # contrib: (n_color_faces, n_sps, 5), cell_indices: (n_color_faces,)
    # 需要处理多个面写同一个 cell 的情况（虽然同色面不冲突，但保险起见用 add）
    cp.scatter_add(correction, (cell_indices, slice(None), slice(None)), contrib)
