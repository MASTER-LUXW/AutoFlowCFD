"""AutoFlowCFD V2.0 - 标量场面外插与面校正分配(GPU)

从 `src/autoflowcfd/core/gpu/turbulence/gpu_scalar_transport.py`(原 743 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""


from autoflowcfd.core.gpu.residual.gpu_inviscid import _native_self_extrap


def _lift_native_contrib_scalar(
    cp, cube_face_code, lift_native, true_area_weight_face, jump,
):
    """标量版 `gpu_inviscid.py::_lift_native_contrib`：
    `lift_native[code-6] @ (true_area_weight ⊙ jump)`——与 CPU 版
    `transport_kernel.py::distribute_corrections_to_cells_kernel` 逐字
    对应，仅去掉 5 变量的末轴（标量场没有这一维）。

    **权重是物理面积权重，与无粘/粘性那两路的参考求积权重不同，而且
    两边都是对的**：本路的 `jump` 是物理通量密度差，那两路的 `jump` 是
    参考空间的法向通量差 —— 完整对照见 CPU 版
    `transport_kernel.py::_weighted_jump_native` 文档。

    Args:
        cube_face_code: (n,)
        lift_native: (9, n_sps, n_fp)
        true_area_weight_face: (n, n_fp)
        jump: (n, n_fp) 未加权原始跳变量

    Returns:
        contrib: (n, n_sps)
    """
    lift = lift_native[cube_face_code - 6]              # (n, n_sps, n_fp)
    weighted_jump = true_area_weight_face * jump        # (n, n_fp)
    return cp.einsum('nsf,nf->ns', lift, weighted_jump)  # (n, n_sps)


def _extrapolate_scalar_to_faces_gpu(
    cp, ff, n_prism, scalar_sps,
    wall_dirichlet_zero_face=None,
    wall_dirichlet_value_face=None,
    has_wall_dirichlet_value=None,
):
    """CuPy 版 `extrapolate_scalar_to_faces_kernel`，对全部面一次性向量化
    处理（不分色、不按 owner/neighbor_is_primary 过滤，见模块文档）。

    与 CPU 版逐字对应的三条边界 ghost 规则（真实边界面，即
    neighbor_src0_cell<0 且 neighbor_src1_idx<0）：
    - wall_dirichlet_zero_face: ghost = -owner（k=0 Dirichlet 镜像）
    - has_wall_dirichlet_value: ghost = 2*target - owner（omega 解析壁面值）
    - 都不是：ghost = owner（Neumann 默认，零梯度）
    以及混合拆分面（B-8）配对边界面同一规则的逐 FP 覆盖。

    Args:
        scalar_sps: (n_cells, n_sps) CuPy 数组

    Returns:
        (phi_owner_fp, phi_neighbor_fp) 各 (n_faces, n_fp)
    """
    n_faces = ff.n_faces

    oc = ff.owner_cell
    # 自身外插按 `code - 6` gather 原生表（四面体 [6,10)、棱柱 [10,15)
    # 同一张表），与 CPU 版 `_extrap_owner_scalar_to_faces` 逐字对应。
    E_o = _native_self_extrap(cp, ff.owner_cube_face, ff.boundary_extrap_native)
    phi_owner = cp.einsum('fps,fs->fp', E_o, scalar_sps[oc])

    c0 = ff.neighbor_src0_cell
    valid0 = c0 >= 0
    c0_safe = cp.maximum(c0, 0)
    phi_neighbor = cp.einsum('fps,fs->fp', ff.neighbor_src0_mat, scalar_sps[c0_safe]) * valid0[:, None]

    idx1 = ff.neighbor_src1_idx
    valid1 = idx1 >= 0
    if bool(cp.any(valid1)):
        idx1_safe = cp.maximum(idx1, 0)
        c1 = ff.neighbor_src1_cell[idx1_safe]
        mat1 = ff.neighbor_src1_mat[idx1_safe]
        phi_neighbor = phi_neighbor + cp.einsum('fps,fs->fp', mat1, scalar_sps[c1]) * valid1[:, None]

    n_fp = phi_owner.shape[1]
    if wall_dirichlet_zero_face is None:
        wall_dirichlet_zero_face = cp.zeros(n_faces, dtype=cp.bool_)
    if has_wall_dirichlet_value is None:
        has_wall_dirichlet_value = cp.zeros(n_faces, dtype=cp.bool_)
    if wall_dirichlet_value_face is None:
        wall_dirichlet_value_face = cp.zeros((n_faces, n_fp), dtype=cp.float64)

    is_true_boundary = (~valid0) & (~valid1)
    dirichlet_zero = is_true_boundary & wall_dirichlet_zero_face
    dirichlet_value = is_true_boundary & has_wall_dirichlet_value & (~wall_dirichlet_zero_face)
    neumann = is_true_boundary & (~wall_dirichlet_zero_face) & (~has_wall_dirichlet_value)

    phi_neighbor = cp.where(dirichlet_zero[:, None], -phi_owner, phi_neighbor)
    phi_neighbor = cp.where(dirichlet_value[:, None], 2.0 * wall_dirichlet_value_face - phi_owner, phi_neighbor)
    phi_neighbor = cp.where(neumann[:, None], phi_owner, phi_neighbor)

    mp = ff.mixed_nb_partner
    has_partner = mp >= 0
    if bool(cp.any(has_partner)):
        mp_safe = cp.maximum(mp, 0)
        partner_dirichlet_zero = wall_dirichlet_zero_face[mp_safe]
        partner_has_value = has_wall_dirichlet_value[mp_safe]
        partner_value = wall_dirichlet_value_face[mp_safe]
        mask = has_partner[:, None] & ff.mixed_nb_mask

        val_dirichlet_zero = -phi_owner
        val_dirichlet_value = 2.0 * partner_value - phi_owner
        val_neumann = phi_owner
        chosen = cp.where(
            partner_dirichlet_zero[:, None], val_dirichlet_zero,
            cp.where(partner_has_value[:, None], val_dirichlet_value, val_neumann),
        )
        phi_neighbor = cp.where(mask, chosen, phi_neighbor)

    return phi_owner, phi_neighbor


def _distribute_scalar_correction_gpu(cp, ff, raw_jump_fp, det_jacs, n_cells, n_sps,
                                       raw_jump_fp_neighbor=None):
    """CuPy 版 `distribute_corrections_to_cells_kernel[_colored]`，标量版，
    对全部面一次性向量化处理（不分色，见模块文档）。

    2026-09-03 重写（见模块文档"真实 bug 修复 + native 完整补全"一节）：
    (1) 新增 owner_is_primary/neighbor_is_primary 过滤，与 CPU 版
        2026-09-02 的修复对齐；(2) 参数从"已按 |adj_row| 预加权"的
        `correction_fp` 改为**未加权**的 `raw_jump_fp`，加权方式（collapsed
        用 |adj_row|，native 用 true_area_weight）延后到本函数内部按面
        类型分派，与 CPU 版 `_weighted_jump_collapsed`/`_weighted_jump_
        native` 逐字对应；(3) 新增 native 分支（`boundary_extrap_native`/
        `lift_native` DG 提升算子）。

    raw_jump_fp_neighbor（真实 bug 修复，2026-09-12，与 CPU 版
    `transport.py::_distribute_correction_to_cells` 同名参数同一处修复，
    完整推导见 `compute_scalar_convection_residual_gpu` 模块文档）：
    neighbor 侧必须用相对 phi_neighbor 计算的独立跳变量，不能复用 owner
    侧那份相对 phi_owner 算出的 `raw_jump_fp`。默认 None 时退化为与
    `raw_jump_fp` 相同（扩散残差的 BR1 跳变量对 owner/neighbor 天然
    对称，调用方不传此参数，行为不变）。

    raw_jump_fp: (n_faces, n_fp) 未加权原始物理跳变量
    Returns: (n_cells, n_sps)
    """
    if raw_jump_fp_neighbor is None:
        raw_jump_fp_neighbor = raw_jump_fp
    correction = cp.zeros((n_cells, n_sps), dtype=cp.float64)

    # ── owner 侧分配（B-8 混合拆分面：只有 owner_is_primary 的记录才
    # 贡献 owner 侧，见模块文档）──
    owner_primary = ff.owner_is_primary
    sel_o = cp.where(owner_primary)[0]
    if bool(sel_o.shape[0] > 0):
        oc = ff.owner_cell[sel_o]
        raw_o = raw_jump_fp[sel_o]
        contrib_o = _lift_native_contrib_scalar(
            cp, ff.owner_cube_face[sel_o], ff.lift_native,
            ff.true_area_weight[sel_o], raw_o,
        )
        contrib_o = contrib_o / det_jacs[oc]
        cp.scatter_add(correction, (oc, slice(None)), -contrib_o)

    # ── neighbor 侧分配（内部面，同一处 B-8 过滤）──
    nc = ff.neighbor_cell
    neighbor_primary = ff.neighbor_is_primary
    has_neighbor = (nc >= 0) & neighbor_primary
    if bool(cp.any(has_neighbor)):
        sel_n = cp.where(has_neighbor)[0]
        nc_sel = nc[sel_n]
        raw_n = raw_jump_fp_neighbor[sel_n]
        contrib_n = _lift_native_contrib_scalar(
            cp, ff.neighbor_cube_face[sel_n], ff.lift_native,
            ff.true_area_weight[sel_n], raw_n,
        )
        contrib_n = contrib_n / det_jacs[nc_sel]
        cp.scatter_add(correction, (nc_sel, slice(None)), contrib_n)

    return correction
