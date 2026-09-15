"""
AutoFlowCFD V2.0 - GPU 版湍流标量（k/omega）输运残差 (#7 第四次评审第四轮)

与 core/turbulence/transport.py + transport_kernel.py 对应的 CuPy 版本，
补齐 GPU SST/DDES/IDDES 长期缺失的输运项（此前 GPUTurbulenceSST 只有
逐点源项 ODE，`update_fields_gpu` 的 `transport_k`/`transport_omega`
参数从未被调用方传入，见 gpu_turbulence_sst.py 模块文档）。

不需要图着色分组：CPU numba 版本用图着色规避多线程写冲突（per-thread
buffer 的替代方案），但 `cp.scatter_add` 本身就正确处理重复索引累加，
对全部面一次性向量化处理即可，颜色分组对 GPU 版本的正确性没有必要。

分配机制复用 `gpu_inviscid_volume.py::distribute_face_correction_to_sps`
（V2.0 专家组盲审第四轮修复的 gather 机制，与 CPU numba kernel
`_distribute_point_scalar` 完全一致，仅 collapsed 面使用），不是矩阵乘法。

真实 bug 修复 + native 四面体（路径C）完整补全（2026-09-03，"把 native
全部补充完整"排查——本机没有真实 CuPy，本模块此前从未被真正执行验证过，
与本次同一排查发现的 gpu_inviscid.py/gpu_viscous.py/gpu_gradients.py 系列
bug 同源）：

1. **owner_is_primary/neighbor_is_primary 过滤缺失**（不是"刻意保留的
   行为差异"——本模块旧文档曾这样声称，但去核对 CPU 参考
   `transport_kernel.py::distribute_corrections_to_cells_kernel` 发现
   CPU 早在 2026-09-02（该函数自己的文档"真实 bug 修复"一节）就已经
   补上了这个过滤，旧文档的说法是过时信息，不是事实）：B-8 混合拆分面
   场景下，同一个物理面会被拆成 2 条记录，只有 `owner_is_primary=True`/
   `neighbor_is_primary=True` 的那条记录才应该贡献对应侧——本模块
   `_distribute_scalar_correction_gpu` 此前对全部面（含非 primary 的
   重复记录）无条件累加，等价于把这批面的贡献重复计入。现在补齐过滤，
   与 CPU 版逐字对应。

2. **完全没有 native 分派**：`_extrapolate_scalar_to_faces_gpu` 自身
   外插、`_distribute_scalar_correction_gpu` 面校正分配，此前都无条件
   走 collapsed 路径（`boundary_extrap[celltype,axis,side_idx]` 查表 +
   `distribute_face_correction_to_sps` 1D 修正函数分布）——对 native
   四面体面，`owner_axis`/`neighbor_axis` 存的是复用的 excluded_vertex
   （0~3），既会在 `axis` 维度只有 3 的表上越界（`==3` 时崩溃），修复
   越界后也仍然是错误结果（native 单纯形基没有"坍缩计算方向"，1D 分布
   机制本身不适用，必须用 `boundary_extrap_native`/`lift_native` DG
   提升算子）。现在按 CPU 版 `_extrap_owner_scalar_to_faces`/
   `distribute_corrections_to_cells_kernel` 的 native 分支逐字补齐：
   - 自身外插复用 `gpu_inviscid.py::_native_self_extrap`（形状签名
     `(n,n_fp,n_sps)` 与变量个数无关，标量场直接复用不需要改写）。
   - 面校正分配新增标量版 `_native_or_collapsed_contrib_scalar`（对照
     `gpu_inviscid.py::_native_or_collapsed_contrib`，去掉 5 变量末轴，
     `jump`/`contrib` 都是 `(n,n_fp)`/`(n,n_sps)`）。与 CPU 版一致，
     加权方式按面类型分派（collapsed 面乘 `|adj_row|`，native 面乘
     `true_area_weight`），加权发生在分配阶段，不再像旧版那样在调用方
     （`compute_scalar_convection/diffusion_residual_gpu`）提前统一乘
     `|owner_adj_row_exact|`——旧版这个提前加权对 native 面是错误的
     （native 面应该用 `true_area_weight` 而不是 `|adj_row|`），必须
     像 CPU 版一样把**未加权**的 `raw_jump_fp` 一路传到分配阶段，加权
     方式才能按面类型正确分派。
   - `compute_scalar_convection_residual_gpu`/
     `compute_scalar_diffusion_residual_gpu` 的四面体段 divergence
     收缩此前无条件用 `ops_data['D_3d_tet']`（坍缩坐标微分算子），与
     `gpu_gradients.py::compute_physical_gradient_gpu` 同一类遗漏——
     改用 `gpu_gradients.py` 已确立的自描述判据
     `'D_native_tet_padded' in ops_data`。

Args/Returns 类型标注、`_distribute_scalar_correction_gpu` 的调用方签名
均已同步更新（`correction_fp` 参数改名 `raw_jump_fp`，语义从"已加权"变
"未加权"，与 CPU 版 `raw_jump_fp` 命名及语义完全对齐）。
"""

from typing import Optional, Tuple

from autoflowcfd.core.gpu import get_cupy
from autoflowcfd.core.gpu.residual.gpu_volume_contract import (
    gpu_contract_shared_operator_1axis, gpu_contract_shared_operator_2axis,
)
from autoflowcfd.core.gpu.residual.gpu_gradients import compute_physical_scalar_gradient_gpu
from autoflowcfd.core.gpu.residual.gpu_inviscid_volume import distribute_face_correction_to_sps
from autoflowcfd.core.gpu.residual.gpu_inviscid import _native_self_extrap


def _native_or_collapsed_contrib_scalar(
    cp, is_native, cube_face_code, lift_native, true_area_weight_face, jump, contrib_collapsed,
):
    """标量版 `gpu_inviscid.py::_native_or_collapsed_contrib`：native 面用
    `lift_native[excluded_vertex] @ (true_area_weight ⊙ jump)`，collapsed
    面用调用方已经算好的 `contrib_collapsed`——与 CPU 版
    `transport_kernel.py::distribute_corrections_to_cells_kernel` 的
    native/collapsed 分支逐字对应，仅去掉 5 变量的末轴（标量场没有这一维）。

    Args:
        is_native, cube_face_code: (n,)
        lift_native: (4, n_sps, n_fp)
        true_area_weight_face: (n, n_fp)
        jump: (n, n_fp) 未加权原始跳变量
        contrib_collapsed: (n, n_sps)

    Returns:
        contrib: (n, n_sps)
    """
    # 与 `_native_self_extrap`/`_native_or_collapsed_contrib` 同一处修复：
    # `tet_basis_mode="collapsed"` 时 `lift_native` 是空数组
    # `(0, n_sps, n_fp)`，`is_native` 恒为 False，短路直接返回
    # `contrib_collapsed`，避免对空数组做越界 gather。
    if lift_native.shape[0] == 0:
        return contrib_collapsed
    excluded_vertex = cp.clip(cube_face_code - 6, 0, lift_native.shape[0] - 1)
    lift = lift_native[excluded_vertex]  # (n, n_sps, n_fp)
    weighted_jump = true_area_weight_face * jump  # (n, n_fp)
    contrib_native = cp.einsum('nsf,nf->ns', lift, weighted_jump)  # (n, n_sps)
    return cp.where(is_native[:, None], contrib_native, contrib_collapsed)


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
    oax = ff.owner_axis
    oside = ff.owner_side
    oside_idx = cp.where(oside <= 0, 0, 1)
    compact_cell_type = getattr(ff, 'compact_cell_type', None)
    if compact_cell_type is not None:
        celltype_o = compact_cell_type[oc]
    else:
        celltype_o = cp.where(oc < n_prism, 0, 1)

    # native 四面体（路径C）自身外插（2026-09-03 补齐，见模块文档）：
    # `oax` 对 native 面存的是复用的 excluded_vertex（0~3），不能无条件
    # gather 只有 3 个轴的 `boundary_extrap`——先 clip 到安全哑值 0，
    # 再用 `_native_self_extrap` 按 `oc_code_o>=6` 分派到
    # `boundary_extrap_native[excluded_vertex]`；纯 collapsed 网格下
    # `boundary_extrap_native` 是空数组，短路直接退化为原有行为。
    oc_code_o = ff.owner_cube_face
    is_native_o = oc_code_o >= 6
    oax_safe = cp.where(is_native_o, 0, oax)
    E_o_collapsed = ff.boundary_extrap[celltype_o, oax_safe, oside_idx]  # (n_faces, n_fp, n_sps)
    E_o = _native_self_extrap(cp, is_native_o, oc_code_o, ff.boundary_extrap_native, E_o_collapsed)
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
        oax = ff.owner_axis[sel_o]
        oside = ff.owner_side[sel_o]
        raw_o = raw_jump_fp[sel_o]

        oc_code_o = ff.owner_cube_face[sel_o]
        is_native_o = oc_code_o >= 6
        oax_safe = cp.where(is_native_o, 0, oax)

        adj_mag_o = cp.linalg.norm(ff.owner_adj_row_exact[sel_o], axis=-1)
        weighted_o_collapsed = adj_mag_o * raw_o
        contrib_o_collapsed = distribute_face_correction_to_sps(
            cp, weighted_o_collapsed, oax_safe, oside, ff.dist_fp_of_sp, ff.dist_axis_coord_of_sp,
            ff.g_left, ff.g_right,
        )  # (nO, n_sps)
        contrib_o = _native_or_collapsed_contrib_scalar(
            cp, is_native_o, oc_code_o, ff.lift_native, ff.true_area_weight[sel_o], raw_o, contrib_o_collapsed,
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
        nax_sel = ff.neighbor_axis[sel_n]
        nside_sel = ff.neighbor_side[sel_n]
        raw_n = raw_jump_fp_neighbor[sel_n]

        nc_code_n = ff.neighbor_cube_face[sel_n]
        is_native_n = nc_code_n >= 6
        nax_safe = cp.where(is_native_n, 0, nax_sel)

        adj_mag_n = cp.linalg.norm(ff.neighbor_adj_row_exact[sel_n], axis=-1)
        weighted_n_collapsed = adj_mag_n * raw_n
        contrib_n_collapsed = distribute_face_correction_to_sps(
            cp, weighted_n_collapsed, nax_safe, nside_sel,
            ff.dist_fp_of_sp, ff.dist_axis_coord_of_sp, ff.g_left, ff.g_right,
        )
        contrib_n = _native_or_collapsed_contrib_scalar(
            cp, is_native_n, nc_code_n, ff.lift_native, ff.true_area_weight[sel_n], raw_n, contrib_n_collapsed,
        )
        contrib_n = contrib_n / det_jacs[nc_sel]
        cp.scatter_add(correction, (nc_sel, slice(None)), contrib_n)

    return correction


from autoflowcfd.core.turbulence.transport import resolve_turb_overintegration

# 过积分上下文已提取到 `core/gpu/gpu_overintegration.py`（2026-09-15，
# 粘性体积项 GPU 侧补齐时共用同一份，避免 residual 模块反向依赖
# turbulence 模块）。这里保留原名别名，调用点不变。
from autoflowcfd.core.gpu.gpu_overintegration import (  # noqa: E402
    OVERINT_OPS_KEYS as _OVERINT_KEYS,
    get_overintegration_segs_gpu as _turb_overint_segs_gpu,
)


def _scalar_volume_div_overintegrated_gpu(cp, factors, adj_j_fine, segs,
                                          n_cells, n_sps):
    """标量体积项 `div(adj(J) * prod(factors))` 的去混叠版（GPU）。

    `factors` 是一串 (n_cells, n_sps, k) 的场，k 为 1 或 3；各自精确插值
    到 FINE 点后**在 FINE 点相乘**（去混叠的全部内容就是"先插值再相乘"），
    再用细点度量算逆变通量、细网格微分矩阵求散度、精确限制回 coarse。

    对流用 (rho, phi, u)、扩散用 (gamma, grad_phi)，与 CPU 端
    `_scalar_convection_volume_overintegrated` /
    `_scalar_diffusion_volume_overintegrated` 逐字对应。
    """
    div = cp.zeros((n_cells, n_sps), dtype=cp.float64)
    for lo, hi, c2f, D_fine, f2c in segs:
        if hi <= lo:
            continue
        prod = None
        for f in factors:
            ff_ = gpu_contract_shared_operator_1axis(c2f, f[lo:hi])
            prod = ff_ if prod is None else prod * ff_
        F_tilde = cp.matmul(adj_j_fine[lo:hi], prod[..., None]).squeeze(-1)
        div_f = gpu_contract_shared_operator_2axis(D_fine, F_tilde[..., None])[..., 0]
        div[lo:hi] = gpu_contract_shared_operator_1axis(
            f2c, div_f[..., None])[..., 0]
    return div


def compute_scalar_convection_residual_gpu(
    scalar_field, rho, velocity, mesh_data, ops_data, ff, n_cells, n_prism, n_sps,
    wall_dirichlet_zero_face=None, wall_dirichlet_value_face=None, has_wall_dirichlet_value=None,
):
    """标量对流 FR 残差（体积项 + 界面上风校正），与 CPU 版
    `compute_scalar_convection_residual` 逐字对应。"""
    cp = get_cupy()
    det_jacs = mesh_data['det_jacs']
    adj_j = mesh_data['adj_j']

    # 去混叠（AFCFD_TURB_OVERINT，默认 on）：与 CPU 端
    # `compute_scalar_convection_residual` 同一个开关、同一条链路。
    # 此前 GPU 侧完全没有这一层，导致同一个环境变量在两个后端意味着
    # 不同的数值方案——本项目不接受这种静默不一致（同一原则见
    # fr_solver/filter.py::resolve_filter_mode）。
    _segs = (_turb_overint_segs_gpu(mesh_data, ops_data, n_cells, n_prism)
             if resolve_turb_overintegration() == "on" else None)
    if _segs is not None:
        div_F = _scalar_volume_div_overintegrated_gpu(
            cp, (rho[..., None], scalar_field[..., None], velocity),
            mesh_data['adj_j_fine'], _segs, n_cells, n_sps)
    else:
        rho_u_phi = rho[..., None] * velocity * scalar_field[..., None]  # (n_cells,n_sps,3)
        F_tilde = cp.matmul(adj_j, rho_u_phi[..., None]).squeeze(-1)  # (n_cells,n_sps,3)

        div_F = cp.zeros((n_cells, n_sps), dtype=cp.float64)
        if n_prism > 0:
            div_F[:n_prism] = gpu_contract_shared_operator_2axis(
                ops_data['D_3d_prism'], F_tilde[:n_prism, :, :, None]
            )[..., 0]
        if n_cells > n_prism:
            # native 四面体 D 矩阵分派（2026-09-03 补齐，见模块文档 /
            # gpu_gradients.py::compute_physical_gradient_gpu 同一处判据）。
            D_tet_op = (
                ops_data['D_native_tet_padded']
                if 'D_native_tet_padded' in ops_data
                else ops_data['D_3d_tet']
            )
            div_F[n_prism:] = gpu_contract_shared_operator_2axis(
                D_tet_op, F_tilde[n_prism:, :, :, None]
            )[..., 0]

    residual = -div_F / det_jacs

    rho_o, rho_n = _extrapolate_scalar_to_faces_gpu(cp, ff, n_prism, rho)
    n_fp = rho_o.shape[1]
    vel_o = cp.zeros((ff.n_faces, n_fp, 3), dtype=cp.float64)
    for d in range(3):
        vo, _ = _extrapolate_scalar_to_faces_gpu(cp, ff, n_prism, velocity[..., d])
        vel_o[..., d] = vo
    phi_o, phi_n = _extrapolate_scalar_to_faces_gpu(
        cp, ff, n_prism, scalar_field,
        wall_dirichlet_zero_face, wall_dirichlet_value_face, has_wall_dirichlet_value,
    )

    mass_flux = cp.sum(rho_o[..., None] * vel_o * ff.true_normal, axis=-1)  # (n_faces,n_fp)
    phi_upwind = cp.where(mass_flux >= 0, phi_o, phi_n)
    # 真实 bug 修复（2026-09-12，与 CPU 版 `transport.py::
    # compute_scalar_convection_residual` 同一处修复，完整推导见该函数
    # 模块文档"owner/neighbor 跳变量不对称"一节）：owner/neighbor 两侧
    # 的正确校正必须分别相对各自的面值计算——owner 侧沿用
    # `phi_upwind-phi_o`；neighbor 侧此前错误地复用了同一个 owner 参照
    # 的跳变量，在 owner 恰好是上风侧（mass_flux>=0，phi_upwind==phi_o）
    # 时该跳变量恒为 0，等价于 neighbor（真实网格中占全部内部面一半）
    # 完全收不到这个面本该有的对流稀释/浓缩效果。
    delta_phi_owner = phi_upwind - phi_o
    delta_phi_neighbor = phi_upwind - phi_n

    # 未加权原始跳变量（2026-09-03 修复，见 `_distribute_scalar_correction_
    # gpu` 文档）：不再在这里提前乘 |owner_adj_row_exact|——加权方式（
    # collapsed 用 |adj_row|，native 用 true_area_weight）延后到分配阶段
    # 按面类型分派，与 CPU 版 `raw_jump_fp = mass_flux * delta_phi_owner`
    # 逐字对应。
    raw_jump_fp = mass_flux * delta_phi_owner
    raw_jump_fp_neighbor = mass_flux * delta_phi_neighbor

    interface_correction = _distribute_scalar_correction_gpu(
        cp, ff, raw_jump_fp, det_jacs, n_cells, n_sps,
        raw_jump_fp_neighbor=raw_jump_fp_neighbor,
    )
    return residual + interface_correction


def compute_scalar_diffusion_residual_gpu(
    scalar_field, gamma_field, mesh_data, ops_data, ff, n_cells, n_prism, n_sps,
):
    """标量扩散 FR 残差（体积项 + BR1 梯度差界面校正），与 CPU 版
    `compute_scalar_diffusion_residual` 逐字对应（2026-08-25 梯度差
    符号约定修复后的版本，见该函数文档）。"""
    cp = get_cupy()
    det_jacs = mesh_data['det_jacs']
    adj_j = mesh_data['adj_j']

    grad_phi = compute_physical_scalar_gradient_gpu(scalar_field, mesh_data, ops_data)  # (n_cells,n_sps,3)

    # 去混叠，与 CPU 端 `_scalar_diffusion_volume_overintegrated` 对应
    # （含那边写明的"Gamma 自身混叠仍在"这条已量化、刻意不实施的局限）。
    _segs = (_turb_overint_segs_gpu(mesh_data, ops_data, n_cells, n_prism)
             if resolve_turb_overintegration() == "on" else None)
    if _segs is not None:
        div_G = _scalar_volume_div_overintegrated_gpu(
            cp, (gamma_field[..., None], grad_phi),
            mesh_data['adj_j_fine'], _segs, n_cells, n_sps)
    else:
        G_phys = gamma_field[..., None] * grad_phi
        G_tilde = cp.matmul(adj_j, G_phys[..., None]).squeeze(-1)  # (n_cells,n_sps,3)

        div_G = cp.zeros((n_cells, n_sps), dtype=cp.float64)
        if n_prism > 0:
            div_G[:n_prism] = gpu_contract_shared_operator_2axis(
                ops_data['D_3d_prism'], G_tilde[:n_prism, :, :, None]
            )[..., 0]
        if n_cells > n_prism:
            # native 四面体 D 矩阵分派（2026-09-03 补齐，同上）。
            D_tet_op = (
                ops_data['D_native_tet_padded']
                if 'D_native_tet_padded' in ops_data
                else ops_data['D_3d_tet']
            )
            div_G[n_prism:] = gpu_contract_shared_operator_2axis(
                D_tet_op, G_tilde[n_prism:, :, :, None]
            )[..., 0]

    residual = div_G / det_jacs

    gamma_o, gamma_n = _extrapolate_scalar_to_faces_gpu(cp, ff, n_prism, gamma_field)
    n_fp = gamma_o.shape[1]
    grad_o = cp.zeros((ff.n_faces, n_fp, 3), dtype=cp.float64)
    grad_n = cp.zeros((ff.n_faces, n_fp, 3), dtype=cp.float64)
    for d in range(3):
        go, gn = _extrapolate_scalar_to_faces_gpu(cp, ff, n_prism, grad_phi[..., d])
        grad_o[..., d] = go
        grad_n[..., d] = gn

    gamma_face = 0.5 * (gamma_o + gamma_n)
    delta_grad = 0.5 * (grad_n - grad_o)
    flux_jump_phys = gamma_face * cp.sum(delta_grad * ff.true_normal, axis=-1)

    # 未加权原始跳变量（2026-09-03 修复，同 convection 侧，见
    # `_distribute_scalar_correction_gpu` 文档），与 CPU 版
    # `raw_jump_fp = flux_jump_phys` 逐字对应。
    raw_jump_fp = flux_jump_phys

    interface_correction = _distribute_scalar_correction_gpu(cp, ff, raw_jump_fp, det_jacs, n_cells, n_sps)
    return residual - interface_correction


def compute_omega_wall_target_gpu(cp, ff, wall_mask, wall_distance_gpu, Q_gpu, mu, beta1, omega_max=1e6):
    """CuPy 版 `_compute_omega_wall_target`：omega_wall = 60*nu/(beta1*d1^2)
    （Wilcox 解析式），逐字对应 CPU 版同名函数——全程 GPU 原生实现（不像
    边界幽灵态那样需要 CPU round-trip：wall_distance_gpu/Q_gpu/owner_cell
    都已经常驻显存，没有必要为这个小计算专门下载/上传）。

    真实 bug 修复（2026-09-05，代码复审发现，与 CPU 版
    `transport.py::_compute_omega_wall_target` 2026-09-05 那次真实网格
    验证决定性发现的 bug 同一个根因，此前只修了 CPU 版、GPU 镜像版本
    漏了）：`d1 = cp.maximum(d1, 1e-8)` 只防止除零，不防止结果本身失控
    ——cube_demo 真实网格上确认存在 wall_distance 恰好卡在这个 1e-8
    下限的退化 WALL 面 owner 单元，代入公式算出 `omega_wall~1e14`，比
    "应急安全上限" `omega_max`（SST 模型 `k_max`/`omega_max` 属性，
    默认 1e6）大 8 个数量级。此前唯一的消费者（对流项上风 ghost）碰巧
    被壁面处趋零的对流通量掩盖，没暴露这个缺口；GPU 版
    `enforce_omega_wall_relaxation_gpu`（本文件同日新增，直接把这个值
    混合进 omega_field 本身）不再有这层天然保护，必须在这里、唯一的
    真值来源处夹到 `omega_max`。

    Args:
        wall_mask: (n_faces,) CuPy bool，WALL 边界面掩码（由
            `compute_wall_dirichlet_masks_gpu` 一次性算出并缓存）
        omega_max: 湍流模型的应急安全上限（`solver.turb_model_gpu.omega_max`，
            调用方未提供时退回 1e6，与 CPU 版同一个默认值）

    Returns:
        (omega_wall_value_face, has_value_face)，与 CPU 版返回语义一致
    """
    n_faces = ff.n_faces
    n_fp = ff.boundary_extrap.shape[-2]
    omega_wall_value_face = cp.zeros((n_faces, n_fp), dtype=cp.float64)
    wall_idx = cp.where(wall_mask)[0]
    if wall_idx.shape[0] > 0:
        owner_cells = ff.owner_cell[wall_idx]
        # 长度尺度口径可切换，与 CPU 端 core/turbulence/transport.py::
        # _compute_omega_wall_target 同一处 2026-09-15 发现逐字对应
        # （`min` 既不是单元中心也不是单元高度，会让这个经标定的壁面
        # 函数产生随阶数变化的系统性高估：order=1/2/3 分别 5.60x /
        # 19.68x / 51.86x）。**默认仍为 `min`**，理由见 CPU 端注释。
        import os as _os
        _d1_mode = _os.environ.get("AFCFD_OMEGA_WALL_D1", "min").lower()
        if _d1_mode not in ("min", "mean"):
            raise ValueError(
                f"AFCFD_OMEGA_WALL_D1={_d1_mode!r} 不是合法取值（min | mean）")
        _wd = wall_distance_gpu[owner_cells]
        d1 = _wd.min(axis=1) if _d1_mode == "min" else _wd.mean(axis=1)
        d1 = cp.maximum(d1, 1e-8)
        # 只统计真实自由度，与 CPU 端 transport.py::
        # _compute_omega_wall_target 同一处 2026-09-15 审计逐字对应。
        # 阶数从 SP 数反解（棱柱是张量积 n_sps=(order+1)^3），棱柱数从
        # flat face 几何取——都不需要改本函数签名。
        # 用与 CPU 端**同一个**共享辅助（2026-09-15 统一）：原先这里是
        # 一份手写的等价逻辑，等价性只能靠人工比对维护；共享版还附带
        # "order 与 n_sps 必须自洽"的显式校验。
        from autoflowcfd.fr.native_tet_padding import (
            order_from_n_sps, reduce_rows_over_real_sps,
        )
        _np_prism = getattr(ff, 'n_prism', None)
        _rows = Q_gpu[owner_cells, :, 0]
        _order = order_from_n_sps(_rows.shape[1])
        if _np_prism is None:
            # 拿不到棱柱数就不能判断行的单元类型；此时退回全场平均并
            # 保持既有行为（flat face 几何一定带 n_prism，这条分支只在
            # 测试替身缺字段时才会走到）。
            rho_owner = cp.mean(_rows, axis=1)
        else:
            rho_owner = reduce_rows_over_real_sps(
                _rows, owner_cells < _np_prism, _order, 'mean', xp=cp)
        nu_owner = mu / cp.maximum(rho_owner, 1e-10)
        omega_wall = 60.0 * nu_owner / (beta1 * d1 ** 2)
        omega_wall = cp.minimum(omega_wall, omega_max)
        omega_wall_value_face[wall_idx, :] = omega_wall[:, None]
    return omega_wall_value_face, wall_mask


def enforce_omega_wall_relaxation_gpu(cp, solver, relax=None):
    """CuPy 版 `transport.py::enforce_omega_wall_relaxation`——GPU SST/
    DDES/IDDES 输运路径此前完全没有移植这个修复（2026-09-05 代码复审
    发现，不是本次新引入的差异）：`compute_omega_wall_target_gpu` 算出
    的 Wilcox 解析值此前只喂给对流项上风 ghost（`compute_scalar_
    convection_residual_gpu` 的 `wall_dirichlet_value_face` 参数），
    扩散项同样没有把这个约束传递进去——与 CPU 版被修复前完全同一个
    架构缺口（见 CPU 版 `enforce_omega_wall_relaxation` 文档的完整
    推导：边界层单元 omega 长期不受约束衰减到下界，经 nu_t 近零分母
    奇点放大湍流粘性比，持续向平均流注入过量粘性应力）。GPU SST/DDES/
    IDDES 长期运行（真实生产场景，例如 cube_demo 这类真实网格）会
    出现与 CPU 版修复前完全相同的中长期发散机制。

    实现逐字对应 CPU 版（同样的固定 relax=0.5 事后松弛，不是 CPU 版
    2026-09-05 那次被真实数据证伪撤销的"点隐式"动态松弛——不要重复
    那次已经证伪的尝试，见 CPU 版文档完整失败记录）：`update_fields_gpu`
    之后，对 WALL 面 owner 单元的 `omega_field` 做一次向解析壁面目标值
    的固定比例松弛。`np.add.at`（CPU 版处理"同一 owner 单元是多个 WALL
    面的 owner（角部单元）"）在这里换成 GPU 原生的 `cp.scatter_add`
    （本模块模块文档已说明：正确处理重复索引累加，不需要图着色）。

    Args:
        solver: `GPUFRSolver` 实例
        relax: 松弛系数，默认 0.5（与 CPU 版同一个经验证的安全值）
    """
    if relax is None:
        relax = 0.5
    wall_mask = getattr(solver, "_wall_mask_k_gpu", None)
    if wall_mask is None or not cp.any(wall_mask):
        return

    Q = solver.Q_gpu
    turb = solver.turb_model_gpu
    ff = solver.flat_face_gpu
    omega_max = getattr(turb, "omega_max", 1e6)
    omega_wall_value_face, has_wall = compute_omega_wall_target_gpu(
        cp, ff, wall_mask, solver.wall_distance_gpu, Q, solver.mu_molecular,
        getattr(turb, "beta1", 0.075), omega_max=omega_max,
    )

    wall_face_idx = cp.where(has_wall)[0]
    if wall_face_idx.shape[0] == 0:
        return
    owner_cells = ff.owner_cell[wall_face_idx]
    target = omega_wall_value_face[wall_face_idx, 0]  # 同一面上恒为同一常数，见函数文档

    n_cells = solver.mesh.n_cells
    # 同一个 owner 单元可能是多个 WALL 面的 owner（角部单元）——用
    # scatter_add 累加再除以命中次数取平均目标值，不能直接花式索引赋值
    # 覆盖（后写的面会覆盖先写的面，不是真正的平均），与 CPU 版
    # `np.add.at` 同一个理由。
    sum_target = cp.zeros(n_cells, dtype=cp.float64)
    count = cp.zeros(n_cells, dtype=cp.float64)
    cp.scatter_add(sum_target, owner_cells, target)
    cp.scatter_add(count, owner_cells, 1.0)
    hit_cells = cp.where(count > 0)[0]
    avg_target = sum_target[hit_cells] / count[hit_cells]

    turb.omega_field[hit_cells, :] = (
        (1.0 - relax) * turb.omega_field[hit_cells, :] + relax * avg_target[:, None]
    )


def compute_wall_dirichlet_mask_gpu(mesh, boundary_ghost_provider):
    """CuPy 版 `_compute_wall_dirichlet_face_mask`：纯拓扑查询（哪些面是
    **真实无滑移** WALL 类型边界组），与 wall_distance/流场状态无关，
    只依赖网格自身，求解过程中不变——调用方（gpu_solver_io.py）应该只在
    初始化时调用一次并缓存结果，不是每步都重新算（CPU 版每步都重算是
    因为它本身开销可忽略；这里直接给出 numpy 版本供调用方自行决定何时
    上传/缓存）。

    真实 bug 修复（2026-09-12）：与 CPU 版 `transport.py::_compute_wall_
    dirichlet_face_mask` 同一处、同一理由——`is_no_slip=False` 的滑移壁
    （如 cube_demo 的 "tunnel" 远场边界）物理上零剪切、没有真实边界层，
    不应享受 Wilcox omega 壁面解析式/k 的 Dirichlet-zero 处理，否则会
    把这些单元的 omega 强行拉向物理上荒谬的近壁目标值（完整推导见 CPU
    版同名函数文档）。只把 `is_no_slip` 非 False 的 WALL 编码计入。

    Returns:
        wall_mask: (n_faces,) numpy bool 数组
    """
    import numpy as np
    n_faces = mesh.face_connectivity.n_faces
    group_code = getattr(boundary_ghost_provider, "group_code", None)
    code_to_config = getattr(boundary_ghost_provider, "code_to_config", None)
    if group_code is None or code_to_config is None:
        return np.zeros(n_faces, dtype=np.bool_)
    wall_codes = [
        code for code, cfg in code_to_config.items()
        if cfg.get("type") == "WALL" and cfg.get("is_no_slip", True)
    ]
    if not wall_codes:
        return np.zeros(n_faces, dtype=np.bool_)
    return np.isin(group_code, wall_codes)


def compute_turbulence_transport_residual_gpu(
    solver, grad_vel=None,
) -> Tuple:
    """GPU 版 k/omega 完整输运残差入口（对流+扩散），与 CPU 版
    `compute_turbulence_transport_residual` 逐字对应。

    2026-09-02 补齐：此前这里不做 CPU 版末尾的 `suppress_residual_
    outliers`（troubled_cell.py 的中位数离群值抑制机制3），理由是"该
    函数是纯 numba/numpy 实现，本身就没有 GPU 版本"——但 `gpu_inviscid.
    py::compute_inviscid_residual_fr_gpu` 处理平均流残差的同一个问题
    时，从未真正重新实现一份 GPU 版本，而是直接 `cp.asnumpy` 把残差
    倒回 CPU、调用现成的 numba 版 `suppress_residual_outliers`、再
    `cp.asarray` 传回 GPU——本函数此前没有照抄这个已经在生产路径上使用
    的既有模式，是一处遗漏，不是"没有对应实现"（对应实现本来就不需要
    在 GPU 上重写，跨设备复制小数组的开销远小于跳过这层安全网的风险）。
    现在补齐，与 `gpu_inviscid.py` 同一个模式：小规模跨设备拷贝（形状
    (n_cells,n_sps)，不是大数组，每步 2 次可忽略的 H2D/D2H 往返），
    换来与 CPU 路径逐位一致的离群值抑制行为。

    Args:
        solver: GPUFRSolver 实例，需要 turb_model_gpu 已初始化
            （SST/DDES/IDDES 均可，都是 GPUTurbulenceSST 实例）
        grad_vel: 可选，调用方已经算好的速度梯度（CuPy (n_cells,n_sps,3,3)），
            复用避免重复计算物理梯度这个真实热点，与 CPU 版同名参数
            同一个性能考量。

    Returns:
        (dk_dt_transport, domega_dt_transport)，各自 (n_cells, n_sps)
    """
    cp = get_cupy()
    turb = solver.turb_model_gpu
    n_cells = solver.mesh.n_cells
    n_sps = solver.mesh.n_sps_per_cell
    if turb is None:
        z = cp.zeros((n_cells, n_sps), dtype=cp.float64)
        return z, z

    Q = solver.Q_gpu
    rho = Q[:, :, 0]
    vel = Q[:, :, 1:4]
    mu = solver.mu_molecular
    rho_nu_t = rho * turb.nu_t

    if grad_vel is None:
        # 真实 bug 修复（2026-09-03，同 fr_solver/turbulence.py::
        # compute_turbulence_source 文档同一处）：不能对*守恒*变量 U_gpu
        # 求梯度再切片动量分量冒充速度梯度——`vel`（上面已从 Q 取出）
        # 本来就是真正的速度，直接对它求梯度。
        from autoflowcfd.core.gpu.residual.gpu_gradients import compute_physical_gradient_gpu
        grad_vel = compute_physical_gradient_gpu(vel, solver.mesh_data, solver.ops_data)

    nu = mu / cp.maximum(rho, 1e-10)

    grad_k = compute_physical_scalar_gradient_gpu(turb.k_field, solver.mesh_data, solver.ops_data)
    grad_omega = compute_physical_scalar_gradient_gpu(turb.omega_field, solver.mesh_data, solver.ops_data)

    max_grad_mag = 1e6
    grad_k_mag = cp.linalg.norm(grad_k, axis=-1)
    grad_omega_mag = cp.linalg.norm(grad_omega, axis=-1)
    scale_k = cp.clip(max_grad_mag / cp.maximum(grad_k_mag, 1e-10), 0, 1)
    grad_k = cp.where((grad_k_mag > max_grad_mag)[..., None], grad_k * scale_k[..., None], grad_k)
    scale_omega = cp.clip(max_grad_mag / cp.maximum(grad_omega_mag, 1e-10), 0, 1)
    grad_omega = cp.where(
        (grad_omega_mag > max_grad_mag)[..., None], grad_omega * scale_omega[..., None], grad_omega
    )

    grad_dot = cp.sum(grad_k * grad_omega, axis=-1)
    omega_safe = cp.maximum(turb.omega_field, 1e-10)
    CD_kw = cp.maximum(2.0 * rho * turb.sigma_w2 / omega_safe * grad_dot, 1e-10)

    d_wall = solver.wall_distance_gpu
    F1 = turb.compute_blending_F1_gpu(turb.k_field, turb.omega_field, d_wall, nu, rho, CD_kw)

    sigma_k = F1 * turb.sigma_k1 + (1.0 - F1) * turb.sigma_k2
    sigma_w = F1 * turb.sigma_w1 + (1.0 - F1) * turb.sigma_w2
    gamma_k = mu + sigma_k * rho_nu_t
    gamma_w = mu + sigma_w * rho_nu_t

    ff = solver.flat_face_gpu
    n_prism = solver.mesh_data.get('n_prism', solver.mesh.n_prism_cells)

    wall_mask_k = solver._wall_mask_k_gpu  # 见 gpu_solver_io.py 缓存点文档

    conv_k = compute_scalar_convection_residual_gpu(
        turb.k_field, rho, vel, solver.mesh_data, solver.ops_data, ff, n_cells, n_prism, n_sps,
        wall_dirichlet_zero_face=wall_mask_k,
    )
    diff_k = compute_scalar_diffusion_residual_gpu(
        turb.k_field, gamma_k, solver.mesh_data, solver.ops_data, ff, n_cells, n_prism, n_sps,
    )
    dk_dt_transport = (conv_k + diff_k) / cp.maximum(rho, 1e-10)

    omega_wall_value_face, has_omega_wall = compute_omega_wall_target_gpu(
        cp, ff, wall_mask_k, d_wall, Q, mu, getattr(turb, "beta1", 0.075),
        omega_max=getattr(turb, "omega_max", 1e6),
    )

    conv_w = compute_scalar_convection_residual_gpu(
        turb.omega_field, rho, vel, solver.mesh_data, solver.ops_data, ff, n_cells, n_prism, n_sps,
        wall_dirichlet_value_face=omega_wall_value_face, has_wall_dirichlet_value=has_omega_wall,
    )
    diff_w = compute_scalar_diffusion_residual_gpu(
        turb.omega_field, gamma_w, solver.mesh_data, solver.ops_data, ff, n_cells, n_prism, n_sps,
    )
    domega_dt_transport = (conv_w + diff_w) / cp.maximum(rho, 1e-10)

    # 机制3离群值抑制（2026-09-02 补齐，与 gpu_inviscid.py 同一个"跨设备
    # 拷贝复用 CPU numba 实现"模式，见上方函数文档）——CPU 版
    # reference_field 用的是 k_field/omega_field 自身（不是残差本身），
    # 逐字对应 transport.py 里的同一处调用。
    from autoflowcfd.core.fr_operators.troubled_cell import suppress_residual_outliers
    dk_dt_np = cp.asnumpy(dk_dt_transport)
    domega_dt_np = cp.asnumpy(domega_dt_transport)
    k_field_np = cp.asnumpy(turb.k_field)
    omega_field_np = cp.asnumpy(turb.omega_field)
    dk_dt_np = suppress_residual_outliers(dk_dt_np[:, :, None], k_field_np[:, :, None])[:, :, 0]
    domega_dt_np = suppress_residual_outliers(
        domega_dt_np[:, :, None], omega_field_np[:, :, None]
    )[:, :, 0]
    dk_dt_transport = cp.asarray(dk_dt_np)
    domega_dt_transport = cp.asarray(domega_dt_np)

    dk_dt_transport = cp.where(cp.isfinite(dk_dt_transport), dk_dt_transport, 0.0)
    domega_dt_transport = cp.where(cp.isfinite(domega_dt_transport), domega_dt_transport, 0.0)

    return dk_dt_transport, domega_dt_transport
