"""
AutoFlowCFD V2.0 - 湍流标量输运 numba kernel（阶段一 HPC 优化）

将 `turbulence_transport.py` 中两个纯 Python 串行瓶颈 numba 化：

1. `_extrapolate_scalar_to_faces_kernel`：将 SPs 标量场外插到面通量点
   （owner 侧用 boundary_extrap 矩阵，neighbor 侧用 neighbor_sources 矩阵）
2. `_distribute_correction_to_cells_kernel`：将面通量点校正量分配回 SPs
   （prange + per-thread buffer + sum 归约，与 fr_residual_inviscid_kernel.py
   相同的并行模式）

体积项（BLAS gemm 收缩）保持 numpy 不变——已经是多线程 BLAS 加速。

多核并行约束（与 fr_residual_inviscid_kernel.py 完全一致）：
- `n_threads` 由调用方紧邻调用前取 `numba.get_num_threads()` 传入
- per-thread buffer 内存 = n_threads * n_cells * n_sps * 8 bytes（标量，
  比 5 变量欧拉方程小 5 倍）
- scatter-add 使用 per-thread buffer + sum(axis=0) 归约，避免原子操作
"""

import numpy as np
from numba import njit, prange, get_thread_id


@njit(cache=True)
def _extrap_owner_scalar_to_faces(
    scalar_sps, boundary_extrap,
    owner_cell, owner_axis, owner_side,
    n_prism, n_faces, n_fp, n_sps,
    owner_cube_face, boundary_extrap_native,
):
    """owner 侧标量场外插到面通量点。

    对每个面 f，根据 owner_cell 的单元类型（prism/tet）和 axis/side 选择
    对应的 boundary_extrap 矩阵，将 owner 单元的 SPs 标量值外插到面通量点。

    native 四面体（路径C）真实 bug 修复（2026-08-30，真实 cube_demo 生产
    网格 P2+SST 首次冒烟测试段错误崩溃后发现——整条湍流标量输运链路
    此前完全没有 native 分支，与 fr_residual/viscous_flux_kernel.py 此前
    的遗漏是同一类问题，见该模块修复记录）：`owner_cube_face[f]>=6` 判定
    native 面，改用 `boundary_extrap_native[excluded_vertex]`（不使用
    `owner_axis`/`owner_side`——对 native 面它们是复用槽位哑值，直接
    索引 `boundary_extrap` 会越界读取，是真实的段错误根源）。

    Args:
        scalar_sps: (n_cells, n_sps)
        boundary_extrap: (2, 3, 2, n_fp, n_sps) [celltype, axis, side_idx]
        owner_cell/owner_axis/owner_side: (n_faces,) 面几何数组
        n_prism: 棱柱单元数
        n_faces, n_fp, n_sps: 维度
        owner_cube_face: (n_faces,) 原始 cube face 编码，>=6 即 native 面
        boundary_extrap_native: (4, n_fp, n_sps) native 自身面外插矩阵

    Returns:
        phi_owner_fp: (n_faces, n_fp)
    """
    phi_owner_fp = np.zeros((n_faces, n_fp))
    for f in range(n_faces):
        oc = owner_cell[f]
        oc_code = owner_cube_face[f]
        if oc_code >= 6:
            E = boundary_extrap_native[oc_code - 6]  # (n_fp, n_sps)
        else:
            oax = owner_axis[f]
            oside = owner_side[f]
            oside_idx = 0 if oside <= 0.0 else 1
            celltype_o = 0 if oc < n_prism else 1
            E = boundary_extrap[celltype_o, oax, oside_idx]  # (n_fp, n_sps)
        # phi_owner_fp[f, :] = E @ scalar_sps[oc, :]
        for i in range(n_fp):
            val = 0.0
            for s in range(n_sps):
                val += E[i, s] * scalar_sps[oc, s]
            phi_owner_fp[f, i] = val
    return phi_owner_fp


@njit(cache=True)
def _extrap_neighbor_scalar_to_faces(
    scalar_sps,
    neighbor_src0_cell, neighbor_src0_mat,
    neighbor_src1_idx, neighbor_src1_cell, neighbor_src1_mat,
    n_faces, n_fp, n_sps,
):
    """neighbor 侧标量场外插到面通量点。

    通过 neighbor_sources 矩阵组装 neighbor 侧在面通量点的标量值。
    src0 是稠密槽（每面一个来源），src1 是稀疏槽（部分面有第二个来源）。

    Returns:
        phi_neighbor_fp: (n_faces, n_fp)
    """
    phi_neighbor_fp = np.zeros((n_faces, n_fp))
    for f in range(n_faces):
        c0 = neighbor_src0_cell[f]
        if c0 >= 0:
            mat0 = neighbor_src0_mat[f]  # (n_fp, n_sps)
            for i in range(n_fp):
                val = 0.0
                for s in range(n_sps):
                    val += mat0[i, s] * scalar_sps[c0, s]
                phi_neighbor_fp[f, i] = val

        idx1 = neighbor_src1_idx[f]
        if idx1 >= 0:
            c1 = neighbor_src1_cell[idx1]
            mat1 = neighbor_src1_mat[idx1]  # (n_fp, n_sps)
            for i in range(n_fp):
                val = 0.0
                for s in range(n_sps):
                    val += mat1[i, s] * scalar_sps[c1, s]
                phi_neighbor_fp[f, i] += val
    return phi_neighbor_fp


@njit(cache=True)
def extrapolate_scalar_to_faces_kernel(
    scalar_sps, boundary_extrap,
    neighbor_src0_cell, neighbor_src0_mat,
    neighbor_src1_idx, neighbor_src1_cell, neighbor_src1_mat,
    owner_cell, owner_axis, owner_side,
    n_prism, n_faces, n_fp, n_sps,
    wall_dirichlet_zero_face,
    mixed_nb_partner, mixed_nb_mask,
    has_wall_dirichlet_value,
    wall_dirichlet_value_face,
    owner_cube_face, boundary_extrap_native,
):
    """将 SPs 标量场外插到所有面的通量点（owner + neighbor 两侧）。

    组合 _extrap_owner_scalar_to_faces 和 _extrap_neighbor_scalar_to_faces。

    边界面 ghost 值修复（真实 bug，已修复，真实复现：合成 Couette+SST
    小算例，P0 单个显式步内 omega 场的输运残差就已经达到 ~2e4 量级，
    而当时整个 omega 场仍是完全均匀的初值 1.0——均匀场理论上不该产生
    任何非零输运残差）：`_extrap_neighbor_scalar_to_faces` 对没有真实
    neighbor 单元的边界面（`neighbor_src0_cell[f]==-1` 且没有
    `neighbor_src1`），循环体直接跳过，把 `phi_neighbor_fp` 留在数组
    初始化时的 0——对 WALL/OUTLET/SYMMETRY 任何边界类型都一样，等于
    悄悄把边界处的 k、omega ghost 值当成 0 处理，与这两个物理量的真实
    边界条件毫无关系（尤其 omega 在近壁应该很大，不是 0）。与平均流
    残差组装不同，这里从未接入 `solver.boundary_ghost_provider`——这个
    人工的"ghost=0"在扩散残差里表现为一个恒定虚假跳跃
    delta_phi=0.5*(0-phi_owner)，不随场是否已经物理收敛而消失，从第一
    步开始就往湍流标量方程里注入等效于虚假源项的边界通量，是本文档
    开头描述的 P0->P1->P2 残差链式失控的最初触发点（后续被 CFL/湍流
    粘性比等其它环节放大，但源头在这里）。
    真正实现每种边界类型各自正确的 k/omega ghost 值（WALL 上 k=0、
    omega 按 Wilcox 解析式随近壁距离变化，OUTLET/FARFIELD 用来流值，
    SYMMETRY 零法向梯度等）需要把 boundary_ghost_provider 的分组信息
    接入这个纯标量输运模块。这里先修正最基本、对所有边界类型都成立的
    最小合理默认——零梯度（Neumann）ghost：ghost 值取 owner 侧外插值
    本身，边界处 delta_phi=0，不再凭空引入虚假跳跃/虚假源项。这不是
    "引入新阈值掩盖问题"，是把一个连"当前场是否收敛"都不敏感、对任何
    输入都会触发的错误 ghost 值改成没有额外假设的中性默认（对 SYMMETRY
    严格物理正确；对 OUTLET/FARFIELD 是比"隐式当作0"更保守、更不容易
    引入数值毛刺的近似，目前仍是这两种边界类型的实际处理方式）。
    **WALL 上 k/omega 的解析 Dirichlet 值已经补齐**（见下方两段，
    2026-08-21/2026-08-28 两次真实修复）——本段开头列出的"真正实现每种
    边界类型各自正确的 ghost 值"这个目标，WALL 部分已完成，OUTLET/
    FARFIELD/SYMMETRY 仍是本段描述的 Neumann 默认（SYMMETRY 本来就是
    严格正确的物理边界条件，不需要另外补齐；OUTLET/FARFIELD 用来流值
    这项仍是未来可选的精化方向，不是当前已知有问题的近似）。

    WALL 上 k=0 的 Dirichlet 处理（真实修复，2026-08-21）：k 在无滑移
    壁面上严格为零，这是标准 k-omega/SST 边界条件（Wilcox《Turbulence
    Modeling for CFD》），不是近似——不需要任何额外输入，可以用与均流场
    无滑移壁面幽灵态完全相同的镜像手法独立实现：ghost = 2*k_wall -
    k_owner = -k_owner（k_wall=0）。`wall_dirichlet_zero_face[f]` 由
    调用方（`compute_turbulence_transport_residual`，据 solver.boundary_
    ghost_provider 的边界分组信息算出）标记这个面是否要用这种
    Dirichlet-zero 处理而不是上面的 Neumann 默认——只在对 k 场调用本
    kernel 时为 True，omega/rho/velocity/gamma_field 等其它场的调用
    仍传全 False 数组，行为不变。

    WALL 上 omega 解析壁面值的 Dirichlet 处理（真实修复，V2.0 专家组
    盲审发现，2026-08-28）：omega 在壁面的正确边界条件不是零梯度，而是
    Wilcox 解析式 `omega_wall = 60*nu/(beta1*d1^2)`（d1=壁面到第一层
    SP 的距离）——此前因为"需要额外的壁面距离数据"被搁置，现在
    `has_wall_dirichlet_value[f]`/`wall_dirichlet_value_face[f,:]` 由
    调用方按同一套 wall_mask 逻辑、外插 solver.wall_distance 到壁面算出
    （见 compute_turbulence_transport_residual 里 `_compute_omega_wall_
    target` 的说明）提供：`ghost = 2*target - owner`，与 k=0 分支同一个
    通式（target=0 时就是 k 的情形），只是这里 target 逐 FP 给定而非
    恒为 0。与 `wall_dirichlet_zero_face` 互斥（一个面只会被两者之一
    标记）；omega/velocity/gamma_field 等不适用的场调用本 kernel 时，
    `has_wall_dirichlet_value` 传全 False 数组，行为与此前完全一致。
    """
    phi_owner = _extrap_owner_scalar_to_faces(
        scalar_sps, boundary_extrap,
        owner_cell, owner_axis, owner_side,
        n_prism, n_faces, n_fp, n_sps,
        owner_cube_face, boundary_extrap_native,
    )
    phi_neighbor = _extrap_neighbor_scalar_to_faces(
        scalar_sps,
        neighbor_src0_cell, neighbor_src0_mat,
        neighbor_src1_idx, neighbor_src1_cell, neighbor_src1_mat,
        n_faces, n_fp, n_sps,
    )
    for f in range(n_faces):
        if neighbor_src0_cell[f] < 0 and neighbor_src1_idx[f] < 0:
            if wall_dirichlet_zero_face[f]:
                for i in range(n_fp):
                    phi_neighbor[f, i] = -phi_owner[f, i]
            elif has_wall_dirichlet_value[f]:
                # 非零 Dirichlet 目标值（例如 omega 壁面解析式，见
                # compute_turbulence_transport_residual 调用处文档）：
                # ghost = 2*target - owner，使 (ghost+owner)/2 = target
                # 恰好等于目标值——k=0 的 Dirichlet-zero 分支是这个通式
                # target=0 的特例，两者用同一套镜像原理，只是这里 target
                # 逐 FP 给定而不是恒为 0。
                for i in range(n_fp):
                    phi_neighbor[f, i] = 2.0 * wall_dirichlet_value_face[f, i] - phi_owner[f, i]
            else:
                for i in range(n_fp):
                    phi_neighbor[f, i] = phi_owner[f, i]
    # 混合拆分面（B-8，见 fr/face_flux_points/merge.py）：混合配对的内部面在边界半区
    # 没有真实邻居单元，neighbor 侧值按配对边界面自身获得的同一规则逐 FP 覆盖——
    # Dirichlet-zero 壁面用镜像（-phi_owner），非零 Dirichlet 用配对面自己的
    # target 值镜像，其余用零梯度（phi_owner）；内部半区保持上方多源插值结果不动。
    for f in range(n_faces):
        mp = mixed_nb_partner[f]
        if mp >= 0:
            dirichlet = wall_dirichlet_zero_face[mp]
            has_value = has_wall_dirichlet_value[mp]
            for i in range(n_fp):
                if mixed_nb_mask[f, i]:
                    if dirichlet:
                        phi_neighbor[f, i] = -phi_owner[f, i]
                    elif has_value:
                        phi_neighbor[f, i] = 2.0 * wall_dirichlet_value_face[mp, i] - phi_owner[f, i]
                    else:
                        phi_neighbor[f, i] = phi_owner[f, i]
    return phi_owner, phi_neighbor


@njit(cache=True)
def _distribute_point_scalar(fp_data, fp_of_sp_axis, axis_coord_of_sp_axis, g_prime):
    """将面通量点数据分配到 SPs（标量版本）。

    与 fr_residual_inviscid_kernel.py::_distribute_point 相同逻辑，
    但处理标量（n_vars=1）而非 5 变量。

    fp_data: (n_fp,); 输出 (n_sps,)
    """
    n_sps = fp_of_sp_axis.shape[0]
    out = np.zeros(n_sps)
    for s in range(n_sps):
        fp_i = fp_of_sp_axis[s]
        g = g_prime[axis_coord_of_sp_axis[s]]
        out[s] = g * fp_data[fp_i]
    return out


@njit(cache=True, inline='always')
def _weighted_jump_collapsed(raw_jump_f, adj_row_f, n_fp):
    """collapsed 面：跳变量按 |adj_row|（度量张量的 adj 行模长，即体积项
    里 `adj(J)` 逆变通量投影用的同一个度量因子）加权——与
    inviscid_kernel.py/viscous_flux_kernel.py 送进 `_distribute_point`/
    lift 之前"物理通量密度差 × 面元幅值因子"是同一原则，这里只是标量场
    版本。"""
    weighted = np.empty(n_fp)
    for i in range(n_fp):
        rx = adj_row_f[i, 0]
        ry = adj_row_f[i, 1]
        rz = adj_row_f[i, 2]
        adj_mag = np.sqrt(rx * rx + ry * ry + rz * rz)
        weighted[i] = adj_mag * raw_jump_f[i]
    return weighted


@njit(cache=True, inline='always')
def _weighted_jump_native(raw_jump_f, true_area_weight_f, n_fp):
    """native 面：跳变量按真实**物理面积**权重加权，供 DG 提升算子消费。

    **本函数与 inviscid/viscous 那两路的权重不同，而且两边都是对的** ——
    差别在传进来的 `raw_jump` 处在哪个空间（2026-09-18 核实）：

      * 本路（湍流标量输运）：`raw_jump = mass_flux * delta_phi` 是**物理**
        通量密度差（见 `transport.py` 里 `raw_jump_fp` 的构造），所以
        坍缩分支要自己乘 `|adj_row|`（`_weighted_jump_collapsed`）、
        native 分支乘**物理面积权重** `true_area_weight = |adj_row|*w_ref`
        —— 正好是弱形式 `∮ Psi * jump dA` 的正确离散。
      * inviscid/viscous 两路：`jump = adj_row . (F* - F_own)` 已经是
        **参考空间**的法向通量差，所以那边的正确权重是**参考**求积权重
        `ref_area_weight`（那处原先误用物理面积权重、多乘一个 `|adj_row|`，
        已修，完整记录见 `face_kernels.py::FlatFaceGeometry.ref_area_weight`）。

    两路的最终乘积其实是同一个量 `|adj_row| * w_ref * 物理跳跃`，只是
    `|adj_row|` 由谁提供不同。改动任一路之前先看清 `raw_jump` 在哪个空间。
    """
    weighted = np.empty(n_fp)
    for i in range(n_fp):
        weighted[i] = true_area_weight_f[i] * raw_jump_f[i]
    return weighted


@njit(cache=True, parallel=True)
def distribute_corrections_to_cells_kernel(
    raw_jump_fp,
    owner_cell, neighbor_cell,
    owner_axis, owner_side,
    neighbor_axis, neighbor_side,
    det_jacs,
    g_left, g_right,
    dist_fp_of_sp, dist_axis_coord_of_sp,
    n_cells, n_sps, n_faces,
    n_threads,
    owner_cube_face, neighbor_cube_face,
    owner_adj_row_exact, neighbor_adj_row_exact,
    true_area_weight,
    lift_native,
    owner_is_primary, neighbor_is_primary,
    raw_jump_fp_neighbor,
):
    """将面通量点校正量分配回 SPs（prange + per-thread buffer）。

    对每个面 f：
    - owner 侧：correction[oc, sp] -= weighted_jump[f, ...] 分配 / detJ[oc]
    - neighbor 侧（内部面）：correction[nc, sp] += weighted_jump[f, ...] 分配 / detJ[nc]

    使用 per-thread buffer 避免 scatter-add 写冲突，最后 sum(axis=0) 归约。

    native 四面体（路径C）真实 bug 修复（2026-08-30，见
    `_extrap_owner_scalar_to_faces` 同名记录，真实 cube_demo P2+SST 段
    错误崩溃后发现）：此前本函数直接接收"已经预乘 |adj_row| 面元幅值
    因子"的 `correction_fp`，用 `owner_axis`/`owner_side` 索引
    `g_left`/`g_right`/`dist_fp_of_sp` 做 1D 修正函数分布——这套机制
    对 native 单纯形基完全不适用（无"坍缩计算方向"概念，`owner_axis`/
    `owner_side` 对 native 面是复用槽位哑值，索引会越界，是真实的段
    错误根源）。改为接收**未加权**的原始物理跳变量 `raw_jump_fp`（不
    含任何面元幅值因子），按 `owner_cube_face[f]>=6` 分派：collapsed
    面在这里才乘 `|owner_adj_row_exact|` 加权、走原有 1D distribute；
    native 面乘 `true_area_weight` 加权、走 `lift_native[excluded_
    vertex]` DG 提升算子——两条分支算出的 `contrib_owner`/`contrib_
    neighbor` 语义（"待除以 det(J) 累加进 correction 的贡献量"）完全
    一致，只是加权方式和分配矩阵不同，调用方（`transport.py`）外部的
    owner"-="/neighbor"+="符号约定不受影响。

    Args:
        raw_jump_fp: (n_faces, n_fp) 未加权的原始物理通量密度/梯度跳变
        owner_cell, neighbor_cell: (n_faces,) int64
        owner_axis, owner_side: (n_faces,)（仅 collapsed 面使用）
        neighbor_axis, neighbor_side: (n_faces,)（仅 collapsed 面使用）
        det_jacs: (n_cells, n_sps)
        g_left, g_right: (n1d,) 校正函数导数（仅 collapsed 面使用）
        dist_fp_of_sp: (3, n_sps) int64（仅 collapsed 面使用）
        dist_axis_coord_of_sp: (3, n_sps) int64（仅 collapsed 面使用）
        n_cells, n_sps, n_faces: int
        n_threads: int（调用方从 numba.get_num_threads() 取值传入）
        owner_cube_face, neighbor_cube_face: (n_faces,) 原始 cube face
            编码，>=6 即 native 面
        owner_adj_row_exact, neighbor_adj_row_exact: (n_faces, n_fp, 3)
            逐 FP 精确 adj 行（collapsed 面加权用）
        true_area_weight: (n_faces, n_fp) 物理面积权重（native 面加权用；
            本路的 raw_jump 是物理量，所以这里**就该**是物理面积权重，
            见 `_weighted_jump_native` 文档里两路的对比）
        lift_native: (4, n_sps, n_fp) native DG 提升算子

    Returns:
        correction_sps: (n_cells, n_sps)
    """
    correction_per_thread = np.zeros((n_threads, n_cells, n_sps))

    for f in prange(n_faces):
        tid = get_thread_id()

        # --- owner 侧分配（真实 bug 修复，2026-09-02，实现分布式湍流
        # 模型时用非均匀流场端到端测试发现——此前本函数无条件对每个面
        # 都写 owner 侧贡献，完全没有 owner_is_primary 过滤，与
        # inviscid_kernel.py/viscous_flux_kernel.py 早已验证过、必须
        # 有的"owner-primary/neighbor-primary 两段式累加"约定不一致
        # （B-8 混合拆分面场景——同一个物理面被拆成2条记录，只有
        # owner_is_primary=True 的那条记录才应该贡献 owner 侧——本函数
        # 此前对两条记录都无条件累加，等价于把这批面的 owner 侧贡献
        # 重复计入 2 次）。这不是分布式专属问题，单机路径遇到 B-8
        # 混合拆分面同样会中招，只是没有被现有测试覆盖到。）
        if owner_is_primary[f]:
            oc = owner_cell[f]
            oc_code = owner_cube_face[f]
            o_is_native = oc_code >= 6
            if o_is_native:
                weighted_o = _weighted_jump_native(raw_jump_fp[f], true_area_weight[f], raw_jump_fp.shape[1])
                contrib_owner = lift_native[oc_code - 6] @ weighted_o  # (n_sps,)
            else:
                oax = owner_axis[f]
                oside = owner_side[f]
                weighted_o = _weighted_jump_collapsed(raw_jump_fp[f], owner_adj_row_exact[f], raw_jump_fp.shape[1])
                g_prime_owner = g_right if oside > 0 else g_left
                fp_ids_owner = dist_fp_of_sp[oax]
                axis_coords_owner = dist_axis_coord_of_sp[oax]
                contrib_owner = _distribute_point_scalar(
                    weighted_o, fp_ids_owner, axis_coords_owner, g_prime_owner
                )
            for s in range(n_sps):
                dj = det_jacs[oc, s]
                correction_per_thread[tid, oc, s] -= contrib_owner[s] / dj

        # --- neighbor 侧分配（内部面，同一处修复；跳变量改用
        # raw_jump_fp_neighbor，理由见 distribute_corrections_to_cells_
        # kernel_colored 同名参数文档）---
        nc = neighbor_cell[f]
        if nc >= 0 and neighbor_is_primary[f]:
            nc_code = neighbor_cube_face[f]
            n_is_native = nc_code >= 6
            if n_is_native:
                weighted_n = _weighted_jump_native(raw_jump_fp_neighbor[f], true_area_weight[f], raw_jump_fp.shape[1])
                contrib_neighbor = lift_native[nc_code - 6] @ weighted_n
            else:
                nax = neighbor_axis[f]
                nside = neighbor_side[f]
                weighted_n = _weighted_jump_collapsed(raw_jump_fp_neighbor[f], neighbor_adj_row_exact[f], raw_jump_fp.shape[1])
                g_prime_neighbor = g_right if nside > 0 else g_left
                fp_ids_neighbor = dist_fp_of_sp[nax]
                axis_coords_neighbor = dist_axis_coord_of_sp[nax]
                contrib_neighbor = _distribute_point_scalar(
                    weighted_n, fp_ids_neighbor, axis_coords_neighbor, g_prime_neighbor
                )
            for s in range(n_sps):
                dj = det_jacs[nc, s]
                correction_per_thread[tid, nc, s] += contrib_neighbor[s] / dj

    return correction_per_thread.sum(axis=0)


@njit(cache=True, parallel=True)
def distribute_corrections_to_cells_kernel_colored(
    raw_jump_fp,
    owner_cell, neighbor_cell,
    owner_axis, owner_side,
    neighbor_axis, neighbor_side,
    det_jacs,
    g_left, g_right,
    dist_fp_of_sp, dist_axis_coord_of_sp,
    n_cells, n_sps,
    face_indices,    # 当前颜色组的面索引
    correction_sps,  # 共享输出 buffer（同色面无冲突，直接写入）
    owner_cube_face, neighbor_cube_face,
    owner_adj_row_exact, neighbor_adj_row_exact,
    true_area_weight,
    lift_native,
    owner_is_primary, neighbor_is_primary,
    raw_jump_fp_neighbor,
):
    """图着色版本的标量校正分配 kernel。

    与 distribute_corrections_to_cells_kernel 相同逻辑（含 native 四面体
    分支，两处必须同步修改，见该函数模块文档），但：
    1. 只处理 face_indices 指定的面（当前颜色组）
    2. 直接写入共享 correction_sps buffer（同色面无冲突）
    3. 无需 per-thread buffer 和 sum 归约

    调用方按颜色循环调用此函数，每种颜色处理约 n_faces/n_colors 个面。
    内存从 O(n_threads * n_cells * n_sps) 降至 O(n_cells * n_sps)。

    raw_jump_fp_neighbor（真实 bug 修复，2026-09-12，见
    `compute_scalar_convection_residual` 模块文档"owner/neighbor 跳变量
    不对称"一节完整推导）：neighbor 侧的正确校正必须相对 neighbor 自己的
    面值 phi_neighbor_fp 计算（`mass_flux*(phi_upwind-phi_neighbor_fp)`），
    不能复用 owner 侧那份相对 phi_owner_fp 算出的 `raw_jump_fp`——两者只在
    phi_owner_fp==phi_neighbor_fp（面上无跳变）时才恰好相等。调用方
    （`_distribute_correction_to_cells`）现在总是传入这个独立数组；对
    `compute_scalar_diffusion_residual`（跳变量本身按 BR1 公共梯度定义、
    对 owner/neighbor 天然对称）调用方直接传 `raw_jump_fp_neighbor=
    raw_jump_fp`（同一个数组），保持该调用方数值结果不变。
    """
    n_faces_in_color = face_indices.shape[0]
    n_fp = raw_jump_fp.shape[1]

    for fi in prange(n_faces_in_color):
        f = face_indices[fi]

        # --- owner 侧分配（真实 bug 修复，2026-09-02，与
        # distribute_corrections_to_cells_kernel 同一处、同一理由，见
        # 该函数模块文档"owner 侧分配"注释）---
        if owner_is_primary[f]:
            oc = owner_cell[f]
            oc_code = owner_cube_face[f]
            o_is_native = oc_code >= 6
            if o_is_native:
                weighted_o = _weighted_jump_native(raw_jump_fp[f], true_area_weight[f], n_fp)
                contrib_owner = lift_native[oc_code - 6] @ weighted_o
            else:
                oax = owner_axis[f]
                oside = owner_side[f]
                weighted_o = _weighted_jump_collapsed(raw_jump_fp[f], owner_adj_row_exact[f], n_fp)
                g_prime_owner = g_right if oside > 0 else g_left
                fp_ids_owner = dist_fp_of_sp[oax]
                axis_coords_owner = dist_axis_coord_of_sp[oax]
                contrib_owner = _distribute_point_scalar(
                    weighted_o, fp_ids_owner, axis_coords_owner, g_prime_owner
                )
            for s in range(n_sps):
                dj = det_jacs[oc, s]
                correction_sps[oc, s] -= contrib_owner[s] / dj

        # --- neighbor 侧分配（内部面，同一处修复；跳变量改用
        # raw_jump_fp_neighbor，见上方函数文档）---
        nc = neighbor_cell[f]
        if nc >= 0 and neighbor_is_primary[f]:
            nc_code = neighbor_cube_face[f]
            n_is_native = nc_code >= 6
            if n_is_native:
                weighted_n = _weighted_jump_native(raw_jump_fp_neighbor[f], true_area_weight[f], n_fp)
                contrib_neighbor = lift_native[nc_code - 6] @ weighted_n
            else:
                nax = neighbor_axis[f]
                nside = neighbor_side[f]
                weighted_n = _weighted_jump_collapsed(raw_jump_fp_neighbor[f], neighbor_adj_row_exact[f], n_fp)
                g_prime_neighbor = g_right if nside > 0 else g_left
                fp_ids_neighbor = dist_fp_of_sp[nax]
                axis_coords_neighbor = dist_axis_coord_of_sp[nax]
                contrib_neighbor = _distribute_point_scalar(
                    weighted_n, fp_ids_neighbor, axis_coords_neighbor, g_prime_neighbor
                )
            for s in range(n_sps):
                dj = det_jacs[nc, s]
                correction_sps[nc, s] += contrib_neighbor[s] / dj


@njit(cache=True, parallel=True)
def scalar_convection_volume_kernel(phi, rho_u_tilde, op_D, out) -> None:
    """标量对流体积项散度：`out[c,s] = sum_m sum_j op_D[s,j,m] * phi[c,j] * rho_u_tilde[c,j,m]`。

    性能优化（2026-09-13，用户反馈"每步耗时过长、对 CPU 核数不敏感"后的
    真实剖析结论，见 `compute_scalar_convection_residual` 文档"共享几何量"
    一节）：此前体积项在 Python 层按块做
    `adj_j_chunk = det*inv_jacs` -> `rho_u_phi = rho*vel*phi` ->
    `np.matmul(adj_j_chunk, rho_u_phi)` -> 3 次 `np.tensordot` 累加，
    每一步都物化一份块大小的中间数组、且 matmul 那步是逐点 3x3@3x1 的
    批量微型 gemm（不随核数并行）。

    这里把整条链压进一个按 cell `prange` 的 kernel，配合调用方预先算好的
    **与标量无关**的逆变质量通量 `rho_u_tilde`（= adj(J) @ (rho*u)，k 和
    omega 两次调用共享同一份，见调用方文档）：
      F_tilde[c,j,m] = phi[c,j] * rho_u_tilde[c,j,m]
    （数学恒等式——phi 在该点是标量，可以从度量乘法里提出来）
    工作集只有逐 cell 的几十个 double，留在 L1 内；无任何大中间数组。

    Args:
        phi: (n_cells, n_sps) 标量场（k 或 omega）
        rho_u_tilde: (n_cells, n_sps, 3) 逆变质量通量 adj(J) @ (rho*u)
        op_D: (n_sps, n_sps, 3) 该单元类型的微分矩阵
        out: (n_cells, n_sps) 输出（调用方按 prism/tet 段分别传入切片）
    """
    C, S = phi.shape
    for c in prange(C):
        for s in range(S):
            tot = 0.0
            for m in range(3):
                acc = 0.0
                for j in range(S):
                    acc += op_D[s, j, m] * phi[c, j] * rho_u_tilde[c, j, m]
                tot += acc
            out[c, s] = tot


@njit(cache=True, parallel=True)
def scalar_volume_divergence_kernel(tilde, op_D, out) -> None:
    """逆变通量的体积项散度：`out[c,s] = sum_m sum_j op_D[s,j,m] * tilde[c,j,m]`。

    与 `scalar_convection_volume_kernel` 的区别只是没有 phi 因子——对流项
    可以把标量提到度量乘法外面（见该 kernel 文档），扩散项的逆变通量
    `adj(J) @ (Gamma*grad(phi))` 里 Gamma 与 grad(phi) 都随标量变化、提不
    出来，只能先算好 `tilde` 再收缩。

    性能优化（2026-09-13，与对流项同一次剖析）：原实现是 Python 层按块
    `adj_j = det*inv_jacs` -> `np.matmul(adj_j, G_phys[...,None])` ->
    3 次 `np.tensordot` 累加，逐块物化 3 份中间数组、且 matmul 那步是逐点
    3x3@3x1 微型 gemm（不随核数并行）。

    Args:
        tilde: (n_cells, n_sps, 3) 逆变通量（调用方用
            `volume_contract.contravariant_flux_from_metric` 算好）
        op_D: (n_sps, n_sps, 3) 该单元类型的微分矩阵
        out: (n_cells, n_sps) 输出（调用方按 prism/tet 段分别传入切片）
    """
    C, S = out.shape
    for c in prange(C):
        for s in range(S):
            tot = 0.0
            for m in range(3):
                acc = 0.0
                for j in range(S):
                    acc += op_D[s, j, m] * tilde[c, j, m]
                tot += acc
            out[c, s] = tot
