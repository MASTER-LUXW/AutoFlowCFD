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
):
    """owner 侧标量场外插到面通量点。

    对每个面 f，根据 owner_cell 的单元类型（prism/tet）和 axis/side 选择
    对应的 boundary_extrap 矩阵，将 owner 单元的 SPs 标量值外插到面通量点。

    Args:
        scalar_sps: (n_cells, n_sps)
        boundary_extrap: (2, 3, 2, n_fp, n_sps) [celltype, axis, side_idx]
        owner_cell/owner_axis/owner_side: (n_faces,) 面几何数组
        n_prism: 棱柱单元数
        n_faces, n_fp, n_sps: 维度

    Returns:
        phi_owner_fp: (n_faces, n_fp)
    """
    phi_owner_fp = np.zeros((n_faces, n_fp))
    for f in range(n_faces):
        oc = owner_cell[f]
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
    接入这个纯标量输运模块，是一项更大的独立工作。这里先修正最基本、
    对所有边界类型都成立的最小合理默认——零梯度（Neumann）ghost：
    ghost 值取 owner 侧外插值本身，边界处 delta_phi=0，不再凭空引入
    虚假跳跃/虚假源项。这不是"引入新阈值掩盖问题"，是把一个连"当前
    场是否收敛"都不敏感、对任何输入都会触发的错误 ghost 值改成没有
    额外假设的中性默认（对 SYMMETRY 严格物理正确；对 OUTLET/WALL 是
    比"隐式当作0"更保守、更不容易引入数值毛刺的近似，真正的解析壁面
    condition 留待后续工作补齐)。

    WALL 上 k=0 的 Dirichlet 处理（真实修复，2026-08-21）：k 在无滑移
    壁面上严格为零，这是标准 k-omega/SST 边界条件（Wilcox《Turbulence
    Modeling for CFD》），不是近似——不像 omega 解析壁面值需要额外的
    壁面距离数据，k=0 不需要任何额外输入，可以用与均流场无滑移壁面
    幽灵态完全相同的镜像手法独立实现：ghost = 2*k_wall - k_owner =
    -k_owner（k_wall=0）。`wall_dirichlet_zero_face[f]` 由调用方
    （`compute_turbulence_transport_residual`，据 solver.boundary_
    ghost_provider 的边界分组信息算出）标记这个面是否要用这种
    Dirichlet-zero 处理而不是上面的 Neumann 默认——只在对 k 场调用本
    kernel 时为 True，omega/rho/velocity/gamma_field 等其它场的调用
    仍传全 False 数组，行为不变。
    """
    phi_owner = _extrap_owner_scalar_to_faces(
        scalar_sps, boundary_extrap,
        owner_cell, owner_axis, owner_side,
        n_prism, n_faces, n_fp, n_sps,
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
            else:
                for i in range(n_fp):
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


@njit(cache=True, parallel=True)
def distribute_corrections_to_cells_kernel(
    correction_fp,
    owner_cell, neighbor_cell,
    owner_axis, owner_side,
    neighbor_axis, neighbor_side,
    det_jacs,
    g_left, g_right,
    dist_fp_of_sp, dist_axis_coord_of_sp,
    n_cells, n_sps, n_faces,
    n_threads,
):
    """将面通量点校正量分配回 SPs（prange + per-thread buffer）。

    对每个面 f：
    - owner 侧：correction[oc, sp] -= correction_fp[f, ...] * g'_owner / detJ[oc]
    - neighbor 侧（内部面）：correction[nc, sp] += correction_fp[f, ...] * g'_neighbor / detJ[nc]

    使用 per-thread buffer 避免 scatter-add 写冲突，最后 sum(axis=0) 归约。

    Args:
        correction_fp: (n_faces, n_fp) 面通量点校正量
        owner_cell, neighbor_cell: (n_faces,) int64
        owner_axis, owner_side: (n_faces,)
        neighbor_axis, neighbor_side: (n_faces,)
        det_jacs: (n_cells, n_sps)
        g_left, g_right: (n1d,) 校正函数导数
        dist_fp_of_sp: (3, n_sps) int64
        dist_axis_coord_of_sp: (3, n_sps) int64
        n_cells, n_sps, n_faces: int
        n_threads: int（调用方从 numba.get_num_threads() 取值传入）

    Returns:
        correction_sps: (n_cells, n_sps)
    """
    correction_per_thread = np.zeros((n_threads, n_cells, n_sps))

    for f in prange(n_faces):
        tid = get_thread_id()
        oc = owner_cell[f]
        oax = owner_axis[f]
        oside = owner_side[f]

        # --- owner 侧分配 ---
        g_prime_owner = g_right if oside > 0 else g_left
        fp_ids_owner = dist_fp_of_sp[oax]
        axis_coords_owner = dist_axis_coord_of_sp[oax]

        contrib_owner = _distribute_point_scalar(
            correction_fp[f], fp_ids_owner, axis_coords_owner, g_prime_owner
        )
        for s in range(n_sps):
            dj = det_jacs[oc, s]
            correction_per_thread[tid, oc, s] -= contrib_owner[s] / dj

        # --- neighbor 侧分配（内部面）---
        nc = neighbor_cell[f]
        if nc >= 0:
            nax = neighbor_axis[f]
            nside = neighbor_side[f]
            g_prime_neighbor = g_right if nside > 0 else g_left

            fp_ids_neighbor = dist_fp_of_sp[nax]
            axis_coords_neighbor = dist_axis_coord_of_sp[nax]

            contrib_neighbor = _distribute_point_scalar(
                correction_fp[f], fp_ids_neighbor, axis_coords_neighbor, g_prime_neighbor
            )
            for s in range(n_sps):
                dj = det_jacs[nc, s]
                correction_per_thread[tid, nc, s] += contrib_neighbor[s] / dj

    return correction_per_thread.sum(axis=0)


@njit(cache=True, parallel=True)
def distribute_corrections_to_cells_kernel_colored(
    correction_fp,
    owner_cell, neighbor_cell,
    owner_axis, owner_side,
    neighbor_axis, neighbor_side,
    det_jacs,
    g_left, g_right,
    dist_fp_of_sp, dist_axis_coord_of_sp,
    n_cells, n_sps,
    face_indices,    # 当前颜色组的面索引
    correction_sps,  # 共享输出 buffer（同色面无冲突，直接写入）
):
    """图着色版本的标量校正分配 kernel。

    与 distribute_corrections_to_cells_kernel 相同逻辑，但：
    1. 只处理 face_indices 指定的面（当前颜色组）
    2. 直接写入共享 correction_sps buffer（同色面无 owner_cell 冲突）
    3. 无需 per-thread buffer 和 sum 归约

    调用方按颜色循环调用此函数，每种颜色处理约 n_faces/n_colors 个面。
    内存从 O(n_threads * n_cells * n_sps) 降至 O(n_cells * n_sps)。
    """
    n_faces_in_color = face_indices.shape[0]

    for fi in prange(n_faces_in_color):
        f = face_indices[fi]
        oc = owner_cell[f]
        oax = owner_axis[f]
        oside = owner_side[f]

        # --- owner 侧分配 ---
        g_prime_owner = g_right if oside > 0 else g_left
        fp_ids_owner = dist_fp_of_sp[oax]
        axis_coords_owner = dist_axis_coord_of_sp[oax]

        contrib_owner = _distribute_point_scalar(
            correction_fp[f], fp_ids_owner, axis_coords_owner, g_prime_owner
        )
        for s in range(n_sps):
            dj = det_jacs[oc, s]
            correction_sps[oc, s] -= contrib_owner[s] / dj

        # --- neighbor 侧分配（内部面）---
        nc = neighbor_cell[f]
        if nc >= 0:
            nax = neighbor_axis[f]
            nside = neighbor_side[f]
            g_prime_neighbor = g_right if nside > 0 else g_left

            fp_ids_neighbor = dist_fp_of_sp[nax]
            axis_coords_neighbor = dist_axis_coord_of_sp[nax]

            contrib_neighbor = _distribute_point_scalar(
                correction_fp[f], fp_ids_neighbor, axis_coords_neighbor, g_prime_neighbor
            )
            for s in range(n_sps):
                dj = det_jacs[nc, s]
                correction_sps[nc, s] += contrib_neighbor[s] / dj
