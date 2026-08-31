"""无粘残差界面项 —— numba 逐点标量 kernel (性能优化，替代
fr_residual_inviscid.py 里原来的纯 Python `for f in range(fc.n_faces)` 循环)。

逐字复刻原循环体的控制流（owner_is_primary / alignment<0.5 回退真实
法向 / is_boundary 走幽灵态或 sources 求和 / neighbor_is_primary），
只是把执行方式从"Python 解释器 + 逐次小 numpy 调用"换成 numba 编译的
逐点标量代码。数学公式与 fr_residual_inviscid.py 的向量化版本必须
保持完全一致，改动一处必须同步检查另一处。

**关键正确性约束（不能"优化掉"）**：owner 侧和 neighbor 侧对同一个
内部面各自独立调用一次 AUSM+up（各自用自己的度量法向、自己原生 FP
位置上插值出的状态），不能因为看起来像同一个黎曼问题就合并成一次
公共通量复用给两侧——这个"优化"已经在本项目里被真实网格验证证伪过
（自由流场残差从 9e-5 恶化到 3.1e7，见 fr_residual_inviscid.py 里
`F_common_n_owner` 附近的注释），必须保留两次独立调用。

**多核并行（阶段二）与 scatter-add 的正确处理**：`correction[oc/nc,
s, v] += ...` 是典型的 scatter-add——同一个 cell 会被多个不同的面
（不同的 f）累加，owner_cell[f]/neighbor_cell[f] 在不同 f 之间会重复。
直接对 `for f in range(n_faces)` 套 `prange` 会有多线程写冲突（已用
实验证实：不加保护会静默丢失约 23% 的累加更新，无任何报错）。这里
用"每线程私有累加缓冲区 `correction_per_thread[tid, ...]` + 循环结束
后按 thread 轴 `sum(axis=0)` 归约"的标准做法。**由此带来两条不能违反
的约束**：
1. `n_threads` 参数必须由调用方在调用本函数之前、紧邻调用处取
   `numba.get_num_threads()` 得到，不能缓存旧值、不能与其他地方并发
   修改的线程数不一致——`correction_per_thread[tid, ...]` 在 nopython
   模式下默认关闭 bounds check，`tid >= n_threads` 会静默内存越界，
   而不是报错。`numba.set_num_threads(...)` 只应该在求解器启动时
   调用一次（见 core/fr_solver.py），不要在其他地方并发修改。
2. `get_num_threads()` 本身**不能在这个 `@njit(cache=True,
   parallel=True)` 函数内部调用**——会导致 numba 静默放弃磁盘缓存
   （`NumbaWarning: uses dynamic globals`），已用实验证实：修正为
   "调用方取值、作为普通 int 参数传入，kernel 内部只用
   `get_thread_id()`"之后缓存行为完全恢复（`get_thread_id()` 本身
   不影响缓存）。

**并行化后累加顺序的变化（合法，不是 bug）**：串行版本的历史约束是
"严格保持与 range(n_faces) 相同的处理顺序"，因为退化 Jacobian 单元
处的舍入误差量级对求和结合顺序敏感。并行化后不同线程并发处理不同的
f，最终归约顺序（每线程内部保持子区间顺序，最后按 thread id 顺序
求和）必然不再是严格的 `range(n_faces)` 顺序——这是并行化本身不可
避免的、合法的浮点重结合来源，用与本项目一贯处理"numpy/BLAS vs
numba 标量重结合"同一套"相对 p_inf 容差"方法论验证（见
`tests/unit/test_fr_residual_inviscid_kernel_crosscheck.py` 的
`nt=16` 用例），不是 `nt=1` 时才用的逐位相等判据。
"""

import numpy as np
from numba import njit, prange, get_thread_id

from autoflowcfd.core.fr_operators.kernels import compute_ausm_up_flux
from autoflowcfd.core.fr_operators.flux_kernels import euler_physical_flux_point


@njit(cache=True, inline='always')
def _extrap_matmul(field_cell: np.ndarray, E: np.ndarray) -> np.ndarray:
    """外插矩阵乘法：field_cell (n_sps, k), E (n_fp, n_sps) -> (n_fp, k)。

    Python 边界幽灵态预处理（本文件外）和这里的主 kernel 共用同一个
    函数，不允许出现两份需要永远保持一致的独立实现。
    """
    return E @ field_cell


@njit(cache=True, inline='always')
def _distribute_point(fp_data: np.ndarray, fp_of_sp_axis: np.ndarray,
                       axis_coord_of_sp_axis: np.ndarray, g_prime: np.ndarray) -> np.ndarray:
    """`_distribute_from_face` 的逐点等价形式，见
    fr_face_kernels_flat.py::_derive_distribute_mapping 文档。

    fp_data: (n_fp, n_vars); fp_of_sp_axis/axis_coord_of_sp_axis: (n_sps,)
    （已经按 axis 选好的那一份）; g_prime: (n1d,)。
    """
    n_sps = fp_of_sp_axis.shape[0]
    n_vars = fp_data.shape[1]
    out = np.zeros((n_sps, n_vars))
    for s in range(n_sps):
        fp_i = fp_of_sp_axis[s]
        g = g_prime[axis_coord_of_sp_axis[s]]
        for v in range(n_vars):
            out[s, v] = g * fp_data[fp_i, v]
    return out


@njit(cache=True, parallel=True)
def compute_inviscid_interface_correction_kernel(
    Q: np.ndarray, det_jacs: np.ndarray,
    owner_cell: np.ndarray, neighbor_cell: np.ndarray, is_boundary: np.ndarray,
    owner_axis: np.ndarray, owner_side: np.ndarray,
    neighbor_axis: np.ndarray, neighbor_side: np.ndarray,
    owner_is_primary: np.ndarray, neighbor_is_primary: np.ndarray,
    true_normal: np.ndarray,
    owner_adj_row_exact: np.ndarray, neighbor_adj_row_exact: np.ndarray,
    neighbor_src0_cell: np.ndarray, neighbor_src0_mat: np.ndarray,
    neighbor_src1_idx: np.ndarray, neighbor_src1_cell: np.ndarray, neighbor_src1_mat: np.ndarray,
    owner_src0_cell: np.ndarray, owner_src0_mat: np.ndarray,
    owner_src1_idx: np.ndarray, owner_src1_cell: np.ndarray, owner_src1_mat: np.ndarray,
    mixed_nb_partner: np.ndarray, mixed_nb_mask: np.ndarray,
    mixed_ow_partner: np.ndarray, mixed_ow_mask: np.ndarray,
    boundary_extrap: np.ndarray,
    g_left: np.ndarray, g_right: np.ndarray,
    Q_ghost: np.ndarray,
    dist_fp_of_sp: np.ndarray, dist_axis_coord_of_sp: np.ndarray,
    n_prism: int,
    n_threads: int,
    mach_ref: float,
    owner_cube_face: np.ndarray, neighbor_cube_face: np.ndarray,
    true_area_weight: np.ndarray,
    boundary_extrap_native: np.ndarray, lift_native: np.ndarray,
) -> np.ndarray:
    """返回 correction，形状 (n_cells, n_sps, 5)，与
    fr_residual_inviscid.py::compute_inviscid_residual_fr 里"--- 界面项
    ---"那一段算出的 correction 逐位对应。

    native 四面体（路径C）支持（Part8 文档"三、本次会话实现范围"）：
    `owner_cube_face`/`neighbor_cube_face`（原始 cube face 编码，>=6
    即 native 真实面，excluded_vertex=code-6）用于分派——`owner_axis`/
    `owner_side` 对 native 面存的是复用槽位，不能用来判断/当 axis/side
    语义使用。native 分支：(a) 自身面外插矩阵改用 `boundary_extrap_
    native[excluded_vertex]`；(b) 方向定向系数固定为 +1（`face_flux_
    points_exact_normal.py::_native_tet_adj_row_batched` 已经给出正确
    outward 定向的 adj 行，不需要像坍缩坐标那样再乘 side 翻转——这是
    Part7 文档记录过的同一个坑，这里必须复刻同一个原则）；(c) 面修正项
    改用 DG 提升算子 `lift_native[excluded_vertex] @ (true_area_weight
    ⊙ jump)`（`native_simplex_basis.py::build_native_tet_lift`
    "弱形式提升定义"，替代坍缩坐标 1D Radau/VCJH 修正函数
    `_distribute_point`——native 单纯形基没有"坍缩计算方向"，那套 1D
    分布机制不适用）。`n_prism`+`boundary_extrap`+`dist_fp_of_sp` 等
    坍缩坐标专属参数对 native 面完全不使用，但仍然按原签名传入（不含
    native 四面体的既有网格调用完全不变，见调用处 `face_kernels.py`
    "自动探测/零占位"说明）。

    `n_threads` 必须是调用方紧邻本次调用之前取的 `numba.
    get_num_threads()`，理由见模块文档"多核并行"一节——不能在这个函数
    内部自己查询（会破坏磁盘缓存）。多线程下累加顺序不再是严格的
    `range(n_faces)` 顺序，验证判据也相应分层，见模块文档。

    真实 bug 修复（2026-08-23，见 fr/face_flux_points_exact_normal.py
    模块文档完整原理）：`owner_adj_row_exact`/`neighbor_adj_row_exact`
    取代了此前这里对 `adj_j`（SP 网格上的度量）用 `E_o`/`E_n` 做
    Lagrange 外插到 FP 得到"自洽方向"的做法——外插对坍缩坐标下本质是
    有理函数的 adj(J) 有不可忽略的截断误差（P1 尤其明显）。改为在
    mesh 加载阶段一次性预计算好的、每个 FP 自己精确参考坐标处的解析
    Jacobian 值，直接查表读取，消除这部分截断误差——原来的 `adj_j`
    参数（SP 网格度量）因此不再被本函数需要，已从签名中移除（同文件
    的 `compute_boundary_ghost_states` 是另一个独立函数，自己的
    `adj_j` 参数不受影响）。
    """
    n_cells = Q.shape[0]
    n_sps = Q.shape[1]
    n_faces = owner_cell.shape[0]
    n_fp = true_normal.shape[1]

    correction_per_thread = np.zeros((n_threads, n_cells, n_sps, 5))

    for f in prange(n_faces):
        tid = get_thread_id()
        oc = owner_cell[f]
        oc_code = owner_cube_face[f]
        o_is_native = oc_code >= 6
        oax = owner_axis[f]
        oside = owner_side[f]
        # native 面：_native_tet_adj_row_batched 已给出正确 outward 定向，
        # side 因子固定 +1（不能沿用 oside 那个复用槽位的哑值，见函数文档）。
        side_factor_o = 1.0 if o_is_native else oside
        oside_idx = 0 if oside < 0 else 1
        celltype_o = 0 if oc < n_prism else 1

        if owner_is_primary[f]:
            if o_is_native:
                E_o = boundary_extrap_native[oc_code - 6]  # (n_fp, n_sps)
            else:
                E_o = boundary_extrap[celltype_o, oax, oside_idx]  # (n_fp, n_sps)

            Q_o = _extrap_matmul(Q[oc], E_o)  # (n_fp, 5)
            adjrow_o = owner_adj_row_exact[f]  # (n_fp, 3)，逐 FP 精确值，见函数文档

            jump_owner = np.zeros((n_fp, 5))
            for i in range(n_fp):
                a0 = adjrow_o[i, 0]
                a1 = adjrow_o[i, 1]
                a2 = adjrow_o[i, 2]
                adj_mag = np.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
                if adj_mag < 1e-300:
                    adj_mag = 1e-300
                dirx = a0 / adj_mag * side_factor_o
                diry = a1 / adj_mag * side_factor_o
                dirz = a2 / adj_mag * side_factor_o

                alignment = dirx * true_normal[f, i, 0] + diry * true_normal[f, i, 1] + dirz * true_normal[f, i, 2]
                if alignment < 0.5:
                    dirx = true_normal[f, i, 0]
                    diry = true_normal[f, i, 1]
                    dirz = true_normal[f, i, 2]

                # --- Q_neighbor 在这个 owner FP 处的取值 ---
                if is_boundary[f]:
                    Q_n = Q_ghost[f, i]
                else:
                    Q_n = np.zeros(5)
                    c0 = neighbor_src0_cell[f]
                    if c0 >= 0:
                        mat0 = neighbor_src0_mat[f]  # (n_fp, n_sps)
                        for s in range(n_sps):
                            w = mat0[i, s]
                            if w != 0.0:
                                for v in range(5):
                                    Q_n[v] += w * Q[c0, s, v]
                    idx1 = neighbor_src1_idx[f]
                    if idx1 >= 0:
                        c1 = neighbor_src1_cell[idx1]
                        mat1 = neighbor_src1_mat[idx1]
                        for s in range(n_sps):
                            w = mat1[i, s]
                            if w != 0.0:
                                for v in range(5):
                                    Q_n[v] += w * Q[c1, s, v]
                    # 混合拆分面（B-8，见 fr/face_flux_points_merge.py）：内部半区由上方多源插值覆盖，
                    # 边界半区逐 FP 取配对边界面的幽灵态（两条记录共享同一 owner 棱柱与立方体面，
                    # FP 网格逐点重合）。掩码行内多源插值矩阵权重为 0，先算再覆盖不冲突。
                    mp = mixed_nb_partner[f]
                    if mp >= 0 and mixed_nb_mask[f, i]:
                        Q_n = Q_ghost[mp, i]

                normal = np.empty(3)
                normal[0] = dirx
                normal[1] = diry
                normal[2] = dirz
                F_common_n = compute_ausm_up_flux(Q_o[i], Q_n, normal, mach_ref)  # (5,)

                F_tilde_common = np.empty(5)
                for v in range(5):
                    F_tilde_common[v] = F_common_n[v] * adj_mag * side_factor_o

                F_phys_o = euler_physical_flux_point(Q_o[i])  # (3,5)
                F_tilde_own = np.zeros(5)
                for v in range(5):
                    F_tilde_own[v] = a0 * F_phys_o[0, v] + a1 * F_phys_o[1, v] + a2 * F_phys_o[2, v]

                for v in range(5):
                    jump_owner[i, v] = F_tilde_common[v] - F_tilde_own[v]

            if o_is_native:
                # DG 提升算子（见函数文档）：物理面积权重逐 FP 加权跳跃量，
                # 再用提升算子映射回体积节点——替代坍缩坐标的 1D 修正函数
                # 分布机制，见 native_simplex_basis.py::build_native_tet_lift
                # "弱形式提升定义"。
                weighted_jump_o = np.empty((n_fp, 5))
                for i in range(n_fp):
                    w_area = true_area_weight[f, i]
                    for v in range(5):
                        weighted_jump_o[i, v] = w_area * jump_owner[i, v]
                contrib_owner = lift_native[oc_code - 6] @ weighted_jump_o  # (n_sps, 5)
            else:
                g_prime_owner = g_left if oside < 0 else g_right
                contrib_owner = _distribute_point(
                    jump_owner, dist_fp_of_sp[oax], dist_axis_coord_of_sp[oax], g_prime_owner
                )  # (n_sps, 5)
            for s in range(n_sps):
                dj = det_jacs[oc, s]
                for v in range(5):
                    correction_per_thread[tid, oc, s, v] += -contrib_owner[s, v] / dj

        if (not is_boundary[f]) and neighbor_is_primary[f]:
            nc = neighbor_cell[f]
            nc_code = neighbor_cube_face[f]
            n_is_native = nc_code >= 6
            nax = neighbor_axis[f]
            nside = neighbor_side[f]
            side_factor_n = 1.0 if n_is_native else nside
            nside_idx = 0 if nside < 0 else 1
            celltype_n = 0 if nc < n_prism else 1

            if n_is_native:
                E_n = boundary_extrap_native[nc_code - 6]
            else:
                E_n = boundary_extrap[celltype_n, nax, nside_idx]

            Q_n_native = _extrap_matmul(Q[nc], E_n)  # (n_fp,5)
            adjrow_n_native = neighbor_adj_row_exact[f]  # (n_fp,3)，逐 FP 精确值，见函数文档

            jump_neighbor = np.zeros((n_fp, 5))
            for i in range(n_fp):
                a0 = adjrow_n_native[i, 0]
                a1 = adjrow_n_native[i, 1]
                a2 = adjrow_n_native[i, 2]
                adj_mag = np.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
                if adj_mag < 1e-300:
                    adj_mag = 1e-300
                dirx = a0 / adj_mag * side_factor_n
                diry = a1 / adj_mag * side_factor_n
                dirz = a2 / adj_mag * side_factor_n

                # neighbor 视角外法向恒为 -true_normal（平面直边网格）
                ntnx = -true_normal[f, i, 0]
                ntny = -true_normal[f, i, 1]
                ntnz = -true_normal[f, i, 2]
                alignment_n = dirx * ntnx + diry * ntny + dirz * ntnz
                if alignment_n < 0.5:
                    dirx = ntnx
                    diry = ntny
                    dirz = ntnz

                Q_o_at_n = np.zeros(5)
                c0 = owner_src0_cell[f]
                if c0 >= 0:
                    mat0 = owner_src0_mat[f]
                    for s in range(n_sps):
                        w = mat0[i, s]
                        if w != 0.0:
                            for v in range(5):
                                Q_o_at_n[v] += w * Q[c0, s, v]
                idx1 = owner_src1_idx[f]
                if idx1 >= 0:
                    c1 = owner_src1_cell[idx1]
                    mat1 = owner_src1_mat[idx1]
                    for s in range(n_sps):
                        w = mat1[i, s]
                        if w != 0.0:
                            for v in range(5):
                                Q_o_at_n[v] += w * Q[c1, s, v]
                # 混合拆分面（B-8）：neighbor-primary 分支对称处理——若本面是 neighbor 侧的
                # 混合配对内部面，边界半区处的对侧状态同样取配对边界面的幽灵态。
                # 逐元素拷贝而非整体赋值：Q_o_at_n 首次赋值为 np.zeros(5)（C 布局），
                # numba 不允许再把 A 布局视图赋给 C 布局变量。
                mp_o = mixed_ow_partner[f]
                if mp_o >= 0 and mixed_ow_mask[f, i]:
                    for v in range(5):
                        Q_o_at_n[v] = Q_ghost[mp_o, i, v]

                normal = np.empty(3)
                normal[0] = dirx
                normal[1] = diry
                normal[2] = dirz
                F_common_n_native = compute_ausm_up_flux(Q_n_native[i], Q_o_at_n, normal, mach_ref)

                F_tilde_common_n = np.empty(5)
                for v in range(5):
                    F_tilde_common_n[v] = F_common_n_native[v] * adj_mag * side_factor_n

                F_phys_n = euler_physical_flux_point(Q_n_native[i])
                F_tilde_own_n = np.zeros(5)
                for v in range(5):
                    F_tilde_own_n[v] = a0 * F_phys_n[0, v] + a1 * F_phys_n[1, v] + a2 * F_phys_n[2, v]

                for v in range(5):
                    jump_neighbor[i, v] = F_tilde_common_n[v] - F_tilde_own_n[v]

            if n_is_native:
                weighted_jump_n = np.empty((n_fp, 5))
                for i in range(n_fp):
                    w_area = true_area_weight[f, i]
                    for v in range(5):
                        weighted_jump_n[i, v] = w_area * jump_neighbor[i, v]
                contrib_neighbor = lift_native[nc_code - 6] @ weighted_jump_n
            else:
                g_prime_neighbor = g_left if nside < 0 else g_right
                contrib_neighbor = _distribute_point(
                    jump_neighbor, dist_fp_of_sp[nax], dist_axis_coord_of_sp[nax], g_prime_neighbor
                )
            for s in range(n_sps):
                dj = det_jacs[nc, s]
                for v in range(5):
                    correction_per_thread[tid, nc, s, v] += -contrib_neighbor[s, v] / dj

    return correction_per_thread.sum(axis=0)



def compute_boundary_ghost_states(flat, Q: np.ndarray, adj_j: np.ndarray, ghost_provider) -> np.ndarray:
    """边界面幽灵态预处理（纯 Python，只跑边界面这一小部分——约占全部
    面的 3%，`boundary_ghost_provider` 是任意 Python 可调用对象，numba
    调不了，见模块文档）。与主 kernel 共用同一个 `_extrap_matmul`。

    Returns:
        Q_ghost: (n_faces, n_fp, 5)，只有边界面对应的行有意义。
    """
    Q_ghost = np.zeros((flat.n_faces, flat.n_fp, 5))
    for f in range(flat.n_faces):
        if not flat.is_boundary[f]:
            continue
        # 混合拆分面（B-8）：混合配对的边界子面 owner_primary 已被置 False（不参与残差累加），
        # 但它的幽灵态仍被配对的内部面逐 FP 读取，不能跳过计算。
        if not flat.owner_is_primary[f] and not flat.mixed_bnd_face[f]:
            continue
        oc = flat.owner_cell[f]
        oc_code = flat.owner_cube_face[f]
        if oc_code >= 6:
            # native 四面体（路径C）：自身面外插改用 boundary_extrap_native
            # （excluded_vertex=code-6），见 compute_inviscid_interface_
            # correction_kernel 文档同一个分派原则。
            E_o = flat.boundary_extrap_native[oc_code - 6]
        else:
            oax = flat.owner_axis[f]
            oside = flat.owner_side[f]
            oside_idx = 0 if oside < 0 else 1
            celltype_o = 0 if oc < flat.n_prism else 1
            E_o = flat.boundary_extrap[celltype_o, oax, oside_idx]
        Q_o = _extrap_matmul(Q[oc], E_o)
        Q_ghost[f] = ghost_provider(f, Q_o, flat.true_normal[f])
    return Q_ghost
