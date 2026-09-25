"""无粘残差界面项 —— numba 逐点标量 kernel (性能优化，替代
fr_residual_inviscid.py 里原来的纯 Python `for f in range(fc.n_faces)` 循环)。

逐字复刻原循环体的控制流（owner_is_primary / 本侧精确度量法向
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

from autoflowcfd.core.fr_operators.small_dense import matmul_small
from autoflowcfd.core.fr_operators.kernels import compute_ausm_up_flux
from autoflowcfd.core.fr_operators.flux_kernels import euler_physical_flux_point


@njit(cache=True, inline='always')
def _extrap_matmul(field_cell: np.ndarray, E: np.ndarray) -> np.ndarray:
    """外插矩阵乘法：field_cell (n_sps, k), E (n_fp, n_sps) -> (n_fp, k)。

    Python 边界幽灵态预处理（本文件外）和这里的主 kernel 共用同一个
    函数，不允许出现两份需要永远保持一致的独立实现
    （不调 BLAS：并行核里逐面调用 BLAS 线程越多越慢，见 small_dense.py）
    """
    return matmul_small(E, field_cell)


@njit(cache=True, parallel=True)
def compute_inviscid_interface_correction_kernel(
    Q: np.ndarray, det_jacs: np.ndarray,
    owner_cell: np.ndarray, neighbor_cell: np.ndarray, is_boundary: np.ndarray,
    owner_is_primary: np.ndarray, neighbor_is_primary: np.ndarray,
    owner_adj_row_exact: np.ndarray, neighbor_adj_row_exact: np.ndarray,
    neighbor_src0_cell: np.ndarray, neighbor_src0_mat: np.ndarray,
    neighbor_src1_idx: np.ndarray, neighbor_src1_cell: np.ndarray, neighbor_src1_mat: np.ndarray,
    owner_src0_cell: np.ndarray, owner_src0_mat: np.ndarray,
    owner_src1_idx: np.ndarray, owner_src1_cell: np.ndarray, owner_src1_mat: np.ndarray,
    mixed_nb_partner: np.ndarray, mixed_nb_mask: np.ndarray,
    mixed_ow_partner: np.ndarray, mixed_ow_mask: np.ndarray,
    Q_ghost: np.ndarray,
    n_threads: int,
    mach_ref: float,
    precond_mode: int,   # AUSM+up 预处理声速作用域，见 kernels.py::
                         # compute_ausm_up_flux 文档；必须由纯 Python 层
                         # 用 resolve_ausm_precond_mode() 解析后传入
    owner_cube_face: np.ndarray, neighbor_cube_face: np.ndarray,
    ref_area_weight: np.ndarray,
    boundary_extrap_native: np.ndarray, lift_native: np.ndarray,
) -> np.ndarray:
    """返回 correction，形状 (n_cells, n_sps, 5)，与
    fr_residual_inviscid.py::compute_inviscid_residual_fr 里"--- 界面项
    ---"那一段算出的 correction 逐位对应。

    native 四面体（路径C）支持（Part8 文档"三、本次会话实现范围"）：
    `owner_cube_face`/`neighbor_cube_face` 是原始 cube face 编码：四面体
    真实面 `[6,10)`、棱柱真实面 `[10,15)`，`excluded_vertex = code - 6`
    统一索引 `boundary_extrap_native`/`lift_native`（棱柱那 5 个键在同一张
    表里，见 `fr/operators/container.py`）。`owner_axis`/`owner_side` 对
    原生面存的是复用槽位，**不是** axis/side 语义，本函数不读它们。

    三条原生基专属做法：(a) 自身面外插矩阵用
    `boundary_extrap_native[code-6]`；(b) 方向定向系数固定为 +1
    （`fr/face_flux_points/exact_normal.py` 已经给出正确 outward 定向的
    adj 行，不需要像坍缩坐标那样再乘 side 翻转——Part7 文档记录过的坑，
    2026-09-18 无滑移壁 IP 罚项又在同一处踩了一次）；(c) 面修正项用
    DG 提升算子 `lift_native[code-6] @ (ref_area_weight ⊙ jump)`
    （`native_tet/basis.py::build_native_tet_lift` "弱形式提升定义"）。

    **2026-09-23**：坍缩坐标那条并行路径（`boundary_extrap[celltype,
    axis,side]` 外插 + 1D Radau/VCJH 修正函数 `_distribute_point` 分布）
    已整体删除。坍缩四面体基 2026-09-03 删除、坍缩棱柱基 2026-09-23 删除
    之后，`FRFaceConnectivity.with_native_face_codes` 把**每一个**真实面
    都翻译成原生编码，实测混合网格（P1/P2）的面编码集合是
    `{6,7,8,9,10,11,12,13,14}`、`code < 6` 的非边界面**为 0 个**，所以那
    条分支在生产上不可达。不可达的第二套离散方案留在最热的 kernel 里，
    正是本项目明确要删的冗余。构造期护栏见
    `core/fr_operators/face_kernels.py::build_flat_face_geometry`。

    `n_threads` 必须是调用方紧邻本次调用之前取的 `numba.
    get_num_threads()`，理由见模块文档"多核并行"一节——不能在这个函数
    内部自己查询（会破坏磁盘缓存）。多线程下累加顺序不再是严格的
    `range(n_faces)` 顺序，验证判据也相应分层，见模块文档。

    **法向一律取自本侧精确度量行**（2026-09-24）：此前两侧各有一个
    `alignment = dir(adj_row).true_normal < 0.5` 时改用 `true_normal` 的
    兜底。它是原生基之前的遗留（那时法向由 Lagrange 外插的度量得到，坍缩
    顶点附近可能是垃圾方向）；原生基下 adj 行是逐 FP 的解析精确值，夹角
    >60° 说明几何本身就是那样（严重扭曲的棱柱四边形面对两个平面三角形），
    兜底不再保护任何合法情形，只剩一个效果 —— 破坏自由流保持：换了方向
    之后拥有侧投影仍用 adj_row，对均匀流
        jump = |adj| F(Q).(n_ref - dir(adj)) != 0。
    plate_demo_volume_les（179,237 单元）上它在 11 个面、42 个 FP 触发
    （全部在板锐边的扭曲棱柱上），在"全边界取远场"的均匀流里产生
    4~5e6 /s 的残差（其余单元为 1e-7 量级），正是 P1 长程运行在第 4134 步
    单步爆炸那个单元的源头：它的一个解点密度在恒定源项下按恒定斜率下降，
    过零后被逐点硬钳，一步放大到 1e53。粘性核本来就没有这个兜底，所以
    粘性残差在同一测试里处处精确。

    真实 bug 修复（2026-08-23，见 fr/face_flux_points/exact_normal.py
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
    n_fp = owner_adj_row_exact.shape[1]

    correction_per_thread = np.zeros((n_threads, n_cells, n_sps, 5))

    for f in prange(n_faces):
        tid = get_thread_id()
        oc = owner_cell[f]
        oc_code = owner_cube_face[f]
        # 原生面的 adj 行已是正确 outward 定向（见函数文档），所以不再有
        # side 翻转因子——`owner_axis`/`owner_side` 是复用槽位，不读。

        if owner_is_primary[f]:
            E_o = boundary_extrap_native[oc_code - 6]  # (n_fp, n_sps)

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
                dirx = a0 / adj_mag
                diry = a1 / adj_mag
                dirz = a2 / adj_mag


                # 法向恒用本侧**精确度量行**的方向（2026-09-24 删除了此前的
                # `alignment < 0.5` 兜底——它在夹角过大时把方向换成
                # true_normal、但拥有侧投影仍用 adj_row，于是对均匀流
                # 跳跃量 = |adj| F.(n_ref - dir(adj)) != 0，凭空注入压力
                # 量级的源项。完整依据见本函数文档"法向一律取自本侧度量"。）

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
                    # 混合拆分面（B-8，见 fr/face_flux_points/merge.py）：内部半区由上方多源插值覆盖，
                    # 边界半区逐 FP 取配对边界面的幽灵态（两条记录共享同一 owner 棱柱与立方体面，
                    # FP 网格逐点重合）。掩码行内多源插值矩阵权重为 0，先算再覆盖不冲突。
                    mp = mixed_nb_partner[f]
                    if mp >= 0 and mixed_nb_mask[f, i]:
                        Q_n = Q_ghost[mp, i]

                normal = np.empty(3)
                normal[0] = dirx
                normal[1] = diry
                normal[2] = dirz
                F_common_n = compute_ausm_up_flux(Q_o[i], Q_n, normal, mach_ref, precond_mode)  # (5,)

                F_tilde_common = np.empty(5)
                for v in range(5):
                    F_tilde_common[v] = F_common_n[v] * adj_mag

                F_phys_o = euler_physical_flux_point(Q_o[i])  # (3,5)
                F_tilde_own = np.zeros(5)
                for v in range(5):
                    F_tilde_own[v] = a0 * F_phys_o[0, v] + a1 * F_phys_o[1, v] + a2 * F_phys_o[2, v]

                for v in range(5):
                    jump_owner[i, v] = F_tilde_common[v] - F_tilde_own[v]

            # DG 提升算子（见函数文档）：物理面积权重逐 FP 加权跳跃量，
            # 再用提升算子映射回体积节点，见
            # native_tet/basis.py::build_native_tet_lift "弱形式提升定义"。
            weighted_jump_o = np.empty((n_fp, 5))
            for i in range(n_fp):
                w_area = ref_area_weight[i]
                for v in range(5):
                    weighted_jump_o[i, v] = w_area * jump_owner[i, v]
            contrib_owner = matmul_small(lift_native[oc_code - 6], weighted_jump_o)  # (n_sps, 5)
            for s in range(n_sps):
                dj = det_jacs[oc, s]
                for v in range(5):
                    correction_per_thread[tid, oc, s, v] += -contrib_owner[s, v] / dj

        if (not is_boundary[f]) and neighbor_is_primary[f]:
            nc = neighbor_cell[f]
            nc_code = neighbor_cube_face[f]
            E_n = boundary_extrap_native[nc_code - 6]

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
                dirx = a0 / adj_mag
                diry = a1 / adj_mag
                dirz = a2 / adj_mag


                # 法向恒用本侧**精确度量行**的方向（2026-09-24 删除了此前的
                # `alignment < 0.5` 兜底——它在夹角过大时把方向换成
                # true_normal、但本侧投影仍用 adj_row，于是对均匀流
                # 跳跃量 = |adj| F.(n_ref - dir(adj)) != 0，凭空注入压力
                # 量级的源项。完整依据见本函数文档"法向一律取自本侧度量"。）

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
                F_common_n_native = compute_ausm_up_flux(Q_n_native[i], Q_o_at_n, normal, mach_ref, precond_mode)

                F_tilde_common_n = np.empty(5)
                for v in range(5):
                    F_tilde_common_n[v] = F_common_n_native[v] * adj_mag

                F_phys_n = euler_physical_flux_point(Q_n_native[i])
                F_tilde_own_n = np.zeros(5)
                for v in range(5):
                    F_tilde_own_n[v] = a0 * F_phys_n[0, v] + a1 * F_phys_n[1, v] + a2 * F_phys_n[2, v]

                for v in range(5):
                    jump_neighbor[i, v] = F_tilde_common_n[v] - F_tilde_own_n[v]

            weighted_jump_n = np.empty((n_fp, 5))
            for i in range(n_fp):
                w_area = ref_area_weight[i]
                for v in range(5):
                    weighted_jump_n[i, v] = w_area * jump_neighbor[i, v]
            contrib_neighbor = matmul_small(lift_native[nc_code - 6], weighted_jump_n)
            for s in range(n_sps):
                dj = det_jacs[nc, s]
                for v in range(5):
                    correction_per_thread[tid, nc, s, v] += -contrib_neighbor[s, v] / dj

    return correction_per_thread.sum(axis=0)



def _compute_boundary_ghost_states_per_face(flat, Q: np.ndarray, ghost_provider) -> np.ndarray:
    """边界面幽灵态预处理的通用逐面回退实现——只依赖 `ghost_provider`
    满足最小鸭子类型接口 `(face_idx, Q_owner_fp, true_normal) -> Q_ghost`，
    对任意实现都正确（含 `DefaultGhostProvider`、测试里的自定义 stub 等，
    见 `compute_boundary_ghost_states` 分派说明）。约占全部面的 3%，
    `ghost_provider` 是任意 Python 可调用对象，numba 调不了。
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
        # 自身面外插用 `boundary_extrap_native[code-6]`（四面体 [6,10)、
        # 棱柱 [10,15) 同一张表），与 compute_inviscid_interface_correction_
        # kernel 同一个分派原则。
        E_o = flat.boundary_extrap_native[flat.owner_cube_face[f] - 6]
        Q_o = _extrap_matmul(Q[oc], E_o)
        Q_ghost[f] = ghost_provider(f, Q_o, flat.true_normal[f])
    return Q_ghost


def _compute_boundary_ghost_states_batched(flat, Q: np.ndarray, ghost_provider) -> np.ndarray:
    """`compute_boundary_ghost_states` 的向量化实现——只在 `ghost_provider`
    是 `BoundaryGhostStateProvider` 实例时使用（见该分派条件说明）：按
    `group_code` 分组批量调用底层幽灵态函数（`wall_ghost_state` 等，
    本身已经是接受任意长度批量的向量化函数），替代逐面 Python 调度
    （2026-09-03 性能优化，真实 profile 显示每步 ~24万次这类逐面调用，
    约占单步耗时 3~6%，见排查记录）。

    数学结果与 `_compute_boundary_ghost_states_per_face` 逐位一致（同一套
    E_o 外插 + 同一批底层幽灵态函数，只是把"逐面调用"改成"先向量化算出
    全部活跃边界面的 Q_o，再按边界组一次性批量调用"，不改变任何计算
    本身）——已用真实含 WALL/INLET/OUTLET/棱柱四边形混合拆分面的合成
    网格与逐面版本决定性交叉验证到逐位相等，见
    tests/unit/test_boundary_ghost_states_batched_crosscheck.py。

    INLET + SEM（合成湍流入口，逐面涡核位置相关）不做批量化，组内退回
    逐面调用——SEM 组内面数远小于总边界面数，不是性能热点，见
    `InletSEMGhostState.__call__` 文档。
    """
    from autoflowcfd.boundary.fr_ghost_state import (
        wall_ghost_state, farfield_ghost_state, inlet_ghost_state,
        outlet_ghost_state, symmetry_ghost_state,
    )

    n_fp = flat.n_fp
    Q_ghost = np.zeros((flat.n_faces, n_fp, 5))

    active_mask = flat.is_boundary & (flat.owner_is_primary | flat.mixed_bnd_face)
    active_faces = np.where(active_mask)[0]
    if active_faces.size == 0:
        return Q_ghost

    # --- 向量化 Q_o 外插（与逐面版本同一套 E_o 选择：原生面编码减 6
    # 直接索引 `boundary_extrap_native`）---
    oc = flat.owner_cell[active_faces]
    E_o = flat.boundary_extrap_native[flat.owner_cube_face[active_faces] - 6]

    Q_o_all = np.einsum('fps,fsv->fpv', E_o, Q[oc])  # (n_active,n_fp,5)
    normal_all = flat.true_normal[active_faces]  # (n_active,n_fp,3)

    # --- 按边界组编码分组（同一 group_code 保证共享同一份 cfg，含
    # WALL 的 is_no_slip/wall_velocity 这类逐组参数，不是全局常量）---
    codes = ghost_provider.group_code[active_faces]
    for code in np.unique(codes):
        in_group = np.where(codes == code)[0]
        faces_in_group = active_faces[in_group]
        cfg = ghost_provider.code_to_config.get(int(code), ghost_provider.default_config)
        bc_type = cfg["type"]

        Q_o_group = Q_o_all[in_group]        # (n_group,n_fp,5)
        normal_group = normal_all[in_group]  # (n_group,n_fp,3)
        n_group = Q_o_group.shape[0]

        if bc_type == "INLET" and cfg.get("sem") is not None:
            sem_ghost = cfg["sem"]
            for k in range(n_group):
                f = faces_in_group[k]
                Q_ghost[f] = sem_ghost(f, Q_o_group[k], normal_group[k])
            continue

        Q_o_flat = Q_o_group.reshape(-1, 5)
        normal_flat = normal_group.reshape(-1, 3)

        if bc_type == "WALL":
            result = wall_ghost_state(
                Q_o_flat, normal_flat,
                is_no_slip=cfg.get("is_no_slip", True),
                wall_velocity=cfg.get("wall_velocity"),
            )
        elif bc_type == "FARFIELD":
            result = farfield_ghost_state(Q_o_flat, cfg["Q_free"])
        elif bc_type == "INLET":
            result = inlet_ghost_state(Q_o_flat, cfg["Q_inlet"], normal_flat)
        elif bc_type == "OUTLET":
            result = outlet_ghost_state(Q_o_flat, cfg["p_outlet"], normal_flat)
        elif bc_type == "SYMMETRY":
            result = symmetry_ghost_state(Q_o_flat, normal_flat)
        else:
            raise ValueError(f"Unknown boundary condition type '{bc_type}' for group code {code}")

        Q_ghost[faces_in_group] = result.reshape(n_group, n_fp, 5)

    return Q_ghost


def compute_boundary_ghost_states(flat, Q: np.ndarray, adj_j: np.ndarray, ghost_provider) -> np.ndarray:
    """边界面幽灵态预处理（纯 Python，只跑边界面这一小部分——约占全部
    面的 3%，`boundary_ghost_provider` 是任意 Python 可调用对象，numba
    调不了）。

    分派（2026-09-03 性能优化新增）：`ghost_provider` 是
    `BoundaryGhostStateProvider`（生产求解器实际使用的实现，见
    `fr_solver/boundary.py::build_boundary_ghost_provider`）时走按边界组
    批量化的 `_compute_boundary_ghost_states_batched`；否则（
    `DefaultGhostProvider`、测试里的自定义 stub 等只满足最小鸭子类型
    接口的实现）回退到逐面版本——不假设"任意 ghost_provider"都能被
    按 group_code 分组，那样会破坏这个函数一直保持的"boundary_ghost_
    provider 是任意 Python 可调用对象"的通用契约。

    Returns:
        Q_ghost: (n_faces, n_fp, 5)，只有边界面对应的行有意义。
    """
    from autoflowcfd.boundary.fr_ghost_state import BoundaryGhostStateProvider
    if isinstance(ghost_provider, BoundaryGhostStateProvider):
        return _compute_boundary_ghost_states_batched(flat, Q, ghost_provider)
    return _compute_boundary_ghost_states_per_face(flat, Q, ghost_provider)
