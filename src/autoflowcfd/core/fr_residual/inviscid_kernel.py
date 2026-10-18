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

**多核并行**：按面着色（同色面的 owner/neighbor 单元互不重复），调用方逐颜色调用
`compute_inviscid_interface_correction_kernel`，组内 `prange` 直接累加进共享 buffer——不加保护地对全部面
`prange` 会静默丢失约 23% 的累加更新（已实验证实）。并行后累加顺序不再是 `range(n_faces)` 顺序，这是合法的
浮点重结合，验证用相对 p_inf 的容差（`tests/unit/test_fr_residual_inviscid_kernel_crosscheck.py`）。
"""

import numpy as np
from numba import njit, prange

from autoflowcfd.core.fr_operators.small_dense import matmul_small
from autoflowcfd.core.fr_residual.face_point_jumps import inviscid_common_flux_point


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
    neighbor_src0_cell: np.ndarray, neighbor_src0_tpl: np.ndarray, neighbor_src0_tid: np.ndarray,
    neighbor_src1_idx: np.ndarray, neighbor_src1_cell: np.ndarray, neighbor_src1_mat: np.ndarray,
    owner_src0_cell: np.ndarray, owner_src0_tpl: np.ndarray, owner_src0_tid: np.ndarray,
    owner_src1_idx: np.ndarray, owner_src1_cell: np.ndarray, owner_src1_mat: np.ndarray,
    mixed_nb_partner: np.ndarray, mixed_nb_mask: np.ndarray,
    mixed_ow_partner: np.ndarray, mixed_ow_mask: np.ndarray,
    Q_ghost: np.ndarray,
    face_indices: np.ndarray,  # 当前颜色组的面索引
    correction: np.ndarray,    # 共享输出 buffer（同色面无冲突，直接写入）
    mach_ref: float,
    owner_face_op: np.ndarray, neighbor_face_op: np.ndarray,
    ref_area_weight: np.ndarray,
    boundary_extrap_native: np.ndarray, lift_native: np.ndarray,
) -> None:
    """把颜色组 `face_indices` 里各面的无粘界面修正项累加进 `correction`（(n_cells, n_sps, 5)，就地）。

    **按面着色并行**：调用方按颜色循环，每次只处理一个颜色组（同色面的 owner/neighbor 单元互不重复，构造见
    `core/utils/face_coloring.py`），所以可以对组内的面 `prange`、直接累加进共享的 `correction`，没有写冲突、
    不需要每线程缓冲与归约（内存 O(n_cells*n_sps*5)，与线程数无关）。2026-10-18 以前另有一份每线程缓冲版
    （环境变量切换），两份逐字同步维护，随开关删除。

    native 四面体（路径C）支持（Part8 文档"三、本次会话实现范围"）：
    `owner_face_op`/`neighbor_face_op` 是两侧的面算子索引（面编码 + 三角形面的
    坍缩顶点槽位，`fr/operators/face_ops.py`），统一索引
    `boundary_extrap_native`/`lift_native` 两张整表。`owner_axis`/`owner_side` 对
    原生面存的是复用槽位，**不是** axis/side 语义，本函数不读它们。

    三条原生基专属做法：(a) 自身面外插矩阵用
    `boundary_extrap_native[op]`；(b) 方向定向系数固定为 +1
    （`fr/face_flux_points/exact_normal.py` 已经给出正确 outward 定向的
    adj 行，不需要像坍缩坐标那样再乘 side 翻转——Part7 文档记录过的坑，
    2026-09-18 无滑移壁 IP 罚项又在同一处踩了一次）；(c) 面修正项用
    DG 提升算子 `lift_native[op] @ (ref_area_weight ⊙ |a| F_common)`（本侧通量迹已并进体积算子，见 `fr/face_flux_trace.py`）
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
    n_sps = Q.shape[1]
    n_fp = owner_adj_row_exact.shape[1]
    n_faces_in_color = face_indices.shape[0]

    for fi in prange(n_faces_in_color):
        f = face_indices[fi]
        oc = owner_cell[f]
        oc_op = owner_face_op[f]

        if owner_is_primary[f]:
            E_o = boundary_extrap_native[oc_op]

            Q_o = _extrap_matmul(Q[oc], E_o)
            adjrow_o = owner_adj_row_exact[f]  # (n_fp, 3)，逐 FP 精确值，见函数文档

            flux_owner = np.empty((n_fp, 5))
            for i in range(n_fp):
                # 另一侧状态（幽灵态 / sources 插值 / 混合拆分面配对幽灵态）
                if is_boundary[f]:
                    Q_n = Q_ghost[f, i]
                else:
                    Q_n = np.zeros(5)
                    c0 = neighbor_src0_cell[f]
                    if c0 >= 0:
                        mat0 = neighbor_src0_tpl[neighbor_src0_tid[f]]
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
                    # 混合拆分面（B-8，与主 kernel 同步修改，见 inviscid_kernel.py 同名注释）
                    mp = mixed_nb_partner[f]
                    if mp >= 0 and mixed_nb_mask[f, i]:
                        Q_n = Q_ghost[mp, i]
                flux_owner[i] = inviscid_common_flux_point(Q_o[i], Q_n, adjrow_o[i], mach_ref)

            weighted_flux_o = np.empty((n_fp, 5))
            for i in range(n_fp):
                w_area = ref_area_weight[i]
                for v in range(5):
                    weighted_flux_o[i, v] = w_area * flux_owner[i, v]
            contrib_owner = matmul_small(lift_native[oc_op], weighted_flux_o)
            for s in range(n_sps):
                dj = det_jacs[oc, s]
                for v in range(5):
                    correction[oc, s, v] += -contrib_owner[s, v] / dj

        if (not is_boundary[f]) and neighbor_is_primary[f]:
            nc = neighbor_cell[f]
            nc_op = neighbor_face_op[f]
            E_n = boundary_extrap_native[nc_op]

            Q_n_native = _extrap_matmul(Q[nc], E_n)
            adjrow_n_native = neighbor_adj_row_exact[f]  # (n_fp,3)，逐 FP 精确值，见函数文档

            flux_neighbor = np.empty((n_fp, 5))
            for i in range(n_fp):
                # 另一侧状态（幽灵态 / sources 插值 / 混合拆分面配对幽灵态）
                Q_o_at_n = np.zeros(5)
                c0 = owner_src0_cell[f]
                if c0 >= 0:
                    mat0 = owner_src0_tpl[owner_src0_tid[f]]
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
                # 混合拆分面（B-8，与主 kernel 同步；逐元素拷贝避免 numba C/A 布局赋值冲突）
                mp_o = mixed_ow_partner[f]
                if mp_o >= 0 and mixed_ow_mask[f, i]:
                    for v in range(5):
                        Q_o_at_n[v] = Q_ghost[mp_o, i, v]
                flux_neighbor[i] = inviscid_common_flux_point(Q_n_native[i], Q_o_at_n, adjrow_n_native[i], mach_ref)

            weighted_flux_n = np.empty((n_fp, 5))
            for i in range(n_fp):
                w_area = ref_area_weight[i]
                for v in range(5):
                    weighted_flux_n[i, v] = w_area * flux_neighbor[i, v]
            contrib_neighbor = matmul_small(lift_native[nc_op], weighted_flux_n)
            for s in range(n_sps):
                dj = det_jacs[nc, s]
                for v in range(5):
                    correction[nc, s, v] += -contrib_neighbor[s, v] / dj


def compute_boundary_ghost_states(flat, Q: np.ndarray, ghost_provider) -> np.ndarray:
    """全部活跃边界面（含混合拆分面的边界子面记录）通量点上的幽灵态。

    本侧状态按面算子 `boundary_extrap_native[owner_face_op]` 外插到通量点（四面体 [6,10)、
    棱柱 [10,15) 同一张表），再交给 `ghost_provider.ghost_states` 一次性批量构造。
    混合拆分面的边界子面 owner_is_primary 为 False（不参与残差累加），但它的幽灵态仍被配对的
    内部面逐通量点读取，所以同样要算。

    Returns:
        Q_ghost: (n_faces, n_fp, 5)，只有活跃边界面的行有意义。
    """
    Q_ghost = np.zeros((flat.n_faces, flat.n_fp, 5))
    faces = np.flatnonzero(flat.is_boundary & (flat.owner_is_primary | flat.mixed_bnd_face))
    if faces.size == 0:
        return Q_ghost
    E_o = flat.boundary_extrap_native[flat.owner_face_op[faces]]
    Q_o = np.einsum('fps,fsv->fpv', E_o, Q[flat.owner_cell[faces]])
    Q_ghost[faces] = ghost_provider.ghost_states(faces, Q_o, flat.true_normal[faces])
    return Q_ghost


