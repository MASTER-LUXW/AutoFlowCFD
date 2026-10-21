"""无粘残差界面项 —— numba 逐点标量 kernel（`core/fr_residual/inviscid.py` 调用）。

控制流：owner_is_primary / 本侧精确度量法向 / is_boundary 走幽灵态或 sources 求和 / neighbor_is_primary。
逐点公共通量与解析 Jacobian、GPU 版共用 `face_point_jumps.inviscid_common_flux_point`。

**关键正确性约束（不能"优化掉"）**：owner 侧和 neighbor 侧对同一个
内部面各自独立调用一次 AUSM+up（各自用自己的度量法向、自己原生 FP
位置上插值出的状态），不能因为看起来像同一个黎曼问题就合并成一次
公共通量复用给两侧——这个"优化"已在真实网格上证伪（自由流场残差从 9e-5 恶化到 3.1e7），
必须保留两次独立调用。

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

    `owner_face_op`/`neighbor_face_op` 是两侧的面算子索引（面编码 + 三角形面的
    坍缩顶点槽位，`fr/operators/face_ops.py`），统一索引
    `boundary_extrap_native`/`lift_native` 两张整表。`owner_axis`/`owner_side` 对
    原生面存的是复用槽位，**不是** axis/side 语义，本函数不读它们。

    (a) 自身面外插矩阵用 `boundary_extrap_native[op]`；(b) 方向定向系数固定为 +1
    （`fr/face_flux_points/exact_normal.py` 已经给出外向定向的 adj 行，再乘一次 side 会让 `side = -1` 的面
    反号）；(c) 面修正项用
    DG 提升算子 `lift_native[op] @ (ref_area_weight ⊙ |a| F_common)`（本侧通量迹已并进体积算子，见 `fr/face_flux_trace.py`）
    （`native_tet/basis.py::build_native_tet_lift` "弱形式提升定义"）。

    构造期护栏（`core/fr_operators/face_kernels.py::build_flat_face_geometry`）保证全部面编码都是原生编码
    （四面体 6~9、棱柱 10~14）。

    **法向一律取自本侧精确度量行**：adj 行是逐 FP 的解析精确值，夹角 >60° 说明几何本身就是那样（严重扭曲的
    棱柱四边形面对两个平面三角形）。不能在夹角大时改用 `true_normal`：拥有侧投影仍用 adj_row，对均匀流
        jump = |adj| F(Q).(n_ref - dir(adj)) != 0。
    plate_demo_volume_les（179,237 单元）上它在 11 个面、42 个 FP 触发
    （全部在板锐边的扭曲棱柱上），在"全边界取远场"的均匀流里产生
    4~5e6 /s 的残差（其余单元为 1e-7 量级），是 P1 长程运行单步爆炸到 1e53 的源头。

    两侧的 adj 行（`owner_adj_row_exact`/`neighbor_adj_row_exact`）是 mesh 加载阶段在每个 FP 自己的精确参考
    坐标处预计算的解析 Jacobian 值（`fr/face_flux_points/exact_normal.py`），不经过解点度量的外插。
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


def boundary_ghost_faces(flat) -> np.ndarray:
    """需要幽灵态的面（活跃边界面 + 混合拆分面的边界子面记录），主机 int64 数组，按面几何对象缓存（面表构造后
    不变；设备端面几何只在第一次调用时把三个掩码取回主机）。"""
    cached = getattr(flat, "_ghost_face_ids", None)
    if cached is not None:
        return cached
    from autoflowcfd.core.utils.array_module import to_host

    mask = to_host(flat.is_boundary).astype(bool) & (
        to_host(flat.owner_is_primary).astype(bool) | to_host(flat.mixed_bnd_face).astype(bool))
    flat._ghost_face_ids = np.flatnonzero(mask).astype(np.int64)
    return flat._ghost_face_ids


def compute_boundary_ghost_states(flat, Q, ghost_provider):
    """全部活跃边界面（含混合拆分面的边界子面记录）通量点上的幽灵态。

    本侧状态按面算子 `boundary_extrap_native[owner_face_op]` 外插到通量点（四面体 [6,10)、
    棱柱 [10,15) 同一张表），再交给 `ghost_provider.ghost_states` 一次性批量构造。
    混合拆分面的边界子面 owner_is_primary 为 False（不参与残差累加），但它的幽灵态仍被配对的
    内部面逐通量点读取，所以同样要算。

    CPU 与 GPU 共用（`flat` 为主机 `FlatFaceGeometry` 或设备 `GPUFlatFaceGeometry`，`Q` 在同一端）：GPU 的外插与
    常规边界条件都在设备上算。2026-10-19（I-119）以前 GPU 残差（无粘、粘性、源项梯度提升）每次求值都把整份 Q
    拷回主机、在主机上算、再把全部面大小的数组传回设备——隐式下每次 J v 两次全场 PCIe 往返。

    Returns:
        Q_ghost: (n_faces, n_fp, 5)，只有活跃边界面的行有意义（与 `Q` 同一数组模块）。
    """
    from autoflowcfd.core.utils.array_module import array_module

    xp = array_module(Q)
    n_fp = flat.true_normal.shape[1]
    Q_ghost = xp.zeros((flat.n_faces, n_fp, 5))
    faces = boundary_ghost_faces(flat)
    if faces.size == 0:
        return Q_ghost
    idx = faces if xp is np else xp.asarray(faces)
    E_o = flat.boundary_extrap_native[flat.owner_face_op[idx]]
    Q_o = xp.einsum('fps,fsv->fpv', E_o, Q[flat.owner_cell[idx]])
    Q_ghost[idx] = ghost_provider.ghost_states(faces, Q_o, flat.true_normal[idx])
    return Q_ghost


