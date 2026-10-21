"""
AutoFlowCFD - FR 单元-面 Flux Points 几何组装

棱柱四边形侧面被网格生成器恒定拆分成 2 个三角形子面，约 5% 的棱柱其
侧面 2 条子面记录指向 2 个不同的相邻四面体（multi-source）。本模块正确
处理这种“一个 owner/neighbor 立方体面对应 1~2 个不同真实相邻单元”的
情形：每个 (cell, 立方体面) 分组只让一条子面记录（primary）触发原生
FP 外插，FP 按对角线解析分类分别匹配到正确的相邻单元。

核心类和辅助函数位于 face_flux_points_data 模块。
"""

import numpy as np
from loguru import logger

from .basis_inverses import build_basis_inverses
from ..native_padding import sps_width
from .face_metrics import build_face_metrics
from .grouping import _second_half_frac, build_face_groups
from .data import _KernelFaceData, _classify_half, prism_quad_local_idx
from .validation import validate_face_flux_point_residuals
from .templates import deduplicate_face_matrices
from autoflowcfd.grid.connectivity.face_connectivity import FRFaceConnectivity

# numba 并行 kernel（延迟导入避免启动时 numba 编译阻塞）
_build_fp_newton_parallel = None
_build_ms_interp_parallel = None


def _get_numba_kernel():
    """延迟导入并返回 numba 并行 kernel。"""
    global _build_fp_newton_parallel
    if _build_fp_newton_parallel is None:
        from .kernel import build_fp_newton_parallel
        _build_fp_newton_parallel = build_fp_newton_parallel
    return _build_fp_newton_parallel


def _get_ms_numba_kernel():
    """延迟导入并返回 multi-source 插值矩阵 numba kernel。"""
    global _build_ms_interp_parallel
    if _build_ms_interp_parallel is None:
        from .kernel_multisource import build_ms_interp_parallel
        _build_ms_interp_parallel = build_ms_interp_parallel
    return _build_ms_interp_parallel


def _merge_single_cell_halves(faces, mixed, pn_cell, sec_cell, mask, interp, extra_cells, extra_mats, src1_idx):
    """多源面里两半来自同一个单元的（共形棱柱-棱柱四边形面，两条三角形子面的跨单元邻居是同一个棱柱）：
    次源矩阵（只在另一半的通量点行非零）并进主源矩阵（只在本半的行非零）、改为单一来源；次源表只保留
    真正来自两个不同单元的面。返回压缩后的 `(extra_cells, extra_mats)`，`src1_idx` 原地重排。

    两个矩阵在对方的行上都是零，并入后每行的权重与求和顺序都不变；并入的那一半随主源矩阵一起进模板去重
    （`templates.py`，同一模板内逐元素差 < 1e-13），所以与并入前只差模板容差：cube_demo 粗网格（28.6 万
    单元）P2 基态残差最大相对差 1.8e-13。两侧各约 13.6 万个多源面，真正两个单元的只有 297 个：次源表从各
    0.25 GiB 降到约 0.5 MiB，界面核在这些面上也少一次来源插值。混合分组（另半区是域边界）没有次源，不进表。"""
    same = ~mixed & (pn_cell == sec_cell)
    if same.any():
        fs = faces[same]
        interp[fs] = np.where(mask[fs][:, :, None], interp[fs], extra_mats[np.nonzero(same)[0]])
        src1_idx[fs] = -1
    keep = np.nonzero(~mixed & ~same)[0]
    src1_idx[faces[keep]] = np.arange(keep.size, dtype=np.int64)
    return extra_cells[keep], np.ascontiguousarray(extra_mats[keep])


def _require_native_face_codes(face_conn: FRFaceConnectivity) -> None:
    """全部面记录（边界面只看 owner 侧）的面编码必须是原生面编码 [6,15)：四面体 [6,10)、棱柱 [10,15)。

    numba kernel 只实现原生基的插值；面编码 0~5（坍缩立方体面）随坍缩基一并删除，若出现说明面连接关系没有经过
    原生面编码翻译，必须报错而不是在 kernel 里静默走错分支。
    """
    from autoflowcfd.grid.connectivity.face_connectivity import NATIVE_PRISM_FACE_CODE_RANGE, NATIVE_TET_FACE_CODE_RANGE

    lo, hi = NATIVE_TET_FACE_CODE_RANGE[0], NATIVE_PRISM_FACE_CODE_RANGE[1]
    codes = [np.asarray(face_conn.owner_cube_face),
             np.asarray(face_conn.neighbor_cube_face)[~np.asarray(face_conn.is_boundary, dtype=bool)]]
    bad = sum(int(np.count_nonzero((c < lo) | (c >= hi))) for c in codes)
    if bad:
        raise ValueError(f"{bad} 个面记录的面编码不在原生区间 [{lo},{hi})：面连接关系没有翻译成原生面编码")


def build_face_flux_points(face_conn: FRFaceConnectivity, mesh) -> _KernelFaceData:
    """为 face_conn 里的每个面预计算 Flux Points 几何（正确处理棱柱四边形
    侧面被拆分成 1~2 个真实相邻单元的情形，见模块文档）。"""
    from autoflowcfd.fr.operators import gauss_legendre

    n1d = mesh.n_points_1d
    sps_1d, weights_1d = gauss_legendre(n1d)
    n_fp = n1d * n1d
    n_faces = face_conn.n_faces
    n_prism = mesh.n_prism_cells

    # 原生 FP 网格 (u,v) 参考坐标（u=其中一个自由轴，v=另一个，升序，
    # 与 face_ref_grid 的构造顺序一致）：解析判断每个 FP 落在四边形对角线
    # 哪一侧。真实对角线可能是"标准"情形（连接第0、2个角点，参考坐标线
    # u=v）或"翻转"情形（连接第1、3个角点，参考坐标线 u=-v）之一，取决于
    # 该棱柱局部存储顺序是否与按全局节点编号排序后的顺序一致（见
    # `_prism_quad_diagonal_local` 文档）——两种情形都要预先算好对应的
    # FP 掩码，供下面多源面按每个四边形的实际对角线选用。
    g1, g2 = np.meshgrid(sps_1d, sps_1d, indexing="ij")
    is_lower_fp_standard = g1.ravel() >= g2.ravel()
    is_lower_fp_flipped = (g1.ravel() + g2.ravel()) <= 0.0

    grp = build_face_groups(
        face_conn, mesh, n_faces, n_fp, n_prism,
        is_lower_fp_standard, is_lower_fp_flipped)
    owner_groups = grp.owner_groups
    neighbor_groups = grp.neighbor_groups
    owner_primary = grp.owner_primary
    neighbor_primary = grp.neighbor_primary
    mixed_nb_partner = grp.mixed_nb_partner
    mixed_ow_partner = grp.mixed_ow_partner
    mixed_nb_mask = grp.mixed_nb_mask
    mixed_ow_mask = grp.mixed_ow_mask
    mixed_bnd_face = grp.mixed_bnd_face
    p0_second_half_frac = grp.p0_second_half_frac

    # ---- numba 并行 Newton + 插值矩阵预计算 ----
    logger.info("Running numba parallel Newton+interp kernel for FP geometry...")
    kernel = _get_numba_kernel()
    # 准备 flat 连接数组（numba 需要连续内存）。单一单元类型的网格
    # （纯棱柱 BL 通道、纯四面体 TGV 等验证算例）另一种 connectivity
    # 在 HighOrderMesh.load_from_volume_mesh 里按设计留 None（不是空
    # 数组，见该方法文档）——下面的 kernel 调用同时显式传了 n_prism
    # （棱柱数）来界定循环范围，用空数组占位不会让 kernel 越界访问
    # 不存在的那一类单元，只是避免在这里对 None 调用 .ravel() 崩溃。
    prism_conn_flat = np.ascontiguousarray(
        (mesh._fixed_prism_conn if mesh._fixed_prism_conn is not None
         else np.empty((0, 6), dtype=np.int64)).ravel().astype(np.int32)
    )
    tet_conn_flat = np.ascontiguousarray(
        (mesh._fixed_tet_conn if mesh._fixed_tet_conn is not None
         else np.empty((0, 4), dtype=np.int64)).ravel().astype(np.int32)
    )
    node_coords = np.ascontiguousarray(mesh._node_coords.astype(np.float64))
    n_sps = sps_width(n1d - 1)
    _require_native_face_codes(face_conn)
    binv = build_basis_inverses(n1d - 1)
    v_sps_inv_native = binv.v_sps_inv_native
    native_mode_i = binv.native_mode_i
    native_mode_j = binv.native_mode_j
    native_mode_k = binv.native_mode_k
    v_sps_inv_np = binv.v_sps_inv_np
    np_mode_i = binv.np_mode_i
    np_mode_j = binv.np_mode_j
    np_mode_k = binv.np_mode_k
    # 三角形面的坍缩顶点槽位（`fr/triangle_apex.py`）：只由单元连接关系决定，
    # 共享面两侧因此取同一批物理通量点
    from ..triangle_apex import cell_triangle_slots, face_record_slots

    cell_slots = cell_triangle_slots(mesh._fixed_tet_conn, mesh._fixed_prism_conn)
    owner_tri_slot = face_record_slots(cell_slots, face_conn.owner_cell, face_conn.owner_cube_face)
    neighbor_tri_slot = face_record_slots(
        cell_slots, np.where(face_conn.is_boundary, -1, face_conn.neighbor_cell), face_conn.neighbor_cube_face)
    (
        _nb_fc, _nb_resid, _ow_fc, _ow_resid,
        _nb_interp, _ow_interp, _nb_cell_id, _ow_cell_id,
    ) = kernel(
        n_faces, n_prism, n1d, n_fp, n_sps,
        np.ascontiguousarray(sps_1d.astype(np.float64)),
        np.ascontiguousarray(face_conn.is_boundary),
        np.ascontiguousarray(face_conn.owner_cell.astype(np.int32)),
        np.ascontiguousarray(face_conn.owner_cube_face.astype(np.int32)),
        np.ascontiguousarray(face_conn.neighbor_cell.astype(np.int32)),
        np.ascontiguousarray(face_conn.neighbor_cube_face.astype(np.int32)),
        np.ascontiguousarray(face_conn.area.astype(np.float64)),
        np.ascontiguousarray(face_conn.normal.astype(np.float64)),
        np.ascontiguousarray(face_conn.face_translation.astype(np.float64)),
        prism_conn_flat,
        tet_conn_flat,
        node_coords,
        np.ascontiguousarray(owner_primary),
        np.ascontiguousarray(neighbor_primary),
        v_sps_inv_native,
        native_mode_i, native_mode_j, native_mode_k,
        v_sps_inv_np, np_mode_i, np_mode_j, np_mode_k,
        owner_tri_slot, neighbor_tri_slot,
    )
    logger.info("Numba parallel kernel completed.")

    # 缓存常用数组引用
    _is_bnd = face_conn.is_boundary
    _op = owner_primary
    _np_ = neighbor_primary
    fm = build_face_metrics(
        face_conn, mesh, n_faces, n1d, n_prism, sps_1d, weights_1d, _is_bnd)
    _all_normals = fm.true_normal
    _all_area_w = fm.true_area_weight
    _owner_adj_row_exact = fm.owner_adj_row_exact
    _neighbor_adj_row_exact = fm.neighbor_adj_row_exact

    # ---- 直接构建 flat 源数组 ----
    # 内存说明（P3 阶数 OOM 排查，2026-08-21）：nb_src0_mat/ow_src0_mat 曾经
    # 各自独立 np.zeros((n_faces,n_fp,n_sps)) 分配、再从 kernel 输出的
    # _nb_interp/_ow_interp（同形状、同 dtype）逐元素拷贝进去——但真实数据
    # 追踪证实 _nb_interp/_ow_interp 在 kernel（含下面的 multi-source
    # kernel 原地写入）跑完之后，其内容与最终应存入 nb_src0_mat/ow_src0_mat
    # 的值逐位相等（对每个面 f：kernel 只在 owner_primary[f] 为真、且非
    # 多源棱柱四边形面时写入 _nb_interp[f]，其余情形 _nb_interp[f] 保持
    # 初始化的全零；nb_src0_mat 的赋值范围恰好覆盖同一个集合，非该集合的
    # 面在原实现里也是保持全零——两者对全部 n_faces 逐位恒等，不是巧合，
    # 是因为本来就是同一份计算结果的两次冗余存储）。这个逐元素拷贝纯粹是
    # 浪费——在 P3（n_fp=16,n_sps=64）下每个矩阵约 14.3GiB，"kernel 输出
    # 缓冲区"+"这里的拷贝目标"同时存活会让这两个数组的峰值内存翻倍到约
    # 57GiB，是真实 OOM 崩溃的直接原因之一（而不仅是矩阵本身"必须稠密"）。
    # 修复：不再单独分配 nb_src0_mat/ow_src0_mat 并拷贝，kernel 输出的
    # _nb_interp/_ow_interp（已被下面的向量化赋值和 multi-source kernel
    # 原地写满）在处理完 multi-source 面之后直接作为 nb_src0_mat/
    # ow_src0_mat 本身使用（零拷贝别名，不是近似）——数值上与旧实现逐位
    # 相同,只是消除了一次完全冗余的 14.3GiB x2 拷贝。
    logger.info("Building flat source arrays from kernel output...")
    nb_src0_cell = np.full(n_faces, -1, dtype=np.int64)
    nb_src1_idx = np.full(n_faces, -1, dtype=np.int64)
    ow_src0_cell = np.full(n_faces, -1, dtype=np.int64)
    ow_src1_idx = np.full(n_faces, -1, dtype=np.int64)

    # ---- 向量化：单源面直接批量拷贝（~95% 的面） ----
    _nb_single = _nb_cell_id >= 0  # 单源 neighbor 掩码
    _ow_single = _ow_cell_id >= 0  # 单源 owner 掩码
    nb_src0_cell[_nb_single] = _nb_cell_id[_nb_single]
    ow_src0_cell[_ow_single] = _ow_cell_id[_ow_single]
    # 矩阵本身不再拷贝：_nb_interp[_nb_single]/_ow_interp[_ow_single] 已经是
    # kernel 直接写入的最终值，见上面"内存说明"——nb_src0_mat/ow_src0_mat
    # 在函数末尾直接别名到 _nb_interp/_ow_interp。
    logger.info(f"  Single-source vectorized: nb={np.sum(_nb_single)}, ow={np.sum(_ow_single)}")

    # ---- 仅遍历 multi-source 面（~5% 棱柱四边形面）+ 边界面跳过 ----
    _multi_nb_faces = np.where(~_nb_single & ~_is_bnd & _op)[0]
    _multi_ow_faces = np.where(~_ow_single & ~_is_bnd & _np_)[0]
    logger.info(f"  Multi-source faces: nb={len(_multi_nb_faces)}, ow={len(_multi_ow_faces)}")

    # ---- Multi-source 面：numba 并行 kernel 替代 Python 串行循环 ----
    if len(_multi_nb_faces) > 0 or len(_multi_ow_faces) > 0:
        logger.info("Running multi-source numba kernel for multi-source faces...")

        # 预计算 secondary cell 信息（每组第二条子面的跨单元邻居）
        _ms_nb_sec_cell = np.empty(len(_multi_nb_faces), dtype=np.int32)
        _ms_nb_sec_cf = np.empty(len(_multi_nb_faces), dtype=np.int32)
        _ms_nb_mixed = np.zeros(len(_multi_nb_faces), dtype=np.bool_)
        # secondary 子面自身的 face_conn 索引（不是 cell/code，是面记录本身）——
        # 供残差校验按 secondary 子面*自己的*面积算特征尺度（逐半区各自用
        # `face_conn.area[gf]`；不能借用 primary 子面 f 的面积，两个三角子面
        # 面积一般不相等）。
        _ms_nb_other_face = np.full(len(_multi_nb_faces), -1, dtype=np.int64)
        for i, f in enumerate(_multi_nb_faces):
            if mixed_nb_partner[f] >= 0:
                # 混合分组：无真实次源（另半区是域边界），ms kernel 按标志跳过；
                # 占位值不会被读取（src1_idx 保持 -1）。
                _ms_nb_mixed[i] = True
                _ms_nb_sec_cell[i] = 0
                _ms_nb_sec_cf[i] = 0
                continue
            key = (int(face_conn.owner_cell[f]), int(face_conn.owner_cube_face[f]))
            group = sorted(owner_groups[key])
            other = group[1] if group[0] == f else group[0]
            _ms_nb_sec_cell[i] = face_conn.neighbor_cell[other]
            _ms_nb_sec_cf[i] = face_conn.neighbor_cube_face[other]
            _ms_nb_other_face[i] = other
            p0_second_half_frac[f] = _second_half_frac(face_conn.area, f, other)

        _ms_ow_sec_cell = np.empty(len(_multi_ow_faces), dtype=np.int32)
        _ms_ow_sec_cf = np.empty(len(_multi_ow_faces), dtype=np.int32)
        _ms_ow_mixed = np.zeros(len(_multi_ow_faces), dtype=np.bool_)
        _ms_ow_other_face = np.full(len(_multi_ow_faces), -1, dtype=np.int64)
        for i, f in enumerate(_multi_ow_faces):
            if mixed_ow_partner[f] >= 0:
                # 混合分组（neighbor 角色）：处理同 nb 分支。
                _ms_ow_mixed[i] = True
                _ms_ow_sec_cell[i] = 0
                _ms_ow_sec_cf[i] = 0
                continue
            key = (int(face_conn.neighbor_cell[f]), int(face_conn.neighbor_cube_face[f]))
            group = sorted(neighbor_groups[key])
            other = group[1] if group[0] == f else group[0]
            _ms_ow_sec_cell[i] = face_conn.owner_cell[other]
            _ms_ow_sec_cf[i] = face_conn.owner_cube_face[other]
            _ms_ow_other_face[i] = other
            p0_second_half_frac[f] = _second_half_frac(face_conn.area, f, other)

        # 计算对角线掩码（primary half = 分组首条子面覆盖的半侧）
        nb_mask = np.zeros((n_faces, n_fp), dtype=bool)
        for i, f in enumerate(_multi_nb_faces):
            key = (int(face_conn.owner_cell[f]), int(face_conn.owner_cube_face[f]))
            group = sorted(owner_groups[key])
            primary_f = group[0]
            cell_node_ids = mesh._fixed_prism_conn[key[0]]
            quad_local_idx = prism_quad_local_idx(key[1])
            half, is_std = _classify_half(
                cell_node_ids, quad_local_idx, face_conn.face_node_ids[primary_f]
            )
            is_lower = is_lower_fp_standard if is_std else is_lower_fp_flipped
            nb_mask[f] = is_lower if half == "lower" else ~is_lower

        ow_mask = np.zeros((n_faces, n_fp), dtype=bool)
        for i, f in enumerate(_multi_ow_faces):
            key = (int(face_conn.neighbor_cell[f]), int(face_conn.neighbor_cube_face[f]))
            group = sorted(neighbor_groups[key])
            primary_f = group[0]
            cell_node_ids = mesh._fixed_prism_conn[key[0]]
            quad_local_idx = prism_quad_local_idx(key[1])
            half, is_std = _classify_half(
                cell_node_ids, quad_local_idx, face_conn.face_node_ids[primary_f]
            )
            is_lower = is_lower_fp_standard if is_std else is_lower_fp_flipped
            ow_mask[f] = is_lower if half == "lower" else ~is_lower

        # 分配 extra 数组（float64——同一套跨单元插值矩阵，与 nb_interp/
        # ow_interp 一样存在坍缩坐标模态 Vandermonde 条件数病态问题，
        # 见 face_flux_points/kernel.py::build_fp_newton_parallel 文档，
        # 不能降精度）
        n_extra_nb = len(_multi_nb_faces)
        n_extra_ow = len(_multi_ow_faces)
        _nb_extra_mats_arr = np.zeros((max(n_extra_nb, 1), n_fp, n_sps), dtype=np.float64)
        _ow_extra_mats_arr = np.zeros((max(n_extra_ow, 1), n_fp, n_sps), dtype=np.float64)
        _ms_nb_extra_idx = np.arange(n_extra_nb, dtype=np.int64)
        _ms_ow_extra_idx = np.arange(n_extra_ow, dtype=np.int64)
        # secondary Newton 逐 FP 残差输出（此前算出即丢弃，见
        # face_flux_points/kernel_multisource.py::build_ms_interp_parallel 文档）；
        # mixed 分组（无 secondary Newton）对应行保持全零，下面校验时按
        # ~_ms_{nb,ow}_mixed 排除。
        _nb_sec_resid_arr = np.zeros((max(n_extra_nb, 1), n_fp), dtype=np.float64)
        _ow_sec_resid_arr = np.zeros((max(n_extra_ow, 1), n_fp), dtype=np.float64)

        # Primary neighbor/owner cell & code（分组首条子面的跨单元邻居）
        _ms_nb_pn_cell = np.empty(len(_multi_nb_faces), dtype=np.int32)
        _ms_nb_pn_code = np.empty(len(_multi_nb_faces), dtype=np.int32)
        for i, f in enumerate(_multi_nb_faces):
            key = (int(face_conn.owner_cell[f]), int(face_conn.owner_cube_face[f]))
            primary_f = sorted(owner_groups[key])[0]
            _ms_nb_pn_cell[i] = face_conn.neighbor_cell[primary_f]
            _ms_nb_pn_code[i] = face_conn.neighbor_cube_face[primary_f]

        _ms_ow_pn_cell = np.empty(len(_multi_ow_faces), dtype=np.int32)
        _ms_ow_pn_code = np.empty(len(_multi_ow_faces), dtype=np.int32)
        for i, f in enumerate(_multi_ow_faces):
            key = (int(face_conn.neighbor_cell[f]), int(face_conn.neighbor_cube_face[f]))
            primary_f = sorted(neighbor_groups[key])[0]
            _ms_ow_pn_cell[i] = face_conn.owner_cell[primary_f]
            _ms_ow_pn_code[i] = face_conn.owner_cube_face[primary_f]

        # 调用 multi-source numba kernel
        ms_kernel = _get_ms_numba_kernel()
        ms_kernel(
            len(_multi_nb_faces), len(_multi_ow_faces),
            _multi_nb_faces, _multi_ow_faces,
            _ms_nb_sec_cell, _ms_nb_sec_cf, _ms_nb_extra_idx,
            _ms_ow_sec_cell, _ms_ow_sec_cf, _ms_ow_extra_idx,
            nb_mask, ow_mask,
            _ms_nb_mixed, _ms_ow_mixed,
            n_prism, n1d, n_fp, n_sps, np.ascontiguousarray(sps_1d.astype(np.float64)),
            np.ascontiguousarray(face_conn.owner_cell.astype(np.int32)),
            np.ascontiguousarray(face_conn.owner_cube_face.astype(np.int32)),
            np.ascontiguousarray(face_conn.neighbor_cell.astype(np.int32)),
            np.ascontiguousarray(face_conn.neighbor_cube_face.astype(np.int32)),
            np.ascontiguousarray(face_conn.area.astype(np.float64)),
            np.ascontiguousarray(face_conn.face_translation.astype(np.float64)),
            prism_conn_flat, tet_conn_flat, node_coords,
            _nb_fc, _ow_fc,
            v_sps_inv_native, native_mode_i, native_mode_j, native_mode_k,
            v_sps_inv_np, np_mode_i, np_mode_j, np_mode_k,
            _nb_interp, _ow_interp,
            _nb_cell_id, _ow_cell_id,
            _nb_extra_mats_arr, _ow_extra_mats_arr,
            _ms_nb_pn_cell, _ms_nb_pn_code,
            _ms_ow_pn_cell, _ms_ow_pn_code,
            _nb_sec_resid_arr, _ow_sec_resid_arr,
        )
        logger.info("Multi-source numba kernel completed.")

        # 次源单元 = 次源插值矩阵构造时用的那个单元（另一条子面的跨单元邻居 `_ms_*_sec_cell`）。2026-10-15
        # 之前这里记的是面 f 自己的邻居：f 是分组首条子面，于是次源单元与主源相同，另一半通量点用 A 单元的
        # 解点值配 B 单元的插值权重（两侧公共通量不相等、离散不守恒）。共形的棱柱-棱柱四边形面两条子面邻居
        # 相同，看不出来；棱柱四边形侧面挨着两个四面体时错（cube_demo 粗网格 297 处，
        # tests/unit/test_prism_quad_two_tet_sources.py）。混合分组没有次源（src1_idx 保持 -1），值不被读取。
        _nb_extra_cells, _nb_extra_mats_arr = _merge_single_cell_halves(
            _multi_nb_faces, _ms_nb_mixed, _ms_nb_pn_cell, _ms_nb_sec_cell, nb_mask, _nb_interp,
            _ms_nb_sec_cell.astype(np.int64), _nb_extra_mats_arr[:len(_multi_nb_faces)], nb_src1_idx)
        _ow_extra_cells, _ow_extra_mats_arr = _merge_single_cell_halves(
            _multi_ow_faces, _ms_ow_mixed, _ms_ow_pn_cell, _ms_ow_sec_cell, ow_mask, _ow_interp,
            _ms_ow_sec_cell.astype(np.int64), _ow_extra_mats_arr[:len(_multi_ow_faces)], ow_src1_idx)
        logger.info(f"  Two-cell multi-source faces kept as second sources: nb={len(_nb_extra_cells)}, "
                    f"ow={len(_ow_extra_cells)}")
        # src1_idx 由 `_merge_single_cell_halves` 定（混合分组无次源，保持 -1；另半区由残差 kernel 的
        # mixed_{nb,ow}_partner/mask 分支逐 FP 取边界幽灵态）
        # src0 cell（矩阵已由 ms_kernel 原地写进 _nb_interp/_ow_interp 的
        # multi-source 面对应位置，见上面"内存说明"，不再单独拷贝）
        nb_src0_cell[_multi_nb_faces] = _ms_nb_pn_cell.astype(np.int64)
        ow_src0_cell[_multi_ow_faces] = _ms_ow_pn_cell.astype(np.int64)
    else:
        _nb_extra_mats_arr = np.empty((0, n_fp, n_sps), dtype=np.float64)
        _ow_extra_mats_arr = np.empty((0, n_fp, n_sps), dtype=np.float64)
        _nb_extra_cells = np.empty(0, dtype=np.int64)
        _ow_extra_cells = np.empty(0, dtype=np.int64)
        nb_mask = np.zeros((n_faces, n_fp), dtype=bool)
        ow_mask = np.zeros((n_faces, n_fp), dtype=bool)
        _ms_nb_mixed = np.zeros(0, dtype=np.bool_)
        _ms_ow_mixed = np.zeros(0, dtype=np.bool_)
        _ms_nb_other_face = np.full(0, -1, dtype=np.int64)
        _ms_ow_other_face = np.full(0, -1, dtype=np.int64)
        _nb_sec_resid_arr = np.zeros((1, n_fp), dtype=np.float64)
        _ow_sec_resid_arr = np.zeros((1, n_fp), dtype=np.float64)

    # kernel 输出的 _nb_interp/_ow_interp 此时已经历完主 kernel 的向量化赋值范围与
    # multi-source kernel 的原地写入，就是逐面的 src0 矩阵（详见本函数开头"内存说明"）；
    # 再归并成模板表 + 逐面编号（`templates.py`：同一模板内逐元素差 < 1e-13，
    # P3 两份从 6.15 GiB 降到约 0.35 GiB），全尺寸数组随即释放
    nb_src0_tpl, nb_src0_tid = deduplicate_face_matrices(_nb_interp)
    ow_src0_tpl, ow_src0_tid = deduplicate_face_matrices(_ow_interp)
    del _nb_interp, _ow_interp

    nb_extra_cell = _nb_extra_cells
    nb_extra_mat = _nb_extra_mats_arr
    ow_extra_cell = _ow_extra_cells
    ow_extra_mat = _ow_extra_mats_arr

    # 校验 Newton/精确点位定位残差（拆到 face_flux_points/validation.py，
    # 见该模块文档——本文件此前超过 600 行硬性拆分阈值，这段"接回此前
    # 被丢弃的 kernel 残差输出、按阈值分级、超差即 raise"的收尾校验只读
    # 上面已经算好的残差数组和分组掩码，不产生 build_face_flux_points
    # 后续需要的返回值，是清晰的拆分边界）。
    validate_face_flux_point_residuals(
        face_conn, n_prism, n_faces,
        _nb_resid, _ow_resid,
        _nb_single, _ow_single,
        _multi_nb_faces, _multi_ow_faces,
        nb_mask, ow_mask,
        _ms_nb_mixed, _ms_ow_mixed,
        _nb_sec_resid_arr, _ow_sec_resid_arr,
        _ms_nb_other_face, _ms_ow_other_face,
    )

    logger.info("Flux Points geometry built (flat array format).")
    return _KernelFaceData(
        n_faces=n_faces, n_fp=n_fp, n_sps=n_sps, n1d=n1d,
        owner_is_primary=_op,
        neighbor_is_primary=_np_,
        true_normal=_all_normals,
        owner_adj_row_exact=_owner_adj_row_exact,
        neighbor_adj_row_exact=_neighbor_adj_row_exact,
        true_area_weight=_all_area_w,
        nb_src0_cell=nb_src0_cell,
        nb_src0_tpl=nb_src0_tpl, nb_src0_tid=nb_src0_tid,
        nb_src1_idx=nb_src1_idx,
        ow_src0_cell=ow_src0_cell,
        ow_src0_tpl=ow_src0_tpl, ow_src0_tid=ow_src0_tid,
        ow_src1_idx=ow_src1_idx,
        nb_extra_cell=nb_extra_cell,
        nb_extra_mat=nb_extra_mat,
        ow_extra_cell=ow_extra_cell,
        ow_extra_mat=ow_extra_mat,
        mixed_nb_partner=mixed_nb_partner,
        mixed_nb_mask=mixed_nb_mask,
        mixed_ow_partner=mixed_ow_partner,
        mixed_ow_mask=mixed_ow_mask,
        mixed_bnd_face=mixed_bnd_face,
        p0_second_half_frac=p0_second_half_frac,
        owner_tri_slot=owner_tri_slot,
        neighbor_tri_slot=neighbor_tri_slot,
        _face_conn=face_conn,
        _sps_1d=sps_1d,
        _owner_groups=owner_groups,
        _neighbor_groups=neighbor_groups,
    )
