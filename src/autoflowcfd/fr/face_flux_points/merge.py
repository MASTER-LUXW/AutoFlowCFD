"""
AutoFlowCFD - FR 单元-面 Flux Points 几何组装

棱柱四边形侧面被网格生成器恒定拆分成 2 个三角形子面，约 5% 的棱柱其
侧面 2 条子面记录指向 2 个不同的相邻四面体（multi-source）。本模块正确
处理这种“一个 owner/neighbor 立方体面对应 1~2 个不同真实相邻单元”的
情形：每个 (cell, 立方体面) 分组只让一条子面记录（primary）触发原生
FP 外插，FP 按对角线解析分类分别匹配到正确的相邻单元。

核心类和辅助函数位于 face_flux_points_data 模块。
"""

from typing import Dict, List, Tuple

import numpy as np
from loguru import logger

from .geometry import (
    CUBE_FACE_AXIS_SIDE,
    FaceFluxPointGeometry,
)
from .basis_inverses import build_basis_inverses
from .face_metrics import build_face_metrics
from .grouping import build_face_groups
from .data import (
    _KernelFaceData, _PRISM_QUAD_CODES, _classify_half, prism_quad_local_idx,
)
from .exact_normal import (
    compute_exact_face_normals_and_weights,
)
from .validation import validate_face_flux_point_residuals
from autoflowcfd.grid.curved_mapping.curved_mapping import PRISM_CUBE_FACES
from autoflowcfd.grid.connectivity.face_connectivity import (
    CUBE_FACE_NAMES,
    NATIVE_PRISM_FACE_CODE_RANGE,
    FRFaceConnectivity,
)

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

def build_face_flux_points(face_conn: FRFaceConnectivity, mesh) -> List[FaceFluxPointGeometry]:
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
    # FP 掩码，供 `_resolve_multi_source` 按每个四边形的实际对角线选用。
    g1, g2 = np.meshgrid(sps_1d, sps_1d, indexing="ij")
    is_lower_fp_standard = g1.ravel() >= g2.ravel()
    is_lower_fp_flipped = (g1.ravel() + g2.ravel()) <= 0.0

    grp = build_face_groups(
        face_conn, mesh, n_faces, n_fp, n_prism,
        is_lower_fp_standard, is_lower_fp_flipped)
    owner_groups = grp.owner_groups
    neighbor_groups = grp.neighbor_groups
    boundary_owner_groups = grp.boundary_owner_groups
    owner_primary = grp.owner_primary
    neighbor_primary = grp.neighbor_primary
    mixed_nb_partner = grp.mixed_nb_partner
    mixed_ow_partner = grp.mixed_ow_partner
    mixed_nb_mask = grp.mixed_nb_mask
    mixed_ow_mask = grp.mixed_ow_mask
    mixed_bnd_face = grp.mixed_bnd_face
    mixed_p0_bnd_frac = grp.mixed_p0_bnd_frac

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
    n_sps = n1d ** 3
    binv = build_basis_inverses(face_conn, n1d, sps_1d)
    v_sps_inv_tet = binv.v_sps_inv_tet
    v_sps_inv_prism = binv.v_sps_inv_prism
    v_sps_inv_native = binv.v_sps_inv_native
    native_mode_i = binv.native_mode_i
    native_mode_j = binv.native_mode_j
    native_mode_k = binv.native_mode_k
    v_sps_inv_np = binv.v_sps_inv_np
    np_mode_i = binv.np_mode_i
    np_mode_j = binv.np_mode_j
    np_mode_k = binv.np_mode_k
    (
        _nb_fc, _nb_resid, _ow_fc, _ow_resid,
        _nb_interp, _ow_interp, _nb_cell_id, _ow_cell_id,
        _geom_oa, _geom_os, _geom_na, _geom_ns,
        _geom_aw, _geom_n,
    ) = kernel(
        n_faces, n_prism, n1d, n_fp,
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
        v_sps_inv_tet,
        v_sps_inv_prism,
        v_sps_inv_native,
        native_mode_i, native_mode_j, native_mode_k,
        v_sps_inv_np, np_mode_i, np_mode_j, np_mode_k,
    )
    logger.info("Numba parallel kernel completed.")

    # ---- 预计算 axis/side 数组（消除逐面字典查找） ----
    _CF_AXIS = np.array([v[0] for v in CUBE_FACE_AXIS_SIDE.values()], dtype=np.int32)
    _CF_SIDE = np.array([v[1] for v in CUBE_FACE_AXIS_SIDE.values()])
    _o_axis_arr = _CF_AXIS[np.asarray(face_conn.owner_cube_face, dtype=np.int32)]
    _o_side_arr = _CF_SIDE[np.asarray(face_conn.owner_cube_face, dtype=np.int32)]
    # kernel 已计算 neighbor axis/side（含边界面 -1/0.0），直接使用
    _n_axis_arr = _geom_na
    _n_side_arr = _geom_ns
    # 缓存常用数组引用
    _is_bnd = face_conn.is_boundary
    _op = owner_primary
    _np_ = neighbor_primary
    fm = build_face_metrics(
        face_conn, mesh, n_faces, n1d, n_prism, sps_1d, weights_1d,
        _o_axis_arr, _o_side_arr, _n_axis_arr, _n_side_arr, _is_bnd)
    _all_normals = fm.true_normal
    _all_area_w = fm.true_area_weight
    _owner_adj_row_exact = fm.owner_adj_row_exact
    _neighbor_adj_row_exact = fm.neighbor_adj_row_exact

    # ---- 直接构建 flat 源数组（跳过 180 万 FaceFluxPointGeometry 对象创建）----
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
        # 供残差校验按 secondary 子面*自己的*面积算特征尺度，与旧慢速路径
        # `_resolve_multi_source` 逐半区各自用 `face_conn.area[gf]` 完全一致
        # （不能借用 primary 子面 f 的面积，两个三角子面面积一般不相等）。
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
            n_prism, n1d, n_fp, np.ascontiguousarray(sps_1d.astype(np.float64)),
            np.ascontiguousarray(face_conn.owner_cell.astype(np.int32)),
            np.ascontiguousarray(face_conn.owner_cube_face.astype(np.int32)),
            np.ascontiguousarray(face_conn.neighbor_cell.astype(np.int32)),
            np.ascontiguousarray(face_conn.neighbor_cube_face.astype(np.int32)),
            np.ascontiguousarray(face_conn.area.astype(np.float64)),
            np.ascontiguousarray(face_conn.face_translation.astype(np.float64)),
            prism_conn_flat, tet_conn_flat, node_coords,
            _nb_fc, _ow_fc,
            v_sps_inv_tet, v_sps_inv_prism,
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

        # 将 kernel 输出拷贝到紧凑 extra 列表
        _nb_extra_cells = face_conn.neighbor_cell[_multi_nb_faces].astype(np.int64) if len(_multi_nb_faces) > 0 else np.empty(0, dtype=np.int64)
        _ow_extra_cells = face_conn.owner_cell[_multi_ow_faces].astype(np.int64) if len(_multi_ow_faces) > 0 else np.empty(0, dtype=np.int64)
        # 更新 src1_idx（混合分组无次源，保持 -1；另半区由残差 kernel 的
        # mixed_{nb,ow}_partner/mask 分支逐 FP 取边界幽灵态）
        for i, f in enumerate(_multi_nb_faces):
            if mixed_nb_partner[f] < 0:
                nb_src1_idx[f] = i
        for i, f in enumerate(_multi_ow_faces):
            if mixed_ow_partner[f] < 0:
                ow_src1_idx[f] = i
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

    # nb_src0_mat/ow_src0_mat：零拷贝别名到 kernel 输出的 _nb_interp/
    # _ow_interp——此时两者已经历完主 kernel 的向量化赋值范围与
    # multi-source kernel 的原地写入，逐位等于旧实现里 nb_src0_mat/
    # ow_src0_mat 该有的值（详见本函数开头"内存说明"），不再重新分配、
    # 不再逐元素拷贝，避免 P3 阶数下这两个 ~14.3GiB 矩阵各自双份同时
    # 存活导致的 OOM。
    nb_src0_mat = _nb_interp
    ow_src0_mat = _ow_interp

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
        owner_axis=_o_axis_arr.astype(np.int64),
        owner_side=_o_side_arr.astype(np.float64),
        neighbor_axis=_geom_na.astype(np.int64),
        neighbor_side=_geom_ns.astype(np.float64),
        owner_is_primary=_op,
        neighbor_is_primary=_np_,
        true_normal=_all_normals,
        owner_adj_row_exact=_owner_adj_row_exact,
        neighbor_adj_row_exact=_neighbor_adj_row_exact,
        true_area_weight=_all_area_w,
        nb_src0_cell=nb_src0_cell,
        nb_src0_mat=nb_src0_mat,
        nb_src1_idx=nb_src1_idx,
        ow_src0_cell=ow_src0_cell,
        ow_src0_mat=ow_src0_mat,
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
        mixed_p0_bnd_frac=mixed_p0_bnd_frac,
        _mesh=mesh,
        _face_conn=face_conn,
        _sps_1d=sps_1d,
        _nb_fc=_nb_fc, _nb_resid=_nb_resid,
        _ow_fc=_ow_fc, _ow_resid=_ow_resid,
        _nb_cell_id=_nb_cell_id, _ow_cell_id=_ow_cell_id,
        _is_lower_fp_standard=is_lower_fp_standard,
        _is_lower_fp_flipped=is_lower_fp_flipped,
        _owner_groups=owner_groups,
        _neighbor_groups=neighbor_groups,
    )
