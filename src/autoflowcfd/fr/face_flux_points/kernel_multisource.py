"""
AutoFlowCFD - Multi-source 面插值矩阵 numba 并行 kernel

将棱柱四边形侧面（multi-source）的插值矩阵构建从纯 Python 串行循环
（每面调用 build_cross_interp，含 scipy lu_solve + 大量小数组分配）
改为 numba @njit(parallel=True) + prange 并行。

依赖主模块 face_flux_points_numba 中的辅助函数：
- 坐标变换：_FACE_AXIS, _FACE_SIDE
- 物理映射：_face_ref_grid_nb, _map_ref_nb
- Newton 迭代：_newton_locate_nb
- 模态基：经 `interp_matrix_from_cube_coords_nb` 间接使用（三条基共用一个入口）
"""

import numpy as np
from numba import njit, prange

from .kernel import (
    _FACE_AXIS,
    _FACE_SIDE,
    _face_ref_grid_nb,
    _map_ref_nb,
    _newton_locate_nb,
)
from .face_code_tables import (
    _NATIVE_PRISM_LO, _NATIVE_TET_HI, _NATIVE_TET_LO,
)
from .native_geometry_nb import (
    _tet_native_locate_nb,
    _native_alpha_beta_to_rst_nb,
    _native_interp_matrix_nb, interp_matrix_from_cube_coords_nb,
)


@njit(parallel=True, cache=True)
def build_ms_interp_parallel(
    n_ms_nb, n_ms_ow, ms_nb_faces, ms_ow_faces,
    ms_nb_sec_cell, ms_nb_sec_cube_face, ms_nb_extra_idx,
    ms_ow_sec_cell, ms_ow_sec_cube_face, ms_ow_extra_idx,
    nb_mask, ow_mask,
    ms_nb_mixed, ms_ow_mixed,
    n_prism, n1d, n_fp, sps_1d,
    owner_cell_arr, owner_cube_face_arr,
    neighbor_cell_arr, neighbor_cube_face_arr,
    area_arr, face_translation_arr,
    prism_conn, tet_conn, node_coords,
    nb_fc, ow_fc,
    v_sps_inv_tet, v_sps_inv_prism,
    v_sps_inv_native, native_mode_i, native_mode_j, native_mode_k,
    v_sps_inv_np, np_mode_i, np_mode_j, np_mode_k,
    nb_interp, ow_interp,
    nb_cell_id, ow_cell_id,
    nb_extra_mat, ow_extra_mat,
    ms_nb_pn_cell, ms_nb_pn_code,
    ms_ow_pn_cell, ms_ow_pn_code,
    nb_sec_resid, ow_sec_resid,
):
    """为 multi-source 面构建主/次插值矩阵（含 Newton + 对角线掩码）。

    native 四面体（路径C）支持：本 kernel 处理的棱柱四边形侧面自身
    （`oc_code`/`nc_code`）恒为 0~5（棱柱专属，见调用方 face_flux_points_
    merge.py 的分组条件），但跨单元的目标（`pn_code`/`sec_code`）可能是
    与之相邻的 native 四面体（code>=6，excluded_vertex=code-6）——两处
    都需要与 face_flux_points/kernel.py::build_fp_newton_parallel 同样的
    native 分支，见下方 pn_code/sec_code 判断处。primary interp 用的
    `nb_fc`/`ow_fc` 已经是主 kernel 算好的 native (alpha,beta) 自由坐标
    （原样存进同一个数组槽位，见 _tet_native_locate_nb 文档），这里只需
    `_native_alpha_beta_to_rst_nb` 还原成 (r,s,t)，不需要重新定位。

    直接写入 nb_interp/ow_interp（primary half）和 nb_extra_mat/ow_extra_mat
    （secondary half），避免额外内存分配。

    ms_{nb,ow}_pn_cell/code：primary 插值使用的 primary neighbor/owner
    （分组内第一条子面的跨单元邻居），而非 face f 自身的 neighbor/owner。
    Newton 自由坐标是相对 primary 邻居的参考空间计算的，primary interp
    必须使用同一邻居的节点。

    ms_{nb,ow}_mixed：混合分组标志（B-8，2026-08-25）。棱柱四边形侧面
    的两条子面一条在域边界、一条为内部界面时，内部子面记录作为整张面
    的 primary：primary half（对角线掩码内）照常构建跨单元插值；另半区
    没有真实相邻单元，由残差 kernel 逐 FP 取边界子面记录的幽灵态（见
    inviscid_kernel.py 的 mixed_{nb,ow}_partner/mask 分支），因此这里跳过
    Secondary Newton + interp，只写入 nb_cell_id/ow_cell_id 后结束。

    nb_sec_resid/ow_sec_resid：(max(n_ms_{nb,ow},1), n_fp) 输出数组，
    Secondary Newton 的逐 Flux Point 残差（sec_rs，绝对长度单位，未归约、
    未归一化）——之前算出来直接丢弃，现在写回供 face_flux_points/merge.py
    按 ~{nb,ow}_mask（secondary 半区掩码）取值校验。混合分组（Secondary
    Newton 被跳过）对应行保持初始化的全零，调用方必须只在非 mixed 索引上
    读取。
    """
    n_sps = n1d * n1d * n1d

    # ---- Neighbor multi-source ----
    for idx in prange(n_ms_nb):
        f = ms_nb_faces[idx]
        oc = owner_cell_arr[f]
        oc_code = owner_cube_face_arr[f]
        o_axis = _FACE_AXIS[oc_code]
        o_side = _FACE_SIDE[oc_code]
        nc = neighbor_cell_arr[f]
        nc_code = neighbor_cube_face_arr[f]
        n_axis = _FACE_AXIS[nc_code]
        n_side = _FACE_SIDE[nc_code]
        o_is_prism = oc < n_prism
        n_is_prism = nc < n_prism

        # 获取 owner 节点
        if o_is_prism:
            o_nd = np.empty((6, 3))
            for ni in range(6):
                nid = prism_conn[oc * 6 + ni]
                for d in range(3):
                    o_nd[ni, d] = node_coords[nid, d]
        else:
            o_nd = np.empty((4, 3))
            for ni in range(4):
                nid = tet_conn[(oc - n_prism) * 4 + ni]
                for d in range(3):
                    o_nd[ni, d] = node_coords[nid, d]

        # Owner 物理 FP（self-side = owner for nb multi-source）
        ref_o = _face_ref_grid_nb(n1d, o_axis, o_side, sps_1d)
        phys_o = _map_ref_nb(o_is_prism, ref_o, o_nd)
        a_val = area_arr[f]
        cl = np.sqrt(max(a_val, 1e-300))

        # ---- Primary interp（primary neighbor cell, 用预计算自由坐标）----
        # 使用 primary neighbor（分组内第一条子面的跨单元邻居），而非 face f 的
        # neighbor——Newton 自由坐标 nb_fc[f] 是相对 primary neighbor 的参考空间
        # 计算的，primary interp 必须使用同一邻居的节点和参考坐标
        pn_c = ms_nb_pn_cell[idx]
        pn_code = ms_nb_pn_code[idx]
        # 两类原生面分开判（`code >= 6` 原本等价于"是原生四面体面"，
        # 加了原生棱柱编码 [10,15) 之后必须区分，否则按 `code-6` 会越界
        # 取四面体的表；numba 不做边界检查。见主 kernel 同一处说明。
        pn_is_native_tet = _NATIVE_TET_LO <= pn_code < _NATIVE_TET_HI
        pn_is_native_prism = pn_code >= _NATIVE_PRISM_LO
        if pn_is_native_tet:
            rst_pn = _native_alpha_beta_to_rst_nb(pn_code - _NATIVE_TET_LO, nb_fc[f])
            interp_pn = _native_interp_matrix_nb(
                rst_pn, native_mode_i, native_mode_j, native_mode_k, v_sps_inv_native, n_sps
            )
            for p in range(n_fp):
                if nb_mask[f, p]:
                    nb_interp[f, p] = interp_pn[p]
        else:
            # 坍缩面与**原生棱柱面**共用一条（见主 kernel 同一处说明与
            # `interp_matrix_from_cube_coords_nb` 文档）。
            pn_axis = _FACE_AXIS[pn_code]
            pn_side = _FACE_SIDE[pn_code]
            pn_is_prism = pn_c < n_prism
            abc_n = np.empty((n_fp, 3))
            for p in range(n_fp):
                abc_n[p, pn_axis] = pn_side
                ix2 = 0
                for ax in range(3):
                    if ax != pn_axis:
                        abc_n[p, ax] = nb_fc[f, p, ix2]
                        ix2 += 1
            interp_pn = interp_matrix_from_cube_coords_nb(
                abc_n, pn_is_prism, pn_is_native_prism, n1d, n_sps,
                v_sps_inv_prism, v_sps_inv_tet,
                v_sps_inv_np, np_mode_i, np_mode_j, np_mode_k)
            for p in range(n_fp):
                if nb_mask[f, p]:
                    nb_interp[f, p] = interp_pn[p]

        # ---- Secondary Newton + interp ----
        sec_cell = ms_nb_sec_cell[idx]
        sec_code = ms_nb_sec_cube_face[idx]
        # 两类原生面分开判（`code >= 6` 原本等价于"是原生四面体面"，
        # 加了原生棱柱编码 [10,15) 之后必须区分，否则按 `code-6` 会越界
        # 取四面体的表；numba 不做边界检查。见主 kernel 同一处说明。
        sec_is_native_tet = _NATIVE_TET_LO <= sec_code < _NATIVE_TET_HI
        sec_is_native_prism = sec_code >= _NATIVE_PRISM_LO
        sec_is_prism = sec_cell < n_prism
        if sec_is_prism:
            sec_nd = np.empty((6, 3))
            for ni in range(6):
                nid = prism_conn[sec_cell * 6 + ni]
                for d in range(3):
                    sec_nd[ni, d] = node_coords[nid, d]
        else:
            sec_nd = np.empty((4, 3))
            for ni in range(4):
                nid = tet_conn[(sec_cell - n_prism) * 4 + ni]
                for d in range(3):
                    sec_nd[ni, d] = node_coords[nid, d]

        t = face_translation_arr[f]
        ht = abs(t[0]) > 1e-300 or abs(t[1]) > 1e-300 or abs(t[2]) > 1e-300
        if ht:
            # 真实 bug 修复（同 face_flux_points/kernel.py::build_fp_newton_parallel
            # 的 owner_primary 分支，见该处详细说明）：把 owner 侧物理 FP
            # 平移到 secondary（neighbor 一侧）单元所在区域，必须加
            # translation，此前写成减，残差恰好偏了 2*|translation|。
            search_sec = np.empty((n_fp, 3))
            for p in range(n_fp):
                search_sec[p, 0] = phys_o[p, 0] + t[0]
                search_sec[p, 1] = phys_o[p, 1] + t[1]
                search_sec[p, 2] = phys_o[p, 2] + t[2]
        else:
            search_sec = phys_o

        ei = ms_nb_extra_idx[idx]
        if sec_is_native_tet:
            sec_free, sec_rst, sec_rs = _tet_native_locate_nb(sec_nd, sec_code - _NATIVE_TET_LO, search_sec)
            interp_sec = _native_interp_matrix_nb(
                sec_rst, native_mode_i, native_mode_j, native_mode_k, v_sps_inv_native, n_sps
            )
            for p in range(n_fp):
                nb_sec_resid[ei, p] = sec_rs[p]
                if not nb_mask[f, p]:
                    nb_extra_mat[ei, p] = interp_sec[p]
        else:
            sec_axis = _FACE_AXIS[sec_code]
            sec_side = _FACE_SIDE[sec_code]
            sec_fc, sec_rs = _newton_locate_nb(
                sec_is_prism, sec_nd, sec_axis, sec_side, search_sec, cl
            )

            abc_sec = np.empty((n_fp, 3))
            for p in range(n_fp):
                abc_sec[p, sec_axis] = sec_side
                ix2 = 0
                for ax in range(3):
                    if ax != sec_axis:
                        abc_sec[p, ax] = sec_fc[p, ix2]
                        ix2 += 1
            interp_sec = interp_matrix_from_cube_coords_nb(
                abc_sec, sec_is_prism, sec_is_native_prism, n1d, n_sps,
                v_sps_inv_prism, v_sps_inv_tet,
                v_sps_inv_np, np_mode_i, np_mode_j, np_mode_k)
            for p in range(n_fp):
                nb_sec_resid[ei, p] = sec_rs[p]
                if not nb_mask[f, p]:
                    nb_extra_mat[ei, p] = interp_sec[p]
        nb_cell_id[f] = pn_c

    # ---- Owner multi-source ----
    for idx in prange(n_ms_ow):
        f = ms_ow_faces[idx]
        oc = owner_cell_arr[f]
        oc_code = owner_cube_face_arr[f]
        o_axis = _FACE_AXIS[oc_code]
        o_side = _FACE_SIDE[oc_code]
        nc = neighbor_cell_arr[f]
        nc_code = neighbor_cube_face_arr[f]
        n_axis = _FACE_AXIS[nc_code]
        n_side = _FACE_SIDE[nc_code]
        o_is_prism = oc < n_prism
        n_is_prism = nc < n_prism

        # 获取 neighbor 节点
        if n_is_prism:
            n_nd = np.empty((6, 3))
            for ni in range(6):
                nid = prism_conn[nc * 6 + ni]
                for d in range(3):
                    n_nd[ni, d] = node_coords[nid, d]
        else:
            n_nd = np.empty((4, 3))
            for ni in range(4):
                nid = tet_conn[(nc - n_prism) * 4 + ni]
                for d in range(3):
                    n_nd[ni, d] = node_coords[nid, d]

        # Neighbor 物理 FP（self-side = neighbor for ow multi-source）
        ref_n = _face_ref_grid_nb(n1d, n_axis, n_side, sps_1d)
        phys_n = _map_ref_nb(n_is_prism, ref_n, n_nd)
        a_val = area_arr[f]
        cl_o = np.sqrt(max(a_val, 1e-300))

        # ---- Primary interp（primary owner cell, 用预计算自由坐标）----
        pn_c = ms_ow_pn_cell[idx]
        pn_code = ms_ow_pn_code[idx]
        # 两类原生面分开判（`code >= 6` 原本等价于"是原生四面体面"，
        # 加了原生棱柱编码 [10,15) 之后必须区分，否则按 `code-6` 会越界
        # 取四面体的表；numba 不做边界检查。见主 kernel 同一处说明。
        pn_is_native_tet = _NATIVE_TET_LO <= pn_code < _NATIVE_TET_HI
        pn_is_native_prism = pn_code >= _NATIVE_PRISM_LO
        if pn_is_native_tet:
            rst_pn_o = _native_alpha_beta_to_rst_nb(pn_code - _NATIVE_TET_LO, ow_fc[f])
            interp_pn_o = _native_interp_matrix_nb(
                rst_pn_o, native_mode_i, native_mode_j, native_mode_k, v_sps_inv_native, n_sps
            )
            for p in range(n_fp):
                if ow_mask[f, p]:
                    ow_interp[f, p] = interp_pn_o[p]
        else:
            pn_axis = _FACE_AXIS[pn_code]
            pn_side = _FACE_SIDE[pn_code]
            pn_is_prism = pn_c < n_prism
            abc_o = np.empty((n_fp, 3))
            for p in range(n_fp):
                abc_o[p, pn_axis] = pn_side
                ix2 = 0
                for ax in range(3):
                    if ax != pn_axis:
                        abc_o[p, ax] = ow_fc[f, p, ix2]
                        ix2 += 1
            interp_pn_o = interp_matrix_from_cube_coords_nb(
                abc_o, pn_is_prism, pn_is_native_prism, n1d, n_sps,
                v_sps_inv_prism, v_sps_inv_tet,
                v_sps_inv_np, np_mode_i, np_mode_j, np_mode_k)
            for p in range(n_fp):
                if ow_mask[f, p]:
                    ow_interp[f, p] = interp_pn_o[p]

        if ms_ow_mixed[idx]:
            # 混合分组（neighbor 角色）：另半区是域边界，处理同 nb 分支。
            ow_cell_id[f] = pn_c
            continue

        # ---- Secondary Newton + interp ----
        sec_cell = ms_ow_sec_cell[idx]
        sec_code = ms_ow_sec_cube_face[idx]
        # 两类原生面分开判（`code >= 6` 原本等价于"是原生四面体面"，
        # 加了原生棱柱编码 [10,15) 之后必须区分，否则按 `code-6` 会越界
        # 取四面体的表；numba 不做边界检查。见主 kernel 同一处说明。
        sec_is_native_tet = _NATIVE_TET_LO <= sec_code < _NATIVE_TET_HI
        sec_is_native_prism = sec_code >= _NATIVE_PRISM_LO
        sec_is_prism = sec_cell < n_prism
        if sec_is_prism:
            sec_nd = np.empty((6, 3))
            for ni in range(6):
                nid = prism_conn[sec_cell * 6 + ni]
                for d in range(3):
                    sec_nd[ni, d] = node_coords[nid, d]
        else:
            sec_nd = np.empty((4, 3))
            for ni in range(4):
                nid = tet_conn[(sec_cell - n_prism) * 4 + ni]
                for d in range(3):
                    sec_nd[ni, d] = node_coords[nid, d]

        t = face_translation_arr[f]
        ht = abs(t[0]) > 1e-300 or abs(t[1]) > 1e-300 or abs(t[2]) > 1e-300
        if ht:
            # 对称的符号修复（见 build_fp_newton_parallel 的
            # neighbor_primary 分支说明）：把 neighbor 侧物理 FP 平移回
            # secondary（owner 一侧）单元所在区域，必须减 translation。
            search_sec = np.empty((n_fp, 3))
            for p in range(n_fp):
                search_sec[p, 0] = phys_n[p, 0] - t[0]
                search_sec[p, 1] = phys_n[p, 1] - t[1]
                search_sec[p, 2] = phys_n[p, 2] - t[2]
        else:
            search_sec = phys_n

        ei = ms_ow_extra_idx[idx]
        if sec_is_native_tet:
            sec_free_o, sec_rst_o, sec_rs = _tet_native_locate_nb(sec_nd, sec_code - _NATIVE_TET_LO, search_sec)
            interp_sec_o = _native_interp_matrix_nb(
                sec_rst_o, native_mode_i, native_mode_j, native_mode_k, v_sps_inv_native, n_sps
            )
            for p in range(n_fp):
                ow_sec_resid[ei, p] = sec_rs[p]
                if not ow_mask[f, p]:
                    ow_extra_mat[ei, p] = interp_sec_o[p]
        else:
            sec_axis = _FACE_AXIS[sec_code]
            sec_side = _FACE_SIDE[sec_code]
            sec_fc, sec_rs = _newton_locate_nb(
                sec_is_prism, sec_nd, sec_axis, sec_side, search_sec, cl_o
            )

            abc_sec = np.empty((n_fp, 3))
            for p in range(n_fp):
                abc_sec[p, sec_axis] = sec_side
                ix2 = 0
                for ax in range(3):
                    if ax != sec_axis:
                        abc_sec[p, ax] = sec_fc[p, ix2]
                        ix2 += 1
            interp_sec_o2 = interp_matrix_from_cube_coords_nb(
                abc_sec, sec_is_prism, sec_is_native_prism, n1d, n_sps,
                v_sps_inv_prism, v_sps_inv_tet,
                v_sps_inv_np, np_mode_i, np_mode_j, np_mode_k)
            for p in range(n_fp):
                ow_sec_resid[ei, p] = sec_rs[p]
                if not ow_mask[f, p]:
                    ow_extra_mat[ei, p] = interp_sec_o2[p]
        ow_cell_id[f] = pn_c
