"""
AutoFlowCFD - Multi-source 面插值矩阵 numba 并行 kernel

将棱柱四边形侧面（multi-source）的插值矩阵构建从纯 Python 串行循环
（每面调用 build_cross_interp，含 scipy lu_solve + 大量小数组分配）
改为 numba @njit(parallel=True) + prange 并行。

依赖主模块 face_flux_points_numba 中的辅助函数：
- 坐标变换：_FACE_AXIS, _FACE_SIDE
- 物理映射：_face_ref_grid_nb, _map_ref_nb
- Newton 迭代：_newton_locate_nb
- 模态基：prism_modal_basis_and_grad, tet_modal_basis_and_grad（间接）
"""

import numpy as np
from numba import njit, prange

from autoflowcfd.fr.collapsed_basis import (
    tet_modal_basis_and_grad,
    prism_modal_basis_and_grad,
)
from autoflowcfd.fr.face_flux_points_numba import (
    _FACE_AXIS,
    _FACE_SIDE,
    _face_ref_grid_nb,
    _map_ref_nb,
    _newton_locate_nb,
)


@njit(parallel=True, cache=True)
def build_ms_interp_parallel(
    n_ms_nb, n_ms_ow, ms_nb_faces, ms_ow_faces,
    ms_nb_sec_cell, ms_nb_sec_cube_face, ms_nb_extra_idx,
    ms_ow_sec_cell, ms_ow_sec_cube_face, ms_ow_extra_idx,
    nb_mask, ow_mask,
    n_prism, n1d, n_fp, sps_1d,
    owner_cell_arr, owner_cube_face_arr,
    neighbor_cell_arr, neighbor_cube_face_arr,
    area_arr, face_translation_arr,
    prism_conn, tet_conn, node_coords,
    nb_fc, ow_fc,
    v_sps_inv_tet, v_sps_inv_prism,
    nb_interp, ow_interp,
    nb_cell_id, ow_cell_id,
    nb_extra_mat, ow_extra_mat,
    ms_nb_pn_cell, ms_nb_pn_code,
    ms_ow_pn_cell, ms_ow_pn_code,
):
    """为 multi-source 面构建主/次插值矩阵（含 Newton + 对角线掩码）。

    直接写入 nb_interp/ow_interp（primary half）和 nb_extra_mat/ow_extra_mat
    （secondary half），避免额外内存分配。

    ms_{nb,ow}_pn_cell/code：primary 插值使用的 primary neighbor/owner
    （分组内第一条子面的跨单元邻居），而非 face f 自身的 neighbor/owner。
    Newton 自由坐标是相对 primary 邻居的参考空间计算的，primary interp
    必须使用同一邻居的节点。
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
        if pn_is_prism:
            V_t = prism_modal_basis_and_grad(
                abc_n[:, 0], abc_n[:, 1], abc_n[:, 2], n1d - 1
            )[0]
            V_inv = v_sps_inv_prism
        else:
            V_t = tet_modal_basis_and_grad(
                abc_n[:, 0], abc_n[:, 1], abc_n[:, 2], n1d - 1
            )[0]
            V_inv = v_sps_inv_tet
        for p in range(n_fp):
            if nb_mask[f, p]:
                for s in range(n_sps):
                    val = 0.0
                    for m in range(n_sps):
                        val += V_t[p, m] * V_inv[m, s]
                    nb_interp[f, p, s] = val

        # ---- Secondary Newton + interp ----
        sec_cell = ms_nb_sec_cell[idx]
        sec_code = ms_nb_sec_cube_face[idx]
        sec_axis = _FACE_AXIS[sec_code]
        sec_side = _FACE_SIDE[sec_code]
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
            search_sec = np.empty((n_fp, 3))
            for p in range(n_fp):
                search_sec[p, 0] = phys_o[p, 0] - t[0]
                search_sec[p, 1] = phys_o[p, 1] - t[1]
                search_sec[p, 2] = phys_o[p, 2] - t[2]
        else:
            search_sec = phys_o
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
        if sec_is_prism:
            V_sec = prism_modal_basis_and_grad(
                abc_sec[:, 0], abc_sec[:, 1], abc_sec[:, 2], n1d - 1
            )[0]
            V_inv_sec = v_sps_inv_prism
        else:
            V_sec = tet_modal_basis_and_grad(
                abc_sec[:, 0], abc_sec[:, 1], abc_sec[:, 2], n1d - 1
            )[0]
            V_inv_sec = v_sps_inv_tet
        ei = ms_nb_extra_idx[idx]
        for p in range(n_fp):
            if not nb_mask[f, p]:
                for s in range(n_sps):
                    val = 0.0
                    for m in range(n_sps):
                        val += V_sec[p, m] * V_inv_sec[m, s]
                    nb_extra_mat[ei, p, s] = val
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
        if pn_is_prism:
            V_to = prism_modal_basis_and_grad(
                abc_o[:, 0], abc_o[:, 1], abc_o[:, 2], n1d - 1
            )[0]
            V_inv_o = v_sps_inv_prism
        else:
            V_to = tet_modal_basis_and_grad(
                abc_o[:, 0], abc_o[:, 1], abc_o[:, 2], n1d - 1
            )[0]
            V_inv_o = v_sps_inv_tet
        for p in range(n_fp):
            if ow_mask[f, p]:
                for s in range(n_sps):
                    val = 0.0
                    for m in range(n_sps):
                        val += V_to[p, m] * V_inv_o[m, s]
                    ow_interp[f, p, s] = val

        # ---- Secondary Newton + interp ----
        sec_cell = ms_ow_sec_cell[idx]
        sec_code = ms_ow_sec_cube_face[idx]
        sec_axis = _FACE_AXIS[sec_code]
        sec_side = _FACE_SIDE[sec_code]
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
            search_sec = np.empty((n_fp, 3))
            for p in range(n_fp):
                search_sec[p, 0] = phys_n[p, 0] + t[0]
                search_sec[p, 1] = phys_n[p, 1] + t[1]
                search_sec[p, 2] = phys_n[p, 2] + t[2]
        else:
            search_sec = phys_n
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
        if sec_is_prism:
            V_sec = prism_modal_basis_and_grad(
                abc_sec[:, 0], abc_sec[:, 1], abc_sec[:, 2], n1d - 1
            )[0]
            V_inv_sec = v_sps_inv_prism
        else:
            V_sec = tet_modal_basis_and_grad(
                abc_sec[:, 0], abc_sec[:, 1], abc_sec[:, 2], n1d - 1
            )[0]
            V_inv_sec = v_sps_inv_tet
        ei = ms_ow_extra_idx[idx]
        for p in range(n_fp):
            if not ow_mask[f, p]:
                for s in range(n_sps):
                    val = 0.0
                    for m in range(n_sps):
                        val += V_sec[p, m] * V_inv_sec[m, s]
                    ow_extra_mat[ei, p, s] = val
        ow_cell_id[f] = pn_c
