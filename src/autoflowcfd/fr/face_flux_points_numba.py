"""
AutoFlowCFD - FP 几何构建 numba 主并行 kernel

将 build_face_flux_points 中最耗时的 Newton 迭代 + 插值矩阵构建从纯 Python
串行循环改为 numba @njit(parallel=True) + prange 并行。

辅助函数（坐标变换、物理映射、Newton 迭代、查找表）位于
face_flux_points_helpers_numba 模块，被本 kernel 和 ms_numba kernel 共用。
"""

import numpy as np
from numba import njit, prange

from autoflowcfd.fr.collapsed_basis import (
    tet_modal_basis_and_grad,
    prism_modal_basis_and_grad,
)
from autoflowcfd.fr.face_flux_points_helpers_numba import (
    _FACE_AXIS, _FACE_SIDE, _PQ_CODES,
    _face_ref_grid_nb, _map_ref_nb, _newton_locate_nb,
)


# ============================================================================
# 主并行 kernel
# ============================================================================


@njit(parallel=True, cache=True)
def build_fp_newton_parallel(
    n_faces, n_prism, n1d, n_fp, sps_1d,
    is_boundary, owner_cell, owner_cube_face,
    neighbor_cell, neighbor_cube_face,
    area, normal, face_translation,
    prism_conn, tet_conn, node_coords,
    owner_primary, neighbor_primary,
    v_sps_inv_tet, v_sps_inv_prism,
):
    """并行计算所有面的 Newton 自由坐标 + 插值矩阵。

    对每个 interior primary 面，在 Newton 定位后立即构建插值矩阵，
    避免 Python 循环中的逐面 build_cross_interp 调用。
    棱柱四边形面 (multi-source) 的插值矩阵由 Python 端重新构建。

    Returns: (nb_fc, nb_resid, ow_fc, ow_resid,
              nb_interp, ow_interp, nb_cell_id, ow_cell_id,
              geom_oa, geom_os, geom_na, geom_ns, geom_aw, geom_n)
    """
    n_sps = n1d * n1d * n1d
    nb_fc = np.zeros((n_faces, n_fp, 2))
    nb_resid = np.zeros(n_faces)
    ow_fc = np.zeros((n_faces, n_fp, 2))
    ow_resid = np.zeros(n_faces)
    # nb_interp/ow_interp 必须是 float64，不能降到 float32——已经真实
    # 验证过 float32 在这里不安全，不是假设：cross-interpolation 用的
    # 坍缩坐标模态 Vandermonde 矩阵条件数随阶数快速增长（collapsed_basis.py
    # 文档实测 P2 时 cond~1e5、P3 时 cond~1e9），这意味着矩阵单个条目的
    # 量级可以比"重构常数场应得的求和结果"大出条件数那么多倍——float64
    # 的 lu_solve 正是为控制这个病态而不用显式求逆引入的（见
    # face_flux_points.py::_get_v_sps_lu 文档），把算出来的矩阵条目本身
    # 再降到 float32 存储，等于在最后一步重新引入同样量级的舍入误差，
    # 白费了前面用 lu_solve 控制条件数的努力。真实回归测试证实了这一点：
    # 把这两个数组改成 float32 后，tests/unit/test_fr_residual_inviscid.py
    # 的自由流场保持性测试在 P=2（生产阶数）从 3e-5 容差內失败到
    # rel_res 量级压根不通过，线性剪切流去混叠测试在 P=3 下 max|mass
    # residual| 从应有的 <1e-2 暴涨到 1.512e+05——不是"略微变差"，是完全
    # 破坏了这两个此前专门为控制舍入误差而做的修复(G-04 跨单元插值统一 +
    # S-02 体积项去混叠)，因此保留 float64，只保留"消除冗余拷贝"这一个
    # 真正安全的内存优化（见 fr/face_flux_points_merge.py 顶部内存说明）。
    nb_interp = np.zeros((n_faces, n_fp, n_sps), dtype=np.float64)
    ow_interp = np.zeros((n_faces, n_fp, n_sps), dtype=np.float64)
    nb_cell_id = np.full(n_faces, -1, dtype=np.int32)
    ow_cell_id = np.full(n_faces, -1, dtype=np.int32)
    geom_oa = np.empty(n_faces, dtype=np.int32)
    geom_os = np.empty(n_faces)
    geom_na = np.empty(n_faces, dtype=np.int32)
    geom_ns = np.empty(n_faces)
    geom_aw = np.empty((n_faces, n_fp))
    geom_n = np.empty((n_faces, n_fp, 3))

    for f in prange(n_faces):
        oc = owner_cell[f]
        oc_code = owner_cube_face[f]
        o_axis = _FACE_AXIS[oc_code]
        o_side = _FACE_SIDE[oc_code]

        a_val = area[f]
        for fp in range(n_fp):
            geom_aw[f, fp] = a_val
            geom_n[f, fp, 0] = normal[f, 0]
            geom_n[f, fp, 1] = normal[f, 1]
            geom_n[f, fp, 2] = normal[f, 2]
        geom_oa[f] = o_axis
        geom_os[f] = o_side

        # 边界面
        if is_boundary[f]:
            geom_na[f] = -1
            geom_ns[f] = 0.0
            continue

        nc = neighbor_cell[f]
        nc_code = neighbor_cube_face[f]
        n_axis = _FACE_AXIS[nc_code]
        n_side = _FACE_SIDE[nc_code]
        geom_na[f] = n_axis
        geom_ns[f] = n_side

        # 获取 owner 节点
        o_is_prism = oc < n_prism
        if o_is_prism:
            o_nd = np.empty((6, 3))
            for ni in range(6):
                nid = prism_conn[oc * 6 + ni]
                o_nd[ni, 0] = node_coords[nid, 0]
                o_nd[ni, 1] = node_coords[nid, 1]
                o_nd[ni, 2] = node_coords[nid, 2]
        else:
            o_nd = np.empty((4, 3))
            for ni in range(4):
                nid = tet_conn[(oc - n_prism) * 4 + ni]
                o_nd[ni, 0] = node_coords[nid, 0]
                o_nd[ni, 1] = node_coords[nid, 1]
                o_nd[ni, 2] = node_coords[nid, 2]

        # 获取 neighbor 节点
        n_is_prism = nc < n_prism
        if n_is_prism:
            n_nd = np.empty((6, 3))
            for ni in range(6):
                nid = prism_conn[nc * 6 + ni]
                n_nd[ni, 0] = node_coords[nid, 0]
                n_nd[ni, 1] = node_coords[nid, 1]
                n_nd[ni, 2] = node_coords[nid, 2]
        else:
            n_nd = np.empty((4, 3))
            for ni in range(4):
                nid = tet_conn[(nc - n_prism) * 4 + ni]
                n_nd[ni, 0] = node_coords[nid, 0]
                n_nd[ni, 1] = node_coords[nid, 1]
                n_nd[ni, 2] = node_coords[nid, 2]

        # Owner 物理 FP
        ref_o = _face_ref_grid_nb(n1d, o_axis, o_side, sps_1d)
        phys_o = _map_ref_nb(o_is_prism, ref_o, o_nd)

        # ---- Neighbor 侧 Newton (owner_primary 才需要) ----
        if owner_primary[f]:
            cl = np.sqrt(max(a_val, 1e-300))
            t = face_translation[f]
            ht = abs(t[0]) > 1e-300 or abs(t[1]) > 1e-300 or abs(t[2]) > 1e-300
            if ht:
                search = np.empty((n_fp, 3))
                for p in range(n_fp):
                    search[p, 0] = phys_o[p, 0] - t[0]
                    search[p, 1] = phys_o[p, 1] - t[1]
                    search[p, 2] = phys_o[p, 2] - t[2]
            else:
                search = phys_o
            fc, rs = _newton_locate_nb(n_is_prism, n_nd, n_axis, n_side, search, cl)
            for p in range(n_fp):
                nb_fc[f, p, 0] = fc[p, 0]
                nb_fc[f, p, 1] = fc[p, 1]
            nb_resid[f] = rs

            # 检查是否棱柱四边形面 (multi-source)，若是则跳过插值矩阵
            is_pq = False
            if o_is_prism:
                for pq in range(3):
                    if oc_code == _PQ_CODES[pq]:
                        is_pq = True
                        break
            if not is_pq:
                # 构建插值矩阵: interp = V_target @ V_sps_inv
                abc = np.empty((n_fp, 3))
                for p in range(n_fp):
                    abc[p, n_axis] = n_side
                    ix2 = 0
                    for ax in range(3):
                        if ax != n_axis:
                            abc[p, ax] = fc[p, ix2]
                            ix2 += 1
                if n_is_prism:
                    V_t = prism_modal_basis_and_grad(abc[:,0], abc[:,1], abc[:,2], n1d-1)[0]
                    V_inv = v_sps_inv_prism
                else:
                    V_t = tet_modal_basis_and_grad(abc[:,0], abc[:,1], abc[:,2], n1d-1)[0]
                    V_inv = v_sps_inv_tet
                # interp = V_t @ V_inv  (n_fp, n_sps)
                for p in range(n_fp):
                    for s in range(n_sps):
                        val = 0.0
                        for m in range(n_sps):
                            val += V_t[p, m] * V_inv[m, s]
                        nb_interp[f, p, s] = val
                nb_cell_id[f] = nc

        # ---- Owner 侧 Newton (neighbor_primary 才需要) ----
        if neighbor_primary[f]:
            ref_n = _face_ref_grid_nb(n1d, n_axis, n_side, sps_1d)
            phys_n = _map_ref_nb(n_is_prism, ref_n, n_nd)
            cl_o = np.sqrt(max(a_val, 1e-300))
            t = face_translation[f]
            ht = abs(t[0]) > 1e-300 or abs(t[1]) > 1e-300 or abs(t[2]) > 1e-300
            if ht:
                search_o = np.empty((n_fp, 3))
                for p in range(n_fp):
                    search_o[p, 0] = phys_n[p, 0] + t[0]
                    search_o[p, 1] = phys_n[p, 1] + t[1]
                    search_o[p, 2] = phys_n[p, 2] + t[2]
            else:
                search_o = phys_n
            fc_o, rs_o = _newton_locate_nb(o_is_prism, o_nd, o_axis, o_side, search_o, cl_o)
            for p in range(n_fp):
                ow_fc[f, p, 0] = fc_o[p, 0]
                ow_fc[f, p, 1] = fc_o[p, 1]
            ow_resid[f] = rs_o

            # 检查是否棱柱四边形面 (multi-source)
            is_pq_n = False
            if n_is_prism:
                for pq in range(3):
                    if nc_code == _PQ_CODES[pq]:
                        is_pq_n = True
                        break
            if not is_pq_n:
                abc_o = np.empty((n_fp, 3))
                for p in range(n_fp):
                    abc_o[p, o_axis] = o_side
                    ix2 = 0
                    for ax in range(3):
                        if ax != o_axis:
                            abc_o[p, ax] = fc_o[p, ix2]
                            ix2 += 1
                if o_is_prism:
                    V_to = prism_modal_basis_and_grad(abc_o[:,0], abc_o[:,1], abc_o[:,2], n1d-1)[0]
                    V_inv_o = v_sps_inv_prism
                else:
                    V_to = tet_modal_basis_and_grad(abc_o[:,0], abc_o[:,1], abc_o[:,2], n1d-1)[0]
                    V_inv_o = v_sps_inv_tet
                for p in range(n_fp):
                    for s in range(n_sps):
                        val = 0.0
                        for m in range(n_sps):
                            val += V_to[p, m] * V_inv_o[m, s]
                        ow_interp[f, p, s] = val
                ow_cell_id[f] = oc

    return (nb_fc, nb_resid, ow_fc, ow_resid,
            nb_interp, ow_interp, nb_cell_id, ow_cell_id,
            geom_oa, geom_os, geom_na, geom_ns, geom_aw, geom_n)
