"""
AutoFlowCFD - FP 几何构建 numba 主并行 kernel

将 build_face_flux_points 中最耗时的 Newton 迭代 + 插值矩阵构建从纯 Python
串行循环改为 numba @njit(parallel=True) + prange 并行。

辅助函数（坐标变换、物理映射、Newton 迭代、查找表）位于
face_flux_points_helpers_numba 模块，被本 kernel 和 ms_numba kernel 共用。
"""

import numpy as np
from numba import njit, prange

from .face_code_tables import (
    _FACE_AXIS, _FACE_SIDE, _PQ_CODES,
    _NATIVE_PRISM_LO, _NATIVE_TET_HI, _NATIVE_TET_LO,
)
from .ref_geometry_nb import (
    _face_ref_grid_nb, _map_ref_nb, _newton_locate_nb,
)
from .native_geometry_nb import (
    _native_tet_face_points_nb, _tet_native_locate_nb,
    _native_interp_matrix_nb, interp_matrix_from_cube_coords_nb,
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
    v_sps_inv_native, native_mode_i, native_mode_j, native_mode_k,
    v_sps_inv_np, np_mode_i, np_mode_j, np_mode_k,
):
    """并行计算所有面的 Newton 自由坐标 + 插值矩阵。

    对每个 interior primary 面，在 Newton 定位后立即构建插值矩阵，
    避免 Python 循环中的逐面 build_cross_interp 调用。
    棱柱四边形面 (multi-source) 的插值矩阵由 Python 端重新构建。

    Returns: (nb_fc, nb_resid, ow_fc, ow_resid,
              nb_interp, ow_interp, nb_cell_id, ow_cell_id,
              geom_oa, geom_os, geom_na, geom_ns, geom_aw, geom_n)
    nb_resid/ow_resid 形状 (n_faces, n_fp)：逐 Flux Point 的 Newton 精确
    点位定位残差（绝对长度单位，未按面特征尺度归一化，未按容差分级）——
    调用方 face_flux_points/merge.py 负责归一化、按 ACCEPT_STRICT_REL/
    _ACCEPT_WARN_REL 分级、以及对多源棱柱四边形侧面按半区掩码取值。

    native 四面体（路径C）支持（Part7 文档阶段2 numba 核函数移植）：
    `owner_cube_face`/`neighbor_cube_face` 里 code>=6 的项目标记该侧是
    native 四面体的真实面（`excluded_vertex = code-6`），本函数据此
    分派到 `_tet_native_locate_nb`/`_native_tet_face_points_nb`/
    `simplex3d_value`（`fr/face_flux_points/native_geometry_nb.py`/
    `fr/native_tet/basis.py` 的 numba 版本，与坍缩坐标分支平行），
    不再假设 `_FACE_AXIS`/`_FACE_SIDE`（长度仅 6，对 code>=6 越界）
    覆盖所有 cube face code。code>=6 只可能出现在 native 四面体的真实面
    （棱柱四边形侧面恒为 0~5，见 grid/connectivity/face_connectivity.py::
    with_native_face_codes 文档），因此本函数所有 native 分支只需要判断
    `code>=6`，不需要额外与 is_prism 组合判断。`v_sps_inv_native`/
    `native_mode_{i,j,k}` 在整个网格不含任何 native 四面体时（既有
    默认坍缩坐标路径）传入零长度占位数组即可，对应分支永远不会被执行，
    不改变任何现有行为——见 face_flux_points/merge.py 调用处说明。
    """
    n_sps = n1d * n1d * n1d
    nb_fc = np.zeros((n_faces, n_fp, 2))
    # nb_resid/ow_resid：逐 Flux Point 残差（不在这里归约成单一标量），
    # 供 face_flux_points/merge.py 对多源棱柱四边形侧面按对角线半区
    # 分别掩码取 max——见 _newton_locate_nb 文档，同一批点里混有真正
    # 属于该 cell 和根本不属于该 cell（对角线另一半）的目标点，提前
    # 归约成全批次单一 max 会把两者混在一起，对多源面产生系统性误报。
    nb_resid = np.zeros((n_faces, n_fp))
    ow_fc = np.zeros((n_faces, n_fp, 2))
    ow_resid = np.zeros((n_faces, n_fp))
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
    # 真正安全的内存优化（见 fr/face_flux_points/merge.py 顶部内存说明）。
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
        # code>=6 只可能是 native 四面体真实面（见函数文档），此时 axis/
        # side 语义不适用——_FACE_AXIS/_FACE_SIDE 长度只有 6，不能对
        # code>=6 索引（numba 不做边界检查，会读到未定义内存）。o_axis/
        # o_side 在 native 分支下只是未使用的占位值（真正用到的是下面
        # o_is_native 分支各自独立处理），仍然写入 geom_oa/geom_os 但那
        # 两个数组本身在下游（face_flux_points/merge.py）已确认从不被
        # 消费（owner 侧 axis/side 改用 CUBE_FACE_AXIS_SIDE 字典查表）。
        # 两类原生面必须**分开**判（2026-09-19）：`code >= 6` 原本等价于
        # "是原生四面体面"，加了原生棱柱编码 [10,15) 之后它变成了"是任意
        # 原生面"，而两者要分派到**不同**的算子组；按 `code - 6` 去取四面体
        # 的表会拿到 4~8 行，而那些数组只有 4 行（numba 不做边界检查）。
        o_is_nat_tet = _NATIVE_TET_LO <= oc_code < _NATIVE_TET_HI
        o_is_nat_prism = oc_code >= _NATIVE_PRISM_LO
        if o_is_nat_tet:
            # 四面体原生面没有 (axis,side) 语义（axis 槽位复用成
            # excluded_vertex），这里给占位值；真正的分派在下面各分支里。
            o_axis = 0
            o_side = -1.0
        else:
            # 坍缩面与**原生棱柱面**都用真实 (axis,side)：原生棱柱面的通量点
            # 与坍缩立方体面的通量点已验证是同一批物理点、同一顺序（三项
            # 实测机器精度，见 fr/native_prism/__init__.py 模块文档），所以
            # 面点生成与 Newton 定位继续走坍缩那条，只有插值矩阵换成原生。
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
        n_is_nat_tet = _NATIVE_TET_LO <= nc_code < _NATIVE_TET_HI
        n_is_nat_prism = nc_code >= _NATIVE_PRISM_LO
        if n_is_nat_tet:
            n_axis = 0
            n_side = -1.0
        else:
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

        # Owner 物理 FP —— native 四面体没有 (axis,side) 概念，面点物理
        # 位置改用与棱柱三角形封盖相同的坍缩三角形采样（见
        # _native_tet_face_points_nb 文档），不是 _face_ref_grid_nb 的
        # 张量积网格。
        if o_is_nat_tet:
            phys_o = _native_tet_face_points_nb(n1d, oc_code - _NATIVE_TET_LO,
                                                o_nd, sps_1d)
        else:
            ref_o = _face_ref_grid_nb(n1d, o_axis, o_side, sps_1d)
            phys_o = _map_ref_nb(o_is_prism, ref_o, o_nd)

        # ---- Neighbor 侧 Newton (owner_primary 才需要) ----
        if owner_primary[f]:
            cl = np.sqrt(max(a_val, 1e-300))
            t = face_translation[f]
            ht = abs(t[0]) > 1e-300 or abs(t[1]) > 1e-300 or abs(t[2]) > 1e-300
            if ht:
                # 真实 bug 修复（V2.0 专家组盲审发现，2026-08-28）：周期边界
                # 配对约定 owner.center + translation ≈ neighbor.center
                # （见 face_connectivity_periodic.py::pair_periodic_boundary_faces
                # "centers_a_shifted = center[idx_a] + translation 匹配
                # center[idx_b]"，配对后 idx_a 侧留作 owner、idx_b 的
                # owner 变成 neighbor_cell）。要把 owner 侧物理 FP 平移到
                # neighbor 单元所在的区域去定位，必须 **加** translation，
                # 此前这里写成减——这个安全网此前从未真正被激活过（见
                # face_flux_points/validation.py 模块文档"safety net
                # 架空"一节），直到本次评审把 _classify_and_record 真正
                # 接上，才第一次在真实周期网格上暴露：残差恰好等于
                # 2*|translation|（符号取反导致的偏差是 -t 相对正确值 +t
                # 偏了 2t，不是巧合），此前完全静默——任何用到周期边界的
                # 算例（TGV/Couette/periodic_bc 等验证用例）在周期面上的
                # Flux Point 插值矩阵实际上一直是错的，只是从未被检查过。
                search = np.empty((n_fp, 3))
                for p in range(n_fp):
                    search[p, 0] = phys_o[p, 0] + t[0]
                    search[p, 1] = phys_o[p, 1] + t[1]
                    search[p, 2] = phys_o[p, 2] + t[2]
            else:
                search = phys_o

            # rst_nb 只在 n_is_native 分支被赋值、也只在后面同样被
            # n_is_native 分支守卫的地方被读取（运行时永远不会读到这个
            # 占位初值）——预先给一个形状/类型一致的哑值，只是为了让
            # numba 的类型推断在两条分支路径上都能确定该变量类型（它按
            # 控制流图形状做定义可达性检查，不做"同一个未被重新赋值的
            # 布尔变量在两处 if 判断结果必然一致"这种值层面的推理）。
            rst_nb = np.zeros((n_fp, 3))
            if n_is_nat_tet:
                # native 四面体目标：解析闭式解直接给出 (r,s,t)，与坍缩坐标
                # 版本假设 (axis,side) 语义的 _newton_locate_nb 不兼容，见
                # locate_native_tet_face_point 的 numba 移植文档。
                fc, rst_nb, rs = _tet_native_locate_nb(
                    n_nd, nc_code - _NATIVE_TET_LO, search)
            else:
                # 坍缩面与原生棱柱面共用这条（几何映射逐位恒等）。
                fc, rs = _newton_locate_nb(n_is_prism, n_nd, n_axis, n_side, search, cl)
            for p in range(n_fp):
                nb_fc[f, p, 0] = fc[p, 0]
                nb_fc[f, p, 1] = fc[p, 1]
            nb_resid[f] = rs

            # 检查是否棱柱四边形面 (multi-source)，若是则跳过插值矩阵
            is_pq = False
            if o_is_prism:
                for pq in range(_PQ_CODES.shape[0]):
                    if oc_code == _PQ_CODES[pq]:
                        is_pq = True
                        break
            if not is_pq:
                if n_is_nat_tet:
                    # native 四面体目标插值矩阵：直接用上面闭式解已给出的
                    # (r,s,t)（内部经 _rst_to_abc_nb 转换只用于取值，安全，
                    # 不经过求导链式法则），不会重新引入坍缩坐标退化轴
                    # 病态；按 Part6/7"补位对齐"原则自动只写前 n_native
                    # 列，其余列保持零（见 _native_interp_matrix_nb 文档）。
                    interp_native = _native_interp_matrix_nb(
                        rst_nb, native_mode_i, native_mode_j, native_mode_k, v_sps_inv_native, n_sps
                    )
                    nb_interp[f] = interp_native
                else:
                    # 坍缩面与**原生棱柱面**共用一条：两者的目标点都由坍缩
                    # Newton 给出（几何映射逐位恒等），唯一差别是求哪个
                    # Vandermonde —— 由 `interp_matrix_from_cube_coords_nb`
                    # 内部按 `is_native_prism` 分派，见那边文档（它同时替掉了
                    # 本文件与 ms kernel 里原先 6 处逐字重复的"构造 V_t + 手写
                    # 矩阵乘"）。
                    abc = np.empty((n_fp, 3))
                    for p in range(n_fp):
                        abc[p, n_axis] = n_side
                        ix2 = 0
                        for ax in range(3):
                            if ax != n_axis:
                                abc[p, ax] = fc[p, ix2]
                                ix2 += 1
                    nb_interp[f] = interp_matrix_from_cube_coords_nb(
                        abc, n_is_prism, n_is_nat_prism, n1d, n_sps,
                        v_sps_inv_prism, v_sps_inv_tet,
                        v_sps_inv_np, np_mode_i, np_mode_j, np_mode_k)
                nb_cell_id[f] = nc

        # ---- Owner 侧 Newton (neighbor_primary 才需要) ----
        if neighbor_primary[f]:
            if n_is_nat_tet:
                phys_n = _native_tet_face_points_nb(
                    n1d, nc_code - _NATIVE_TET_LO, n_nd, sps_1d)
            else:
                ref_n = _face_ref_grid_nb(n1d, n_axis, n_side, sps_1d)
                phys_n = _map_ref_nb(n_is_prism, ref_n, n_nd)
            cl_o = np.sqrt(max(a_val, 1e-300))
            t = face_translation[f]
            ht = abs(t[0]) > 1e-300 or abs(t[1]) > 1e-300 or abs(t[2]) > 1e-300
            if ht:
                # 对称的符号修复（见上方 owner_primary 分支的详细说明）：
                # 把 neighbor 侧物理 FP 平移回 owner 单元所在区域，必须
                # **减** translation。
                search_o = np.empty((n_fp, 3))
                for p in range(n_fp):
                    search_o[p, 0] = phys_n[p, 0] - t[0]
                    search_o[p, 1] = phys_n[p, 1] - t[1]
                    search_o[p, 2] = phys_n[p, 2] - t[2]
            else:
                search_o = phys_n

            # 同上 rst_nb 的说明。
            rst_o_nb = np.zeros((n_fp, 3))
            if o_is_nat_tet:
                fc_o, rst_o_nb, rs_o = _tet_native_locate_nb(
                    o_nd, oc_code - _NATIVE_TET_LO, search_o)
            else:
                fc_o, rs_o = _newton_locate_nb(o_is_prism, o_nd, o_axis, o_side, search_o, cl_o)
            for p in range(n_fp):
                ow_fc[f, p, 0] = fc_o[p, 0]
                ow_fc[f, p, 1] = fc_o[p, 1]
            ow_resid[f] = rs_o

            # 检查是否棱柱四边形面 (multi-source)
            is_pq_n = False
            if n_is_prism:
                for pq in range(_PQ_CODES.shape[0]):
                    if nc_code == _PQ_CODES[pq]:
                        is_pq_n = True
                        break
            if not is_pq_n:
                if o_is_nat_tet:
                    interp_o_native = _native_interp_matrix_nb(
                        rst_o_nb, native_mode_i, native_mode_j, native_mode_k, v_sps_inv_native, n_sps
                    )
                    ow_interp[f] = interp_o_native
                else:
                    # 与 neighbor 侧对称，同一个共用入口（见那边说明）。
                    abc_o = np.empty((n_fp, 3))
                    for p in range(n_fp):
                        abc_o[p, o_axis] = o_side
                        ix2 = 0
                        for ax in range(3):
                            if ax != o_axis:
                                abc_o[p, ax] = fc_o[p, ix2]
                                ix2 += 1
                    ow_interp[f] = interp_matrix_from_cube_coords_nb(
                        abc_o, o_is_prism, o_is_nat_prism, n1d, n_sps,
                        v_sps_inv_prism, v_sps_inv_tet,
                        v_sps_inv_np, np_mode_i, np_mode_j, np_mode_k)
                ow_cell_id[f] = oc

    return (nb_fc, nb_resid, ow_fc, ow_resid,
            nb_interp, ow_interp, nb_cell_id, ow_cell_id,
            geom_oa, geom_os, geom_na, geom_ns, geom_aw, geom_n)
