"""AutoFlowCFD V2.0 - native 四面体（路径C）numba 主并行核函数
(`face_flux_points_numba.py::build_fp_newton_parallel`) 与纯 Python
参考实现（`build_cross_interp`/`native_tet_face_points_physical`）的
决定性一致性验证。

背景：Part7 文档阶段2 发现 numba 核函数内部有自己独立、编译过的
点位定位/插值矩阵构建逻辑（`_newton_locate_nb`/`_FACE_AXIS` 假设坍缩
坐标 (axis,side) 语义），与 Python 层已经验证过的 native 分支
（`locate_native_tet_face_point`/`build_cross_interp`）完全不共享代码，
必须独立移植（`face_flux_points_helpers_numba.py` 新增的 `_tet_native_
locate_nb`/`_native_tet_face_points_nb`/`_native_interp_matrix_nb`）并
独立验证数值一致——本文件直接调用核函数（构造最小合成连接数组，绕开
完整网格加载/求解器接入，那是仍然独立、尚未做的后续工作），覆盖两类
真实拓扑：四面体(native)-四面体(native)、四面体(native)-棱柱(collapsed)。
"""

import numpy as np
from scipy.linalg import lu_solve

from autoflowcfd.fr.face_flux_points_numba import build_fp_newton_parallel
from autoflowcfd.fr.face_flux_points import (
    _get_v_sps_lu_native,
    build_cross_interp,
    native_tet_face_points_physical,
    face_ref_grid,
)
from autoflowcfd.fr.face_flux_points_locate import map_ref_points
from autoflowcfd.fr.quadrature_points import gauss_legendre


class _MockMesh:
    """`build_cross_interp`/`cell_info` 只需要这 4 个属性，不需要完整
    HighOrderMesh。"""

    def __init__(self, n_prism_cells, fixed_prism_conn, fixed_tet_conn, node_coords):
        self.n_prism_cells = n_prism_cells
        self._fixed_prism_conn = fixed_prism_conn
        self._fixed_tet_conn = fixed_tet_conn
        self._node_coords = node_coords


def _run_kernel_single_face(
    order, n_prism, prism_conn, tet_conn, node_coords,
    owner_cell, owner_code, neighbor_cell, neighbor_code, area, normal,
):
    n1d = order + 1
    sps_1d, _ = gauss_legendre(n1d)
    n_fp = n1d * n1d
    n_sps = n1d ** 3

    lu_native, modes_native = _get_v_sps_lu_native(order)
    n_native = len(modes_native)
    v_sps_inv_native = np.ascontiguousarray(lu_solve(lu_native, np.eye(n_native)).T)
    native_mode_i = np.array([m[0] for m in modes_native], dtype=np.int32)
    native_mode_j = np.array([m[1] for m in modes_native], dtype=np.int32)
    native_mode_k = np.array([m[2] for m in modes_native], dtype=np.int32)
    # 坍缩坐标 V_sps_inv 必须真算（不能用零占位）：即便本测试的 owner/
    # neighbor 至少一侧恒为 native，另一侧（棱柱或坍缩四面体）仍然是
    # 真实的坍缩坐标插值目标，与 face_flux_points_merge.py 生产调用处
    # 完全一样的构造方式——这里只有真正"整个网格不含该单元类型"时才
    # 应该用零占位（那种情况下对应分支确实永远不会被执行到）。
    from autoflowcfd.fr.face_flux_points import _get_v_sps_lu
    I_n = np.eye(n_sps)
    v_sps_inv_tet = np.ascontiguousarray(lu_solve(_get_v_sps_lu("tet", n1d, sps_1d), I_n).T)
    v_sps_inv_prism = np.ascontiguousarray(lu_solve(_get_v_sps_lu("prism", n1d, sps_1d), I_n).T)

    result = build_fp_newton_parallel(
        1, n_prism, n1d, n_fp,
        np.ascontiguousarray(sps_1d.astype(np.float64)),
        np.array([False]),
        np.array([owner_cell], dtype=np.int32), np.array([owner_code], dtype=np.int32),
        np.array([neighbor_cell], dtype=np.int32), np.array([neighbor_code], dtype=np.int32),
        np.array([area], dtype=np.float64), np.array([normal], dtype=np.float64),
        np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        np.ascontiguousarray(prism_conn.ravel().astype(np.int32)),
        np.ascontiguousarray(tet_conn.ravel().astype(np.int32)),
        np.ascontiguousarray(node_coords.astype(np.float64)),
        np.array([True]), np.array([True]),
        v_sps_inv_tet, v_sps_inv_prism,
        v_sps_inv_native, native_mode_i, native_mode_j, native_mode_k,
    )
    return result, n1d, sps_1d


def test_native_tet_tet_interface_kernel_matches_python_reference():
    """两个真实相邻 native 四面体共享同一个物理面——核函数算出的
    owner/neighbor 双向插值矩阵必须与纯 Python `build_cross_interp`
    native 分支逐位一致（机器精度）。"""
    order = 2
    tetA = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    # tetB 与 tetA 共享物理面 {(1,0,0),(0,1,0),(0,0,1)}——两侧局部顶点
    # 编号顺序不需要一致（重心坐标解算在物理空间进行，与局部编号无关），
    # excluded_vertex 都恰好是局部下标 0。
    tetB = np.array([[1, 1, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    node_coords = np.vstack([tetA, tetB])
    tet_conn = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)
    n_prism = 0
    prism_conn = np.zeros((0, 6), dtype=np.int64)
    mesh = _MockMesh(n_prism, prism_conn, tet_conn, node_coords)

    p1, p2, p3 = tetA[1], tetA[2], tetA[3]
    area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
    normal = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)

    result, n1d, sps_1d = _run_kernel_single_face(
        order, n_prism, prism_conn, tet_conn, node_coords,
        owner_cell=0, owner_code=6, neighbor_cell=1, neighbor_code=6,
        area=area, normal=normal,
    )
    (nb_fc, nb_resid, ow_fc, ow_resid, nb_interp, ow_interp,
     nb_cell_id, ow_cell_id, geom_oa, geom_os, geom_na, geom_ns, geom_aw, geom_n) = result

    assert nb_cell_id[0] == 1
    assert ow_cell_id[0] == 0
    np.testing.assert_allclose(nb_resid[0], 0.0, atol=1e-9)
    np.testing.assert_allclose(ow_resid[0], 0.0, atol=1e-9)

    char_length = float(np.sqrt(max(area, 1e-300)))
    phys_o = native_tet_face_points_physical(n1d, 0, tetA, sps_1d)
    interp_ref_nb, _ = build_cross_interp(
        mesh, n1d, sps_1d, target_cell=1, target_face_code=6,
        source_phys=phys_o, char_length=char_length,
    )
    np.testing.assert_allclose(nb_interp[0], interp_ref_nb, atol=1e-10)

    phys_n = native_tet_face_points_physical(n1d, 0, tetB, sps_1d)
    interp_ref_ow, _ = build_cross_interp(
        mesh, n1d, sps_1d, target_cell=0, target_face_code=6,
        source_phys=phys_n, char_length=char_length,
    )
    np.testing.assert_allclose(ow_interp[0], interp_ref_ow, atol=1e-10)


def test_native_tet_prism_interface_kernel_matches_python_reference():
    """棱柱(collapsed)三角形封盖面与相邻 native 四面体共享同一个物理
    面——核函数的坍缩坐标分支（棱柱侧）与 native 分支（四面体侧）必须
    分别与各自的纯 Python 参考实现一致，且互不干扰彼此的 axis/side
    与 excluded_vertex 语义（同一个面记录里 owner 是坍缩坐标、neighbor
    是 native，两套语义必须在同一次核函数调用里同时正确）。几何与
    `test_native_tet_prism_cross_basis.py` 完全一致的共享三角形构造。"""
    order = 2
    prism_nodes = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [0, 0, -1], [1, 0, -1], [0, 1, -1],
    ], dtype=float)
    tet_nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    node_coords = np.vstack([prism_nodes, tet_nodes])
    prism_conn = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64)
    tet_conn = np.array([[6, 7, 8, 9]], dtype=np.int64)
    n_prism = 1
    mesh = _MockMesh(n_prism, prism_conn, tet_conn, node_coords)

    p0, p1, p2 = prism_nodes[0], prism_nodes[1], prism_nodes[2]
    area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
    normal = np.array([0.0, 0.0, 1.0])

    # 棱柱 owner 的 c=-1 面（code 4）；四面体 neighbor 的 excluded_vertex=3
    # 面（code 6+3=9）——与共享三角形 (tet_nodes[0:3]==prism_nodes[0:3])
    # 对应，见模块文档。
    result, n1d, sps_1d = _run_kernel_single_face(
        order, n_prism, prism_conn, tet_conn, node_coords,
        owner_cell=0, owner_code=4, neighbor_cell=1, neighbor_code=9,
        area=area, normal=normal,
    )
    (nb_fc, nb_resid, ow_fc, ow_resid, nb_interp, ow_interp,
     nb_cell_id, ow_cell_id, geom_oa, geom_os, geom_na, geom_ns, geom_aw, geom_n) = result

    assert nb_cell_id[0] == 1
    assert ow_cell_id[0] == 0

    char_length = float(np.sqrt(max(area, 1e-300)))
    ref_o = face_ref_grid(n1d, 2, -1.0, sps_1d)
    phys_o = map_ref_points(True, ref_o, prism_nodes)
    interp_ref_nb, _ = build_cross_interp(
        mesh, n1d, sps_1d, target_cell=1, target_face_code=9,
        source_phys=phys_o, char_length=char_length,
    )
    np.testing.assert_allclose(nb_interp[0], interp_ref_nb, atol=1e-10)

    phys_n = native_tet_face_points_physical(n1d, 3, tet_nodes, sps_1d)
    interp_ref_ow, _ = build_cross_interp(
        mesh, n1d, sps_1d, target_cell=0, target_face_code=4,
        source_phys=phys_n, char_length=char_length,
    )
    np.testing.assert_allclose(ow_interp[0], interp_ref_ow, atol=1e-10)
