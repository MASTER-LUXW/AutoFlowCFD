"""AutoFlowCFD V2.0 - native 四面体（路径C）numba 主并行核函数
(`face_flux_points/kernel.py::build_fp_newton_parallel`) 与纯 Python
参考实现（`build_cross_interp`/`native_tet_face_points_physical`）的
决定性一致性验证。

背景：Part7 文档阶段2 发现 numba 核函数内部有自己独立、编译过的
点位定位/插值矩阵构建逻辑（`_newton_locate_nb`/`_FACE_AXIS` 假设坍缩
坐标 (axis,side) 语义），与 Python 层已经验证过的 native 分支
（`locate_native_tet_face_point`/`build_cross_interp`）完全不共享代码，
必须独立移植（`face_flux_points/native_geometry_nb.py` 新增的 `_tet_native_
locate_nb`/`_native_tet_face_points_nb`/`_native_interp_matrix_nb`）并
独立验证数值一致——本文件直接调用核函数（构造最小合成连接数组，绕开
完整网格加载/求解器接入，那是仍然独立、尚未做的后续工作），覆盖两类
真实拓扑：四面体(native)-四面体(native)、四面体(native)-棱柱(native)。
"""

import numpy as np

from autoflowcfd.fr.face_flux_points.kernel import build_fp_newton_parallel
from autoflowcfd.fr.face_flux_points import build_cross_interp, native_tet_face_points_physical
from autoflowcfd.fr.face_flux_points.basis_inverses import build_basis_inverses
from autoflowcfd.fr.face_flux_points.native_geometry_nb import native_prism_cap_points_nb
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

    # 与生产（face_flux_points/merge.py）同一个构造：原生四面体与原生棱柱的 V^{-1} 与模态索引
    b = build_basis_inverses(order)

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
        b.v_sps_inv_native, b.native_mode_i, b.native_mode_j, b.native_mode_k,
        b.v_sps_inv_np, b.np_mode_i, b.np_mode_j, b.np_mode_k,
        # 两侧坍缩顶点槽位 0：Python 参考实现按槽位 0 生成通量点
        np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64),
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
    """原生棱柱三角形封盖面（c=-1，面编码 10）与相邻原生四面体共享同一个物理面——核函数的棱柱侧（坍缩立方体面
    Newton 定位 + 原生棱柱模态）与四面体侧必须分别与各自的纯 Python 参考实现一致，且同一个面记录里两侧的
    (axis,side) 与 excluded_vertex 语义在同一次核函数调用里同时正确。"""
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

    # 棱柱 owner 的 c=-1 封盖（原生面编码 10）；四面体 neighbor 的 excluded_vertex=3 面（编码 6+3=9），
    # 与共享三角形 (tet_nodes[0:3]==prism_nodes[0:3]) 对应。
    result, n1d, sps_1d = _run_kernel_single_face(
        order, n_prism, prism_conn, tet_conn, node_coords,
        owner_cell=0, owner_code=10, neighbor_cell=1, neighbor_code=9,
        area=area, normal=normal,
    )
    (nb_fc, nb_resid, ow_fc, ow_resid, nb_interp, ow_interp,
     nb_cell_id, ow_cell_id, geom_oa, geom_os, geom_na, geom_ns, geom_aw, geom_n) = result

    assert nb_cell_id[0] == 1
    assert ow_cell_id[0] == 0

    char_length = float(np.sqrt(max(area, 1e-300)))
    # owner（棱柱封盖）的通量点：kernel 自己的生成函数（cap 0 = c=-1 封盖，坍缩顶点槽位 0）
    phys_o = native_prism_cap_points_nb(n1d, 0, 0, prism_nodes, sps_1d)
    interp_ref_nb, _ = build_cross_interp(
        mesh, n1d, sps_1d, target_cell=1, target_face_code=9,
        source_phys=phys_o, char_length=char_length,
    )
    np.testing.assert_allclose(nb_interp[0], interp_ref_nb, atol=1e-10)

    phys_n = native_tet_face_points_physical(n1d, 3, tet_nodes, sps_1d)
    interp_ref_ow, _ = build_cross_interp(
        mesh, n1d, sps_1d, target_cell=0, target_face_code=10,
        source_phys=phys_n, char_length=char_length,
    )
    np.testing.assert_allclose(ow_interp[0], interp_ref_ow, atol=1e-10)
