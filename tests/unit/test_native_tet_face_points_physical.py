"""AutoFlowCFD V2.0 - `native_tet_face_points_physical` 单元测试
（Part7 阶段2"二·五"节修正：native 四面体自身面 Flux Points 数量必须
与全网格统一的 n_fp=n1d*n1d 一致）。
"""

import numpy as np

from autoflowcfd.fr.face_flux_points import native_tet_face_points_physical
from autoflowcfd.fr.quadrature_points import gauss_legendre

TET_NODES = np.array([[0.1, -0.3, 0.2], [1.2, 0.0, -0.1], [-0.1, 1.1, 0.0], [0.0, 0.1, 1.3]], dtype=float)


def test_returns_n1d_squared_points_matching_global_n_fp():
    for order in [1, 2, 3]:
        n1d = order + 1
        sps_1d, _ = gauss_legendre(n1d)
        for excluded_vertex in range(4):
            pts = native_tet_face_points_physical(n1d, excluded_vertex, TET_NODES, sps_1d)
            assert pts.shape == (n1d * n1d, 3)


def test_points_lie_exactly_on_the_given_face_plane():
    """所有生成的点必须恰好落在该面对应的三角形所在平面内——用重心坐标
    验证：排除的那个顶点分量应为零。"""
    n1d = 3
    sps_1d, _ = gauss_legendre(n1d)
    from autoflowcfd.grid.curved_mapping.curved_mapping import tet_barycentric

    for excluded_vertex in range(4):
        pts = native_tet_face_points_physical(n1d, excluded_vertex, TET_NODES, sps_1d)
        p0, p1, p2, p3 = TET_NODES
        J = np.column_stack([p1 - p0, p2 - p0, p3 - p0])
        Jinv = np.linalg.inv(J)
        L234 = (pts - p0) @ Jinv.T
        L = np.column_stack([1.0 - L234.sum(axis=1), L234])
        np.testing.assert_allclose(L[:, excluded_vertex], 0.0, atol=1e-9)
        # 其余三个重心坐标应在 [0,1] 范围内（点确实在三角形内部，不是
        # 延长线上——坍缩三角形网格的凸组合构造保证这一点）
        for k in range(4):
            if k != excluded_vertex:
                assert np.all(L[:, k] >= -1e-9) and np.all(L[:, k] <= 1.0 + 1e-9)
