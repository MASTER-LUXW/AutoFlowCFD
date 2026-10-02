"""原生面算子整表与三角形面坍缩顶点槽位（`fr/operators/face_ops.py`、`fr/triangle_apex.py`）。

1. 槽位规则：坍缩顶点 = 面上全局编号最大的顶点，与单元局部编号无关（同一个面从两个
   不同局部编号的单元看过去，选出的坍缩顶点是同一个全局节点）；
2. 槽位 `s` 的通量点集是槽位 0 点集的同一组物理点吗——**不是**（换坍缩顶点点集整体
   不同），但三个槽位都是同一个三角形上的合法求积：权重之和等于面积、对 `2*order`
   次多项式精确；
3. 整表：每个（面, 槽位）行与逐个构造的算子逐位相等、填充块恰为零、四边形面不存在
   的槽位是 NaN（误用立刻暴露）；`native_face_extrap` 拒绝给四边形面传非零槽位。
"""

import numpy as np
import pytest

from autoflowcfd.fr.face_flux_trace import face_reference_weights
from autoflowcfd.fr.native_prism.face import build_native_prism_boundary_extrap, native_prism_face_points
from autoflowcfd.fr.native_tet.basis import build_native_tet_boundary_extrap, build_native_tet_lift, native_tet_face_points
from autoflowcfd.fr.operators import generate_fr_operators
from autoflowcfd.fr.operators.face_ops import N_FACE_OPS, face_op_index
from autoflowcfd.fr.triangle_apex import apex_slots, cell_triangle_slots, rotate_triangle


def test_apex_is_the_largest_global_id_regardless_of_local_order():
    rng = np.random.default_rng(0)
    tri = rng.permutation(1000)[:3 * 200].reshape(200, 3)
    for perm in ([0, 1, 2], [1, 2, 0], [2, 0, 1], [1, 0, 2]):
        fv = tri[:, perm]
        slots = apex_slots(fv)
        apex = np.array([rotate_triangle(row, s)[2] for row, s in zip(fv, slots)])
        np.testing.assert_array_equal(apex, tri.max(axis=1))


def test_cell_slots_agree_across_the_shared_face_of_two_tets():
    # 两个四面体共享面 {10, 20, 30}，局部编号顺序不同
    tets = np.array([[10, 20, 30, 5], [7, 30, 10, 20]])
    slots = cell_triangle_slots(tets, None)
    # 共享面在单元 0 排除局部顶点 3、在单元 1 排除局部顶点 0
    fv0 = tets[0, [0, 1, 2]]
    fv1 = tets[1, [1, 2, 3]]
    assert rotate_triangle(fv0, slots[0, 3])[2] == rotate_triangle(fv1, slots[1, 0])[2] == 30
    assert np.all(slots[:, 4] == 0)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_every_slot_is_an_exact_triangle_quadrature(order):
    """参考四面体面 v=3（z=-1 平面上的直角三角形）：三个槽位的点集各不相同，
    但都对 2*order 次多项式精确（解析积分 ∫x^a y^b 对比）。"""
    from math import factorial

    w = face_reference_weights(order)
    pts = [native_tet_face_points(order, 3, s) for s in range(3)]
    assert not np.allclose(np.sort(pts[0], axis=0), np.sort(pts[1], axis=0))
    # 参考三角形 (r,s) ∈ {r,s>=-1, r+s<=0}；换到 (u,v)=((1+r)/2,(1+s)/2) 单纯形，面积元 4
    for p in pts:
        u, v = (1 + p[:, 0]) / 2, (1 + p[:, 1]) / 2
        # 面权重带 Duffy 因子：用参考面积矢量模长（四面体面 v=3 法向 z）
        from autoflowcfd.fr.face_flux_points.exact_normal import _native_tet_adj_row_batched
        from autoflowcfd.fr.quadrature_points import gauss_legendre

        ref = np.array([[-1.0, -1, -1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
        g, _ = gauss_legendre(order + 1)
        dA = np.linalg.norm(_native_tet_adj_row_batched(3, ref[None], order + 1, g)[0], axis=1) * w
        for a in range(2 * order + 1):
            for b in range(2 * order + 1 - a):
                exact = 4.0 * factorial(a) * factorial(b) / factorial(a + b + 2)
                np.testing.assert_allclose(np.sum(dA * u ** a * v ** b), exact, rtol=1e-12)


@pytest.mark.parametrize("order", [1, 2])
def test_op_tables_match_individual_builders_and_padding(order):
    ops = generate_fr_operators(order)
    E, L = ops.face_extrap_by_op, ops.face_lift_by_op
    assert E.shape[0] == L.shape[0] == N_FACE_OPS
    nt, npr = ops.n_native_sps_tet, ops.n_native_sps_prism
    for v in range(4):
        for s in range(3):
            op = int(face_op_index(6 + v, s))
            np.testing.assert_array_equal(E[op][:, :nt], build_native_tet_boundary_extrap(order, v, s))
            np.testing.assert_array_equal(L[op][:nt], build_native_tet_lift(order, v, s))
            assert np.all(E[op][:, nt:] == 0.0) and np.all(L[op][nt:] == 0.0)
    for fid in range(5):
        for s in range(3):
            op = int(face_op_index(10 + fid, s))
            if fid >= 2 and s > 0:
                assert np.isnan(E[op]).all() and np.isnan(L[op]).all()
                with pytest.raises(ValueError):
                    ops.native_face_extrap(10 + fid, s)
                continue
            np.testing.assert_array_equal(E[op][:, :npr], build_native_prism_boundary_extrap(order, fid, s))
            assert np.all(E[op][:, npr:] == 0.0) and np.all(L[op][npr:] == 0.0)


@pytest.mark.parametrize("order", [1, 2])
def test_prism_cap_slot_points_are_the_rotated_triangle(order):
    """封盖槽位 s 的点 = 槽位 0 的重心坐标作用在轮换后的三个参考顶点上。"""
    verts = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]])
    p0 = native_prism_face_points(order, 0, 0)[:, :2]
    lam = np.linalg.solve(np.vstack([verts.T, np.ones(3)]), np.vstack([p0.T, np.ones(len(p0))])).T
    for s in range(3):
        rv = verts[list(rotate_triangle((0, 1, 2), s))]
        np.testing.assert_allclose(native_prism_face_points(order, 0, s)[:, :2], lam @ rv, atol=1e-14)
