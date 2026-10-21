"""AutoFlowCFD V2.0 - native 四面体面的法向量 / 面积：`compute_exact_face_normals_and_weights` 与独立叉积公式
完全一致，覆盖全部 4 个 excluded_vertex。"""

import numpy as np

from autoflowcfd.fr.face_flux_points.exact_normal import compute_exact_face_normals_and_weights
from autoflowcfd.fr.quadrature_points import gauss_legendre

# 一个真实、非退化、正体积（右手系）四面体
TET_NODES = np.array([
    [0.1, -0.2, 0.3],
    [1.3, 0.1, -0.1],
    [-0.2, 1.1, 0.2],
    [0.0, 0.2, 1.4],
], dtype=float)


def _reference_outward_normal_and_area(excluded_vertex):
    verts = [v for v in range(4) if v != excluded_vertex]
    pa, pb, pc = TET_NODES[verts[0]], TET_NODES[verts[1]], TET_NODES[verts[2]]
    p_excluded = TET_NODES[excluded_vertex]
    n = np.cross(pb - pa, pc - pa)
    if np.dot(n, p_excluded - pa) > 0:
        n = -n  # 翻转到指向被排除顶点的反方向（outward）
    area = 0.5 * np.linalg.norm(n)
    return n / np.linalg.norm(n), area


def test_native_tet_face_normal_matches_cross_product_reference_all_excluded_vertices():
    order = 2
    n1d = order + 1
    sps_1d, weights_1d = gauss_legendre(n1d)
    n_fp = n1d * n1d
    n_prism = 0
    tet_conn = np.array([[0, 1, 2, 3]], dtype=np.int64)
    prism_conn = np.empty((0, 6), dtype=np.int64)

    for ev in range(4):
        code = 6 + ev
        owner_cell = np.array([0], dtype=np.int64)
        owner_code = np.array([code], dtype=np.int64)

        true_normal, true_area_weight = compute_exact_face_normals_and_weights(
            n_faces=1, n1d=n1d, sps_1d=sps_1d, weights_1d=weights_1d, n_prism=n_prism,
            owner_cell=owner_cell, owner_code=owner_code,
            prism_conn=prism_conn, tet_conn=tet_conn, node_coords=TET_NODES,
        )

        ref_normal, ref_area = _reference_outward_normal_and_area(ev)
        # 直边四面体：所有 Flux Points 法向量必须完全相同（常数 Jacobian）
        for p in range(n_fp):
            np.testing.assert_allclose(
                true_normal[0, p], ref_normal, atol=1e-10,
                err_msg=f"excluded_vertex={ev}, fp={p} 法向量与叉积参考值不一致",
            )
        # 面积微元求和必须精确等于参考三角形面积（叉积法直接算的是
        # 真实局部面积 Jacobian，两个 Gauss-Legendre 点足以精确积分这个
        # 关于 (a,b) 至多二次的被积函数，不需要额外换算因子）。
        total_area = true_area_weight[0].sum()
        np.testing.assert_allclose(
            total_area, ref_area, rtol=1e-9,
            err_msg=f"excluded_vertex={ev} 面积微元求和与参考三角形面积不一致",
        )


def test_non_native_face_code_is_refused():
    """面编码必须是原生编码：0~5（立方体面编码，未经原生翻译）直接报错，而不是按别的公式算出错误的法向。"""
    import pytest

    sps_1d, weights_1d = gauss_legendre(2)
    with pytest.raises(ValueError, match="不是原生编码"):
        compute_exact_face_normals_and_weights(
            n_faces=1, n1d=2, sps_1d=sps_1d, weights_1d=weights_1d, n_prism=0,
            owner_cell=np.array([0], dtype=np.int64), owner_code=np.array([0], dtype=np.int64),
            prism_conn=np.empty((0, 6), dtype=np.int64), tet_conn=np.array([[0, 1, 2, 3]], dtype=np.int64),
            node_coords=TET_NODES)
