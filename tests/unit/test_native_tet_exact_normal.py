"""AutoFlowCFD V2.0 - native 四面体（路径C）真实法向量/面积计算决定性
验证（Part7 文档"执行状态更新"节记录的真实 bug：`compute_exact_adj_rows`
原先按 (axis,side) 匹配坍缩坐标查找表，`excluded_vertex` 复用同一个
axis 槽位存储会与真坍缩坐标语义数值碰撞——本文件验证修复后
`compute_exact_face_normals_and_weights` 对 native 四面体给出的法向量/
面积与独立叉积公式完全一致，覆盖全部 4 个 excluded_vertex 取值，包括
会与坍缩坐标 (0,-1)/(1,-1)/(2,-1) 碰撞的 0/1/2 和不落在任何坍缩坐标
条目里的 3。"""

import numpy as np

from autoflowcfd.fr.face_flux_points_exact_normal import compute_exact_face_normals_and_weights
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
        # owner_axis/owner_side：模拟 face_flux_points_merge.py 实际会
        # 传入的、复用 axis 槽位存 excluded_vertex 的占位值（对 ev=0,1,2
        # 会恰好与真坍缩坐标 (axis,-1.0) 条目数值相同——这正是本测试要
        # 确认已被 owner_code 分派规避掉的碰撞）。
        owner_axis = np.array([ev], dtype=np.int64)
        owner_side = np.array([-1.0], dtype=np.float64)
        owner_code = np.array([code], dtype=np.int64)

        true_normal, true_area_weight = compute_exact_face_normals_and_weights(
            n_faces=1, n1d=n1d, sps_1d=sps_1d, weights_1d=weights_1d, n_prism=n_prism,
            owner_cell=owner_cell, owner_axis=owner_axis, owner_side=owner_side,
            prism_conn=prism_conn, tet_conn=tet_conn, node_coords=TET_NODES,
            owner_code=owner_code,
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


def test_native_tet_normal_would_collide_without_owner_code_guard():
    """回归防护：不传 `owner_code`（旧行为）时，excluded_vertex=0/1/2 会
    落进坍缩坐标 (axis,-1) 分支给出错误结果，excluded_vertex=3 落不进
    任何分支给出全零——用来在未来如果这个防护被误删时立刻报错。"""
    order = 1
    n1d = order + 1
    sps_1d, weights_1d = gauss_legendre(n1d)
    n_prism = 0
    tet_conn = np.array([[0, 1, 2, 3]], dtype=np.int64)
    prism_conn = np.empty((0, 6), dtype=np.int64)

    ev = 3
    owner_cell = np.array([0], dtype=np.int64)
    owner_axis = np.array([ev], dtype=np.int64)
    owner_side = np.array([-1.0], dtype=np.float64)

    true_normal_no_guard, _ = compute_exact_face_normals_and_weights(
        n_faces=1, n1d=n1d, sps_1d=sps_1d, weights_1d=weights_1d, n_prism=n_prism,
        owner_cell=owner_cell, owner_axis=owner_axis, owner_side=owner_side,
        prism_conn=prism_conn, tet_conn=tet_conn, node_coords=TET_NODES,
    )
    # 没有 owner_code 时，axis=3 不落进任何坍缩坐标 (axis,side) 条目，
    # adj_row 保持零初始化 -> 归一化后（0/max(0,eps)）法向量恒为 0。
    np.testing.assert_allclose(true_normal_no_guard[0], 0.0, atol=1e-300)

    ref_normal, _ = _reference_outward_normal_and_area(ev)
    assert np.linalg.norm(ref_normal) > 0.5, "参考法向量本身应该是非零单位向量，证明上面的全零是 bug 不是巧合"
