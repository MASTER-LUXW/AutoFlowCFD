"""AutoFlowCFD V2.0 - 四面体路径C（native basis）单元-单元界面耦合的
数学原理决定性验证（Part6 阶段2 范围的核心数学问题，尚未接入生产
`face_flux_points/geometry.py`/`inviscid_kernel.py` 管线——本文件只验证"这套
方法在原理上、在真实相邻单元几何上是否成立"，是否接入现有 Newton
迭代匹配/numba 界面核函数的工程实现，留作后续独立工作）。

关键发现（与现有坍缩坐标方案的本质区别）：现有方案用张量积 Gauss 点
（严格单元内部，不含边界），四面体-四面体共享面之间必须解一个
Vandermonde 系统构造"体积->面"外插矩阵。Warp & Blend（GLL 型）节点
天然包含每个面的边界节点子集，数量精确等于二维三角形节点数——这意味着
**四面体-四面体共享面之间，路径C不需要任何插值/外插，两侧各自的面
节点物理坐标集合本身就精确重合**，只需要一次性求出下标对应排列。
"""

import numpy as np

from autoflowcfd.fr.native_tet.basis import (
    build_native_tet_operators,
    map_native_tet_to_physical,
    face_node_indices,
    match_face_nodes_by_physical_position,
)
from autoflowcfd.grid.curved_mapping.curved_mapping import tet_barycentric

from tests.unit.test_fr_residual_inviscid import _build_synthetic_mixed_mesh


def _shared_face_local_vertex(nodes_a: np.ndarray, nodes_b: np.ndarray, tol: float = 1e-9) -> int:
    """两个四面体如果共享一个面，恰好有一个 nodes_a 的局部顶点不在
    nodes_b 里——返回它在 nodes_a 里的局部下标（0~3），即
    `face_node_indices` 要排除的那个顶点。"""
    for i in range(4):
        if not np.any(np.all(np.abs(nodes_b - nodes_a[i]) < tol, axis=1)):
            return i
    raise ValueError("两个单元没有恰好共享一个顶点不同的面——不是真正相邻的四面体")


def test_native_tet_face_node_count_matches_2d_triangle_dimension():
    """每个面的边界节点数必须精确等于二维三角形单纯形维度
    `(order+1)(order+2)/2`，四个面完全对称（不依赖具体排除哪个顶点）。"""
    for order in [1, 2, 3, 4]:
        ref_rst, _ = build_native_tet_operators(order)
        expected = (order + 1) * (order + 2) // 2
        for excluded in range(4):
            idx = face_node_indices(ref_rst, excluded)
            assert len(idx) == expected, f"order={order} excluded={excluded}: got {len(idx)}, expected {expected}"


def test_native_tet_shared_face_physical_points_match_exactly_on_real_adjacent_cells():
    """真实相邻的两个四面体单元（`_build_synthetic_mixed_mesh` 里真正
    共享一个面的 tet0/tet1），各自独立取出面节点、映射到物理坐标，
    经最近邻匹配后必须逐点精确重合（机器精度）——这是路径C下"四面体-
    四面体界面不需要跨基插值"这个结论的决定性证据，不是理论推测。
    """
    mesh = _build_synthetic_mixed_mesh(2)
    nodes = [mesh._node_coords[conn] for conn in mesh._fixed_tet_conn]
    assert len(nodes) == 2

    excluded_a = _shared_face_local_vertex(nodes[0], nodes[1])
    excluded_b = _shared_face_local_vertex(nodes[1], nodes[0])

    for order in [1, 2, 3, 4]:
        ref_rst, _ = build_native_tet_operators(order)
        idx_a = face_node_indices(ref_rst, excluded_a)
        idx_b = face_node_indices(ref_rst, excluded_b)

        phys_a = map_native_tet_to_physical(ref_rst, nodes[0])[idx_a]
        phys_b = map_native_tet_to_physical(ref_rst, nodes[1])[idx_b]

        perm = match_face_nodes_by_physical_position(phys_a, phys_b)
        np.testing.assert_allclose(phys_b[perm], phys_a, atol=1e-10)


def test_native_tet_face_normal_directions_are_opposite_at_matched_points():
    """两侧在共享面上、经过匹配后对应的物理点上，各自算出的面法向量
    （用该面三角形两条边的叉乘，与该单元自身其余顶点方向做符号约定：
    法向必须指向"背离本单元其余部分"的方向）必须互为相反数——这是
    黎曼求解器（AUSM+up 等）正确定义"owner->neighbor"通量方向的
    基本前提，不满足这一点，界面通量的符号会全盘错误。
    """
    mesh = _build_synthetic_mixed_mesh(2)
    nodes = [mesh._node_coords[conn] for conn in mesh._fixed_tet_conn]
    excluded_a = _shared_face_local_vertex(nodes[0], nodes[1])
    excluded_b = _shared_face_local_vertex(nodes[1], nodes[0])

    order = 2
    ref_rst, _ = build_native_tet_operators(order)
    idx_a = face_node_indices(ref_rst, excluded_a)
    idx_b = face_node_indices(ref_rst, excluded_b)
    phys_a_all = map_native_tet_to_physical(ref_rst, nodes[0])
    phys_b_all = map_native_tet_to_physical(ref_rst, nodes[1])
    phys_a = phys_a_all[idx_a]
    phys_b = phys_b_all[idx_b]
    perm = match_face_nodes_by_physical_position(phys_a, phys_b)

    # 用面三角形的三个物理顶点（重心坐标里另外三个非零分量对应的
    # 四面体顶点）算法向量，指向约定：叉乘方向再根据是否指向本单元
    # 排除掉的那个顶点（本单元其余部分所在方向）来定符号——法向应
    # 指向背离它。
    p0, p1, p2, p3 = nodes[0]
    face_verts_a = [n for i, n in enumerate([p0, p1, p2, p3]) if i != excluded_a]
    normal_a = np.cross(face_verts_a[1] - face_verts_a[0], face_verts_a[2] - face_verts_a[0])
    excluded_pt_a = [p0, p1, p2, p3][excluded_a]
    if np.dot(normal_a, excluded_pt_a - face_verts_a[0]) > 0:
        normal_a = -normal_a
    normal_a = normal_a / np.linalg.norm(normal_a)

    q0, q1, q2, q3 = nodes[1]
    face_verts_b = [n for i, n in enumerate([q0, q1, q2, q3]) if i != excluded_b]
    normal_b = np.cross(face_verts_b[1] - face_verts_b[0], face_verts_b[2] - face_verts_b[0])
    excluded_pt_b = [q0, q1, q2, q3][excluded_b]
    if np.dot(normal_b, excluded_pt_b - face_verts_b[0]) > 0:
        normal_b = -normal_b
    normal_b = normal_b / np.linalg.norm(normal_b)

    np.testing.assert_allclose(normal_a, -normal_b, atol=1e-10)
