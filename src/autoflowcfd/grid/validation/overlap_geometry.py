"""三角形-三角形重叠/接近检查的精确几何原语。

基于 Ericson《Real-Time Collision Detection》(2005) 和 Moller《A Fast
Triangle-Triangle Intersection Test》(1997) 的标准解析算法的向量化实现
（numpy，无候选对 Python 循环）——非启发式或近似方法。本模块所有函数
均接受批量输入（每个点/顶点参数形状为 (N, 3)，每行一个候选对），因此
调用方在完成自粗相过滤后有 M 个候选对时，只需少量向量化 numpy 调用即可
运行全部 M 个测试，而非 Python 级 M 次循环。

由 mesh_overlap_check.py 使用，后者负责粗相候选生成和编排；本模块是纯
计算几何，不包含网格特定概念（无单元、无边界组、除三个顶点位置外无面信息）。

点/线段到三角形的最近点/最近距离原语拆到了同目录
overlap_geometry_primitives.py，本文件只保留三角形-三角形相交检测
（自己的 SAT 分离轴逻辑）和基于上述原语组合出的最小距离。
"""

import numpy as np

from .overlap_geometry_primitives import point_to_triangle_distance, segment_to_segment_distance


def _signed_dist_to_plane(pts: np.ndarray, plane_pt: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return np.einsum('ij,ij->i', pts - plane_pt, normal)


def triangle_triangle_intersect(
    a0: np.ndarray, a1: np.ndarray, a2: np.ndarray,
    b0: np.ndarray, b1: np.ndarray, b2: np.ndarray,
    eps: float = 1e-9,
) -> np.ndarray:
    """精确三角形-三角形交集测试，每行一个布尔值。

    先拒绝所有顶点严格位于另一个三角形平面同一侧的面对（快速拒绝）；
    其余非共面对相交当且仅当某个三角形的某条边穿过另一个三角形
    （`_edges_through_triangles`）。仅共享一条边或一个顶点的两个三角形不被报告为相交
    （它们的重叠区间在一个点接触/测度为零，调用方将其视为"相邻"
    而非"重叠"——见 mesh_overlap_check.py 的节点共享预过滤，它在
    面对到达此函数之前就排除了这样的对）。

    共面三角形（快速拒绝平面测试无法仅用带符号距离区分"共面"
    与"无分离"）回退到共享平面中的 2D 分离轴测试。

    Args:
        a0, a1, a2: (N, 3) 第一个三角形的顶点
        b0, b1, b2: (N, 3) 第二个三角形的顶点
        eps: 与输入坐标相同单位的容差（此处为米）——以下距离
            经过归一化，因此这是真实的、与缩放无关的距离容差。
            之前的版本将原始的（未归一化、面积缩放的）距离与
            固定 eps 比较——对大核心三角形太松，对同一网格上
            的微小 BL 碎片太紧，在真实数百万单元网格上产生了
            全 NaN 区间（RuntimeWarning），只要三角形的边没有
            干净地"接触"另一个平面。

    Returns:
        (N,) bool
    """
    n = len(a0)
    result = np.zeros(n, dtype=bool)

    normal_a = np.cross(a1 - a0, a2 - a0)
    normal_b = np.cross(b1 - b0, b2 - b0)
    # 归一化——使下游每个带符号"距离"都是真实的几何距离（米），
    # 而非按三角形自身面积缩放——见上方 eps 参数文档了解为何
    # 那种缩放依赖性在面尺寸范围大的网格上是个真实的 bug。
    # 退化（零面积）三角形保留其原始（零）法向量；任何涉及
    # 退化三角形的对在物理上作为"交集"无意义，让其安全地
    # 落到非相交结果，而非除以零。
    norm_a_mag = np.linalg.norm(normal_a, axis=1, keepdims=True)
    norm_b_mag = np.linalg.norm(normal_b, axis=1, keepdims=True)
    normal_a = np.divide(normal_a, norm_a_mag, out=np.zeros_like(normal_a), where=norm_a_mag > 1e-300)
    normal_b = np.divide(normal_b, norm_b_mag, out=np.zeros_like(normal_b), where=norm_b_mag > 1e-300)

    db0 = _signed_dist_to_plane(b0, a0, normal_a)
    db1 = _signed_dist_to_plane(b1, a0, normal_a)
    db2 = _signed_dist_to_plane(b2, a0, normal_a)

    da0 = _signed_dist_to_plane(a0, b0, normal_b)
    da1 = _signed_dist_to_plane(a1, b0, normal_b)
    da2 = _signed_dist_to_plane(a2, b0, normal_b)

    def _same_sign_nonzero(d0, d1, d2):
        pos = (d0 > eps) & (d1 > eps) & (d2 > eps)
        neg = (d0 < -eps) & (d1 < -eps) & (d2 < -eps)
        return pos | neg

    b_all_one_side = _same_sign_nonzero(db0, db1, db2)
    a_all_one_side = _same_sign_nonzero(da0, da1, da2)
    separated = b_all_one_side | a_all_one_side

    coplanar = (
        (np.abs(db0) <= eps) & (np.abs(db1) <= eps) & (np.abs(db2) <= eps)
        & (np.abs(da0) <= eps) & (np.abs(da1) <= eps) & (np.abs(da2) <= eps)
    )

    generic = ~separated & ~coplanar
    if np.any(generic):
        # 非共面的两个三角形相交，当且仅当某个三角形的某条边穿过另一个三角形（交线段的两个端点各是一条边与
        # 对方三角形的交点）。2026-10-20 以前这里把两三角形投影到两平面交线上比较区间，再用
        # triangle_triangle_min_distance 当"第二意见"修正薄碎片误报——后者只看顶点-面与边-边距离，一条边从对方
        # 面内部穿过（最常见的相交形态）时两者都不为零，真实相交被翻成"不相交"：竖直三角形穿过水平三角形中部
        # 返回 False，BL 前沿自交冻结与网格重叠检查因此漏检。
        g = generic
        hit = np.zeros(int(g.sum()), dtype=bool)
        tri_a = (a0[g], a1[g], a2[g])
        tri_b = (b0[g], b1[g], b2[g])
        for (p, q) in ((tri_a[0], tri_a[1]), (tri_a[1], tri_a[2]), (tri_a[2], tri_a[0])):
            hit |= _edges_through_triangles(p, q, *tri_b, eps)
        for (p, q) in ((tri_b[0], tri_b[1]), (tri_b[1], tri_b[2]), (tri_b[2], tri_b[0])):
            hit |= _edges_through_triangles(p, q, *tri_a, eps)
        result[g] = hit

    if np.any(coplanar):
        result[coplanar] = _coplanar_triangle_overlap(
            a0[coplanar], a1[coplanar], a2[coplanar],
            b0[coplanar], b1[coplanar], b2[coplanar],
            normal_a[coplanar], eps,
        )

    return result


def _edges_through_triangles(
    p: np.ndarray, q: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float,
) -> np.ndarray:
    """逐行：线段 pq 是否穿过三角形 abc——两端点到三角形平面的真实距离严格异号（超出 eps），且穿越点到三条边
    的面内有向距离都不小于 -eps。只用三重积与真实距离，不计算两平面的交线方向，薄碎片三角形也良态；
    退化（零面积）三角形不被穿过。"""
    normal = np.cross(b - a, c - a)
    mag = np.linalg.norm(normal, axis=1, keepdims=True)
    unit = np.divide(normal, mag, out=np.zeros_like(normal), where=mag > 1e-300)
    dp = _signed_dist_to_plane(p, a, unit)
    dq = _signed_dist_to_plane(q, a, unit)
    crosses = (mag[:, 0] > 1e-300) & (((dp > eps) & (dq < -eps)) | ((dp < -eps) & (dq > eps)))
    t = np.divide(dp, dp - dq, out=np.zeros_like(dp), where=crosses)
    x = p + t[:, None] * (q - p)
    inside = crosses
    for u, v in ((a, b), (b, c), (c, a)):
        edge = v - u
        length = np.linalg.norm(edge, axis=1)
        side = np.einsum('ij,ij->i', np.cross(edge, x - u), unit)
        inside = inside & (side >= -eps * length)
    return inside


def _coplanar_triangle_overlap(
    a0: np.ndarray, a1: np.ndarray, a2: np.ndarray,
    b0: np.ndarray, b1: np.ndarray, b2: np.ndarray,
    normal: np.ndarray,
    eps: float = 1e-9,
) -> np.ndarray:
    """共面三角形的二维分离轴检验：投影到平面的主坐标对上（去掉 |normal| 分量最大的坐标，投影畸变最小），
    检验 6 条候选分离轴（两个三角形各自 3 条棱在二维中的法向）。"""
    n = len(a0)
    abs_normal = np.abs(normal)
    drop_axis = np.argmax(abs_normal, axis=1)  # (N,), 0/1/2

    keep = np.zeros((n, 2), dtype=np.int64)
    for axis in range(3):
        mask = drop_axis == axis
        remaining = [i for i in range(3) if i != axis]
        keep[mask] = remaining

    def _to_2d(pts: np.ndarray) -> np.ndarray:
        rows = np.arange(n)[:, None]
        return pts[rows, keep]

    tri_a = np.stack([_to_2d(a0), _to_2d(a1), _to_2d(a2)], axis=1)  # (N,3,2)
    tri_b = np.stack([_to_2d(b0), _to_2d(b1), _to_2d(b2)], axis=1)

    def _edge_normals_2d(tri: np.ndarray) -> np.ndarray:
        edges = np.roll(tri, -1, axis=1) - tri  # (N,3,2)
        return np.stack([-edges[:, :, 1], edges[:, :, 0]], axis=2)  # (N,3,2)

    axes = np.concatenate([_edge_normals_2d(tri_a), _edge_normals_2d(tri_b)], axis=1)  # (N,6,2)

    # 用 `<`（严格）分离判据会把两个仅沿共享边接触的共面三角形误判为
    # "未分离"（它们在该边自身法向轴上的投影恰好相接、但并未真正穿越
    # 对方）——也就是会把普通的共享边相邻关系误报成重叠。用 `<=` 配合
    # 一个小的 eps 容差，把恰好接触视为"已分离"（不算重叠），与
    # triangle_triangle_intersect 文档承诺的约定一致。`ax` 在投影前先做
    # 单位归一化，这样 `proj_a`/`proj_b` 才是真实的距离（米），`eps` 才能
    # 在不同长度的边之间直接比较——若不归一化，投影值会被该边自身的长度
    # 放大/缩小，导致同一个 eps 对长边实际上偏松、对短边偏紧（与
    # triangle_triangle_intersect 自身法向归一化那一类问题同源，见其
    # eps 相关文档）。
    separated = np.zeros(n, dtype=bool)
    for k in range(6):
        ax = axes[:, k, :]  # (N,2)
        ax_mag = np.linalg.norm(ax, axis=1, keepdims=True)
        ax = np.divide(ax, ax_mag, out=np.zeros_like(ax), where=ax_mag > 1e-300)
        proj_a = np.einsum('nij,nj->ni', tri_a, ax)
        proj_b = np.einsum('nij,nj->ni', tri_b, ax)
        min_a, max_a = proj_a.min(axis=1), proj_a.max(axis=1)
        min_b, max_b = proj_b.min(axis=1), proj_b.max(axis=1)
        separated |= (max_a <= min_b + eps) | (max_b <= min_a + eps)

    return ~separated


def triangle_triangle_min_distance(
    a0: np.ndarray, a1: np.ndarray, a2: np.ndarray,
    b0: np.ndarray, b1: np.ndarray, b2: np.ndarray,
) -> np.ndarray:
    """两个（假定不相交）三角形之间的最小距离，每行一个值。

    对不相交凸多边形对精确：最近点对总是要么一个三角形的顶点
    对另一个三角形的面/边/顶点（由 6 个顶点各自的
    point_to_triangle_distance 覆盖），要么是不涉及任一三角形
    顶点的真正边-边最近接近（由 9 个边-边组合覆盖）——所有
    15 个的全局最小值就是真实答案。调用方应先在此之上用
    triangle_triangle_intersect 为假进行门控；距离对真实重叠
    不是有意义的 ~0 信号（此函数不尝试计算穿透深度）。
    """
    dists = [
        point_to_triangle_distance(a0, b0, b1, b2),
        point_to_triangle_distance(a1, b0, b1, b2),
        point_to_triangle_distance(a2, b0, b1, b2),
        point_to_triangle_distance(b0, a0, a1, a2),
        point_to_triangle_distance(b1, a0, a1, a2),
        point_to_triangle_distance(b2, a0, a1, a2),
    ]
    a_edges = [(a0, a1), (a1, a2), (a2, a0)]
    b_edges = [(b0, b1), (b1, b2), (b2, b0)]
    for (ea0, ea1) in a_edges:
        for (eb0, eb1) in b_edges:
            dists.append(segment_to_segment_distance(ea0, ea1, eb0, eb1))

    return np.min(np.stack(dists, axis=0), axis=0)
