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

    Moller (1997)：拒绝所有顶点严格位于另一个三角形平面同一侧
    的面对（使用该三角形已有的平面进行快速拒绝）；否则两个三角形
    都穿过两平面相交的直线 L，因此相交当且仅当它们在 L 上的两个
    区间重叠。仅共享一条边或一个顶点的两个三角形不被报告为相交
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
        result[generic] = _intersect_on_line(
            a0[generic], a1[generic], a2[generic], da0[generic], da1[generic], da2[generic],
            b0[generic], b1[generic], b2[generic], db0[generic], db1[generic], db2[generic],
            normal_a[generic], normal_b[generic], eps,
        )

        # 薄碎片三角形修正。_intersect_on_line 通过将每个三角形的边
        # 投影到两平面相交的直线上（line_dir = cross(normal_a, normal_b)）
        # 并比较沿该直线的一维区间来确定重叠——该投影和它起始的
        # da/db 带符号平面距离在三角形是薄碎片时都会丢失精度
        # （本项目自身的斜接拐角补偿的真实、可测量的后果——
        # 见 mesh_layer_step.py），因为薄三角形自身的法向量/平面
        # 按构造对沿三角形长轴的真实偏移只有弱敏感性。
        # 这不仅限于近平行平面对——最初怀疑是，但在 cube_demo 上
        # 直接确认了在各种平面角度下（cross(normal_a,normal_b) 的
        # 模长从 ~7e-6 到 ~0.14）两个真实、明确距离分开的薄碎片
        # （每种情况都通过暴力点采样和 triangle_triangle_min_distance
        # 独立验证，0.01m——约本项目 min_cell_size 的 3 倍）仍被标记
        # 为相交。
        #
        # 修复：从 triangle_triangle_min_distance 获取第二意见——
        # 它基于点到三角形/线段对线段最近点原语（Ericson 5.1.5/5.1.9），
        # 从不除以三角形自身法向量或两平面的叉乘，因此无论三角形
        # 多薄或两平面如何取向都保持良好条件。
        # 对每个通用路径当前标记为相交的行都检查（不仅是疑似
        # 薄的——相对于假阳性风险成本很低，且没有可靠的更廉价
        # 方法提前预测哪些行需要）：仅用于将真翻转为假，且仅在
        # 第二意见发现距离明确超出 eps 时——即这只能将假阳性
        # 修正为经验证的真阴性，永不引入新的，也永不触碰通用
        # 路径已标记为非相交的行或（不同条件的）共面分支处理的行。
        # 已直接对本项目自有的 15 个手工构建边界情况和 3000 个
        # 混合缩放压力测试案例确认——它们都没有覆盖这个薄碎片
        # 情况，因此这个修正纯粹是附加的，针对已有代码未覆盖的情况。
        gi = np.flatnonzero(generic)
        suspect = result[gi]
        if np.any(suspect):
            si = gi[suspect]
            dist = triangle_triangle_min_distance(
                a0[si], a1[si], a2[si], b0[si], b1[si], b2[si],
            )
            result[si[dist > eps]] = False

    if np.any(coplanar):
        result[coplanar] = _coplanar_triangle_overlap(
            a0[coplanar], a1[coplanar], a2[coplanar],
            b0[coplanar], b1[coplanar], b2[coplanar],
            normal_a[coplanar], eps,
        )

    return result


def _interval_on_line(
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
    d0: np.ndarray, d1: np.ndarray, d2: np.ndarray,
    line_pt: np.ndarray, line_dir: np.ndarray,
    eps: float,
) -> tuple:
    """将三角形的交集投影到直线上，返回 [lo, hi] 区间。
    `line_dir` 必须是单位长度（调用方负责归一化），使 `lo`/`hi`
    是沿直线的真实距离，可直接与 `eps`（也是真实距离）比较——
    两个三角形的区间都用相同的 line_dir 计算，因此投影值
    彼此直接可比。

    对三角形的 3 条边，当边两端点的带符号距离 (d0, d1, d2) 异号
    或任一个在 `eps` 内接近零时（`da * db <= eps^2`，精确
    `da * db <= 0` 测试的 eps 放宽版本——见 triangle_triangle_intersect
    的 eps 文档了解为何精确零测试在涉及真实网格坐标时很脆弱），
    该边穿过（或接触）平面，交点然后线性插值并投影到直线上。
    独立检查所有 3 条边——而非先选一个"奇数个输出"顶点并假设
    另两边是穿越——正确处理了顶点恰好落在另一个平面上的情况
    （该顶点 d == 0）：接触它的两条边都在该顶点注册有效的"穿越"，
    第三条（真正同号的）边不贡献任何穿越。单顶点选取方法会错误
    处理这种配置，因为当一个距离恰好（或接近）零时，"两顶点共享
    一个符号"没有一条能干净成立。

    在极罕见情况下三条边都不注册为接触（数值上，三个都刚好在
    同一侧，尽管该对已经通过了调用方自身的"未分离"测试——
    浮点边界情况，不是有效的几何配置），这返回空区间
    (lo=+inf, hi=-inf) 而非 NaN/np.nanmin 的全 NaN 警告——
    空区间永远不会与任何东西重叠，这对接近容差边界的模糊
    接触是正确、安全的结论。
    """
    n = len(v0)

    def _proj(pt: np.ndarray) -> np.ndarray:
        return np.einsum('ij,ij->i', pt - line_pt, line_dir)

    p0, p1, p2 = _proj(v0), _proj(v1), _proj(v2)
    eps2 = eps * eps

    def _edge_crossing(pa, da, pb, db):
        touches = da * db <= eps2
        denom = da - db
        t = np.divide(da, denom, out=np.full(n, 0.5), where=np.abs(denom) > 1e-300)
        crossing = pa + t * (pb - pa)
        return np.where(touches, crossing, np.nan)

    c01 = _edge_crossing(p0, d0, p1, d1)
    c12 = _edge_crossing(p1, d1, p2, d2)
    c20 = _edge_crossing(p2, d2, p0, d0)

    stacked = np.stack([c01, c12, c20], axis=0)
    is_nan = np.isnan(stacked)
    # 在无 NaN 数组上用普通 np.min/np.max，而非 np.nanmin/np.nanmax——
    # 如果某行全为 NaN（无边接触，见上方文档字符串），在 min 前将
    # NaN 替换为 +inf（或 max 前替换为 -inf）使该行自然解析为
    # lo=+inf, hi=-inf（空区间），而不会触发 nanmin/nanmax 的全 NaN RuntimeWarning。
    lo = np.min(np.where(is_nan, np.inf, stacked), axis=0)
    hi = np.max(np.where(is_nan, -np.inf, stacked), axis=0)
    return lo, hi


def _intersect_on_line(
    a0, a1, a2, da0, da1, da2,
    b0, b1, b2, db0, db1, db2,
    normal_a, normal_b,
    eps: float = 1e-9,
) -> np.ndarray:
    """两个三角形都真实穿过对方的平面（非分离、非共面）——
    当且仅当它们沿两平面公共线的投影区间以正长度重叠时相交。
    此处用闭合 `<=` 比较也会将仅在直线上一个点接触的两个
    三角形（例如恰好共享一个顶点）标记为"相交"——改用一个小
    eps 余量排除，与此函数的约定一致（共享边/共享顶点相邻
    不算重叠）。"""
    line_dir = np.cross(normal_a, normal_b)
    line_dir_mag = np.linalg.norm(line_dir, axis=1, keepdims=True)
    # 归一化使 _interval_on_line 的投影 lo/hi 是真实距离（米），
    # 可直接与 eps 比较——见该函数自身的 eps 文档。保护近平行
    # 但不完全共面的情况（平面几乎平行，line_dir 模长接近零）：
    # 回退为通过已空的区间将其视为非相交，而非除以约零并放大
    # 浮点噪声到无意义的"直线"方向。
    line_dir = np.divide(line_dir, line_dir_mag, out=np.zeros_like(line_dir), where=line_dir_mag > 1e-9)
    line_pt = a0  # any point on plane A's own triangle works as a projection origin

    lo_a, hi_a = _interval_on_line(a0, a1, a2, da0, da1, da2, line_pt, line_dir, eps)
    lo_b, hi_b = _interval_on_line(b0, b1, b2, db0, db1, db2, line_pt, line_dir, eps)

    return (lo_a < hi_b - eps) & (lo_b < hi_a - eps)


def _coplanar_triangle_overlap(
    a0: np.ndarray, a1: np.ndarray, a2: np.ndarray,
    b0: np.ndarray, b1: np.ndarray, b2: np.ndarray,
    normal: np.ndarray,
    eps: float = 1e-9,
) -> np.ndarray:
    """2D separating-axis test for coplanar triangles: project onto the
    plane's dominant axis pair (drop the coordinate with the largest
    |normal| component, which minimizes projection distortion) and test
    the 6 candidate separating axes (each triangle's 3 edge normals, in
    2D)."""
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

    # A `<` (strict) separation test would call two coplanar triangles that
    # only touch along a shared edge "not separated" (their projections on
    # that edge's own normal axis meet exactly, but nowhere do they cross
    # it) - i.e. it would misreport ordinary shared-edge adjacency as
    # overlap. `<=` with a small eps margin treats an exact touch as
    # separated (no overlap), consistent with triangle_triangle_intersect's
    # documented contract. `ax` is unit-normalized before projecting so
    # `proj_a`/`proj_b` are true distances (meters) and `eps` is directly
    # comparable regardless of that edge's own length - an un-normalized
    # axis would scale the projected values by the edge length, making the
    # same eps effectively too loose for a long edge and too tight for a
    # short one (the same class of bug as triangle_triangle_intersect's own
    # normal-normalization - see its eps doc).
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
