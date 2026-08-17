"""点/线段与三角形之间的基础几何原语。

从 overlap_geometry.py 拆分出来：最近点/最近距离这一类构建块
（点到三角形、线段到线段），供 overlap_geometry.py 的
triangle_triangle_min_distance 组合使用。三角形-三角形相交检测本身
（triangle_triangle_intersect）留在 overlap_geometry.py，因为它不依赖
这些原语，而是自己的一套 SAT 分离轴逻辑。
"""

import numpy as np


def closest_point_on_triangle(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """三角形 (a, b, c) 上距离 `p` 最近的点，每行一个候选。

    Ericson 5.1.5 节的七区域 Voronoi 测试，向量化：对每行确定
    三角形的两个顶点区域、三条边区域或内部面区域中哪个包含
    最近点，使用布尔掩码而非 Ericson 原始的提前返回分支
    （第一个匹配的区域"获胜"，掩码按与分支版本相同的优先级顺序应用）。

    Args:
        p: (N, 3) 查询点
        a, b, c: (N, 3) 三角形顶点，每行一个三角形

    Returns:
        (N, 3) 每行三角形上距离该行查询点最近的点
    """
    n = len(p)
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = np.einsum('ij,ij->i', ab, ap)
    d2 = np.einsum('ij,ij->i', ac, ap)

    bp = p - b
    d3 = np.einsum('ij,ij->i', ab, bp)
    d4 = np.einsum('ij,ij->i', ac, bp)

    cp = p - c
    d5 = np.einsum('ij,ij->i', ab, cp)
    d6 = np.einsum('ij,ij->i', ac, cp)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    out = np.zeros((n, 3), dtype=np.float64)
    assigned = np.zeros(n, dtype=bool)

    def _take(mask: np.ndarray, values: np.ndarray) -> None:
        nonlocal assigned
        use = mask & ~assigned
        out[use] = values[use]
        assigned |= use

    # 顶点 regions.
    _take((d1 <= 0) & (d2 <= 0), a)
    _take((d3 >= 0) & (d4 <= d3), b)
    _take((d6 >= 0) & (d5 <= d6), c)

    # 边 AB 区域.
    mask_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0) & ~assigned
    denom_ab = d1 - d3
    v_ab = np.divide(d1, denom_ab, out=np.zeros(n), where=np.abs(denom_ab) > 1e-300)
    _take(mask_ab, a + v_ab[:, None] * ab)

    # 边 AC 区域.
    mask_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0) & ~assigned
    denom_ac = d2 - d6
    w_ac = np.divide(d2, denom_ac, out=np.zeros(n), where=np.abs(denom_ac) > 1e-300)
    _take(mask_ac, a + w_ac[:, None] * ac)

    # 边 BC 区域.
    e_d4d3 = d4 - d3
    e_d5d6 = d5 - d6
    mask_bc = (va <= 0) & (e_d4d3 >= 0) & (e_d5d6 >= 0) & ~assigned
    denom_bc = e_d4d3 + e_d5d6
    w_bc = np.divide(e_d4d3, denom_bc, out=np.zeros(n), where=np.abs(denom_bc) > 1e-300)
    _take(mask_bc, b + w_bc[:, None] * (c - b))

    # Interior face region: whatever's left.
    denom_face = va + vb + vc
    v_face = np.divide(vb, denom_face, out=np.zeros(n), where=np.abs(denom_face) > 1e-300)
    w_face = np.divide(vc, denom_face, out=np.zeros(n), where=np.abs(denom_face) > 1e-300)
    _take(~assigned, a + v_face[:, None] * ab + w_face[:, None] * ac)

    return out


def point_to_triangle_distance(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """Distance from `p` to the closest point on triangle (a, b, c), (N,)."""
    closest = closest_point_on_triangle(p, a, b, c)
    return np.linalg.norm(p - closest, axis=1)


def closest_points_segment_segment(
    p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray, eps: float = 1e-300
) -> tuple:
    """线段 (p1,q1) 与 (p2,q2) 之间的最近点，每行一对。

    Ericson 5.1.9 节的封闭形式解，向量化：求解夹紧的参数位置
    s ∈ [0,1]（沿 d1 = q1-p1）和 t ∈ [0,1]（沿 d2 = q2-p2），
    使 |c1 - c2| 最小化，将零长度线段和近平行线段作为独立情况处理，
    而不是除以接近零的分母。

    Returns:
        (c1, c2): 各为 (N, 3)，每行第一/第二线段上的最近点
    """
    n = len(p1)
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2

    a = np.einsum('ij,ij->i', d1, d1)
    e = np.einsum('ij,ij->i', d2, d2)
    f = np.einsum('ij,ij->i', d2, r)
    c = np.einsum('ij,ij->i', d1, r)
    b = np.einsum('ij,ij->i', d1, d2)

    deg1 = a <= eps
    deg2 = e <= eps

    s = np.zeros(n)
    t = np.zeros(n)

    # Both segments degenerate to points: s=t=0 (already initialized).

    # Only segment 1 degenerate: closest point on segment 2 to p1.
    only2 = deg1 & ~deg2
    t[only2] = np.clip(np.divide(f, e, out=np.zeros(n), where=~deg2)[only2], 0.0, 1.0)

    # Only segment 2 degenerate: closest point on segment 1 to p2.
    only1 = ~deg1 & deg2
    s[only1] = np.clip(np.divide(-c, a, out=np.zeros(n), where=~deg1)[only1], 0.0, 1.0)

    # General case: neither degenerate.
    general = ~deg1 & ~deg2
    denom = a * e - b * b
    nonparallel = general & (np.abs(denom) > eps)

    s_gen = np.zeros(n)
    s_gen[nonparallel] = np.clip(
        (b[nonparallel] * f[nonparallel] - c[nonparallel] * e[nonparallel])
        / denom[nonparallel],
        0.0, 1.0,
    )
    # Parallel (or near-parallel) segments: any s works for the infinite-line
    # solution, pin s=0 and solve for t below - a fixed, deterministic choice
    # rather than an ill-conditioned division.
    parallel = general & ~nonparallel
    s_gen[parallel] = 0.0

    t_raw = np.divide(b * s_gen + f, e, out=np.zeros(n), where=~deg2)
    # Re-clamp t into [0,1] and, if that moved t, re-solve s for the new t
    # (Ericson's own two-step clamp - clamping t first can otherwise leave s
    # outside [0,1] too).
    t_clamped = np.clip(t_raw, 0.0, 1.0)
    below = general & (t_raw < 0.0)
    above = general & (t_raw > 1.0)
    s_gen[below] = np.clip(np.divide(-c, a, out=np.zeros(n), where=~deg1)[below], 0.0, 1.0)
    s_gen[above] = np.clip(np.divide(b - c, a, out=np.zeros(n), where=~deg1)[above], 0.0, 1.0)

    s[general] = s_gen[general]
    t[general] = t_clamped[general]

    c1 = p1 + s[:, None] * d1
    c2 = p2 + t[:, None] * d2
    return c1, c2


def segment_to_segment_distance(
    p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray
) -> np.ndarray:
    """Distance between segments (p1,q1) and (p2,q2), one value per row."""
    c1, c2 = closest_points_segment_segment(p1, q1, p2, q2)
    return np.linalg.norm(c1 - c2, axis=1)
