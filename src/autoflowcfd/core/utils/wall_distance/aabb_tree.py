"""AutoFlowCFD V2.0 - 三角形面片的 AABB 树与精确最近距离查询（numba）。

壁面距离 `d(p) = min_T dist(p, T)` 对全部壁面三角形 T 取最小。树按三角形形心在最长轴上
中位数二分（叶子不超过 `LEAF_SIZE` 个三角形），每个结点存其子树全部三角形的包围盒；查询是
分支定界的深度优先遍历：结点包围盒到点的距离不小于当前最优值就整枝剪掉，叶子上对每个
三角形求精确的点-三角形距离（Ericson, *Real-Time Collision Detection*, 2005, §5.1.5
"Closest point on triangle to point"）。结果是精确最近距离（不是近似），与 SU2 的 ADT
（alternating digital tree）壁距同一类做法。
"""

import numpy as np
from numba import njit, prange

#: 叶子结点最多容纳的三角形数
LEAF_SIZE = 8
#: 遍历栈深度上限。中位数二分的树深不超过 ceil(log2(n / LEAF_SIZE)) + 1，深度优先每层最多
#: 压两个结点，128 足够 2^60 个三角形。
_STACK_DEPTH = 128


@njit(cache=True)
def _segment_d2(px, py, pz, ax, ay, az, bx, by, bz):
    """点到线段 ab 的距离平方。"""
    ex, ey, ez = bx - ax, by - ay, bz - az
    ee = ex * ex + ey * ey + ez * ez
    t = 0.0
    if ee > 0.0:
        t = ((px - ax) * ex + (py - ay) * ey + (pz - az) * ez) / ee
        t = min(1.0, max(0.0, t))
    dx, dy, dz = px - (ax + t * ex), py - (ay + t * ey), pz - (az + t * ez)
    return dx * dx + dy * dy + dz * dz


@njit(cache=True)
def point_triangle_d2(p, a, b, c):
    """点 p 到三角形 abc（含边界）的距离平方（Ericson §5.1.5 的 Voronoi 区域判别）。
    退化三角形（面积为零）取三条边的最小值。"""
    px, py, pz = p[0], p[1], p[2]
    abx, aby, abz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    acx, acy, acz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    apx, apy, apz = px - a[0], py - a[1], pz - a[2]
    d1 = abx * apx + aby * apy + abz * apz
    d2 = acx * apx + acy * apy + acz * apz
    if d1 <= 0.0 and d2 <= 0.0:
        return apx * apx + apy * apy + apz * apz
    bpx, bpy, bpz = px - b[0], py - b[1], pz - b[2]
    d3 = abx * bpx + aby * bpy + abz * bpz
    d4 = acx * bpx + acy * bpy + acz * bpz
    if d3 >= 0.0 and d4 <= d3:
        return bpx * bpx + bpy * bpy + bpz * bpz
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return _segment_d2(px, py, pz, a[0], a[1], a[2], b[0], b[1], b[2])
    cpx, cpy, cpz = px - c[0], py - c[1], pz - c[2]
    d5 = abx * cpx + aby * cpy + abz * cpz
    d6 = acx * cpx + acy * cpy + acz * cpz
    if d6 >= 0.0 and d5 <= d6:
        return cpx * cpx + cpy * cpy + cpz * cpz
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return _segment_d2(px, py, pz, a[0], a[1], a[2], c[0], c[1], c[2])
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        return _segment_d2(px, py, pz, b[0], b[1], b[2], c[0], c[1], c[2])
    denom = va + vb + vc
    if denom <= 0.0:
        return min(_segment_d2(px, py, pz, a[0], a[1], a[2], b[0], b[1], b[2]),
                   min(_segment_d2(px, py, pz, b[0], b[1], b[2], c[0], c[1], c[2]),
                       _segment_d2(px, py, pz, c[0], c[1], c[2], a[0], a[1], a[2])))
    v = vb / denom
    w = vc / denom
    dx = apx - abx * v - acx * w
    dy = apy - aby * v - acy * w
    dz = apz - abz * v - acz * w
    return dx * dx + dy * dy + dz * dz


@njit(cache=True)
def build_aabb_tree(tri):
    """`tri (n, 3, 3)` -> `(order, node_lo, node_hi, left, right, start, count)`。

    `order` 是三角形的重排，结点 `k` 的三角形为 `order[start[k] : start[k] + count[k]]`；
    内部结点 `left[k] >= 0`，叶子 `left[k] = right[k] = -1`。
    """
    n = tri.shape[0]
    centroid = np.empty((n, 3))
    lo_t = np.empty((n, 3))
    hi_t = np.empty((n, 3))
    for i in range(n):
        for d in range(3):
            x0, x1, x2 = tri[i, 0, d], tri[i, 1, d], tri[i, 2, d]
            centroid[i, d] = (x0 + x1 + x2) / 3.0
            lo_t[i, d] = min(x0, min(x1, x2))
            hi_t[i, d] = max(x0, max(x1, x2))
    order = np.arange(n)
    max_nodes = max(1, 2 * n - 1)
    node_lo = np.empty((max_nodes, 3))
    node_hi = np.empty((max_nodes, 3))
    left = np.full(max_nodes, -1, dtype=np.int64)
    right = np.full(max_nodes, -1, dtype=np.int64)
    start = np.zeros(max_nodes, dtype=np.int64)
    count = np.zeros(max_nodes, dtype=np.int64)
    st_node = np.empty(max_nodes, dtype=np.int64)
    st_s = np.empty(max_nodes, dtype=np.int64)
    st_e = np.empty(max_nodes, dtype=np.int64)
    top = 0
    st_node[0], st_s[0], st_e[0] = 0, 0, n
    top = 1
    n_nodes = 1
    while top > 0:
        top -= 1
        node, s, e = st_node[top], st_s[top], st_e[top]
        start[node] = s
        count[node] = e - s
        clo = np.full(3, np.inf)
        chi = np.full(3, -np.inf)
        for d in range(3):
            node_lo[node, d] = np.inf
            node_hi[node, d] = -np.inf
        for k in range(s, e):
            t = order[k]
            for d in range(3):
                node_lo[node, d] = min(node_lo[node, d], lo_t[t, d])
                node_hi[node, d] = max(node_hi[node, d], hi_t[t, d])
                clo[d] = min(clo[d], centroid[t, d])
                chi[d] = max(chi[d], centroid[t, d])
        if e - s <= LEAF_SIZE:
            continue
        axis = 0
        for d in range(1, 3):
            if chi[d] - clo[d] > chi[axis] - clo[axis]:
                axis = d
        if chi[axis] - clo[axis] <= 0.0:
            continue  # 形心全部重合，无法再分：作为（较大的）叶子
        seg = order[s:e].copy()
        keys = np.empty(e - s)
        for k in range(e - s):
            keys[k] = centroid[seg[k], axis]
        idx = np.argsort(keys)
        for k in range(e - s):
            order[s + k] = seg[idx[k]]
        m = (s + e) // 2
        lc, rc = n_nodes, n_nodes + 1
        n_nodes += 2
        left[node], right[node] = lc, rc
        st_node[top], st_s[top], st_e[top] = lc, s, m
        top += 1
        st_node[top], st_s[top], st_e[top] = rc, m, e
        top += 1
    return (order, node_lo[:n_nodes].copy(), node_hi[:n_nodes].copy(), left[:n_nodes].copy(),
            right[:n_nodes].copy(), start[:n_nodes].copy(), count[:n_nodes].copy())


@njit(cache=True)
def _box_d2(p, lo, hi):
    s = 0.0
    for d in range(3):
        if p[d] < lo[d]:
            g = lo[d] - p[d]
            s += g * g
        elif p[d] > hi[d]:
            g = p[d] - hi[d]
            s += g * g
    return s


@njit(cache=True, parallel=True)
def nearest_triangle_distance(points, tri, order, node_lo, node_hi, left, right, start, count):
    """每个点到三角形集合的精确最近距离与最近三角形编号 `(dist (m,), tri_index (m,))`。"""
    m = points.shape[0]
    dist = np.empty(m)
    nearest = np.empty(m, dtype=np.int64)
    for i in prange(m):
        p = points[i]
        stack = np.empty(_STACK_DEPTH, dtype=np.int64)
        stack[0] = 0
        top = 1
        best = np.inf
        best_t = -1
        while top > 0:
            top -= 1
            nd = stack[top]
            if _box_d2(p, node_lo[nd], node_hi[nd]) >= best:
                continue
            if left[nd] < 0:
                for k in range(start[nd], start[nd] + count[nd]):
                    t = order[k]
                    d2 = point_triangle_d2(p, tri[t, 0], tri[t, 1], tri[t, 2])
                    if d2 < best:
                        best = d2
                        best_t = t
                continue
            l, r = left[nd], right[nd]
            dl = _box_d2(p, node_lo[l], node_hi[l])
            dr = _box_d2(p, node_lo[r], node_hi[r])
            # 先压远的、后压近的：近的先出栈，尽早收紧最优值
            if dl <= dr:
                if dr < best:
                    stack[top] = r
                    top += 1
                if dl < best:
                    stack[top] = l
                    top += 1
            else:
                if dl < best:
                    stack[top] = l
                    top += 1
                if dr < best:
                    stack[top] = r
                    top += 1
        dist[i] = np.sqrt(best)
        nearest[i] = best_t
    return dist, nearest
