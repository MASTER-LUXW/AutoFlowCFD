"""BL 挤出的"法向前方最近面片"间隙（`mesh_tetgen_seam.compute_local_thickness_limit` 用）。

对每个可挤出节点 p（外法向 n），间隙 = 下面两者的较小值，只看不含 p 的面片：
- 最近点落在法向前方锥内（与 n 的夹角不超过 `angle_threshold_deg`）的面片里，p 到面片的最短距离；
- 沿 n 的射线与面片的第一个交点距离（最近点偏在侧面、但法向正对着它的大斜面）。

2026-10-20 以前按"锥内最近**节点**"测量：对面是大三角形、顶点都在搜索半径或锥外时（稀疏远场面、细长间隙
对面的粗面片）完全检测不到，挤出层可以穿透对面。

候选面片：锥与半径 R 的球之交包含在一个沿法向前移的球里（见 facing_gap）；面片按外接半径分档（每档半径上限是
下一档的 2 倍），每档对质心建 KD 树、以"前移球半径 + 档上限"查询——质心在这个范围外的面片与前移球不相交；外接
半径的高估不超过 2 倍，候选数随之有界，大三角形也不会漏掉。
再用"外接球与前方锥相交"剪掉同一片网格上的面内邻居，最后精确计算点到三角形的最近点（Ericson，
《Real-Time Collision Detection》5.1.5）。
"""

from typing import Tuple

import numpy as np

_CHUNK_PAIRS = 2_000_000          # 每批精确计算的 (节点, 面片) 对数上限（内存有界）


def _rowdot(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.einsum('ij,ij->i', x, y)


def closest_points_on_triangles(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """(n, 3) 点 p 到 (n, 3) 三角形 (a, b, c) 上的最近点（逐对，按 Voronoi 区域分情况）。"""
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    d1, d2 = _rowdot(ab, ap), _rowdot(ac, ap)
    d3, d4 = _rowdot(ab, bp), _rowdot(ac, bp)
    d5, d6 = _rowdot(ab, cp), _rowdot(ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    out = np.empty_like(p)
    todo = np.ones(len(p), dtype=bool)

    def assign(mask, value):
        m = mask & todo
        out[m] = value[m]
        todo[m] = False

    with np.errstate(divide='ignore', invalid='ignore'):
        assign((d1 <= 0) & (d2 <= 0), a)                                              # 顶点 a
        assign((d3 >= 0) & (d4 <= d3), b)                                             # 顶点 b
        assign((d6 >= 0) & (d5 <= d6), c)                                             # 顶点 c
        assign((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + (d1 / (d1 - d3))[:, None] * ab)  # 边 ab
        assign((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + (d2 / (d2 - d6))[:, None] * ac)  # 边 ac
        e43, e56 = d4 - d3, d5 - d6
        assign((va <= 0) & (e43 >= 0) & (e56 >= 0), b + (e43 / (e43 + e56))[:, None] * (c - b))  # 边 bc
        inv = 1.0 / (va + vb + vc)
        assign(todo, a + (vb * inv)[:, None] * ab + (vc * inv)[:, None] * ac)       # 面内
    return out


def _ray_hit_distance(p, n, a, b, c, tol=1e-12):
    """逐对：射线 p + t n（t > 0）与三角形的交点距离，不相交为 inf（Moller-Trumbore，重心坐标闭区间）。"""
    e1, e2 = b - a, c - a
    h = np.cross(n, e2)
    det = np.einsum('ij,ij->i', e1, h)
    scale = np.linalg.norm(np.cross(e1, e2), axis=1)
    ok = np.abs(det) > tol * scale
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = p - a
    u = inv * np.einsum('ij,ij->i', s, h)
    q = np.cross(s, e1)
    v = inv * np.einsum('ij,ij->i', n, q)
    t = inv * np.einsum('ij,ij->i', e2, q)
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    return np.where(hit, t, np.inf)


def facing_gap(nodes: np.ndarray, faces: np.ndarray, query_idx: np.ndarray, normals: np.ndarray,
               radius: float, angle_threshold_deg: float) -> Tuple[np.ndarray, int]:
    """每个查询节点到法向前方最近面片的距离（`radius` 内，找不到为 inf）。

    Args:
        nodes: (n_nodes, 3)；faces: (m, 3) 全部表面三角形；query_idx: 查询节点；normals: (len(query_idx), 3) 单位法向

    Returns:
        (gap, n_pairs)：gap 形状 (len(query_idx),)；n_pairs 为精确计算过的 (节点, 面片) 对数（诊断用）
    """
    from scipy.spatial import cKDTree

    a, b, c = nodes[faces[:, 0]], nodes[faces[:, 1]], nodes[faces[:, 2]]
    centroid = (a + b + c) / 3.0
    circ = np.max(np.stack([np.linalg.norm(x - centroid, axis=1) for x in (a, b, c)]), axis=0)
    cos_th = np.cos(np.radians(angle_threshold_deg))
    sin_th = np.sin(np.radians(angle_threshold_deg))
    p_all = nodes[query_idx]
    gap = np.full(len(query_idx), np.inf)
    n_pairs = 0

    # 锥（半角 θ < 90°）与半径 R 的球之交，恰好包含在球心 p + n R/(2cosθ)、半径 R/(2cosθ) 的球里（锥顶、锥轴
    # 末端与锥口边缘都在这个球面上）；以它查询，同一片网格上的面内邻居天然落在外面。射线段 [p, p + R n] 也在其中。
    rho = radius / (2.0 * cos_th)
    centers = p_all + normals * rho
    # 外接半径分档：档 k 含 circ ∈ (r0 2^(k-1), r0 2^k]
    r0 = max(float(circ.min()), 1e-300)
    level = np.maximum(np.ceil(np.log2(np.maximum(circ / r0, 1.0))), 0).astype(np.int64)
    for k in np.unique(level):
        members = np.flatnonzero(level == k)
        r_cap = r0 * 2.0 ** k
        tree = cKDTree(centroid[members])
        lists = tree.query_ball_point(centers, r=rho + r_cap, workers=-1)
        counts = np.fromiter((len(x) for x in lists), dtype=np.int64, count=len(lists))
        if counts.sum() == 0:
            continue
        rows = np.repeat(np.arange(len(query_idx)), counts)
        tri = members[np.concatenate([np.asarray(x, dtype=np.int64) for x in lists if len(x)])]
        # 剪枝：不含查询节点自身；外接球与前方锥相交（球心方向与 n 的夹角减去球的角半径不超过锥半角）
        own = (faces[tri] == query_idx[rows][:, None]).any(axis=1)
        d = centroid[tri] - p_all[rows]
        dist_c = np.linalg.norm(d, axis=1)
        along = np.einsum('ij,ij->i', d, normals[rows])
        lateral = np.sqrt(np.maximum(dist_c ** 2 - along ** 2, 0.0))
        # 球与（凸）锥相交：在子午面内，球心到母线的投影在锥顶前方时看球心到母线的有向距离，在锥顶后方时看
        # 球心到锥顶的距离；不超过外接半径即可能相交（射线沿锥轴，同样被覆盖）
        proj = along * cos_th + lateral * sin_th
        cone = np.where(proj >= 0, lateral * cos_th - along * sin_th <= circ[tri], dist_c <= circ[tri])
        near = dist_c - circ[tri] <= radius
        keep = ~own & cone & near
        rows, tri = rows[keep], tri[keep]
        for s in range(0, len(rows), _CHUNK_PAIRS):
            r_s, t_s = rows[s:s + _CHUNK_PAIRS], tri[s:s + _CHUNK_PAIRS]
            p, n = p_all[r_s], normals[r_s]
            q = closest_points_on_triangles(p, a[t_s], b[t_s], c[t_s])
            dq = q - p
            dist = np.linalg.norm(dq, axis=1)
            ahead = (dist > 0) & (np.einsum('ij,ij->i', dq, n) >= cos_th * dist) & (dist <= radius)
            cand = np.where(ahead, dist, np.inf)
            hit = _ray_hit_distance(p, n, a[t_s], b[t_s], c[t_s])
            cand = np.minimum(cand, np.where(hit <= radius, hit, np.inf))
            np.minimum.at(gap, r_s, cand)
            n_pairs += len(r_s)
    return gap, n_pairs
