"""mesh_domain_classify.classify_boundary_groups 用到的底层几何原语。

从 mesh_domain_classify.py 拆分出来：面/边连通分量分析、射线-三角形相交
计数（用于内外判定）、封闭壳体内部点查找（tetgen hole 种子 用）、带符号
体积、以及包围盒接触面判定。这些都是纯几何计算，和"怎么分类边界组"这个
上层策略无关。
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# _bbox_touch_fraction 判定"这批节点是否主要贴在包围盒某一面上"的多数阈值。
_BBOX_TOUCH_MAJORITY = 0.9


def _face_edges(faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (edge_id per face-edge occurrence, unique edge occurrence counts,
    face index per occurrence) for the given face array."""
    n_faces = len(faces)
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    face_of_edge = np.tile(np.arange(n_faces), 3)
    edges_sorted = np.sort(edges, axis=1)
    edge_dtype = np.dtype((np.void, edges_sorted.dtype.itemsize * 2))
    edge_voids = np.ascontiguousarray(edges_sorted).view(edge_dtype).reshape(-1)
    _, inverse, counts = np.unique(edge_voids, return_inverse=True, return_counts=True)
    return inverse, counts, face_of_edge


def _connected_components(faces: np.ndarray, inverse: np.ndarray, face_of_edge: np.ndarray) -> np.ndarray:
    """Label each face row with a connected-component id (shared-edge adjacency)."""
    n_faces = len(faces)
    order = np.argsort(inverse, kind='stable')
    sorted_edge_id = inverse[order]
    sorted_face_idx = face_of_edge[order]
    splits = np.flatnonzero(np.diff(sorted_edge_id)) + 1
    groups = np.split(sorted_face_idx, splits)

    rows: List[int] = []
    cols: List[int] = []
    for group in groups:
        if len(group) >= 2:
            anchor = group[0]
            rows.extend([anchor] * (len(group) - 1))
            cols.extend(group[1:])

    adjacency = coo_matrix(
        (np.ones(len(rows), dtype=bool), (rows, cols)),
        shape=(n_faces, n_faces)
    )
    _, labels = connected_components(adjacency, directed=False)
    return labels


def _ray_triangle_intersect_count(
    origin: np.ndarray, direction: np.ndarray,
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
) -> int:
    """计数 Moller-Trumbore 射线/三角形相交次数（`origin` 前方）。
    对所有三角形向量化，用于射线投射奇偶性内外测试。"""
    eps = 1e-9
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(direction, edge2)
    a = np.einsum('ij,ij->i', edge1, h)
    valid = np.abs(a) > eps
    f = np.zeros_like(a)
    f[valid] = 1.0 / a[valid]
    s = origin - v0
    u = f * np.einsum('ij,ij->i', s, h)
    valid &= (u >= -eps) & (u <= 1 + eps)
    q = np.cross(s, edge1)
    v = f * np.einsum('j,ij->i', direction, q)
    valid &= (v >= -eps) & (u + v <= 1 + eps)
    t = f * np.einsum('ij,ij->i', edge2, q)
    valid &= t > eps
    return int(np.sum(valid))


def _min_dist_to_edges(p: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
    """从 p 到三角形任意边的最小距离——一个廉价且略微保守的
    表面距离代理。需要它是因为候选点可能恰好落在边上（例如对称
    壳体的顶点均值质心正好落在凹特征上），而不靠近任何单个*顶点*，
    此时仅检查顶点距离会遗漏。"""
    best = np.inf
    for a, b in ((v0, v1), (v1, v2), (v2, v0)):
        ab = b - a
        denom = np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-30)
        t = np.clip(np.einsum('ij,ij->i', p - a, ab) / denom, 0.0, 1.0)
        closest = a + t[:, None] * ab
        d = np.linalg.norm(p - closest, axis=1)
        best = min(best, float(d.min()))
    return best


def find_point_inside_closed_shell(
    nodes: np.ndarray,
    faces: np.ndarray,
    n_attempts: int = 20,
    n_directions: int = 5,
    seed: int = 0,
) -> Optional[np.ndarray]:
    """在封闭（水密）三角形壳体内部找一个点，用作 tetgen hole 种子
    （mesh_tetgen_core.fill_core_volume），使孤立嵌入实体的自身内部
    ——以及通过扩展其 BL 块的封闭空腔——被排除在 core 填充之外，
    而不是被重叠 BL 棱柱已经占据的空间中填充虚假四面体。

    先尝试壳体的顶点均值质心（对合理凸/星形实体有效），然后回退到
    多个随机采样面的内侧点（沿该面自身法向向内偏移），用于质心可能
    完全落在壳体外部的非凸形状。

    每个候选点在接受前需通过两种验证：不能太靠近壳体表面本身
    （顶点质心可能恰好命中的退化情况，例如正好落在凹边上——
    普通的顶点距离检查会遗漏，因为最近的*顶点*仍可能很远），
    且射线投射奇偶性（奇数个交点 = 内部）测试必须在多个独立
    随机射线方向上一致，而非仅一个（单条射线可能擦过边/顶点
    而偶然给出错误答案）。

    Args:
        nodes: (n_nodes, 3) 坐标（共享数组；仅 `faces` 引用的行被使用）
        faces: (n_faces, 3) 封闭、水密的三角形连接关系
        n_attempts: 每面回退候选点的尝试次数
        n_directions: 每个候选点在接受前必须一致的独立射线方向数
        seed: 随机数种子（确定性候选/方向采样）

    Returns:
        壳体内部的一个点，若无候选点通过验证则返回 None
        （调用方应跳过该壳体的 hole 标记，而非冒险使用
        不正确的点，否则会破坏整个填充）
    """
    rng = np.random.default_rng(seed)
    node_idx = np.unique(faces)
    pts = nodes[node_idx]
    v0, v1, v2 = nodes[faces[:, 0]], nodes[faces[:, 1]], nodes[faces[:, 2]]

    face_centroids = (v0 + v1 + v2) / 3.0
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    normals = normals / norms
    edge_len = float(np.median(np.linalg.norm(v1 - v0, axis=1)))
    if edge_len <= 0.0:
        return None

    candidates = [pts.mean(axis=0)]
    n_face_try = min(n_attempts, len(faces))
    for i in rng.choice(len(faces), size=n_face_try, replace=False):
        candidates.append(face_centroids[i] - normals[i] * edge_len * 0.5)

    directions = rng.normal(size=(n_directions, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    min_clearance = edge_len * 0.05
    for cand in candidates:
        if _min_dist_to_edges(cand, v0, v1, v2) < min_clearance:
            continue
        hit_parities = [
            _ray_triangle_intersect_count(cand, d, v0, v1, v2) % 2
            for d in directions
        ]
        if all(p == 1 for p in hit_parities):
            return cand
    return None


def _signed_volume(nodes: np.ndarray, faces: np.ndarray) -> float:
    """（近）封闭曲面的包围体积，使用原始（未归一化）的面绕向。
    符号约定与 mesh_prism_to_tet.orient_tetrahedra 相同
    （正值 = 向外一致的绕向）。即使有小的开口也有意义
    （缺失的帽盖相对于整个壳体的净体积贡献约为零）。"""
    v0 = nodes[faces[:, 0]]
    v1 = nodes[faces[:, 1]]
    v2 = nodes[faces[:, 2]]
    return float(np.sum(np.einsum('ij,ij->i', v0, np.cross(v1, v2))) / 6.0)


def _bbox_touch_fraction(
    nodes: np.ndarray,
    node_idx: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    tol: float,
) -> Tuple[Optional[int], Optional[np.ndarray]]:
    """查找这批节点主要贴在包围盒的哪一个面上。

    如果 6 个包围盒面中的某一个占到了给定节点的
    >= _BBOX_TOUCH_MAJORITY，则返回 (axis, inward_direction)，
    否则返回 (None, None)。
    """
    coords = nodes[node_idx]
    n = len(node_idx)
    candidates = []
    for axis in range(3):
        near_min = np.abs(coords[:, axis] - bbox_min[axis]) <= tol
        frac_min = np.count_nonzero(near_min) / n
        if frac_min >= _BBOX_TOUCH_MAJORITY:
            direction = np.zeros(3)
            direction[axis] = 1.0  # inward = away from the min face
            candidates.append((frac_min, direction))

        near_max = np.abs(coords[:, axis] - bbox_max[axis]) <= tol
        frac_max = np.count_nonzero(near_max) / n
        if frac_max >= _BBOX_TOUCH_MAJORITY:
            direction = np.zeros(3)
            direction[axis] = -1.0  # inward = away from the max face
            candidates.append((frac_max, direction))

    if len(candidates) == 1:
        return candidates[0][1]
    return None
