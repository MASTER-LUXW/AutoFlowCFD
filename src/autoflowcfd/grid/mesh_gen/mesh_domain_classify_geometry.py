"""mesh_domain_classify.classify_boundary_groups 用到的底层几何原语。

从 mesh_domain_classify.py 拆分出来：面/边连通分量分析、射线-三角形相交
计数（用于内外判定）、封闭壳体内部点查找（tetgen hole seed 用）、带符号
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
    """Count Moller-Trumbore ray/triangle intersections ahead of `origin`
    (vectorized over all triangles), used for a ray-casting parity
    inside/outside test."""
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
    """Min distance from p to any triangle edge - a cheap, slightly
    conservative proxy for distance-to-surface. Needed because a candidate
    can sit exactly on an edge (e.g. a symmetric shell's vertex-average
    centroid landing precisely on a concave feature) without being close to
    any single *vertex*, which a vertex-only distance check would miss."""
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
    """Find a point strictly inside a closed (watertight) triangle shell,
    for use as a tetgen hole seed (mesh_tetgen_core.fill_core_volume) so an
    isolated embedded solid's own interior - and by extension its BL
    block's enclosed cavity - is excluded from the core fill instead of
    being filled with spurious tetrahedra that overlap the BL prisms
    already occupying that space.

    Tries the shell's vertex-average centroid first (correct for the
    common case: a reasonably convex/star-shaped solid), then falls back to
    points just inside each of several randomly sampled faces (offset
    inward along that face's own normal), for non-convex shapes where the
    centroid can fall outside the solid entirely.

    Each candidate is verified two ways before being accepted: it must not
    sit too close to the shell surface itself (a degenerate case a vertex
    centroid can hit exactly, e.g. landing precisely on a concave edge -
    ordinary vertex-distance checks miss this since the nearest *vertex*
    can still be far away), and a ray-casting parity (odd intersection
    count = inside) test must agree across several independent random ray
    directions, not just one (a single ray can graze an edge/vertex and
    give a wrong answer by chance).

    Args:
        nodes: (n_nodes, 3) coordinates (shared array; only rows referenced
            by `faces` are used)
        faces: (n_faces, 3) closed, watertight triangle connectivity
        n_attempts: number of per-face fallback candidates to try
        n_directions: number of independent ray directions each candidate
            must agree on before being accepted
        seed: RNG seed (deterministic candidate/direction sampling)

    Returns:
        A point inside the shell, or None if no candidate could be
        verified (caller should skip hole-marking for this shell rather
        than risk an incorrect point, which would corrupt the whole fill)
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
    """Enclosed volume of a (near-)closed surface, using raw (unnormalized)
    face winding. Sign follows the same convention as
    mesh_prism_to_tet.orient_tetrahedra (positive = outward-consistent winding).
    Meaningful even with a small opening (missing caps contribute ~0 net
    volume relative to the whole shell).
    """
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
    """Find the single bounding-box face this node set predominantly touches.

    Returns (axis, inward_direction) if one of the 6 bbox faces accounts for
    >= _BBOX_TOUCH_MAJORITY of the given nodes, else (None, None).
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
