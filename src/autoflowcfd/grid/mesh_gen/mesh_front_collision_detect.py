"""BL 前沿自相交检测：广相位候选对 + 精确三角形相交/跨状态检测。

从 mesh_front_collision.py 拆分出来，是该模块"事后检测"这一半的底层
实现：`find_self_colliding_faces`（同一快照自碰撞）和
`find_cross_state_colliding_faces`（新旧两个快照之间的跨状态碰撞，捕捉
单步推进过快导致的"穿透"）。`clamp_budget_for_convergence`（事前预算裁剪）
和 `freeze_self_colliding_nodes`（事后冻结）仍留在 mesh_front_collision.py。
"""

import numpy as np

from ..validation.overlap_geometry import triangle_triangle_intersect


def _face_geometry(nodes: np.ndarray, faces: np.ndarray):
    tri = nodes[faces]  # (n_faces, 3, 3)
    centroids = tri.mean(axis=1)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    cross_norm = np.linalg.norm(cross, axis=1)
    face_size = np.sqrt(np.maximum(0.5 * cross_norm, 1e-300))
    normal = cross / np.maximum(cross_norm, 1e-300)[:, np.newaxis]
    return tri, centroids, face_size, normal


def _iter_candidate_pairs(
    faces: np.ndarray,
    centroids: np.ndarray,
    face_size: np.ndarray,
    search_multiplier: float,
    chunk_size: int,
):
    """Yield (row_idx, col_idx) int64 arrays, one pair of chunk-local
    candidate-index arrays per chunk: non-self, non-node-sharing face
    pairs whose centroids are within `search_multiplier * own sqrt(area)`
    of each other. Shared broad phase for both find_self_colliding_faces
    and clamp_budget_for_convergence - see either's docstring for why the
    radius is per-face (local mesh scale), not a single domain constant,
    and why chunking bounds memory instead of materializing every
    candidate pair at once (same rationale as mesh_overlap_check.py's
    identical pattern, on a real multi-million-face mesh).
    """
    from scipy.spatial import cKDTree

    n_faces = len(faces)
    tree = cKDTree(centroids)
    search_radius = search_multiplier * face_size

    for start in range(0, n_faces, chunk_size):
        end = min(start + chunk_size, n_faces)
        idx_chunk = np.arange(start, end)
        neighbor_lists = tree.query_ball_point(
            centroids[idx_chunk], r=search_radius[idx_chunk], workers=-1
        )
        counts = np.fromiter(
            (len(lst) for lst in neighbor_lists), dtype=np.int64, count=len(neighbor_lists)
        )
        if counts.sum() == 0:
            continue

        row_idx = np.repeat(idx_chunk, counts)
        col_idx = np.concatenate(
            [np.asarray(lst, dtype=np.int64) for lst in neighbor_lists if len(lst) > 0]
        )
        keep = row_idx != col_idx
        row_idx, col_idx = row_idx[keep], col_idx[keep]
        if len(row_idx) == 0:
            continue

        # Faces sharing a node are ordinary adjacent topology, not a
        # defect (same rule as mesh_overlap_check.py's node-sharing filter).
        fi, fj = faces[row_idx], faces[col_idx]
        shares_node = np.zeros(len(row_idx), dtype=bool)
        for a in range(3):
            for b in range(3):
                shares_node |= fi[:, a] == fj[:, b]
        keep = ~shares_node
        if not np.any(keep):
            continue
        row_idx, col_idx = row_idx[keep], col_idx[keep]

        yield row_idx, col_idx


def find_self_colliding_faces(
    nodes: np.ndarray,
    faces: np.ndarray,
    search_multiplier: float = 2.0,
    chunk_size: int = 2000,
) -> np.ndarray:
    """Indices into `faces` of every face involved in a genuine (exact,
    not proximity-based - there is no "closeness" threshold to tune here)
    self-intersection with another, non-adjacent face of the SAME
    triangle soup.

    Narrow phase only: exact triangle_triangle_intersect on each
    broad-phase candidate from _iter_candidate_pairs. No "close" case -
    only an actual intersection is reported; see clamp_budget_for_
    convergence for the complementary, proximity-based, BEFORE-the-step
    mechanism.

    Args:
        nodes: (n_nodes, 3) current node positions
        faces: (n_faces, 3) triangle connectivity (int)
        search_multiplier: broad-phase KD-tree query radius as a multiple
            of each face's own sqrt(area)
        chunk_size: faces processed per KD-tree batch

    Returns:
        int64 array of face indices with at least one genuine
        intersection (empty if none, never None)
    """
    n_faces = len(faces)
    if n_faces == 0:
        return np.array([], dtype=np.int64)

    tri, centroids, face_size, _normal = _face_geometry(nodes, faces)
    colliding = np.zeros(n_faces, dtype=bool)

    for row_idx, col_idx in _iter_candidate_pairs(
        faces, centroids, face_size, search_multiplier, chunk_size
    ):
        a_nodes, b_nodes = tri[row_idx], tri[col_idx]
        intersects = triangle_triangle_intersect(
            a_nodes[:, 0], a_nodes[:, 1], a_nodes[:, 2],
            b_nodes[:, 0], b_nodes[:, 1], b_nodes[:, 2],
        )
        if np.any(intersects):
            colliding[row_idx[intersects]] = True
            colliding[col_idx[intersects]] = True

    return np.flatnonzero(colliding)


def find_cross_state_colliding_faces(
    new_nodes: np.ndarray,
    current_nodes: np.ndarray,
    faces: np.ndarray,
    search_multiplier: float = 2.0,
    chunk_size: int = 2000,
) -> np.ndarray:
    """Indices into `faces` of every face whose NEW position genuinely
    intersects some other, non-adjacent face's CURRENT (this layer's
    starting, pre-step) position.

    find_self_colliding_faces alone - comparing `new_nodes` against
    itself - misses a real failure mode confirmed directly on cube_demo:
    a fast-advancing triangle A and a slow (or differently-curving)
    neighbour triangle B can each individually look fine at both
    snapshots (A-new vs B-new doesn't intersect, by definition it passed
    this same check when B was itself "new" last layer) while A's own
    large step this layer sweeps it through the space B was still
    occupying at the START of that same step - neither the same-layer
    check (only ever compares same-snapshot state) nor clamp_budget_for_
    convergence (a first-order/instantaneous linear approximation
    evaluated once at the step's start - see CONVERGING_CLOSING_RATE_
    THRESHOLD's own comment) is guaranteed to catch this when a single
    step is large relative to the local feature size (Stage 2 transition
    layers can grow up to 4x/layer - see extrude_layers' target_handoff_
    size solve). This is the cross-triangle generalisation of the same
    tunneling concern CONVERGENCE_SAFETY_FRACTION exists for on a single
    pair; verified directly to find real cases on cube_demo that the
    other two mechanisms did not (up to ~20% of surface triangles
    implicated across most of the BL stack's depth - a "whole nearby
    column swept through a slower column" pattern, not isolated slivers).

    A face's own new-vs-current pairing (comparing a triangle against
    ITSELF across the step) is excluded the same way self-pairs always
    are - a triangle's own sweep containing its own prior position is
    exactly what a prism is, not a defect.

    Args:
        new_nodes: (n_nodes, 3) this layer's tentative node positions
        current_nodes: (n_nodes, 3) previous (already-accepted) positions
        faces: (n_faces, 3) triangle connectivity (int)
        search_multiplier: broad-phase KD-tree query radius as a multiple
            of the QUERYING (new-state) face's own sqrt(area)
        chunk_size: faces processed per KD-tree batch

    Returns:
        int64 array of face indices (empty if none, never None)
    """
    n_faces = len(faces)
    if n_faces == 0:
        return np.array([], dtype=np.int64)

    from scipy.spatial import cKDTree

    tri_new, centroids_new, face_size_new, _n1 = _face_geometry(new_nodes, faces)
    tri_cur, centroids_cur, _face_size_cur, _n2 = _face_geometry(current_nodes, faces)

    tree = cKDTree(centroids_cur)
    search_radius = search_multiplier * face_size_new

    colliding = np.zeros(n_faces, dtype=bool)

    for start in range(0, n_faces, chunk_size):
        end = min(start + chunk_size, n_faces)
        idx_chunk = np.arange(start, end)
        neighbor_lists = tree.query_ball_point(
            centroids_new[idx_chunk], r=search_radius[idx_chunk], workers=-1
        )
        counts = np.fromiter(
            (len(lst) for lst in neighbor_lists), dtype=np.int64, count=len(neighbor_lists)
        )
        if counts.sum() == 0:
            continue

        row_idx = np.repeat(idx_chunk, counts)  # indexes new-state (query side)
        col_idx = np.concatenate(
            [np.asarray(lst, dtype=np.int64) for lst in neighbor_lists if len(lst) > 0]
        )  # indexes current-state (tree side)
        keep = row_idx != col_idx  # a face's own sweep is not a defect
        row_idx, col_idx = row_idx[keep], col_idx[keep]
        if len(row_idx) == 0:
            continue

        fi, fj = faces[row_idx], faces[col_idx]
        shares_node = np.zeros(len(row_idx), dtype=bool)
        for a in range(3):
            for b in range(3):
                shares_node |= fi[:, a] == fj[:, b]
        keep = ~shares_node
        if not np.any(keep):
            continue
        row_idx, col_idx = row_idx[keep], col_idx[keep]

        a_nodes = tri_new[row_idx]
        b_nodes = tri_cur[col_idx]
        intersects = triangle_triangle_intersect(
            a_nodes[:, 0], a_nodes[:, 1], a_nodes[:, 2],
            b_nodes[:, 0], b_nodes[:, 1], b_nodes[:, 2],
        )
        if np.any(intersects):
            colliding[row_idx[intersects]] = True
            colliding[col_idx[intersects]] = True

    return np.flatnonzero(colliding)
