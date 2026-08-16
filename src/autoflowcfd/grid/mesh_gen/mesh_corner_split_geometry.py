"""mesh_corner_split.py 用到的纯几何辅助函数 (从该文件拆分)。

从 mesh_corner_split.py 拆出来（该文件原有 446 行，超过 400 行硬性
拆分阈值）：这三个函数都是无副作用的纯几何/拓扑计算，不依赖
split_sharp_corners 内部任何状态，只被它调用，独立成文件是最干净的
拆分点。纯代码搬移，不改变任何行为。

注意：`_face_normals` 与 mesh_tetgen_seam.py 里的同名私有函数、
mesh_utils.py::compute_face_normals 是三份独立实现（拆分前就已如此，
不是本次拆分引入的重复），本次只搬移 mesh_corner_split.py 自己这一份，
不合并/不改动其它两处。
"""

import numpy as np


def _face_normals(nodes: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = nodes[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    cross_norm = np.linalg.norm(cross, axis=1)
    return cross / np.maximum(cross_norm, 1e-300)[:, np.newaxis]


def _implied_edge_radius(edge_length: float, dihedral_angle: float) -> float:
    """Local radius of curvature implied by one triangulated edge, treating
    it as one chord of a circular arc of unknown radius swept through
    `dihedral_angle` (the standard chord-to-radius relation: chord c,
    subtended angle theta, radius r = c / (2 sin(theta/2))).

    The point of this estimate: for a genuinely SHARP CAD crease (two
    flat faces meeting at a fixed G0-discontinuous angle), that angle is a
    property of the two flat faces themselves - it does NOT shrink as the
    mesh is refined, so the radius this formula implies shrinks toward 0
    in direct proportion to edge_length (r = c / const). For an ordinary
    CURVED surface of true physical radius R that is merely under-
    tessellated (few facets across the curve), the SAME formula recovers
    approximately R itself, regardless of edge_length, PROVIDED
    edge_length is small relative to R (a large chord on a fine curve
    starts to noticeably underestimate R - not a concern here since this
    is only ever evaluated on edges that already registered as locally
    "sharp", i.e. small chords by construction). Distinguishing the two
    is exactly "is the implied radius comparable to the probing edge
    length itself (shrinks toward 0 with it - a real crease) or much
    larger than it (roughly resolution-independent - a real, if coarse,
    curve)" - see split_sharp_corners' own min_feature_radius parameter
    for how this is actually used.
    """
    half_angle = dihedral_angle / 2.0
    sin_half = np.sin(half_angle)
    if sin_half < 1e-9:
        return np.inf
    return edge_length / (2.0 * sin_half)


def _unique_edges(faces: np.ndarray):
    """Yield (v0, v1, face_idx_array) per distinct undirected edge - 1
    face for a patch-boundary edge, 2 for a normal interior edge, more
    only for non-manifold input.
    """
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edge_face_idx = np.tile(np.arange(len(faces)), 3)
    sorted_edges = np.sort(edges, axis=1)

    order = np.lexsort((sorted_edges[:, 1], sorted_edges[:, 0]))
    se = sorted_edges[order]
    efi = edge_face_idx[order]

    is_new = np.ones(len(se), dtype=bool)
    is_new[1:] = np.any(se[1:] != se[:-1], axis=1)
    boundaries = np.flatnonzero(is_new)
    boundaries = np.append(boundaries, len(se))

    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        yield se[lo, 0], se[lo, 1], efi[lo:hi]
