"""Pure per-cell geometric quality metric computations.

Vectorized (no Python-level per-cell loop) functions for the raw arrays
behind MeshQualityValidator's (quality_validator.py) aggregate statistics -
split out of that module so the check/orchestration logic there isn't
interleaved with these self-contained geometric formulas. Every function
here is a pure function of (nodes, cells): no mesh-generation or repair
concepts, no state.
"""

import numpy as np


def compute_tetrahedron_volumes(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Signed volume of every tetrahedron: det(p1-p0, p2-p0, p3-p0) / 6.

    Note: kept signed (not absolute) so a negative-volume/inverted-cell
    check upstream is meaningful; magnitude statistics elsewhere use the
    positive subset regardless.
    """
    p0 = nodes[cells[:, 0]]
    p1 = nodes[cells[:, 1]]
    p2 = nodes[cells[:, 2]]
    p3 = nodes[cells[:, 3]]

    v1 = p1 - p0
    v2 = p2 - p0
    v3 = p3 - p0

    return np.einsum('ij,ij->i', v1, np.cross(v2, v3)) / 6.0


def compute_triangle_areas(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Area of every triangle: 0.5 * |cross(p1-p0, p2-p0)|."""
    p0 = nodes[cells[:, 0]]
    p1 = nodes[cells[:, 1]]
    p2 = nodes[cells[:, 2]]

    cross = np.cross(p1 - p0, p2 - p0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def triangle_edge_lengths(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Edge lengths for every triangle, shape=(n_cells, 3)."""
    p0, p1, p2 = nodes[cells[:, 0]], nodes[cells[:, 1]], nodes[cells[:, 2]]
    e1 = np.linalg.norm(p1 - p0, axis=1)
    e2 = np.linalg.norm(p2 - p1, axis=1)
    e3 = np.linalg.norm(p0 - p2, axis=1)
    return np.stack([e1, e2, e3], axis=1)


def tetrahedron_edge_lengths(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """All 6 edge lengths for every tetrahedron, shape=(n_cells, 6)."""
    pts = nodes[cells]  # (n_cells, 4, 3)
    edges = []
    for i in range(4):
        for j in range(i + 1, 4):
            edges.append(np.linalg.norm(pts[:, i] - pts[:, j], axis=1))
    return np.stack(edges, axis=1)


def compute_triangle_aspect_ratios(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """AR = longest_edge / shortest_edge for every triangle (1.0 = equilateral)."""
    edges = triangle_edge_lengths(nodes, cells)
    return np.max(edges, axis=1) / (np.min(edges, axis=1) + 1e-12)


def compute_tetrahedron_aspect_ratios(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """AR = longest_edge / shortest_edge across all 6 edges of every tet."""
    edges = tetrahedron_edge_lengths(nodes, cells)
    return np.max(edges, axis=1) / (np.min(edges, axis=1) + 1e-12)


def compute_triangle_skewness_values(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Skewness for every triangle: max(|angle - 60 deg|) / 60 deg, in [0, 1].

    Based on angle deviation from equilateral (60 deg each), via the law of
    cosines on each triangle's 3 edge lengths.
    """
    p0, p1, p2 = nodes[cells[:, 0]], nodes[cells[:, 1]], nodes[cells[:, 2]]
    a = np.linalg.norm(p1 - p2, axis=1)
    b = np.linalg.norm(p0 - p2, axis=1)
    c = np.linalg.norm(p0 - p1, axis=1)

    degenerate = (a < 1e-12) | (b < 1e-12) | (c < 1e-12)
    # Guard the law-of-cosines division for degenerate triangles; their
    # skewness is overridden to the worst value (1.0) below regardless.
    safe_b = np.where(degenerate, 1.0, b)
    safe_c = np.where(degenerate, 1.0, c)
    safe_a = np.where(degenerate, 1.0, a)

    cos0 = np.clip((safe_b**2 + safe_c**2 - safe_a**2) / (2 * safe_b * safe_c), -1.0, 1.0)
    cos1 = np.clip((safe_a**2 + safe_c**2 - safe_b**2) / (2 * safe_a * safe_c), -1.0, 1.0)
    angle_0 = np.arccos(cos0)
    angle_1 = np.arccos(cos1)
    angle_2 = np.pi - angle_0 - angle_1

    angles_deg = np.degrees(np.stack([angle_0, angle_1, angle_2], axis=1))
    max_dev = np.max(np.abs(angles_deg - 60.0), axis=1)
    skewness = np.minimum(max_dev / 60.0, 1.0)
    skewness[degenerate] = 1.0

    return skewness


def compute_tetrahedron_skewness_values(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Skewness for every tetrahedron via the radius-ratio quality
    measure: 1 - 3*r_in/r_circ (0=regular tetrahedron, ->1=sliver).

    Standard tet shape-quality metric (matches Verdict/CUBIT's
    TetRadiusRatio up to this 0..1 normalization). Verified against known
    cases: regular tet -> r_in/r_circ == 1/3 exactly (skewness=0); a
    near-flat degenerate tet -> skewness ~1.

    r_in = 3V/surface_area (standard tetrahedron inradius formula).
    r_circ via the vector circumradius formula: with edge vectors a,b,c
    from one vertex, R = |a|^2(b x c) + |b|^2(c x a) + |c|^2(a x b) (vector
    sum, then take magnitude) / (12V).
    """
    p0, p1, p2, p3 = nodes[cells[:, 0]], nodes[cells[:, 1]], nodes[cells[:, 2]], nodes[cells[:, 3]]

    def tri_area(A, B, C):
        return 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)

    # 4 faces, each opposite one vertex
    area_opp_p0 = tri_area(p1, p2, p3)
    area_opp_p1 = tri_area(p0, p2, p3)
    area_opp_p2 = tri_area(p0, p1, p3)
    area_opp_p3 = tri_area(p0, p1, p2)
    surface_area = area_opp_p0 + area_opp_p1 + area_opp_p2 + area_opp_p3

    a = p1 - p0
    b = p2 - p0
    c = p3 - p0
    volume = np.abs(np.einsum('ij,ij->i', a, np.cross(b, c))) / 6.0

    r_in = 3.0 * volume / np.maximum(surface_area, 1e-300)

    a2 = np.einsum('ij,ij->i', a, a)
    b2 = np.einsum('ij,ij->i', b, b)
    c2 = np.einsum('ij,ij->i', c, c)
    circum_vec = (
        a2[:, None] * np.cross(b, c)
        + b2[:, None] * np.cross(c, a)
        + c2[:, None] * np.cross(a, b)
    )
    r_circ = np.linalg.norm(circum_vec, axis=1) / np.maximum(12.0 * volume, 1e-300)

    radius_ratio = 3.0 * r_in / np.maximum(r_circ, 1e-300)
    skewness = 1.0 - np.clip(radius_ratio, 0.0, 1.0)

    degenerate = volume < 1e-300
    skewness[degenerate] = 1.0

    return skewness


# ---------------------------------------------------------------------------
# Triangular prism (BL cell) metrics.
#
# Connectivity convention, shape=(n_cells, 6): (v0, v1, v2, w0, w1, w2) -
# v0..v2 is the bottom-layer triangle, w0..w2 the top-layer triangle, with
# w_i the extrusion of v_i (same convention mesh_extrusion.py/
# mesh_prism_to_tet.py already use for a layer's node correspondence -
# w_i is "directly above" v_i, not an arbitrary vertex permutation).
# ---------------------------------------------------------------------------

def compute_prism_volumes(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Unsigned volume of every triangular prism, via the exact
    3-tetrahedron decomposition T1=(v0,v1,v2,w2), T2=(v0,v1,w1,w2),
    T3=(v0,w0,w1,w2) - the same diagonal-consistent split mesh_prism_to_tet.
    convert_layers_to_tetrahedra uses, so a prism's volume here is always
    exactly the sum of what the old split-to-3-tets representation would
    have used, whether or not the prism is a "right" prism (planar quad
    sides, no twist).

    Each sub-tet's contribution is taken as |signed volume|: the raw
    (v0,v1,v2,w2)-style vertex tuples above are NOT individually oriented
    for a consistently-signed result (mesh_prism_to_tet's own tetrahedra
    only get that guarantee from a separate orient_tetrahedra pass this
    function doesn't replicate) - confirmed directly, one of the three
    comes out negative on an ordinary non-degenerate prism. Summing
    magnitudes is still exact for volume (the three sub-tets tile the
    prism without overlap regardless of each one's own index-order sign),
    it just means this function - unlike compute_tetrahedron_volumes -
    cannot double as an inversion/negative-volume check; that needs a
    dedicated orientation test if ever required.
    """
    v0, v1, v2 = nodes[cells[:, 0]], nodes[cells[:, 1]], nodes[cells[:, 2]]
    w0, w1, w2 = nodes[cells[:, 3]], nodes[cells[:, 4]], nodes[cells[:, 5]]

    def tet_vol(p0, p1, p2, p3):
        return np.abs(np.einsum('ij,ij->i', p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0

    return tet_vol(v0, v1, v2, w2) + tet_vol(v0, v1, w1, w2) + tet_vol(v0, w0, w1, w2)


def prism_edge_lengths(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """All 9 edge lengths for every prism, shape=(n_cells, 9): 3 bottom-cap
    + 3 top-cap + 3 vertical (wall-normal-ish) edges, in that order."""
    pts = nodes[cells]  # (n_cells, 6, 3)
    v0, v1, v2 = pts[:, 0], pts[:, 1], pts[:, 2]
    w0, w1, w2 = pts[:, 3], pts[:, 4], pts[:, 5]
    edges = [
        np.linalg.norm(v1 - v0, axis=1), np.linalg.norm(v2 - v1, axis=1), np.linalg.norm(v0 - v2, axis=1),
        np.linalg.norm(w1 - w0, axis=1), np.linalg.norm(w2 - w1, axis=1), np.linalg.norm(w0 - w2, axis=1),
        np.linalg.norm(w0 - v0, axis=1), np.linalg.norm(w1 - v1, axis=1), np.linalg.norm(w2 - v2, axis=1),
    ]
    return np.stack(edges, axis=1)


def compute_prism_aspect_ratios(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """AR = longest_edge / shortest_edge across all 9 edges of every prism.

    Unlike a tet, a HIGH aspect ratio here is often intentional and correct
    (a near-wall BL prism is *supposed* to be thin: cap edges ~mm, vertical
    edge similarly small but the ratio between successive LAYERS' vertical
    edges, not this per-cell ratio, is what governs growth-rate sanity) -
    this is why the validator applies a separate, more permissive BL-region
    threshold to prism aspect ratio (see quality_validator.py), the same
    way it already does for BL-region tet aspect ratio.

    The denominator is floored at a small FRACTION of this cell's own
    longest edge, not a fixed absolute epsilon - a mesh's edge lengths span
    mm to metres depending on min_cell_size, so a constant like 1e-12 is
    orders of magnitude below any legitimate edge and provides no real
    floor at all. This matters concretely for a "collapsed-corner" prism
    (a BL column whose growth froze at exactly one base vertex - see
    mesh_prism_to_tet.py / ProjectFiles Part6 Bug 4 - a valid, nonzero-
    volume cell with one genuinely near-zero vertical edge): with the old
    epsilon this reported a physically meaningless ratio (measured on a
    real case: 5.11e10) that swamped every other number in the quality
    report. Flooring relative to the cell's own scale instead caps any
    such cell's reported ratio at 1e6 - still unambiguously flagged as bad
    (nothing legitimate needs a 6-order-of-magnitude edge spread), just
    bounded and not misleading.
    """
    edges = prism_edge_lengths(nodes, cells)
    max_edge = np.max(edges, axis=1)
    min_edge = np.min(edges, axis=1)
    return max_edge / np.maximum(min_edge, max_edge * 1e-6)


def compute_prism_skewness_values(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Skewness for every prism: max(bottom-cap, top-cap) triangle skewness
    (equilateral-deviation, same formula as compute_triangle_skewness_values).

    Deliberately does NOT fold in "verticality" (how close the 3 vertical
    edges are to the cap normal, i.e. shear/twist) - that is a genuinely
    different defect class (governs non-orthogonality of the prism's own
    side faces, not the sliver-ness of its cross-section) and is covered by
    the existing face-based orthogonality check instead (compute_face_
    diagnostics), which works unchanged on a prism's triangulated side
    faces same as it does on any other internal face. Folding both into one
    number would let a prism with a perfectly regular cap but severe shear
    (or vice versa) hide its worst dimension behind the other's better one.
    """
    n = len(cells)
    bottom = compute_triangle_skewness_values(nodes, cells[:, 0:3])
    top = compute_triangle_skewness_values(nodes, cells[:, 3:6])
    return np.maximum(bottom, top)
