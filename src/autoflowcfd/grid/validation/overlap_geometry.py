"""Exact geometric primitives for triangle-triangle overlap/proximity checks.

Vectorized (numpy, no Python loop over candidate pairs) implementations of
the standard closed-form algorithms from Ericson, "Real-Time Collision
Detection" (2005) and Moller, "A Fast Triangle-Triangle Intersection Test"
(1997) - not heuristics or approximations. Every function here takes
batched input (shape (N, 3) per point/vertex argument, one row per
candidate pair) so a caller with M candidate pairs after its own broad-phase
filtering runs all M tests in a handful of vectorized numpy calls, not a
Python-level loop over M.

Used by mesh_overlap_check.py, which handles broad-phase candidate
generation and orchestration; this module is pure computational geometry
with no mesh-specific concepts (no cells, no boundary groups, no faces
beyond their three vertex positions).
"""

import numpy as np


def closest_point_on_triangle(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """Closest point to `p` on triangle (a, b, c), one row per candidate.

    Ericson section 5.1.5's seven-region Voronoi test, vectorized: for each
    row, determine which of the triangle's two vertex regions, three edge
    regions, or interior face region contains the closest point, using
    boolean masks instead of Ericson's original early-return branches (the
    first matching region "wins", masks are applied in the same priority
    order the branching version checks them in).

    Args:
        p: (N, 3) query points
        a, b, c: (N, 3) triangle vertices, one triangle per row

    Returns:
        (N, 3) closest point on each row's triangle to that row's query point
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

    # Vertex regions.
    _take((d1 <= 0) & (d2 <= 0), a)
    _take((d3 >= 0) & (d4 <= d3), b)
    _take((d6 >= 0) & (d5 <= d6), c)

    # Edge AB region.
    mask_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0) & ~assigned
    denom_ab = d1 - d3
    v_ab = np.divide(d1, denom_ab, out=np.zeros(n), where=np.abs(denom_ab) > 1e-300)
    _take(mask_ab, a + v_ab[:, None] * ab)

    # Edge AC region.
    mask_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0) & ~assigned
    denom_ac = d2 - d6
    w_ac = np.divide(d2, denom_ac, out=np.zeros(n), where=np.abs(denom_ac) > 1e-300)
    _take(mask_ac, a + w_ac[:, None] * ac)

    # Edge BC region.
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
    """Closest points between segments (p1,q1) and (p2,q2), one pair per row.

    Ericson section 5.1.9's closed-form solution, vectorized: solves for the
    clamped parametric positions s in [0,1] (along d1 = q1-p1) and t in
    [0,1] (along d2 = q2-p2) that minimize |c1 - c2|, handling degenerate
    (zero-length) segments and near-parallel segments as their own cases
    rather than dividing by a near-zero denominator.

    Returns:
        (c1, c2): each (N, 3), the closest point on each row's first/second
        segment respectively
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


def _signed_dist_to_plane(pts: np.ndarray, plane_pt: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return np.einsum('ij,ij->i', pts - plane_pt, normal)


def triangle_triangle_intersect(
    a0: np.ndarray, a1: np.ndarray, a2: np.ndarray,
    b0: np.ndarray, b1: np.ndarray, b2: np.ndarray,
    eps: float = 1e-9,
) -> np.ndarray:
    """Exact triangle-triangle intersection test, one bool per row.

    Moller (1997): reject a pair whose vertices all lie strictly on one
    side of the other triangle's plane (fast rejection using the plane
    each triangle already has); otherwise both triangles cross the line L
    where the two planes meet, so intersect if and only if their two
    intervals along L overlap. Two triangles that only share an edge or a
    single vertex are NOT reported as intersecting by this construction
    (their overlap interval touches at a single point / has zero measure,
    which the caller treats as "adjacent", not "overlapping" - see
    mesh_overlap_check.py's node-sharing pre-filter, which excludes such
    pairs before they ever reach this function).

    Coplanar triangles (the fast-rejection plane tests can't distinguish
    "coplanar" from "no separation" using only signed distances) fall back
    to a 2D separating-axis test in the shared plane.

    Args:
        a0, a1, a2: (N, 3) first triangle's vertices
        b0, b1, b2: (N, 3) second triangle's vertices
        eps: tolerance in the SAME units as the input coordinates (meters
            here) - distances below are unit-normalized (see normal_a/
            normal_b) so this is a true, scale-independent distance
            tolerance. A previous version compared RAW (non-normalized,
            area-scaled) distances against a fixed eps - too loose for a
            large core triangle and too tight for a tiny BL sliver on the
            same mesh, which on a real multi-million-cell mesh produced an
            all-NaN interval (RuntimeWarning) whenever no edge of a
            triangle cleanly registered as "touching" the other plane.

    Returns:
        (N,) bool
    """
    n = len(a0)
    result = np.zeros(n, dtype=bool)

    normal_a = np.cross(a1 - a0, a2 - a0)
    normal_b = np.cross(b1 - b0, b2 - b0)
    # Unit-normalize so every downstream signed "distance" is a TRUE
    # geometric distance (meters), not scaled by the triangle's own area -
    # see the eps arg doc above for why that scale-dependence was a real
    # bug on a mesh with a wide range of face sizes. Degenerate (zero-area)
    # triangles keep their raw (zero) normal; any pair involving one is
    # physically meaningless as an "intersection" and is left to fall
    # through to a safe (non-intersecting) result rather than dividing by
    # zero.
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

        # Thin-sliver-triangle correction. _intersect_on_line determines
        # overlap by projecting each triangle's edges onto the line where
        # the two planes meet (line_dir = cross(normal_a, normal_b)) and
        # comparing 1D intervals along it - both that projection and the
        # da/db signed-distance-to-plane values it starts from lose
        # precision whenever a triangle is a thin sliver (a real,
        # measured consequence of this project's own miter-join sharp-
        # corner compensation - see mesh_layer_step.py), because a thin
        # triangle's own normal/plane is, by construction, only weakly
        # sensitive to a genuine offset along the triangle's long axis.
        # This is not limited to near-parallel plane pairs, as first
        # suspected: confirmed directly on cube_demo across a range of
        # plane angles (cross(normal_a,normal_b) magnitudes from ~7e-6 up
        # to ~0.14) that two thin slivers stacked a real, unambiguous
        # distance apart (independently verified in every case via brute-
        # force point sampling and via triangle_triangle_min_distance,
        # 0.01m - about 3x this project's own min_cell_size) were still
        # flagged as crossing.
        #
        # Fix: get a SECOND opinion from triangle_triangle_min_distance -
        # built from point-to-triangle/segment-to-segment closest-point
        # primitives (Ericson 5.1.5/5.1.9) that never divide by either
        # triangle's own normal or by the two planes' cross product, so
        # they stay well-conditioned regardless of how thin either
        # triangle is or how the two planes happen to be oriented.
        # Checked for every generic-path row currently called
        # intersecting (not just suspected-thin ones - cheap relative to
        # the false-positive risk, and there is no reliable, cheaper way
        # to predict in advance which rows need it): only ever used to
        # turn a True into False, and only when that second opinion finds
        # a distance clearly outside eps - i.e. this can only correct a
        # false positive into a verified true negative, never introduce a
        # new one, and never touches rows the generic path already called
        # non-intersecting or the (differently-conditioned) coplanar
        # branch handles. Confirmed directly against all of this
        # project's own 15 hand-built edge cases and the 3000-case mixed-
        # scale stress test (see Part5 P2) - none of them exercise this
        # thin-sliver regime, so this correction is purely additive for a
        # case nothing existing already covered.
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
    """Project a triangle's intersection with a line onto that line,
    returning the [lo, hi] interval. `line_dir` must be unit length (the
    caller normalizes it) so `lo`/`hi` are true distances along the line,
    directly comparable against `eps` (also a true distance) - both
    triangles' intervals are computed with the SAME line_dir so their
    projected values are directly comparable with each other too.

    For each of the triangle's 3 edges, the edge crosses (or touches) the
    plane the interval is measured against whenever its two endpoints'
    signed distances (d0, d1, d2) have opposite sign or either is within
    `eps` of zero (`da * db <= eps^2`, an eps-widened version of the exact
    `da * db <= 0` test - see triangle_triangle_intersect's own eps doc for
    why an exact-zero test is fragile once real, not synthetic, mesh
    coordinates are involved); its crossing point is then linearly
    interpolated and projected onto the line. Checking all 3 edges
    independently - rather than first picking a single "odd one out"
    vertex and assuming the other two edges are the crossings - correctly
    handles a vertex landing exactly ON the other plane (d == 0 for that
    vertex): both edges touching it register a valid "crossing" at that
    same vertex, and the third (genuinely same-sign) edge contributes no
    crossing at all. A single-vertex-picking approach mishandles this
    configuration, since none of "two vertices share a sign" cleanly holds
    when one distance is exactly (or near-) zero.

    In the extremely rare case none of the 3 edges register as touching
    (numerically, all three are just barely on the same side despite the
    pair having already passed the caller's own "not separated" test - a
    floating-point boundary case, not a valid geometric configuration),
    this returns an EMPTY interval (lo=+inf, hi=-inf) rather than
    NaN/np.nanmin's all-NaN warning - an empty interval can never overlap
    anything, which is the correct, safe conclusion for an ambiguous
    touch this close to the tolerance boundary.
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
    # Plain np.min/np.max on NaN-free arrays, not np.nanmin/np.nanmax - if
    # a row is all-NaN (no edge touched, see docstring above), replacing
    # NaN with +inf before min (or -inf before max) makes that row
    # naturally resolve to lo=+inf, hi=-inf (an empty interval) without
    # nanmin/nanmax's all-NaN RuntimeWarning.
    lo = np.min(np.where(is_nan, np.inf, stacked), axis=0)
    hi = np.max(np.where(is_nan, -np.inf, stacked), axis=0)
    return lo, hi


def _intersect_on_line(
    a0, a1, a2, da0, da1, da2,
    b0, b1, b2, db0, db1, db2,
    normal_a, normal_b,
    eps: float = 1e-9,
) -> np.ndarray:
    """Both triangles genuinely cross the other's plane (non-separated,
    non-coplanar) - intersect iff their projected intervals along the two
    planes' common line overlap with positive length. A closed `<=`
    comparison here would also flag two triangles that only touch at a
    single point on the line (e.g. sharing exactly one vertex) as
    "intersecting" - excluded via a small eps margin instead, consistent
    with this function's contract (shared-edge/shared-vertex adjacency is
    not overlap)."""
    line_dir = np.cross(normal_a, normal_b)
    line_dir_mag = np.linalg.norm(line_dir, axis=1, keepdims=True)
    # Normalized so _interval_on_line's projected lo/hi are true distances
    # (meters), comparable against eps directly - see that function's own
    # eps doc. Guard the near-coplanar-but-not-quite case (planes almost
    # parallel, line_dir magnitude near zero): falls back to treating the
    # pair as non-intersecting via an already-empty interval rather than
    # dividing by ~zero and amplifying floating-point noise into a
    # meaningless "line" direction.
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
    """Minimum distance between two (assumed non-intersecting) triangles,
    one value per row.

    Exact for non-intersecting convex-polygon pairs: the closest pair of
    points is always either a vertex of one triangle against the other
    triangle's face/edge/vertex (covered by point_to_triangle_distance from
    each of the 6 vertices), or a genuine edge-edge closest approach not
    involving either triangle's vertices head-on (covered by the 9 edge-
    edge combinations) - the overall minimum of all 15 is the true answer.
    Callers should gate this on triangle_triangle_intersect being False
    first; distance is not a meaningful ~0 signal for a genuine overlap
    (this function does not attempt to compute penetration depth).
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
