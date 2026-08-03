"""Boundary-group classification for volume mesh generation.

Decides which boundary-group faces should get boundary-layer (BL) extrusion
versus which should be used unmodified as part of the outer domain shell fed
to the constrained tetrahedralizer, and fixes each eligible sub-shell's
winding so BL extrusion grows into the fluid domain instead of trusting the
input mesh's raw (unverified) face winding.

Classification operates per *named boundary group*, not per raw globally
connected component: real automotive surface meshes routinely weld a wall
group (e.g. the car body's underbody) to an adjacent group (ground/tunnel)
at a small contact patch, so a naive "connected components over all
candidate faces" pass would fuse body+ground+tunnel into one blob and lose
the ability to tell them apart. Scoping the analysis to each group's own
face subset avoids that.
"""

from typing import Dict, List, NamedTuple, Optional, Tuple, TYPE_CHECKING

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from loguru import logger

if TYPE_CHECKING:
    from ..structures import BoundaryMap

# Boundary types that are always open-flow boundaries or frictionless
# (slip) walls, so their faces are never BL-extruded regardless of
# geometry: there is no near-wall velocity gradient to resolve at a
# free-slip/symmetry surface, and no wall at all at a genuine open
# boundary. SLIP_WALL covers e.g. "tunnel"/"farfield"-named boundaries
# (see nas_parser_boundary.py's keyword table and bc_handler.py's
# _classify) - previously missing here, so a tunnel wall (falling through
# to the 'WALL' bc_type default before that keyword-table fix) could still
# get BL-extruded, which collapses almost immediately for a domain-
# spanning wall (hits the opposite wall/body within 1-2 layers).
NEVER_EXTRUDE_BC_TYPES = {'VELOCITY_INLET', 'PRESSURE_OUTLET', 'SYMMETRY', 'SLIP_WALL'}

# A sub-shell whose own open-edge fraction is below this is treated as a
# closed (embedded) solid for orientation purposes, even with a small real
# opening (e.g. a body welded to the ground at a small contact patch).
_CLOSED_OPEN_EDGE_FRACTION = 0.01

# An open sub-shell is only extruded if a single bounding-box face accounts
# for at least this fraction of its own nodes (a genuine flat floor/wall);
# otherwise it's treated as part of the outer domain shell (core-only).
_BBOX_TOUCH_MAJORITY = 0.9

# Relative tolerance (of the domain's characteristic length) for deciding a
# node lies "on" a bounding-box face, matching mesh_utils.check_reached_boundary's
# existing 1e-6 convention.
_BBOX_TOUCH_RTOL = 1e-6


class SubShell(NamedTuple):
    """One classified, winding-corrected piece of a boundary group."""
    faces: np.ndarray          # (n, 3) int, indices into the shared node array
    extrude: bool
    group_name: str


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


def classify_boundary_groups(
    nodes: np.ndarray,
    surface_faces: np.ndarray,
    boundaries: 'BoundaryMap',
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
    """Split every boundary group's faces into extrude-eligible vs. core-only,
    with extrude-eligible faces winding-corrected for correct BL growth
    direction.

    Args:
        nodes: (n_nodes, 3) surface node coordinates
        surface_faces: (n_faces, 3) surface connectivity
        boundaries: BoundaryMap with groups (cell/face indices) and bc_types
        bbox_min, bbox_max: overall (unpadded) domain extent, shape (3,)

    Returns:
        extrude_faces: (m, 3) winding-corrected faces eligible for BL extrusion
        core_faces: (k, 3) faces to use unmodified as outer-shell PLC input
            (m + k == n_faces; every input face appears in exactly one)
        extruded_group_names: names of boundary groups that got at least
            some faces extruded (for logging/diagnostics)
        extrude_face_groups: (m,) str array, the original boundary-group name
            for each row of extrude_faces (same order/length) - lets the
            caller attribute BL-extruded tets back to their source group
            directly via face position, instead of matching node indices
            against the pre-extrusion surface (which cannot work for
            genuinely-displaced BL nodes; see mesh_boundary.py).
        hole_points: one point per closed embedded-solid sub-component found
            (e.g. a car body, isolated from the domain's outer shell) - must
            be passed to mesh_tetgen_core.fill_core_volume as tetgen hole
            seeds, or tetgen fills that solid's own interior (and its BL
            block's enclosed cavity) with spurious tetrahedra that overlap
            the BL prisms already occupying that space, instead of
            correctly excluding it. A bbox-touching wall (ground/tunnel) is
            never a hole - it's an open sheet terminating at the domain's
            own outer boundary, with no enclosed interior to exclude.
        core_face_groups: (k,) str array, the original boundary-group name
            for each row of core_faces (same order/length) - lets the
            caller attribute core (tetgen-filled) boundary tets back to
            their source group via tetgen facet markers, which survive
            boundary subdivision unlike node-index matching (see
            mesh_tetgen_core.fill_core_volume's `face_markers`/nobisect=False
            path, needed for graded max-cell-size regions to actually
            refine cells near a coarse far-field wall).
        is_closed_solid_face: (m,) bool array, parallel to extrude_faces -
            True for rows from a closed embedded solid (the `hole_points`
            branch, e.g. a car body), False for a bbox-touching wall sheet
            (ground/tunnel-like). Currently unused by mesh_background.py
            (received as `_is_closed_solid_face`) - it was meant to let the
            caller build max-cell-size grading spheres centered on just the
            isolated solid's own geometry, distinct from a bbox-touching
            wall sheet that can span nearly the whole domain footprint, but
            that per-solid grading-sphere approach was abandoned (see
            mesh_tetgen_core.py's note where those functions used to live)
            in favor of one flat core-fill region. Kept here since it's a
            cheap, already-computed byproduct that a future per-solid
            grading scheme could reuse.
    """
    L_char = float(np.max(bbox_max - bbox_min))
    tol = L_char * _BBOX_TOUCH_RTOL

    extrude_face_rows: List[np.ndarray] = []
    extrude_face_group_rows: List[np.ndarray] = []
    is_closed_solid_rows: List[np.ndarray] = []
    core_face_rows: List[np.ndarray] = []
    core_face_group_rows: List[np.ndarray] = []
    extruded_group_names: List[str] = []
    hole_points: List[np.ndarray] = []

    for name, cell_idx in boundaries.groups.items():
        bc_type = boundaries.bc_types.get(name)
        group_faces = surface_faces[cell_idx].copy()

        if bc_type in NEVER_EXTRUDE_BC_TYPES:
            core_face_rows.append(group_faces)
            core_face_group_rows.append(np.full(len(group_faces), name))
            continue

        inverse, counts, face_of_edge = _face_edges(group_faces)
        labels = _connected_components(group_faces, inverse, face_of_edge)

        any_extruded_in_group = False

        for comp_id in np.unique(labels):
            comp_face_mask = labels == comp_id
            comp_faces = group_faces[comp_face_mask]

            # Check bounding-box touch FIRST, before the open-edge-fraction
            # test below. That fraction is not a topological invariant: a
            # large flat sheet (ground/tunnel wall) has far more internal
            # edges than perimeter edges once meshed finely enough, so it
            # can fall under the "closed" threshold by mesh density alone -
            # empirically confirmed to misclassify a >=150x150-division
            # flat plane as a "closed embedded solid", which then gets its
            # orientation decided by a near-zero (numerically-noisy) signed
            # volume instead of the bbox-direction check meant for exactly
            # this shape, and gets BL-extruded when it should stay core-only.
            # A real embedded solid (car body) never predominantly touches a
            # single bbox face even when welded to the ground at a small
            # contact patch (_BBOX_TOUCH_MAJORITY=0.9 of its own nodes), so
            # checking this first doesn't change that case's outcome.
            comp_node_idx = np.unique(comp_faces)
            direction = _bbox_touch_fraction(nodes, comp_node_idx, bbox_min, bbox_max, tol)

            if direction is not None:
                # Predominantly sits on one bbox face: a floor/wall-like
                # sheet that's part of the domain's outer shell. Orientation
                # comes from that bbox direction, not face winding (which is
                # unreliable for a sheet with a real free boundary).
                from .mesh_utils import compute_face_normals
                comp_normals = compute_face_normals(nodes, comp_faces)
                mean_normal = comp_normals.mean(axis=0)
                if np.dot(mean_normal, direction) < 0:
                    comp_faces = comp_faces[:, [1, 0, 2]]  # flip winding
                extrude_face_rows.append(comp_faces)
                extrude_face_group_rows.append(np.full(len(comp_faces), name))
                is_closed_solid_rows.append(np.zeros(len(comp_faces), dtype=bool))
                any_extruded_in_group = True
                continue

            # Doesn't predominantly sit on a single bbox face. Recompute edge
            # stats scoped to this sub-component alone so the open-edge
            # fraction reflects only its own boundary, not the whole group's.
            _, sub_counts, _ = _face_edges(comp_faces)
            n_unique_edges = len(sub_counts)
            n_open_edges = int(np.count_nonzero(sub_counts == 1))
            open_fraction = n_open_edges / max(n_unique_edges, 1)

            if open_fraction < _CLOSED_OPEN_EDGE_FRACTION:
                # Closed-like (embedded solid, e.g. car body): orient by the
                # sign of its own enclosed volume, not by trusting input
                # winding directly.
                volume = _signed_volume(nodes, comp_faces)
                if volume < 0:
                    comp_faces = comp_faces[:, [1, 0, 2]]  # flip winding
                extrude_face_rows.append(comp_faces)
                extrude_face_group_rows.append(np.full(len(comp_faces), name))
                is_closed_solid_rows.append(np.ones(len(comp_faces), dtype=bool))
                any_extruded_in_group = True

                hole_pt = find_point_inside_closed_shell(nodes, comp_faces)
                if hole_pt is not None:
                    hole_points.append(hole_pt)
                else:
                    logger.warning(
                        f"Could not find a reliable interior point for closed "
                        f"solid '{name}' (component with {len(comp_faces)} "
                        f"faces) - skipping its tetgen hole marker. The core "
                        f"fill may include spurious tetrahedra inside this "
                        f"solid's own BL block."
                    )
            else:
                # Open and not bbox-touching: an outer-shell wall
                # (inlet/outlet/tunnel-like) with a genuine free boundary
                # elsewhere. Use unmodified as part of the core PLC.
                core_face_rows.append(comp_faces)
                core_face_group_rows.append(np.full(len(comp_faces), name))

        if any_extruded_in_group:
            extruded_group_names.append(name)

    extrude_faces = (
        np.vstack(extrude_face_rows) if extrude_face_rows
        else np.empty((0, 3), dtype=surface_faces.dtype)
    )
    extrude_face_groups = (
        np.concatenate(extrude_face_group_rows) if extrude_face_group_rows
        else np.empty((0,), dtype=object)
    )
    core_faces = (
        np.vstack(core_face_rows) if core_face_rows
        else np.empty((0, 3), dtype=surface_faces.dtype)
    )
    core_face_groups = (
        np.concatenate(core_face_group_rows) if core_face_group_rows
        else np.empty((0,), dtype=object)
    )
    is_closed_solid_face = (
        np.concatenate(is_closed_solid_rows) if is_closed_solid_rows
        else np.empty((0,), dtype=bool)
    )

    logger.info(
        f"Boundary classification: {len(extrude_faces)} faces eligible for "
        f"BL extrusion (groups: {extruded_group_names}), "
        f"{len(core_faces)} faces used as-is for the outer domain shell, "
        f"{len(hole_points)} isolated embedded solid(s) marked as tetgen holes"
    )

    return (
        extrude_faces, core_faces, extruded_group_names, extrude_face_groups,
        hole_points, core_face_groups, is_closed_solid_face,
    )
