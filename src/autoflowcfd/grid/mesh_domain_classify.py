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

from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from loguru import logger

# Boundary types that are always open-flow boundaries (never solid walls),
# so their faces are never BL-extruded regardless of geometry.
NEVER_EXTRUDE_BC_TYPES = {'VELOCITY_INLET', 'PRESSURE_OUTLET', 'SYMMETRY'}

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


def _signed_volume(nodes: np.ndarray, faces: np.ndarray) -> float:
    """Enclosed volume of a (near-)closed surface, using raw (unnormalized)
    face winding. Sign follows the same convention as
    mesh_extrusion.orient_tetrahedra (positive = outward-consistent winding).
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
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
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
    """
    L_char = float(np.max(bbox_max - bbox_min))
    tol = L_char * _BBOX_TOUCH_RTOL

    extrude_face_rows: List[np.ndarray] = []
    core_face_rows: List[np.ndarray] = []
    extruded_group_names: List[str] = []

    for name, cell_idx in boundaries.groups.items():
        bc_type = boundaries.bc_types.get(name)
        group_faces = surface_faces[cell_idx].copy()

        if bc_type in NEVER_EXTRUDE_BC_TYPES:
            core_face_rows.append(group_faces)
            continue

        inverse, counts, face_of_edge = _face_edges(group_faces)
        labels = _connected_components(group_faces, inverse, face_of_edge)

        any_extruded_in_group = False

        for comp_id in np.unique(labels):
            comp_face_mask = labels == comp_id
            comp_faces = group_faces[comp_face_mask]

            # Recompute edge stats scoped to this sub-component alone so the
            # open-edge fraction reflects only its own boundary, not the
            # whole group's.
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
                any_extruded_in_group = True
            else:
                # Open-like (a flat sheet touching the domain boundary, e.g.
                # ground): direction comes from which single bbox face this
                # sub-component's nodes predominantly sit on - not from face
                # winding, which is unreliable for a sheet with a real free
                # boundary.
                comp_node_idx = np.unique(comp_faces)
                direction = _bbox_touch_fraction(nodes, comp_node_idx, bbox_min, bbox_max, tol)

                if direction is None:
                    # Doesn't predominantly sit on a single bbox face - this
                    # is an outer-shell wall (inlet/outlet/tunnel-like),
                    # not a floor. Use unmodified as part of the core PLC.
                    core_face_rows.append(comp_faces)
                    continue

                from .mesh_utils import compute_face_normals
                comp_normals = compute_face_normals(nodes, comp_faces)
                mean_normal = comp_normals.mean(axis=0)
                if np.dot(mean_normal, direction) < 0:
                    comp_faces = comp_faces[:, [1, 0, 2]]  # flip winding
                extrude_face_rows.append(comp_faces)
                any_extruded_in_group = True

        if any_extruded_in_group:
            extruded_group_names.append(name)

    extrude_faces = (
        np.vstack(extrude_face_rows) if extrude_face_rows
        else np.empty((0, 3), dtype=surface_faces.dtype)
    )
    core_faces = (
        np.vstack(core_face_rows) if core_face_rows
        else np.empty((0, 3), dtype=surface_faces.dtype)
    )

    logger.info(
        f"Boundary classification: {len(extrude_faces)} faces eligible for "
        f"BL extrusion (groups: {extruded_group_names}), "
        f"{len(core_faces)} faces used as-is for the outer domain shell"
    )

    return extrude_faces, core_faces, extruded_group_names
