"""BL 挤出用的两种尖锐特征衰减启发式。

从 mesh_extrusion.py 拆分出来：`_compute_sharp_angle_attenuation`（按节点
自身最尖锐的二面角直接衰减）和 `_compute_edge_distance_field`（按到最近
尖锐边的欧氏距离衰减）。extrude_layers 取两者的逐节点最小值合并使用——
单独任何一个都不足以在稀疏网格化的圆角处可靠地衰减。
"""

from typing import Optional

import numpy as np

# Floor for _compute_edge_distance_field's own attenuation, applied at the
# sharp-edge vertices themselves (distance == 0). That function previously
# had no floor at all (plain `dists / (2*char_length)`, clipped to
# [0, 1]), so any node actually ON a sharp edge attenuated to EXACTLY 0 -
# combined with _compute_sharp_angle_attenuation via np.minimum, that
# meant a whole seam of nodes tracing every sharp edge of the body barely
# extruded at all, for every single BL layer, regardless of bl_layers or
# growth_rate: not a gradual falloff but a near-total local collapse of
# BL coverage exactly where automotive CFD needs good near-wall
# resolution most (character lines, spoiler/mirror/underbody edges - all
# separation-prone features). 0.2 matches _compute_sharp_angle_
# attenuation's own value for a plain 90-degree edge (its 0.2-1.0 linear
# ramp over the 90-150 degree dihedral range starts at exactly 0.2), so
# the two mechanisms now agree at the edge itself instead of the distance
# field silently overriding the angle field's own considered floor via
# their np.minimum combination.
MIN_EDGE_DISTANCE_ATTENUATION = 0.2


def _compute_sharp_angle_attenuation(
    nodes: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    normal_faces: Optional[np.ndarray] = None,
    sharp_angle_threshold: float = 45.0,
) -> np.ndarray:
    """Compute attenuation based on local dihedral angle (ANSA-style).

    Nodes at sharp corners (e.g., 90-degree edges) will have their
    extrusion thickness attenuated to prevent self-intersection and distortion.

    Args:
        nodes: Surface nodes
        faces: Surface connectivity (topology)
        normals: Face normals
        normal_faces: The subset of faces corresponding to the normals
        sharp_angle_threshold: Angles below this (deviation from flat 180) are considered sharp

    Returns:
        attenuation: [0, 1] array. 0 = no extrusion (sharp corner), 1 = full extrusion.
    """
    n_nodes = len(nodes)
    detect_faces = normal_faces if normal_faces is not None else faces

    # 1. Calculate dihedral angle for each edge
    edge_map = {}  # (min_v, max_v) -> list of face indices
    for i, face in enumerate(detect_faces):
        for j in range(3):
            v1, v2 = int(face[j]), int(face[(j + 1) % 3])
            key = (min(v1, v2), max(v1, v2))
            if key not in edge_map:
                edge_map[key] = []
            edge_map[key].append(i)

    # Per node, track the SHARPEST touching edge as the MINIMUM dot(n1,n2)
    # (dot=1.0 <-> normals parallel <-> a flat continuation; dot values
    # further below 1.0 mean the two adjacent faces' normals diverge more,
    # i.e. a sharper fold). Initialize to 1.0 (flat) rather than a sentinel,
    # so a node touching no qualifying 2-face edge (isolated/patch-boundary
    # vertex) safely defaults to "smooth" instead of needing separate
    # handling.
    node_min_cos_angle = np.full(n_nodes, 1.0)

    for (v1, v2), face_indices in edge_map.items():
        if len(face_indices) >= 2:
            # Use the first two adjacent faces to estimate dihedral angle
            n1 = normals[face_indices[0]]
            n2 = normals[face_indices[1]]
            cos_angle = np.dot(n1, n2)  # 1.0 = flat (normals parallel)

            # MIN, not max: we want the node's SHARPEST touching edge (the
            # smallest dot product / most divergent pair of normals) to
            # govern its attenuation - a node touching even one sharp edge
            # should attenuate, regardless of how many other flat edges it
            # also touches. (Fixed from an earlier version that took max()
            # here despite the variable's own name and this comment saying
            # "min" - that bug meant a node attenuated only if EVERY edge
            # touching it was sharp, which on real geometry is close to
            # never, so this attenuation was silently inert.)
            node_min_cos_angle[v1] = min(node_min_cos_angle[v1], cos_angle)
            node_min_cos_angle[v2] = min(node_min_cos_angle[v2], cos_angle)

    # 2. Convert the normal-to-normal angle into the conventional surface
    # dihedral angle (180 deg = flat continuation, 90 deg = a right-angle
    # fold, decreasing further for a sharper fold) before applying the
    # smooth/sharp thresholds below, which are written in THAT convention.
    # arccos(dot(n1,n2)) alone is the angle BETWEEN THE NORMALS, which runs
    # the opposite way (~0 deg for flat, larger for sharper) - conflating
    # the two was a second, independent bug in an earlier version of this
    # function: flat regions (dot~1, normal-angle~0) satisfied
    # "angle < sharp_limit" and got attenuated to 0.1 almost everywhere,
    # not just at genuine sharp features.
    normal_angle_rad = np.arccos(np.clip(node_min_cos_angle, -1.0, 1.0))
    dihedral_rad = np.pi - normal_angle_rad

    # Define a "sharpness" range (in dihedral-angle terms)
    smooth_limit = np.radians(150)  # 150 degrees: treated as flat
    sharp_limit = np.radians(90)    # 90 degrees: treated as fully sharp

    attenuation = np.ones(n_nodes)

    # Mask for sharp regions
    sharp_mask = dihedral_rad < smooth_limit
    if np.any(sharp_mask):
        # Linearly interpolate between sharp_limit (0.2) and smooth_limit (1.0)
        # This creates a smooth taper from the edge
        t = (dihedral_rad[sharp_mask] - sharp_limit) / (smooth_limit - sharp_limit)
        attenuation[sharp_mask] = 0.2 + 0.8 * np.clip(t, 0, 1)

    # For very sharp corners (< 90 deg), keep a minimal thickness to avoid zero-volume cells
    # but prevent large extrusions
    very_sharp_mask = dihedral_rad < sharp_limit
    attenuation[very_sharp_mask] = 0.1

    return attenuation


def _compute_edge_distance_field(
    nodes: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    angle_threshold: float = 45.0,
    normal_faces: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute a distance field from each node to the nearest sharp edge.

    Args:
        nodes: Surface nodes, shape=(n_nodes, 3)
        faces: Surface connectivity, shape=(n_faces, 3) - used for topology
        normals: Face normals, shape=(n_normal_faces, 3) - MUST match `normal_faces`
        angle_threshold: Dihedral angle threshold (degrees) to consider an edge sharp.
        normal_faces: Optional subset of `faces` that the `normals` correspond to.
                      If None, assumes `normals` match `faces`.

    Returns:
        attenuation: Attenuation factor in [MIN_EDGE_DISTANCE_ATTENUATION, 1]
                     for each node. 1.0 means full thickness, down to the
                     floor right at a sharp edge - see that constant's own
                     comment.
    """
    n_nodes = len(nodes)

    # Use normal_faces for edge detection if provided, otherwise use faces
    detect_faces = normal_faces if normal_faces is not None else faces

    # 1. Identify sharp edges based on face normals
    # For each edge, check the angle between adjacent faces
    edge_map = {}  # (min_v, max_v) -> list of face indices (into detect_faces)

    for i, face in enumerate(detect_faces):
        for j in range(3):
            v1, v2 = int(face[j]), int(face[(j + 1) % 3])
            key = (min(v1, v2), max(v1, v2))
            if key not in edge_map:
                edge_map[key] = []
            edge_map[key].append(i)

    sharp_edges = set()
    for (v1, v2), face_indices in edge_map.items():
        if len(face_indices) >= 2:
            # Calculate dihedral angle
            n1 = normals[face_indices[0]]
            n2 = normals[face_indices[1]]
            cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))

            # Sharp if angle is significantly different from 180 (flat)
            # For a cube, we expect 90 degree angles
            if angle > angle_threshold and angle < (180 - angle_threshold):
                sharp_edges.add((v1, v2))

    if not sharp_edges:
        return np.ones(n_nodes)

    # 2. Compute distance from each node to the nearest sharp edge
    # For simplicity, we use distance to the nearest vertex participating in a sharp edge
    sharp_vertex_set = set()
    for v1, v2 in sharp_edges:
        sharp_vertex_set.add(v1)
        sharp_vertex_set.add(v2)

    sharp_vertices = nodes[list(sharp_vertex_set)]

    # Use KDTree for efficient nearest neighbor search
    from scipy.spatial import cKDTree
    tree = cKDTree(sharp_vertices)
    dists, _ = tree.query(nodes, k=1)

    # 3. Convert distance to attenuation using a smooth step function
    # Characteristic length scale: average edge length of the surface
    edge_lengths = []
    for v1, v2 in list(sharp_edges)[:100]: # Sample for performance
        edge_lengths.append(np.linalg.norm(nodes[v1] - nodes[v2]))
    char_length = np.mean(edge_lengths) if edge_lengths else 0.01

    # Smooth attenuation: MIN_EDGE_DISTANCE_ATTENUATION at distance 0
    # (see that constant's own comment for why this floor exists), ramping
    # to 1 at distance > 2*char_length.
    ramp = np.clip(dists / (2.0 * char_length), 0.0, 1.0)
    attenuation = MIN_EDGE_DISTANCE_ATTENUATION + (1.0 - MIN_EDGE_DISTANCE_ATTENUATION) * ramp

    return attenuation
