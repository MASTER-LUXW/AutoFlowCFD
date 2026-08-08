"""Stage B' 局部重铺（cavity retile）用到的共享底层工具。

从 mesh_repair_cavity.py 拆分出来，供 remesh_core_cavity（同目录
mesh_repair_cavity.py）和 patch_nonmanifold_cavity（同目录
mesh_repair_nonmanifold_patch.py）两个局部重新四面体化流程共用：cavity
（待重铺区域）的环形扩张、cavity 自身边界面提取，以及重铺后的质量评分。
"""

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from ..validation.quality_validator import MeshQualityValidator

# Outward-oriented triangular faces of a POSITIVELY-oriented tetrahedron
# (v0,v1,v2,v3), one row per omitted vertex - see mesh_prism_to_tet.orient_tetrahedra
# for the positive-orientation convention this assumes. Verified against a
# reference unit tet (0,0,0)-(1,0,0)-(0,1,0)-(0,0,1): each row's cross-product
# normal points away from the tet's own centroid, i.e. outward.
_CAVITY_FACE_TEMPLATES = np.array([
    [1, 2, 3],
    [0, 3, 2],
    [0, 1, 3],
    [0, 2, 1],
], dtype=np.int64)


def _grow_cavity_rings(
    seed_mask: np.ndarray,
    owner: np.ndarray,
    neighbor: np.ndarray,
    blocked_mask: np.ndarray,
    n_rings: int,
) -> np.ndarray:
    """Expand a seed cell mask outward by `n_rings` face-adjacency hops,
    never crossing into a `blocked_mask` cell (BL cells / cells that touch a
    physical boundary face - see remesh_core_cavity). The buffer rings exist
    so the cavity's own new boundary lands on already-good cells, not
    through an already-degenerate one.

    Args:
        owner, neighbor: (n_interior_faces,) cell indices on either side of
            each INTERIOR face only (a boundary face has no far side to
            adjoin through, so it's not part of this adjacency graph at all)

    Returns:
        Boolean cell mask, same shape as seed_mask, with blocked cells
        guaranteed False even if reachable.
    """
    cavity = seed_mask & ~blocked_mask
    for _ in range(n_rings):
        touches = cavity[owner] | cavity[neighbor]
        if not np.any(touches):
            break
        newly = np.zeros_like(cavity)
        newly[owner[touches]] = True
        newly[neighbor[touches]] = True
        newly &= ~blocked_mask
        if np.array_equal(newly | cavity, cavity):
            break
        cavity |= newly
    return cavity


def _cavity_boundary_faces(cells: np.ndarray, cavity_cell_idx: np.ndarray) -> np.ndarray:
    """Outward-oriented boundary faces of a cell subset (global node
    indices) - a face shared by two cavity cells is purely interior (tetgen
    will retile it away) and is excluded; a face shared with a cell OUTSIDE
    the subset, or with nothing (a real physical boundary face), appears
    exactly once among the subset's own faces and becomes part of the
    cavity's fixed PLC.
    """
    cav_cells = cells[cavity_cell_idx]
    all_faces = cav_cells[:, _CAVITY_FACE_TEMPLATES].reshape(-1, 3)
    sorted_faces = np.sort(all_faces, axis=1)
    face_dtype = np.dtype((np.void, sorted_faces.dtype.itemsize * 3))
    voids = np.ascontiguousarray(sorted_faces).view(face_dtype).reshape(-1)
    _, inverse, counts = np.unique(voids, return_inverse=True, return_counts=True)
    boundary_mask = counts[inverse] == 1
    return all_faces[boundary_mask]


def _count_bad_cells(validator: 'MeshQualityValidator', nodes: np.ndarray, cells: np.ndarray) -> int:
    """How many of `cells` trip skewness, non-orthogonality, or adjacent-
    volume-ratio - the same three criteria mesh_repair.py's own
    `_bad_cell_mask` uses for the whole mesh, evaluated here on a small
    retiled cavity so remesh_core_cavity's acceptance gate (see its call
    site) compares like-for-like against `bad_cell_mask`'s definition of
    "bad", not skewness alone. A fresh local retile is a handful to a few
    thousand cells (bounded by max_cavity_cells) - cheap to fully
    re-extract faces for, unlike re-validating the whole mesh.
    """
    from .face_extractor import FaceExtractor
    from ..schema.grid_nodes import NodeArray

    bad = validator.compute_cell_skewness(nodes, cells) > validator.thresholds['max_skewness']

    node_arr = NodeArray(
        x=np.ascontiguousarray(nodes[:, 0]),
        y=np.ascontiguousarray(nodes[:, 1]),
        z=np.ascontiguousarray(nodes[:, 2]),
    )
    # face_extractor logs several INFO/SUCCESS lines per call unconditionally
    # (no verbose= switch there, unlike fill_core_volume) - fine for the
    # normal one-call-per-mesh case, but this runs once per cavity CANDIDATE
    # (up to max_clusters_attempted of them, mostly rejected), so left
    # enabled it multiplies into tens of thousands of lines of routine noise
    # per repair pass on a real case with many small cavities (confirmed
    # directly: a single Stage B' pass produced 70K+ log lines this way).
    # Only this module's own per-cavity/summary lines (logged separately by
    # remesh_core_cavity itself) are actually useful at this granularity.
    logger.disable("autoflowcfd.grid.mesh_gen.face_extractor")
    try:
        faces = FaceExtractor.extract_faces(cells.astype(np.int32), node_arr)
    finally:
        logger.enable("autoflowcfd.grid.mesh_gen.face_extractor")
    diag = validator.compute_face_diagnostics(nodes, cells, faces)
    if len(diag['angle_deg']) > 0:
        face_bad = (
            (diag['angle_deg'] > validator.thresholds['max_orthogonality_angle'])
            | (diag['volume_ratio'] > validator.thresholds['max_adjacent_volume_ratio'])
        )
        bad[diag['owner'][face_bad]] = True
        bad[diag['neighbor'][face_bad]] = True

    return int(np.sum(bad))
