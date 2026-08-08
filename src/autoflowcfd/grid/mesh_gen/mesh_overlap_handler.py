"""物理重叠检测与坏单元掩码计算。

从 mesh_background.py 拆分出来以控制行数。
"""

import numpy as np
from typing import Optional, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from ..schema.grid_faces import FaceData
    from ..validation.quality_validator import MeshQualityValidator


def compute_prism_aware_overlap_bad_tet_mask(
    merged_nodes: np.ndarray, prism_cells: np.ndarray, merged_cells: np.ndarray,
) -> Optional[np.ndarray]:
    """Physical-overlap check over the FULL mixed (prism+tet) face set,
    returning a (len(merged_cells),) bool mask of which TET cells are
    implicated - the only side Stage A/B' can act on (true prisms are
    entirely outside their scope).

    Why this is necessary and not redundant with the ordinary tet-only
    overlap check already run as part of validator.validate(...): a
    tet-only check can only ever see merged_cells (transition + core
    tets), so a badly-graded sliver tet at the BL/core interface that's
    elongated enough to physically reach back into a nearby true-prism cell
    is invisible to it as an overlap defect.
    """
    n_prism = len(prism_cells)
    if n_prism == 0:
        return None
        
    from ..schema.grid_nodes import NodeArray
    from .face_extractor import FaceExtractor
    from ..validation.mesh_overlap_check import check_face_overlap_and_proximity

    node_arr = NodeArray(
        x=np.ascontiguousarray(merged_nodes[:, 0]),
        y=np.ascontiguousarray(merged_nodes[:, 1]),
        z=np.ascontiguousarray(merged_nodes[:, 2]),
    )
    mixed_faces = FaceExtractor.extract_faces_mixed(
        prism_cells, merged_cells.astype(np.int64), node_arr
    )
    mixed_report = check_face_overlap_and_proximity(merged_nodes, merged_cells, faces=mixed_faces)
    
    if not len(mixed_report.overlapping_cell_ids):
        return None
        
    global_ids = mixed_report.overlapping_cell_ids
    tet_ids = global_ids[global_ids >= n_prism] - n_prism
    
    if len(tet_ids) == 0:
        return None
        
    mask = np.zeros(len(merged_cells), dtype=bool)
    mask[tet_ids] = True
    
    logger.warning(
        f"Prism-aware overlap check: {len(tet_ids)} core tet cell(s) physically "
        f"overlap a BL prism or another tet across the full mixed mesh - "
        f"invisible to the tet-only overlap check, added to the repair target set"
    )
    return mask


def compute_extra_bad_mask(
    validator: 'MeshQualityValidator',
    initial_report,
    merged_nodes: np.ndarray,
    prism_cells: np.ndarray,
    merged_cells: np.ndarray,
) -> Optional[np.ndarray]:
    """Combine tet-only overlap and prism-aware overlap into a single bad mask."""
    overlap_bad_mask = None
    
    if initial_report.overlapping_cell_ids is not None and len(initial_report.overlapping_cell_ids):
        overlap_bad_mask = np.zeros(len(merged_cells), dtype=bool)
        overlap_bad_mask[initial_report.overlapping_cell_ids] = True

    # Supplement with the prism-aware check
    prism_overlap_mask = compute_prism_aware_overlap_bad_tet_mask(merged_nodes, prism_cells, merged_cells)
    if prism_overlap_mask is not None:
        if overlap_bad_mask is None:
            overlap_bad_mask = prism_overlap_mask
        else:
            overlap_bad_mask |= prism_overlap_mask
            
    return overlap_bad_mask
