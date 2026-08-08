"""Stage A：质量门控的拉普拉斯平滑。

从 mesh_background.py 拆分出来以控制行数。
"""

import numpy as np
from typing import List, Optional, Tuple, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from ..schema.grid_faces import FaceData
    from ..validation.quality_validator import MeshQualityValidator


def run_stage_a_repair(
    merged_nodes: np.ndarray,
    merged_cells: np.ndarray,
    validator: 'MeshQualityValidator',
    pre_repair_faces: 'FaceData',
    overlap_bad_mask: Optional[np.ndarray],
    n_bl_cells: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Run Stage A: quality-gated Laplacian smoothing of skewed/non-orthogonal/
    volume-mismatched cells, restricted to movable (non-boundary) nodes.

    Args:
        merged_nodes: Node coordinates.
        merged_cells: Cell connectivity.
        validator: Quality validator instance.
        pre_repair_faces: Pre-extracted face data.
        overlap_bad_mask: Mask of cells with physical overlaps.
        n_bl_cells: Number of BL cells (used to protect interface).

    Returns:
        Tuple of (new_nodes, bad_cell_mask, repair_actions).
    """
    from .mesh_repair import smooth_bad_cells

    logger.info("Running Stage A: Quality-gated smoothing...")
    
    nodes_before_repair = merged_nodes
    merged_nodes, bad_mask, repair_actions = smooth_bad_cells(
        merged_nodes, merged_cells, validator, max_passes=5, 
        initial_faces=pre_repair_faces,
        extra_bad_mask=overlap_bad_mask, n_bl_cells=n_bl_cells,
    )
    
    mesh_changed_by_repair = not np.array_equal(nodes_before_repair, merged_nodes)
    
    if mesh_changed_by_repair:
        logger.info(f"Stage A completed: moved {int(np.sum(~np.all(nodes_before_repair == merged_nodes, axis=1)))} nodes.")
    else:
        logger.info("Stage A completed: no nodes moved.")
        
    return merged_nodes, bad_mask, repair_actions
