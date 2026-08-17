"""阶段 A：质量门控的拉普拉斯平滑。

从 mesh_background.py 拆分出来以控制行数。
"""

import numpy as np
from typing import List, Optional, Tuple, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from ...schema.grid_faces import FaceData
    from ...validation.quality_validator import MeshQualityValidator


def run_stage_a_repair(
    merged_nodes: np.ndarray,
    merged_cells: np.ndarray,
    validator: 'MeshQualityValidator',
    pre_repair_faces: 'FaceData',
    overlap_bad_mask: Optional[np.ndarray],
    n_bl_cells: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """执行阶段 A：质量门控的拉普拉斯平滑。

    对倾斜/非正交/体积不匹配的坏单元进行平滑，仅移动可动节点（非边界节点）。

    Args:
        merged_nodes: 节点坐标数组。
        merged_cells: 单元连接数组。
        validator: 质量验证器实例。
        pre_repair_faces: 预提取的面数据。
        overlap_bad_mask: 物理重叠单元的掩码。
        n_bl_cells: BL 单元数量（用于保护界面）。

    Returns:
        (新节点数组, 坏单元掩码, 修复动作列表) 的元组。
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
