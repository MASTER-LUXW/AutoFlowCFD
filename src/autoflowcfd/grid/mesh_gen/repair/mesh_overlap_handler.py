"""物理重叠检测与坏单元掩码计算。

从 mesh_background.py 拆分出来以控制行数。
"""

import numpy as np
from typing import Optional, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from ...validation.quality_validator import MeshQualityValidator


def compute_prism_aware_overlap_bad_tet_mask(
    merged_nodes: np.ndarray, prism_cells: np.ndarray, merged_cells: np.ndarray,
) -> Optional[np.ndarray]:
    """在全混合（棱柱+四面体）面集上进行物理重叠检查。

    返回一个 (len(merged_cells),) 的 bool 掩码，标记哪些四面体单元被涉及
    ——这是阶段 A/B' 唯一能操作的一侧（真正的棱柱完全不在其范围内）。

    为什么需要这个，而不是与 validator.validate(...) 中已运行的纯 tet 重叠
    检查冗余：纯 tet 检查只能看到 merged_cells（transition + core 四面体），
    因此在 BL/core 界面处一个梯度不良的扁平四面体，如果物理上延伸到附近
    的真正棱柱单元，在纯 tet 检查中是看不到的重叠缺陷。
    """
    n_prism = len(prism_cells)
    if n_prism == 0:
        return None
        
    from ...schema.grid_nodes import NodeArray
    from ..extraction.face_extractor import FaceExtractor
    from ...validation.mesh_overlap_check import check_face_overlap_and_proximity

    node_arr = NodeArray.from_array(merged_nodes)
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
    """合并纯 tet 重叠和棱柱感知重叠为一个统一的坏单元掩码。"""
    overlap_bad_mask = None
    
    if initial_report.overlapping_cell_ids is not None and len(initial_report.overlapping_cell_ids):
        overlap_bad_mask = np.zeros(len(merged_cells), dtype=bool)
        overlap_bad_mask[initial_report.overlapping_cell_ids] = True

    # 补充棱柱感知检查
    prism_overlap_mask = compute_prism_aware_overlap_bad_tet_mask(merged_nodes, prism_cells, merged_cells)
    if prism_overlap_mask is not None:
        if overlap_bad_mask is None:
            overlap_bad_mask = prism_overlap_mask
        else:
            overlap_bad_mask |= prism_overlap_mask
            
    return overlap_bad_mask
