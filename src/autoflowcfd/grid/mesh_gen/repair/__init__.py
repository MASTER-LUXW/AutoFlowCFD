"""
AutoFlowCFD Grid - 网格生成 / 修复模块

网格修复模块，包含非流形修复、空腔处理和重叠处理。
"""

from autoflowcfd.grid.mesh_gen.repair.mesh_repair import (
    compute_movable_node_mask,
    smooth_bad_cells,
)

__all__ = [
    'compute_movable_node_mask',
    'smooth_bad_cells',
]
