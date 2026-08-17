"""
AutoFlowCFD Grid - 网格生成 / 工具模块

网格生成辅助工具模块，包含边界识别、域分类、角点分割等实用函数。
"""

from autoflowcfd.grid.mesh_gen.utils.mesh_utils import (
    validate_surface_mesh,
    validate_bounding_box,
    compute_face_normals,
    check_reached_boundary,
)
from autoflowcfd.grid.mesh_gen.utils.mesh_boundary import (
    identify_boundaries_from_surface,
    map_surface_boundaries,
)

__all__ = [
    'validate_surface_mesh',
    'validate_bounding_box',
    'compute_face_normals',
    'check_reached_boundary',
    'identify_boundaries_from_surface',
    'map_surface_boundaries',
]
