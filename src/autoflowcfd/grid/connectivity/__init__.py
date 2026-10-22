"""
AutoFlowCFD Grid - 连接性模块

网格连接性模块，包含面连接性和节点邻接关系计算。
"""

from autoflowcfd.grid.connectivity.face_connectivity import (
    FRFaceConnectivity,
    build_face_connectivity,
    CUBE_FACE_CODES,
    CUBE_FACE_NAMES,
)
from autoflowcfd.grid.connectivity.face_connectivity_boundary_tags import (  # noqa: F401
    face_group_codes,
    tag_boundary_groups_for_mesh,
    triangle_keys,
)

__all__ = [
    'FRFaceConnectivity',
    'build_face_connectivity',
    'face_group_codes',
    'tag_boundary_groups_for_mesh',
    'triangle_keys',
    'CUBE_FACE_CODES',
    'CUBE_FACE_NAMES',
]
