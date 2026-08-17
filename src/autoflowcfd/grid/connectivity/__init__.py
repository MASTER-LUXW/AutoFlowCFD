"""
AutoFlowCFD Grid - 连接性模块

网格连接性模块，包含面连接性和节点邻接关系计算。
"""

from autoflowcfd.grid.connectivity.face_connectivity import (
    FRFaceConnectivity,
    build_face_connectivity,
    tag_boundary_groups,
    CUBE_FACE_CODES,
    CUBE_FACE_NAMES,
)
from autoflowcfd.grid.connectivity.node_connectivity import build_node_adjacency

__all__ = [
    'FRFaceConnectivity',
    'build_face_connectivity',
    'tag_boundary_groups',
    'CUBE_FACE_CODES',
    'CUBE_FACE_NAMES',
    'build_node_adjacency',
]
