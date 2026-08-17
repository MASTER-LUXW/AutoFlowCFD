"""
AutoFlowCFD Grid - 弯曲映射模块

高阶网格弯曲映射模块，包含四面体/棱柱到物理空间的映射、雅可比计算和方向校正。
"""

from autoflowcfd.grid.curved_mapping.curved_mapping import (
    CurvedMapping,
    MeshDistortionError,
    TET_CUBE_FACES,
    PRISM_CUBE_FACES,
    map_tet_to_physical,
    map_prism_to_physical,
)
from autoflowcfd.grid.curved_mapping.curved_mapping_orientation import (
    fix_tet_orientation,
    fix_prism_orientation,
)

__all__ = [
    'CurvedMapping',
    'MeshDistortionError',
    'TET_CUBE_FACES',
    'PRISM_CUBE_FACES',
    'map_tet_to_physical',
    'map_prism_to_physical',
    'fix_tet_orientation',
    'fix_prism_orientation',
]
