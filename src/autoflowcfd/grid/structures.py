"""CFD 网格数据结构。

本模块通过从子模块重新导出提供向后兼容。
新代码请直接从以下模块导入:
    - autoflowcfd.grid.schema.grid_nodes
    - autoflowcfd.grid.schema.grid_cells
    - autoflowcfd.grid.schema.grid_boundaries
    - autoflowcfd.grid.schema.grid_metadata
    - autoflowcfd.grid.schema.grid_faces
    - autoflowcfd.grid.schema.grid_data
"""

# Re-export from submodules for backward compatibility
from .schema.grid_nodes import NodeArray, CupyNodeArray
from .schema.grid_cells import CellArray, CupyCellArray, TetrahedralCells, PrismCells
from .schema.grid_boundaries import BoundaryMap
from .schema.grid_metadata import GridMetadata
from .schema.grid_faces import FaceData
from .schema.grid_data import GridData, CupyGridData, VolumeMeshData

__all__ = [
    'NodeArray',
    'CupyNodeArray',
    'CellArray',
    'CupyCellArray',
    'TetrahedralCells',
    'PrismCells',
    'BoundaryMap',
    'GridMetadata',
    'FaceData',
    'GridData',
    'CupyGridData',
    'VolumeMeshData',
]
