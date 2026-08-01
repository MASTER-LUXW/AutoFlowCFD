"""Grid data structures for CFD mesh.

This module provides backward compatibility by re-exporting from submodules.
For new code, import directly from:
    - autoflowcfd.grid.schema.grid_nodes
    - autoflowcfd.grid.schema.grid_cells
    - autoflowcfd.grid.schema.grid_boundaries
    - autoflowcfd.grid.schema.grid_metadata
    - autoflowcfd.grid.schema.grid_faces
    - autoflowcfd.grid.schema.grid_data
"""

# Re-export from submodules for backward compatibility
from .schema.grid_nodes import NodeArray, CupyNodeArray
from .schema.grid_cells import CellArray, CupyCellArray, TetrahedralCells
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
    'BoundaryMap',
    'GridMetadata',
    'FaceData',
    'GridData',
    'CupyGridData',
    'VolumeMeshData',
]
