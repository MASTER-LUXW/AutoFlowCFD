"""Grid data structures for CFD mesh.

This module provides backward compatibility by re-exporting from submodules.
For new code, import directly from:
    - autoflowcfd.grid.grid_nodes
    - autoflowcfd.grid.grid_cells
    - autoflowcfd.grid.grid_boundaries
    - autoflowcfd.grid.grid_metadata
    - autoflowcfd.grid.grid_faces
    - autoflowcfd.grid.grid_data
"""

# Re-export from submodules for backward compatibility
from .grid_nodes import NodeArray, CupyNodeArray
from .grid_cells import CellArray, CupyCellArray, TetrahedralCells
from .grid_boundaries import BoundaryMap
from .grid_metadata import GridMetadata
from .grid_faces import FaceData
from .grid_data import GridData, CupyGridData, VolumeMeshData

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
