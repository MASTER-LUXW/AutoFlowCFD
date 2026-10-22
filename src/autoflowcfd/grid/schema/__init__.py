"""网格数据模式：节点、单元、边界、面的 SoA 布局数据类。

外部代码从本包导入（`from autoflowcfd.grid.schema import GridData, VolumeMeshData`）。2026-10-22（I-123）以前另有
`autoflowcfd.grid.structures` 再导出一遍（"向后兼容"垫片）与只在那里出现的 CuPy 版网格类，均已删除：GPU 求解器经
`core/gpu/array_manager.py` 上传求解器网格，不经过这些数据类。
"""

from .grid_boundaries import BoundaryMap
from .grid_cells import CellArray, PrismCells, TetrahedralCells
from .grid_data import GridData, VolumeMeshData
from .grid_faces import FaceData
from .grid_metadata import GridMetadata
from .grid_nodes import NodeArray

__all__ = [
    "BoundaryMap",
    "CellArray",
    "FaceData",
    "GridData",
    "GridMetadata",
    "NodeArray",
    "PrismCells",
    "TetrahedralCells",
    "VolumeMeshData",
]
