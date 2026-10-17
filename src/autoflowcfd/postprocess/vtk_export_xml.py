"""XML VTK（.vtu）写入：委托 pyvista / VTK 自己的 vtkXMLUnstructuredGridWriter（binary 时带 zlib 压缩）。"""

from pathlib import Path
from typing import Dict, List

import numpy as np
from loguru import logger

from .vtk_export_fields import FIELD_LABELS


def _points(grid) -> np.ndarray:
    nodes = grid.nodes
    return np.column_stack([nodes.x, nodes.y, nodes.z]).astype(np.float64)


def export_volume(path: Path, grid, cell_data: Dict[str, np.ndarray], point_data: Dict[str, np.ndarray],
                  binary: bool) -> None:
    import pyvista as pv

    logger.info(f"写 XML VTK（{'二进制' if binary else 'ASCII'}）：{path}")
    cells = {}
    prism = getattr(grid, 'prism_cells', None)
    if prism is not None and len(prism.connectivity):
        cells[pv.CellType.WEDGE] = np.asarray(prism.connectivity, dtype=np.int64)
    conn = np.asarray(grid.cells.connectivity, dtype=np.int64)
    cell_type = {3: pv.CellType.TRIANGLE, 4: pv.CellType.TETRA}.get(conn.shape[1])
    if cell_type is None:
        raise ValueError(f"不支持的单元连接宽度 {conn.shape[1]}（三角形 3 或四面体 4）")
    cells[cell_type] = conn
    out = pv.UnstructuredGrid(cells, _points(grid))
    for key, arr in cell_data.items():
        out.cell_data[FIELD_LABELS.get(key, key)] = arr
    for key, arr in point_data.items():
        out.point_data[FIELD_LABELS.get(key, key)] = arr
    out.save(str(path), binary=binary)


def export_boundaries(path: Path, grid, tri_conn: np.ndarray, face_data: Dict[str, np.ndarray],
                      boundary_id: np.ndarray, type_id: np.ndarray, id_legend: List[str], type_legend: List[str],
                      binary: bool) -> None:
    import pyvista as pv

    logger.info(f"写 XML VTK 边界面片（{'二进制' if binary else 'ASCII'}）：{path}")
    out = pv.UnstructuredGrid({pv.CellType.TRIANGLE: np.asarray(tri_conn, dtype=np.int64)}, _points(grid))
    out.cell_data['BoundaryID'] = boundary_id
    out.cell_data['BoundaryTypeID'] = type_id
    for key, arr in face_data.items():
        out.cell_data[FIELD_LABELS.get(key, key)] = arr
    out.field_data['BoundaryID_to_Name'] = np.array(id_legend)
    out.field_data['BoundaryTypeID_to_Name'] = np.array(type_legend)
    out.save(str(path), binary=binary)
