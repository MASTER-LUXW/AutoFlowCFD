"""VTKExporter 的 XML VTK (.vtu) 写入逻辑。

从 vtk_export.py 中拆分出来（该文件超过 400 行硬性拆分阈值）：XML
格式的写入委托给 pyvista/VTK 自己的写入器，与 legacy 格式的手写
ASCII/二进制写入逻辑（vtk_export_legacy.py）完全独立，是一处干净的
拆分点。原来的 `VTKExporter` 方法体原样搬到这里，改写成以
`exporter`（原来的 `self`）为第一个参数的模块级函数；`VTKExporter`
上仍保留同名方法作为薄委托包装，外部调用方行为不变。
"""

import numpy as np
from pathlib import Path
from typing import Dict, List

from loguru import logger


def export_xml(exporter, output_path: Path, fields: List[str], binary: bool) -> None:
    """导出为基于 XML 的 VTK 格式（.vtu），当前主流 CFD 后处理工具
    采用的现代标准格式。从与 legacy 写入器相同的单元/节点场数据
    构建一个 pyvista.UnstructuredGrid，交给 VTK 自己的
    vtkXMLUnstructuredGridWriter 序列化（binary=True 时带
    binary+zlib 压缩）——这样不需要自己手写 XML appended-data 的
    二进制编码，pyvista/VTK 已经正确实现了这一点，ParaView 自身
    读写用的也是这一套。
    """
    import pyvista as pv

    logger.info(f"Exporting to XML VTK format ({'binary' if binary else 'ASCII'}): {output_path}")

    nodes = exporter.grid_data.nodes
    n_points = nodes.count
    points = np.column_stack([nodes.x, nodes.y, nodes.z]).astype(np.float64)

    conn = np.asarray(exporter.grid_data.cells.connectivity, dtype=np.int64)
    nodes_per_cell = conn.shape[1]
    cell_type = {3: pv.CellType.TRIANGLE, 4: pv.CellType.TETRA}.get(nodes_per_cell)
    if cell_type is None:
        raise ValueError(
            f"Unsupported cell connectivity width {nodes_per_cell} "
            f"(expected 3 for triangles or 4 for tetrahedra)"
        )

    grid = pv.UnstructuredGrid({cell_type: conn}, points)

    cell_fields = exporter._cell_fields(fields)
    point_fields = exporter._point_fields(cell_fields, n_points)
    for key, arr in cell_fields.items():
        grid.cell_data[exporter._FIELD_LABELS[key]] = arr
    for key, arr in point_fields.items():
        grid.point_data[exporter._FIELD_LABELS[key]] = arr

    grid.save(str(output_path), binary=binary)
    logger.info("XML VTK file written successfully")


def export_boundaries_xml(
    exporter, output_path: Path, fields: List[str], tri_conn: np.ndarray,
    boundary_fields: Dict[str, np.ndarray], boundary_id: np.ndarray, type_id: np.ndarray,
    id_legend: List[str], type_legend: List[str], binary: bool,
) -> None:
    """把边界面片导出为 .vtu——见 export_boundaries。
    BoundaryID/BoundaryTypeID -> 名称对照表放在 field_data（全局
    元数据，不是逐单元）里：已实测验证，逐单元的*字符串*类型
    CELL_DATA 数组经过 VTK XML 写入器/读取器往返后不会保留（数组
    列出来了，但读回时是空指针），而 field_data 字符串数组能作为
    vtkStringArray 正确往返——无论是直接通过
    vtkXMLUnstructuredGridReader 还是通过 pyvista.read()。
    """
    import pyvista as pv

    logger.info(f"Exporting boundary patches to XML VTK format ({'binary' if binary else 'ASCII'}): {output_path}")

    nodes = exporter.grid_data.nodes
    points = np.column_stack([nodes.x, nodes.y, nodes.z]).astype(np.float64)

    grid = pv.UnstructuredGrid({pv.CellType.TRIANGLE: np.asarray(tri_conn, dtype=np.int64)}, points)
    grid.cell_data['BoundaryID'] = boundary_id
    grid.cell_data['BoundaryTypeID'] = type_id
    for key, arr in boundary_fields.items():
        grid.cell_data[exporter._FIELD_LABELS[key]] = arr
    grid.field_data['BoundaryID_to_Name'] = np.array(id_legend)
    grid.field_data['BoundaryTypeID_to_Name'] = np.array(type_legend)

    grid.save(str(output_path), binary=binary)
    logger.info("XML VTK boundary file written successfully")
