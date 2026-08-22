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

    # 真实 bug（已修复，2026-08-21）：此前这里只读 `exporter.grid_data.
    # cells.connectivity`（四面体），完全没有处理棱柱——本项目每一份体
    # 网格的边界层都是棱柱挤出（BL prism extrusion 是核心、始终启用的
    # 功能，见 grid/mesh_gen 文档），意味着这条路径此前对*任何*真实
    # 网格都会静默丢弃全部棱柱单元，只导出四面体核心区（真实复现：
    # cube_demo 79万单元网格，136980 个棱柱单元 100% 丢失，只留下
    # 654512 个四面体——不多不少恰好是四面体总数，与逐单元场数组
    # （791492 个值，两种单元都算在内）大小对不上，之前唯一表现是
    # pyvista/VTK 自己的内部校验直接报错拒绝写入，不是"导出了一份看起
    # 来正常、实际缺了近壁边界层的错误文件"，但 legacy 格式写入器
    # （vtk_export_legacy.py::write_cells_mixed）本来就已经正确实现了
    # 棱柱+四面体混合网格的写入，只是 XML（.vtu，这个项目自己文档里
    # 说的"当前主流格式"、`--binary`默认走的格式）这条路径没有对齐。
    # 用同一个 pv.UnstructuredGrid 多单元类型字典构造（WEDGE 在前、
    # TETRA 在后，与本项目全局单元索引约定——棱柱在前、四面体在后，
    # 见 vtk_export.py::_VTK_WEDGE 文档——完全一致，已用 pyvista 0.44.2
    # 验证过重建后的单元顺序确实遵从字典插入顺序），修复为同时处理
    # 两种单元类型，不再假设网格只有四面体。
    prism_cells_obj = getattr(exporter.grid_data, 'prism_cells', None)
    tet_conn = np.asarray(exporter.grid_data.cells.connectivity, dtype=np.int64)

    if prism_cells_obj is not None and len(prism_cells_obj.connectivity) > 0:
        prism_conn = np.asarray(prism_cells_obj.connectivity, dtype=np.int64)
        grid = pv.UnstructuredGrid(
            {pv.CellType.WEDGE: prism_conn, pv.CellType.TETRA: tet_conn}, points
        )
    else:
        nodes_per_cell = tet_conn.shape[1]
        cell_type = {3: pv.CellType.TRIANGLE, 4: pv.CellType.TETRA}.get(nodes_per_cell)
        if cell_type is None:
            raise ValueError(
                f"Unsupported cell connectivity width {nodes_per_cell} "
                f"(expected 3 for triangles or 4 for tetrahedra)"
            )
        grid = pv.UnstructuredGrid({cell_type: tet_conn}, points)

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
