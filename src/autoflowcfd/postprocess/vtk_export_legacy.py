"""VTKExporter 的 legacy VTK (.vtk) ASCII/二进制写入逻辑。

从 vtk_export.py 中拆分出来（该文件超过 400 行硬性拆分阈值）：legacy
VTK 格式（DataFile Version 3.0，CELLS/CELL_TYPES 经典布局）的写入
细节自成一体，是文件里占比最大的一块。原来的 `VTKExporter` 方法体
原样搬到这里，改写成以 `exporter`（原来的 `self`）为第一个参数的
模块级函数；`VTKExporter` 上仍保留同名方法作为薄委托包装，外部调用方
（包括 post_commands.py 里直接调用 `exporter._write_points(...)` 这类
用法）行为不变。
"""

import numpy as np
from pathlib import Path
from typing import Dict, List

from loguru import logger


def export_legacy(exporter, output_path: Path, fields: List[str], binary: bool) -> None:
    """导出为 legacy VTK 格式（VTK Legacy 规范，DataFile Version
    3.0——经典的 CELLS/CELL_TYPES 布局，不是 VTK 9 更新的
    OFFSETS/CONNECTIVITY 变体，以便和旧版读取器保持最大兼容性）。"""
    logger.info(f"Exporting to legacy VTK format ({'binary' if binary else 'ASCII'}): {output_path}")

    n_points = exporter.grid_data.nodes.count
    n_cells = exporter.grid_data.cell_count
    cell_fields = exporter._cell_fields(fields)
    point_fields = exporter._point_fields(cell_fields, n_points)

    try:
        mode = 'wb' if binary else 'w'
        with open(output_path, mode) as f:
            exporter._wl(f, "# vtk DataFile Version 3.0\n", binary)
            exporter._wl(f, f"AutoFlowCFD Export - {output_path.name}\n", binary)
            exporter._wl(f, ("BINARY\n" if binary else "ASCII\n"), binary)
            exporter._wl(f, "\n", binary)
            exporter._wl(f, "DATASET UNSTRUCTURED_GRID\n", binary)
            exporter._wl(f, "\n", binary)

            exporter._write_points(f, binary)
            exporter._write_cells(f, binary)

            # CELL_DATA：原始、未插值的求解器值。
            exporter._wl(f, f"CELL_DATA {n_cells}\n", binary)
            exporter._write_field_block(f, fields, cell_fields, binary)

            # POINT_DATA：体积加权插值，用于平滑等值面渲染。
            exporter._wl(f, f"POINT_DATA {n_points}\n", binary)
            exporter._write_field_block(f, fields, point_fields, binary)

        logger.info("Legacy VTK file written successfully")

    except IOError as e:
        logger.error(f"Failed to write VTK file: {e}")
        raise


def wl(f, text: str, binary: bool) -> None:
    """写一行头部/关键字，二进制模式下编码为字节。"""
    f.write(text.encode('ascii') if binary else text)


def write_points(exporter, f, binary: bool) -> None:
    nodes = exporter.grid_data.nodes
    n_points = nodes.count
    coords = np.column_stack([nodes.x, nodes.y, nodes.z]).astype(np.float64)

    exporter._wl(f, f"POINTS {n_points} double\n", binary)
    if binary:
        f.write(coords.astype('>f8').tobytes())
        f.write(b'\n')
    else:
        np.savetxt(f, coords, fmt="%.6e")
    exporter._wl(f, "\n", binary)


def write_cells(exporter, f, binary: bool) -> None:
    """把单元连接关系写入 VTK 文件。

    从 connectivity 数组自身的形状检测每个单元实际的节点数
    （3=三角形，4=四面体）——如果设置了 grid_data.prism_cells，则
    先写三棱柱（6 节点 wedge，全局索引 [0, n_prism)），再写四面体
    （[n_prism, n_prism+n_tet)），与本项目的全局单元索引约定一致
    （见 PrismCells/face_extractor.extract_faces_mixed）。
    """
    prism_cells_obj = getattr(exporter.grid_data, 'prism_cells', None)
    if prism_cells_obj is not None:
        exporter._write_cells_mixed(f, prism_cells_obj.connectivity, exporter.grid_data.cells.connectivity, binary)
    else:
        exporter._write_cells_from(f, exporter.grid_data.cells.connectivity, binary)


def write_cells_mixed(exporter, f, prism_conn: np.ndarray, tet_conn: np.ndarray, binary: bool) -> None:
    """为三棱柱(wedge)+四面体混合网格写 CELLS/CELL_TYPES——legacy
    VTK 的 CELLS 格式里每行可以有不同的顶点数（每行开头的整数就是
    该行的顶点数），所以三棱柱和四面体可以直接拼接成一个块；
    CELL_TYPES 携带每行的类型代码（_VTK_WEDGE 还是 _VTK_TETRA）。"""
    from .vtk_export import _VTK_WEDGE, _VTK_TETRA

    prism_conn = np.asarray(prism_conn, dtype=np.int32)
    tet_conn = np.asarray(tet_conn, dtype=np.int32)
    n_prism = len(prism_conn)
    n_tet = len(tet_conn)
    n_cells = n_prism + n_tet
    total_ints = n_prism * 7 + n_tet * 5  # (1 个计数 + 6 个顶点) 或 (1 个计数 + 4 个顶点)

    exporter._wl(f, f"CELLS {n_cells} {total_ints}\n", binary)
    if binary:
        if n_prism:
            prism_lines = np.hstack([np.full((n_prism, 1), 6, dtype=np.int32), prism_conn])
            f.write(prism_lines.astype('>i4').tobytes())
        if n_tet:
            tet_lines = np.hstack([np.full((n_tet, 1), 4, dtype=np.int32), tet_conn])
            f.write(tet_lines.astype('>i4').tobytes())
        f.write(b'\n')
    else:
        if n_prism:
            prism_lines = np.hstack([np.full((n_prism, 1), 6, dtype=np.int32), prism_conn])
            np.savetxt(f, prism_lines, fmt="%d")
        if n_tet:
            tet_lines = np.hstack([np.full((n_tet, 1), 4, dtype=np.int32), tet_conn])
            np.savetxt(f, tet_lines, fmt="%d")
    exporter._wl(f, "\n", binary)

    cell_types = np.concatenate([
        np.full(n_prism, _VTK_WEDGE, dtype=np.int32),
        np.full(n_tet, _VTK_TETRA, dtype=np.int32),
    ])
    exporter._wl(f, f"CELL_TYPES {n_cells}\n", binary)
    if binary:
        f.write(cell_types.astype('>i4').tobytes())
        f.write(b'\n')
    else:
        np.savetxt(f, cell_types.reshape(-1, 1), fmt="%d")
    exporter._wl(f, "\n", binary)


def write_cells_from(exporter, f, conn: np.ndarray, binary: bool) -> None:
    """从显式的 connectivity 数组写 CELLS/CELL_TYPES——同时供整体
    体网格导出（_write_cells）和边界面导出（同一份节点数组上的
    另一组更小的三角形）共用。"""
    from .vtk_export import _VTK_TRIANGLE, _VTK_TETRA

    conn = np.asarray(conn, dtype=np.int32)
    n_cells = conn.shape[0]
    nodes_per_cell = conn.shape[1]

    vtk_type = {3: _VTK_TRIANGLE, 4: _VTK_TETRA}.get(nodes_per_cell)
    if vtk_type is None:
        raise ValueError(
            f"Unsupported cell connectivity width {nodes_per_cell} "
            f"(expected 3 for triangles or 4 for tetrahedra)"
        )

    counts = np.full((n_cells, 1), nodes_per_cell, dtype=np.int32)
    cell_lines = np.hstack([counts, conn])

    exporter._wl(f, f"CELLS {n_cells} {n_cells * (nodes_per_cell + 1)}\n", binary)
    if binary:
        f.write(cell_lines.astype('>i4').tobytes())
        f.write(b'\n')
    else:
        np.savetxt(f, cell_lines, fmt="%d")
    exporter._wl(f, "\n", binary)

    exporter._wl(f, f"CELL_TYPES {n_cells}\n", binary)
    types_arr = np.full(n_cells, vtk_type, dtype=np.int32)
    if binary:
        f.write(types_arr.astype('>i4').tobytes())
        f.write(b'\n')
    else:
        np.savetxt(f, types_arr, fmt="%d")
    exporter._wl(f, "\n", binary)


def write_field_block(exporter, f, fields: List[str], values: Dict[str, np.ndarray], binary: bool) -> None:
    if 'velocity' in fields and 'velocity' in values:
        exporter._write_vector(f, "Velocity", values['velocity'], binary)
    if 'pressure' in fields and 'pressure' in values:
        exporter._write_scalar(f, "Pressure", values['pressure'], binary)
    if 'k' in fields and 'k' in values:
        exporter._write_scalar(f, "TurbulentKineticEnergy", values['k'], binary)
    if 'omega' in fields and 'omega' in values:
        exporter._write_scalar(f, "SpecificDissipationRate", values['omega'], binary)
    if 'nut' in fields and 'nut' in values:
        exporter._write_scalar(f, "TurbulentViscosity", values['nut'], binary)


def write_scalar(exporter, f, name: str, values: np.ndarray, binary: bool, int_type: bool = False) -> None:
    vtk_type_name = "int" if int_type else "double"
    np_dtype = '>i4' if int_type else '>f8'
    exporter._wl(f, f"SCALARS {name} {vtk_type_name} 1\n", binary)
    exporter._wl(f, "LOOKUP_TABLE default\n", binary)
    if binary:
        f.write(np.ascontiguousarray(values).astype(np_dtype).tobytes())
        f.write(b'\n')
    else:
        np.savetxt(f, values, fmt="%d" if int_type else "%.6e")
    exporter._wl(f, "\n", binary)


def write_field_data_legacy(exporter, f, entries: Dict[str, List[str]], binary: bool) -> None:
    """写一个 FIELD FieldData 块（全局元数据，例如
    BoundaryID->名称对照表）。

    只在 ASCII 模式下写出：实测发现，VTK 9.3 自己的
    vtkUnstructuredGridReader 只要文件的数据模式是 BINARY，就无法
    解析**任何**包含字符串类型 FIELD 块的 legacy 文件——用一个独立
    于本写入器的最小手写复现确认过（不管 field 块在二进制载荷之前
    还是之后，都是同样失败；同一个块放在 ASCII 模式文件里能正确
    读回）。与其生成一个 VTK 自己的读取器都打不开的二进制 .vtk，
    binary=True 时改为把对照表记录到日志——无论如何，数值型的
    BoundaryID/BoundaryTypeID CELL_DATA 都不受影响。XML（.vtu）
    没有这个问题（见 _export_boundaries_xml），是这类导出推荐使用
    的格式。
    """
    if not entries:
        return
    if binary:
        for name, values in entries.items():
            logger.info(f"{name}: " + ", ".join(values))
        return
    exporter._wl(f, f"FIELD FieldData {len(entries)}\n", binary)
    for name, values in entries.items():
        exporter._wl(f, f"{name} 1 {len(values)} string\n", binary)
        for v in values:
            exporter._wl(f, f"{v}\n", binary)
    exporter._wl(f, "\n", binary)


def write_vector(exporter, f, name: str, values: np.ndarray, binary: bool) -> None:
    exporter._wl(f, f"VECTORS {name} double\n", binary)
    if binary:
        f.write(np.ascontiguousarray(values, dtype=np.float64).astype('>f8').tobytes())
        f.write(b'\n')
    else:
        np.savetxt(f, values, fmt="%.6e")
    exporter._wl(f, "\n", binary)


def export_boundaries_legacy(
    exporter, output_path: Path, fields: List[str], tri_conn: np.ndarray,
    boundary_fields: Dict[str, np.ndarray], boundary_id: np.ndarray, type_id: np.ndarray,
    id_legend: List[str], type_legend: List[str], binary: bool,
) -> None:
    logger.info(f"Exporting boundary patches to legacy VTK format ({'binary' if binary else 'ASCII'}): {output_path}")
    n_tri = tri_conn.shape[0]

    try:
        mode = 'wb' if binary else 'w'
        with open(output_path, mode) as f:
            exporter._wl(f, "# vtk DataFile Version 3.0\n", binary)
            exporter._wl(f, f"AutoFlowCFD Boundary Export - {output_path.name}\n", binary)
            exporter._wl(f, ("BINARY\n" if binary else "ASCII\n"), binary)
            exporter._wl(f, "\n", binary)
            exporter._wl(f, "DATASET UNSTRUCTURED_GRID\n", binary)
            exporter._write_field_data_legacy(f, {
                'BoundaryID_to_Name': id_legend,
                'BoundaryTypeID_to_Name': type_legend,
            }, binary)
            exporter._wl(f, "\n", binary)

            exporter._write_points(f, binary)
            exporter._write_cells_from(f, tri_conn, binary)

            exporter._wl(f, f"CELL_DATA {n_tri}\n", binary)
            exporter._write_scalar(f, "BoundaryID", boundary_id, binary, int_type=True)
            exporter._write_scalar(f, "BoundaryTypeID", type_id, binary, int_type=True)
            exporter._write_field_block(f, fields, boundary_fields, binary)

        logger.info("Legacy VTK boundary file written successfully")

    except IOError as e:
        logger.error(f"Failed to write VTK boundary file: {e}")
        raise
