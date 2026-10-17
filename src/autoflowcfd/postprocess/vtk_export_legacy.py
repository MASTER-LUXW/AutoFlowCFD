"""legacy VTK（.vtk）写入：ASCII 或符合规范的 big-endian 二进制载荷。"""

from pathlib import Path
from typing import Dict, List

import numpy as np
from loguru import logger

from .vtk_export_fields import FIELD_LABELS

_VTK_TRIANGLE = 5
_VTK_TETRA = 10
_VTK_WEDGE = 13  # 三棱柱：VTK 自己的节点顺序与本项目棱柱 (底面 0,1,2 / 顶面 3,4,5) 一致


def wl(f, text: str, binary: bool) -> None:
    """写一行头部文本（二进制模式下按 ASCII 编码）。"""
    f.write(text.encode('ascii') if binary else text)


def write_header(f, title: str, binary: bool) -> None:
    wl(f, "# vtk DataFile Version 3.0\n", binary)
    wl(f, f"{title}\n", binary)
    wl(f, ("BINARY\n" if binary else "ASCII\n"), binary)
    wl(f, "\n", binary)
    wl(f, "DATASET UNSTRUCTURED_GRID\n", binary)


def write_points(f, grid, binary: bool) -> None:
    nodes = grid.nodes
    coords = np.column_stack([nodes.x, nodes.y, nodes.z]).astype(np.float64)
    wl(f, f"POINTS {len(coords)} double\n", binary)
    if binary:
        f.write(coords.astype('>f8').tobytes())
        f.write(b'\n')
    else:
        np.savetxt(f, coords, fmt="%.6e")
    wl(f, "\n", binary)


def write_cells(f, grid, binary: bool) -> None:
    """体网格的全部单元（棱柱在前、四面体在后，与单元场同序）；面网格写三角形。"""
    prism = getattr(grid, 'prism_cells', None)
    blocks = []
    if prism is not None and len(prism.connectivity):
        blocks.append((np.asarray(prism.connectivity, dtype=np.int32), _VTK_WEDGE))
    conn = np.asarray(grid.cells.connectivity, dtype=np.int32)
    vtk_type = {3: _VTK_TRIANGLE, 4: _VTK_TETRA}.get(conn.shape[1])
    if vtk_type is None:
        raise ValueError(f"不支持的单元连接宽度 {conn.shape[1]}（三角形 3 或四面体 4）")
    blocks.append((conn, vtk_type))
    write_cell_blocks(f, blocks, binary)


def write_cell_blocks(f, blocks, binary: bool) -> None:
    """按块写 CELLS / CELL_TYPES：`blocks` 为 [(连接 (n, k), VTK 单元类型)]。"""
    n_cells = sum(len(c) for c, _ in blocks)
    total = sum(len(c) * (c.shape[1] + 1) for c, _ in blocks)
    wl(f, f"CELLS {n_cells} {total}\n", binary)
    for conn, _ in blocks:
        lines = np.hstack([np.full((len(conn), 1), conn.shape[1], dtype=np.int32), conn])
        if binary:
            f.write(lines.astype('>i4').tobytes())
        else:
            np.savetxt(f, lines, fmt="%d")
    if binary:
        f.write(b'\n')
    wl(f, "\n", binary)
    types = np.concatenate([np.full(len(c), t, dtype=np.int32) for c, t in blocks])
    wl(f, f"CELL_TYPES {n_cells}\n", binary)
    if binary:
        f.write(types.astype('>i4').tobytes())
        f.write(b'\n')
    else:
        np.savetxt(f, types.reshape(-1, 1), fmt="%d")
    wl(f, "\n", binary)


def write_scalar(f, name: str, values: np.ndarray, binary: bool, int_type: bool = False) -> None:
    wl(f, f"SCALARS {name} {'int' if int_type else 'double'} 1\n", binary)
    wl(f, "LOOKUP_TABLE default\n", binary)
    if binary:
        f.write(np.ascontiguousarray(values).astype('>i4' if int_type else '>f8').tobytes())
        f.write(b'\n')
    else:
        np.savetxt(f, values, fmt="%d" if int_type else "%.6e")
    wl(f, "\n", binary)


def write_vector(f, name: str, values: np.ndarray, binary: bool) -> None:
    wl(f, f"VECTORS {name} double\n", binary)
    if binary:
        f.write(np.ascontiguousarray(values, dtype=np.float64).astype('>f8').tobytes())
        f.write(b'\n')
    else:
        np.savetxt(f, values, fmt="%.6e")
    wl(f, "\n", binary)


def write_field_block(f, values: Dict[str, np.ndarray], binary: bool, labels: Dict[str, str] = FIELD_LABELS) -> None:
    """按插入顺序写一组场：二维 (n, 3) 写 VECTORS，一维写 SCALARS。"""
    for key, arr in values.items():
        name = labels.get(key, key)
        if np.ndim(arr) == 2:
            write_vector(f, name, arr, binary)
        else:
            write_scalar(f, name, arr, binary)


def write_field_data(f, entries: Dict[str, List[str]], binary: bool) -> None:
    """字符串图例写成 FIELD 块（ASCII）。VTK 9.3 的 legacy 读取器打不开含字符串 FIELD 块的二进制
    文件，二进制模式下图例改记日志（需要嵌入图例请用 XML 格式）。"""
    if not entries:
        return
    if binary:
        for name, values in entries.items():
            logger.info(f"{name}: " + ", ".join(values))
        return
    wl(f, f"FIELD FieldData {len(entries)}\n", binary)
    for name, values in entries.items():
        wl(f, f"{name} 1 {len(values)} string\n", binary)
        for v in values:
            wl(f, f"{v}\n", binary)
    wl(f, "\n", binary)


def export_volume(path: Path, grid, cell_data: Dict[str, np.ndarray], point_data: Dict[str, np.ndarray],
                  binary: bool) -> None:
    """体网格 + CELL_DATA（求解器单元值）+ POINT_DATA（体积加权节点值）。"""
    logger.info(f"写 legacy VTK（{'二进制' if binary else 'ASCII'}）：{path}")
    with open(path, 'wb' if binary else 'w') as f:
        write_header(f, f"AutoFlowCFD Export - {path.name}", binary)
        wl(f, "\n", binary)
        write_points(f, grid, binary)
        write_cells(f, grid, binary)
        if cell_data:
            wl(f, f"CELL_DATA {grid.cell_count}\n", binary)
            write_field_block(f, cell_data, binary)
        if point_data:
            wl(f, f"POINT_DATA {grid.node_count}\n", binary)
            write_field_block(f, point_data, binary)


def export_boundaries(path: Path, grid, tri_conn: np.ndarray, face_data: Dict[str, np.ndarray],
                      boundary_id: np.ndarray, type_id: np.ndarray, id_legend: List[str], type_legend: List[str],
                      binary: bool) -> None:
    """命名边界面片：三角形 + BoundaryID/BoundaryTypeID + owner 单元的场值。"""
    logger.info(f"写 legacy VTK 边界面片（{'二进制' if binary else 'ASCII'}）：{path}")
    with open(path, 'wb' if binary else 'w') as f:
        write_header(f, f"AutoFlowCFD Boundary Export - {path.name}", binary)
        write_field_data(f, {'BoundaryID_to_Name': id_legend, 'BoundaryTypeID_to_Name': type_legend}, binary)
        wl(f, "\n", binary)
        write_points(f, grid, binary)
        write_cell_blocks(f, [(np.asarray(tri_conn, dtype=np.int32), _VTK_TRIANGLE)], binary)
        wl(f, f"CELL_DATA {len(tri_conn)}\n", binary)
        write_scalar(f, "BoundaryID", boundary_id, binary, int_type=True)
        write_scalar(f, "BoundaryTypeID", type_id, binary, int_type=True)
        write_field_block(f, face_data, binary)
