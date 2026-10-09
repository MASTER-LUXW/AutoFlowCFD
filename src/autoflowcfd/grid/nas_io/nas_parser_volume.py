"""外部生成的体网格 NAS 文件解析器。

和 parser_core.NASParser（读取 CTRIA3 面网格，体网格由本项目自己的
生成-体积 流程从零生成）不同，本模块读取的是别的工具已经生成好的
完整体网格（例如 ANSA 自身的体网格导出：GRID/GRID* + CTETRA + CPENTA 卡片；节点与面网格共用同一个解析器，
小字段/大字段/逗号分隔都支持）。

解析出来的 VolumeMeshData 的 BoundaryMap 是空的——外部生成的体网格通常
完全不带边界条件信息（ANSA 自身的体网格导出只标注 PSOLID 材料分区，不带
面边界条件），不像本项目自己的生成流程会在建网格的同时追踪边界来源。
从配套的面网格文件里反推边界分组（inlet/outlet/wall/...）是单独一步，
见 mesh_gen.mesh_boundary.map_boundaries_by_geometry——它必须按位置匹配
（KD-tree 最近质心），而不能按节点编号匹配，因为外部生成的网格自己的节点
编号和任何其他文件都没有对应关系。
"""

import numpy as np
from typing import Tuple
from loguru import logger

from ..structures import (
    NodeArray, TetrahedralCells, PrismCells, GridMetadata, VolumeMeshData, BoundaryMap,
)


def _cell_node_ids(line: str, n_nodes: int) -> list:
    """CTETRA/CPENTA 卡片的节点 id：固定小字段（列 25 起每 8 列一个）或逗号分隔（`CTETRA,EID,PID,G1,...`）。"""
    if "," in line:
        return [int(p) for p in line.split(",")[3:3 + n_nodes]]
    return [int(line[24 + 8 * k:32 + 8 * k]) for k in range(n_nodes)]


def _parse_cards(path: str) -> Tuple[np.ndarray, np.ndarray, list, list]:
    """节点经 `nas_parser_nodes.parse_nodes_from_nas`（与面网格同一个解析器：小字段固定/逗号/空白分隔与大字段
    `GRID*`）；再流式扫描一遍收集 CTETRA / CPENTA 的节点 id 行。

    2026-10-09 以前这里另有一份只认 8 字符小字段的 GRID 解析：本项目导出端改写大字段 `GRID*`（保留 10 位有效
    数字，见 `nas_export.py::_write_nodes`）之后，它在自家导出的体网格上报"找不到 GRID 卡片"。"""
    from .nas_parser_nodes import parse_nodes_from_nas

    nodes, id_to_index = parse_nodes_from_nas(path)
    if not id_to_index:
        raise ValueError(f"No GRID cards found in {path} - not a valid volume-mesh NAS file")
    node_ids_arr = np.empty(len(id_to_index), dtype=np.int64)
    for nid, idx in id_to_index.items():
        node_ids_arr[idx] = nid
    node_xyz_arr = np.column_stack([nodes.x, nodes.y, nodes.z]).astype(np.float64)

    tet_rows = []
    prism_rows = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            card = line[:8].split(",")[0].strip()
            if card == "CTETRA":
                tet_rows.append(_cell_node_ids(line, 4))
            elif card == "CPENTA":
                prism_rows.append(_cell_node_ids(line, 6))

    if not tet_rows and not prism_rows:
        raise ValueError(
            f"No CTETRA/CPENTA cards found in {path} - this looks like a surface mesh "
            f"(CTRIA3-only); use NASParser instead"
        )
    return node_ids_arr, node_xyz_arr, tet_rows, prism_rows


# 与 NASParser.AUTO_UNITS_MM_THRESHOLD 完全匹配——见该类的注释
# 了解推理（汽车外气动域以 mm 为单位读入数千，以米为单位读入几到几十）。
_AUTO_UNITS_MM_THRESHOLD = 50.0


def parse_volume_mesh_nas(path: str, units: str = 'mm') -> VolumeMeshData:
    """解析外部生成的体网格 NAS 文件 (GRID + CTETRA + CPENTA) 为
    VolumeMeshData。

    Args:
        path: 体网格 .nas 文件路径。
        units: 文件中坐标的长度单位——'mm'（默认，匹配 NASParser
            自身的默认值和 ANSA 的典型导出约定）、'm'（不缩放）
            或 'auto'（从原始包围盒范围检测，与 NASParser 自身的
            units='auto' 使用相同阈值/逻辑）。搞错这个不仅会扭曲
            网格——还会静默破坏 mesh_boundary.map_boundaries_by_
            geometry 对配套表面网格（NASParser 总是缩放到米）的
            最近质心匹配，因为每个体网格面然后坐在比任何表面
            边界面远约 1000 倍的位置，即使宽松容差也轻松超出。
            已直接确认：在真实案例上省略此缩放导致 0 of 39,352
            个外表面所属单元匹配到任何表面边界组。

    Returns:
        VolumeMeshData，BoundaryMap 为空 (groups={}, bc_types={})
        ——见本模块的文档字符串了解为什么边界归因是单独一步。
        四面体被重定向到正体积，与本项目自己的生成管线相同的
        方式；任何精确退化（近零体积）的单元被丢弃，匹配本项目
        对自己的生成网格应用的相同清理。

    Raises:
        ValueError: 无 GRID 卡片、无 CTETRA/CPENTA 卡片（例如误传了
            纯表面文件），或无效的 `units` 值。
    """
    from ..mesh_gen.tetgen.mesh_prism_to_tet import orient_tetrahedra
    from ..validation.quality_metrics import compute_prism_volumes

    if units not in ('mm', 'm', 'auto'):
        raise ValueError(f"units must be 'mm', 'm', or 'auto', got {units!r}")

    logger.info(f"Parsing external volume mesh: {path}")
    node_ids, node_xyz, tet_rows, prism_rows = _parse_cards(path)
    logger.info(
        f"Parsed {len(node_ids)} nodes, {len(tet_rows)} CTETRA, {len(prism_rows)} CPENTA"
    )

    raw_extent = float(np.max(node_xyz.max(axis=0) - node_xyz.min(axis=0)))
    if units == 'mm':
        scale_factor = 1e-3
    elif units == 'm':
        scale_factor = 1.0
    else:  # 'auto'
        if raw_extent > _AUTO_UNITS_MM_THRESHOLD:
            scale_factor = 1e-3
            logger.info(
                f"units='auto': raw bounding-box max extent={raw_extent:.4g} > "
                f"{_AUTO_UNITS_MM_THRESHOLD:g} -> assuming millimeters (scaling by 1e-3)"
            )
        else:
            scale_factor = 1.0
            logger.info(
                f"units='auto': raw bounding-box max extent={raw_extent:.4g} <= "
                f"{_AUTO_UNITS_MM_THRESHOLD:g} -> assuming the file is already in "
                f"meters (no scaling)"
            )
    node_xyz = node_xyz * scale_factor

    id_to_idx = np.full(int(node_ids.max()) + 1, -1, dtype=np.int64)
    id_to_idx[node_ids] = np.arange(len(node_ids))

    nodes_obj = NodeArray(
        x=np.ascontiguousarray(node_xyz[:, 0]),
        y=np.ascontiguousarray(node_xyz[:, 1]),
        z=np.ascontiguousarray(node_xyz[:, 2]),
    )

    tet_conn = np.zeros((0, 4), dtype=np.int64)
    if tet_rows:
        tet_conn = id_to_idx[np.array(tet_rows, dtype=np.int64)]
        if tet_conn.min() < 0:
            raise ValueError(f"{path}: CTETRA references a node id not defined by any GRID card")
        tet_conn = orient_tetrahedra(node_xyz, tet_conn.copy())
        tet_vol = TetrahedralCells.compute_volumes(nodes_obj, tet_conn)
        degenerate = np.abs(tet_vol) < 1e-20
        if np.any(degenerate):
            logger.warning(f"Dropping {int(degenerate.sum())} exactly-degenerate CTETRA cell(s)")
            tet_conn = tet_conn[~degenerate]
            tet_vol = tet_vol[~degenerate]
        neg = tet_vol < 0
        if np.any(neg):
            logger.warning(
                f"Dropping {int(neg.sum())} CTETRA cell(s) still negative-volume after "
                f"re-orientation (likely genuinely degenerate, not just misoriented)"
            )
            tet_conn = tet_conn[~neg]
            tet_vol = tet_vol[~neg]
    else:
        tet_vol = np.zeros(0, dtype=np.float64)

    prism_conn = np.zeros((0, 6), dtype=np.int64)
    prism_vol = np.zeros(0, dtype=np.float64)
    if prism_rows:
        prism_conn = id_to_idx[np.array(prism_rows, dtype=np.int64)]
        if prism_conn.min() < 0:
            raise ValueError(f"{path}: CPENTA references a node id not defined by any GRID card")
        prism_vol = compute_prism_volumes(node_xyz, prism_conn)
        degenerate_p = prism_vol < 1e-20
        if np.any(degenerate_p):
            logger.warning(f"Dropping {int(degenerate_p.sum())} exactly-degenerate CPENTA cell(s)")
            prism_conn = prism_conn[~degenerate_p]
            prism_vol = prism_vol[~degenerate_p]

    cells_obj = TetrahedralCells(
        connectivity=tet_conn.astype(np.int32), volumes=tet_vol.astype(np.float64)
    )
    prism_obj = (
        PrismCells(connectivity=prism_conn.astype(np.int32), volumes=prism_vol.astype(np.float64))
        if len(prism_conn) else None
    )

    boundaries_obj = BoundaryMap(groups={}, bc_types={})
    metadata = GridMetadata(
        node_count=len(node_xyz),
        cell_count=cells_obj.count + (prism_obj.count if prism_obj else 0),
        boundary_groups=[],
        file_format="external_volume_mesh",
    )
    volume_mesh = VolumeMeshData(
        nodes=nodes_obj, cells=cells_obj, boundaries=boundaries_obj,
        metadata=metadata, prism_cells=prism_obj,
    )
    logger.success(
        f"External volume mesh parsed: {volume_mesh.node_count} nodes, "
        f"{volume_mesh.cell_count} cells "
        f"({prism_obj.count if prism_obj else 0} prisms + {cells_obj.count} tets), "
        f"total volume {volume_mesh.total_volume:.6e} m^3"
    )
    return volume_mesh
