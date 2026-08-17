"""外部生成的体网格 NAS 文件解析器。

和 parser_core.NASParser（读取 CTRIA3 面网格，体网格由本项目自己的
生成-体积 流程从零生成）不同，本模块读取的是别的工具已经生成好的
完整体网格（例如 ANSA 自身的体网格导出：GRID + CTETRA + CPENTA 卡片，
fixed-width Nastran 小-字段 格式——和 nas_export.py 自己写出来的格式
一致，已经拿真实的 ANSA 导出文件核实过）。

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
from .nas_parser_utils import parse_nastran_float


def _parse_cards(path: str) -> Tuple[np.ndarray, np.ndarray, list, list]:
    """单次流式扫描文件：收集 GRID 节点 id/xyz、
    CTETRA 节点-id 行和 CPENTA 节点-id 行。全程使用固定宽度
    8 字符 Nastran 小字段卡片（与 nas_export.py 自己的
    CTETRA/CPENTA 写入器完全匹配，ANSA 自身的体导出也使用
    相同约定）。"""
    node_ids = []
    node_xyz = []
    tet_rows = []
    prism_rows = []

    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            card = line[:8].strip()
            if card == "GRID":
                nid = int(line[8:16])
                x = parse_nastran_float(line[24:32])
                y = parse_nastran_float(line[32:40])
                z = parse_nastran_float(line[40:48])
                node_ids.append(nid)
                node_xyz.append((x, y, z))
            elif card == "CTETRA":
                tet_rows.append([int(line[24 + 8 * k:32 + 8 * k]) for k in range(4)])
            elif card == "CPENTA":
                prism_rows.append([int(line[24 + 8 * k:32 + 8 * k]) for k in range(6)])

    if not node_ids:
        raise ValueError(f"No GRID cards found in {path} - not a valid volume-mesh NAS file")
    if not tet_rows and not prism_rows:
        raise ValueError(
            f"No CTETRA/CPENTA cards found in {path} - this looks like a surface mesh "
            f"(CTRIA3-only); use NASParser instead"
        )

    node_ids_arr = np.array(node_ids, dtype=np.int64)
    node_xyz_arr = np.array(node_xyz, dtype=np.float64)
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
    from ..mesh_gen.mesh_prism_to_tet import orient_tetrahedra
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
