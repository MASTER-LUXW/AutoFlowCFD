"""mesh_background_merge._build_merged_mesh 用到的两个独立小工具。

从 mesh_background_merge.py 拆分出来：`_refine_large_boundary_faces`
（tetgen 之前按最大边长迭代二Splitting 边界面的过大三角形，避免生成过大的边界四面体）和
`_export_partial_mesh_and_exit`（`--bl-only`/`--core-only` 等调试导出
路径共用的导出后直接退出进程的辅助函数）。两者都只被 _build_merged_mesh
自己调用，没有独立复用需求，纯粹为了控制文件行数而拆开。
"""

import sys
from typing import Dict, Optional, Tuple

import numpy as np
from loguru import logger


def _refine_large_boundary_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    markers: Optional[np.ndarray],
    max_edge_length: float,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """在边界表面上迭代二Splitting 超过 max_edge_length 的边。

    这可以防止 TetGen 生成违反目标单元尺寸约束的巨大边界四面体。
    同时有助于将 BL 外表面的分辨率与核心填充匹配，并通过更精确的
    边长控制改善与 ANSA 网格质量标准的兼容性。
    """
    if max_edge_length <= 0:
        return vertices, faces, markers

    current_verts = vertices.copy()
    current_faces = faces.copy()
    current_markers = markers.copy() if markers is not None else None

    max_iterations = 10
    max_total_vertices = len(vertices) * 5  # 防止内存爆炸

    for iteration in range(max_iterations):
        if len(current_verts) > max_total_vertices:
            logger.warning(f"Boundary refinement stopped: vertex count ({len(current_verts)}) exceeded limit ({max_total_vertices})")
            break

        # 向量化边长计算
        v0 = current_verts[current_faces[:, 0]]
        v1 = current_verts[current_faces[:, 1]]
        v2 = current_verts[current_faces[:, 2]]

        e01 = np.linalg.norm(v1 - v0, axis=1)
        e12 = np.linalg.norm(v2 - v1, axis=1)
        e20 = np.linalg.norm(v0 - v2, axis=1)

        max_edges = np.maximum.reduce([e01, e12, e20])
        needs_refinement_mask = max_edges > max_edge_length

        if not np.any(needs_refinement_mask):
            logger.info(f"Boundary refinement completed after {iteration} iterations. "
                        f"Faces: {len(faces)} -> {len(current_faces)}, "
                        f"Vertices: {len(vertices)} -> {len(current_verts)}")
            return current_verts, current_faces, current_markers

        # 查找每个需要细化的面的最长边
        split_indices = np.argmax(np.stack([e01, e12, e20], axis=1)[needs_refinement_mask], axis=1)
        faces_to_split = current_faces[needs_refinement_mask]
        markers_to_split = current_markers[needs_refinement_mask] if current_markers is not None else None

        new_faces_list = []
        new_markers_list = []
        new_vertices_list = []

        # 分批处理Splitting以避免索引偏移问题
        # 需要将旧顶点索引映射到新顶点索引
        vertex_offset = len(current_verts)

        for i, (face, split_idx) in enumerate(zip(faces_to_split, split_indices)):
            v0, v1, v2 = face
            if split_idx == 0:  # Splitting 0-1 边
                mid_coord = (current_verts[v0] + current_verts[v1]) / 2.0
                mid_idx = vertex_offset + i
                f1, f2 = [mid_idx, v1, v2], [v0, mid_idx, v2]
            elif split_idx == 1:  # Splitting 1-2 边
                mid_coord = (current_verts[v1] + current_verts[v2]) / 2.0
                mid_idx = vertex_offset + i
                f1, f2 = [v0, mid_idx, v2], [v0, v1, mid_idx]
            else:  # Splitting 2-0 边
                mid_coord = (current_verts[v2] + current_verts[v0]) / 2.0
                mid_idx = vertex_offset + i
                f1, f2 = [v0, v1, mid_idx], [mid_idx, v1, v2]

            new_faces_list.extend([f1, f2])
            new_vertices_list.append(mid_coord)
            if current_markers is not None:
                new_markers_list.extend([markers_to_split[i], markers_to_split[i]])

        # 添加新顶点
        if new_vertices_list:
            current_verts = np.vstack([current_verts, np.array(new_vertices_list)])

        # 用新面替换已Splitting的旧面
        remaining_faces = current_faces[~needs_refinement_mask]
        remaining_markers = current_markers[~needs_refinement_mask] if current_markers is not None else None

        if new_faces_list:
            current_faces = np.vstack([remaining_faces, np.array(new_faces_list, dtype=np.int32)]) if len(remaining_faces) > 0 else np.array(new_faces_list, dtype=np.int32)
            if current_markers is not None:
                current_markers = np.concatenate([remaining_markers, np.array(new_markers_list)]) if len(remaining_markers) > 0 else np.array(new_markers_list)
        else:
            current_faces = remaining_faces
            if current_markers is not None:
                current_markers = remaining_markers

        logger.info(f"  Refinement iteration {iteration+1}: Split {len(faces_to_split)} faces, added {len(new_vertices_list)} vertices")

    logger.info(f"Boundary refinement completed after {max_iterations} iterations (limit reached). "
                f"Faces: {len(faces)} -> {len(current_faces)}, "
                f"Vertices: {len(vertices)} -> {len(current_verts)}")
    return current_verts, current_faces, current_markers


def _export_partial_mesh_and_exit(
    nodes: np.ndarray,
    prism_cells: np.ndarray,
    prism_groups: np.ndarray,
    tet_cells: np.ndarray,
    tet_groups: np.ndarray,
    output_path: str,
    label: str,
) -> None:
    """导出局部（仅 BL / 仅过渡层 / 仅核心）调试网格并退出进程。

    所有 `--*-only` CLI 标志的早期停止路径共用此函数
    （参见 cli/grid_commands.py 的 `--bl-only`/`--trans-only`/`--core-only`）。
    这些标志用于在网格查看器（ANSA 等）中直接检查管线各阶段的生成结果——
    在调查 BL/过渡层到核心填充界面问题时反复需要此功能，但此前没有可复用的方式，
    每次都要写临时脚本。

    `prism_groups`/`tet_groups` 中每个非空分组名称会成为独立的 WALL 边界组；
    每个 ''（未归属——例如中间层单元没有暴露的命名面）会被归入一个统一的
    'INTERFACE' 分组，而不是被静默丢弃，因此导出文件始终具有完整的边界分区可供打开。

    Args:
        nodes: (n_nodes, 3) 节点坐标，单位：米
        prism_cells: (n_prism, 6) 棱柱连接关系，或空数组 (0, 6)
        prism_groups: (n_prism,) 与 prism_cells 平行的字符串数组
        tet_cells: (n_tet, 4) 四面体连接关系，或空数组 (0, 4)
        tet_groups: (n_tet,) 与 tet_cells 平行的字符串数组
        output_path: .nas 文件输出路径
        label: 当前阶段的易读名称，仅用于日志
    """
    from ...nas_io.nas_export import export_volume_mesh_to_nas
    from ...schema.grid_nodes import NodeArray, PrismCells, TetrahedralCells, BoundaryMap, GridMetadata, VolumeMeshData

    logger.success(f"Exporting {label} mesh to: {output_path}")
    try:
        nodes_obj = NodeArray.from_array(nodes)

        n_prism = len(prism_cells)
        prism_cells_obj = None
        if n_prism:
            prism_volumes = PrismCells.compute_volumes(nodes_obj, prism_cells.astype(np.int32))
            prism_cells_obj = PrismCells(connectivity=prism_cells.astype(np.int32), volumes=prism_volumes)

        tet_cells32 = tet_cells.astype(np.int32)
        tet_volumes = (
            TetrahedralCells.compute_volumes(nodes_obj, tet_cells32)
            if len(tet_cells32) else np.empty(0, dtype=np.float64)
        )
        cells_obj = TetrahedralCells(connectivity=tet_cells32, volumes=tet_volumes)

        groups: Dict[str, np.ndarray] = {}
        bc_types: Dict[str, str] = {}
        interface_parts = []
        if n_prism:
            for name in np.unique(prism_groups):
                idx = np.flatnonzero(prism_groups == name).astype(np.int32)
                if name:
                    groups[name] = idx
                    bc_types[name] = 'WALL'
                else:
                    interface_parts.append(idx)
        if len(tet_cells32):
            for name in np.unique(tet_groups):
                idx = (np.flatnonzero(tet_groups == name) + n_prism).astype(np.int32)
                if name:
                    if name in groups:
                        groups[name] = np.union1d(groups[name], idx).astype(np.int32)
                    else:
                        groups[name] = idx
                        bc_types[name] = 'WALL'
                else:
                    interface_parts.append(idx)
        if interface_parts:
            groups['INTERFACE'] = np.concatenate(interface_parts).astype(np.int32)
            bc_types['INTERFACE'] = 'INTERFACE'

        boundaries_obj = BoundaryMap(groups=groups, bc_types=bc_types)
        metadata = GridMetadata(
            node_count=len(nodes), cell_count=n_prism + len(tet_cells32),
            boundary_groups=list(groups.keys()), file_format="nas",
        )
        vol_mesh = VolumeMeshData(
            nodes=nodes_obj, cells=cells_obj, boundaries=boundaries_obj,
            metadata=metadata, prism_cells=prism_cells_obj,
        )
        # export_volume_mesh_to_nas 期望输入为米，内部会转换为毫米
        export_volume_mesh_to_nas(vol_mesh, output_path, scale_factor=1000.0)
        logger.success(f"{label} mesh exported successfully.")
    except Exception as e:
        logger.error(f"Failed to export {label} mesh: {e}")
        import traceback
        traceback.print_exc()

    sys.exit(0)
