"""mesh_background_merge._build_merged_mesh 用到的两个独立小工具。

从 mesh_background_merge.py 拆分出来：`_refine_large_boundary_faces`
（tetgen 之前按最大边长迭代二分边界面，避免生成过大的边界四面体）和
`_export_partial_mesh_and_exit`（`--bl-only`/`--core-only` 等调试导出
路径共用的"导出后直接退出进程"辅助函数）。两者都只被 _build_merged_mesh
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
    """Iteratively bisect edges longer than max_edge_length on the boundary surface.

    This prevents TetGen from generating huge boundary tets that violate the
    target cell size constraint. It also helps match the resolution of the
    BL outer surface to the core fill and improves compatibility with ANSA's
    mesh quality standards through more precise edge length control.
    """
    if max_edge_length <= 0:
        return vertices, faces, markers

    current_verts = vertices.copy()
    current_faces = faces.copy()
    current_markers = markers.copy() if markers is not None else None

    max_iterations = 10
    max_total_vertices = len(vertices) * 5  # Prevent memory explosion

    for iteration in range(max_iterations):
        if len(current_verts) > max_total_vertices:
            logger.warning(f"Boundary refinement stopped: vertex count ({len(current_verts)}) exceeded limit ({max_total_vertices})")
            break

        # Vectorized edge length calculation
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

        # Find the longest edge for each face that needs refinement
        split_indices = np.argmax(np.stack([e01, e12, e20], axis=1)[needs_refinement_mask], axis=1)
        faces_to_split = current_faces[needs_refinement_mask]
        markers_to_split = current_markers[needs_refinement_mask] if current_markers is not None else None

        new_faces_list = []
        new_markers_list = []
        new_vertices_list = []

        # Process splits in batches to avoid index shifting issues
        # We need to map old vertex indices to new ones
        vertex_offset = len(current_verts)

        for i, (face, split_idx) in enumerate(zip(faces_to_split, split_indices)):
            v0, v1, v2 = face
            if split_idx == 0: # Split 0-1
                mid_coord = (current_verts[v0] + current_verts[v1]) / 2.0
                mid_idx = vertex_offset + i
                f1, f2 = [mid_idx, v1, v2], [v0, mid_idx, v2]
            elif split_idx == 1: # Split 1-2
                mid_coord = (current_verts[v1] + current_verts[v2]) / 2.0
                mid_idx = vertex_offset + i
                f1, f2 = [v0, mid_idx, v2], [v0, v1, mid_idx]
            else: # Split 2-0
                mid_coord = (current_verts[v2] + current_verts[v0]) / 2.0
                mid_idx = vertex_offset + i
                f1, f2 = [v0, v1, mid_idx], [mid_idx, v1, v2]

            new_faces_list.extend([f1, f2])
            new_vertices_list.append(mid_coord)
            if current_markers is not None:
                new_markers_list.extend([markers_to_split[i], markers_to_split[i]])

        # Add new vertices
        if new_vertices_list:
            current_verts = np.vstack([current_verts, np.array(new_vertices_list)])

        # Replace split faces with new ones
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
    """Export a partial (BL-only/transition-only/core-only) debug mesh and
    exit the process - shared by every `--*-only` CLI flag's early-stop
    path (see cli/grid_commands.py's own `--bl-only`/`--trans-only`/
    `--core-only`). These exist to let a real, generated mesh from any one
    pipeline stage be inspected directly in a mesh viewer (ANSA etc.) -
    this session's own investigation into the BL/transition-to-core-fill
    interface repeatedly needed exactly this and had no reusable way to
    get it short of ad-hoc scripts each time.

    Every distinct non-empty group name in `prism_groups`/`tet_groups`
    becomes its own WALL boundary group; every '' (unattributed - e.g. a
    mid-stack cell with no exposed named face) is lumped into a single
    catch-all 'INTERFACE' group instead of being silently dropped, so the
    exported file always has a complete boundary partition to open.

    Args:
        nodes: (n_nodes, 3) node coordinates, meters
        prism_cells: (n_prism, 6) prism connectivity, or empty (0, 6)
        prism_groups: (n_prism,) str array parallel to prism_cells
        tet_cells: (n_tet, 4) tet connectivity, or empty (0, 4)
        tet_groups: (n_tet,) str array parallel to tet_cells
        output_path: where to write the .nas file
        label: human-readable name for this stage, used only in log lines
    """
    from ..nas_io.nas_export import export_volume_mesh_to_nas
    from ..structures import NodeArray, PrismCells, TetrahedralCells, BoundaryMap, GridMetadata, VolumeMeshData

    logger.success(f"Exporting {label} mesh to: {output_path}")
    try:
        nodes_obj = NodeArray(x=nodes[:, 0].copy(), y=nodes[:, 1].copy(), z=nodes[:, 2].copy())

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
        # export_volume_mesh_to_nas expects meters and converts to mm.
        export_volume_mesh_to_nas(vol_mesh, output_path, scale_factor=1000.0)
        logger.success(f"{label} mesh exported successfully.")
    except Exception as e:
        logger.error(f"Failed to export {label} mesh: {e}")
        import traceback
        traceback.print_exc()

    sys.exit(0)
