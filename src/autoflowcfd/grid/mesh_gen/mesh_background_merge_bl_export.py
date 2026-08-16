"""`--bl-only` 调试导出：只把已挤出的边界层棱柱网格写成 NAS 文件并退出。

从 mesh_background_merge.py 的 _build_merged_mesh_with_bl 分支里拆出来
（原文件超过 400 行上限）——这段逻辑只在 export_bl_only=True 时触发，是
一个自包含的"构造导出对象 -> 写文件 -> sys.exit(0)"流程，和周围的核心
网格合并逻辑没有共享状态，适合单独成一个函数。逐字节保留原逻辑，未做任
何改动。
"""

import sys
import numpy as np
from typing import Optional
from loguru import logger


def _export_bl_only_and_exit(
    export_bl_only_path: Optional[str],
    bl_nodes: np.ndarray,
    bl_prisms: np.ndarray,
    n_bl_cells: int,
    bl_split_offset: int,
    nodes_per_layer: int,
    _effective_bl_layers: int,
) -> None:
    """导出仅含边界层棱柱的网格并终止进程（sys.exit(0)），供调试使用。

    对应 mesh_background_merge._build_merged_mesh 原来 with-BL 分支里的
    `if export_bl_only:` 代码块，逐字搬移。调用方只需在 export_bl_only
    为真时调用本函数——本函数自身会在成功或失败后都以 sys.exit(0) 结束
    进程，调用方之后的代码不会再执行到。
    """
    if not export_bl_only_path:
        raise ValueError("export_bl_only=True requires export_bl_only_path to be set")
    logger.success(f"Exporting BL-only mesh to: {export_bl_only_path}")

    try:
        from ..nas_io.nas_export import export_volume_mesh_to_nas
        from ..structures import NodeArray, PrismCells, BoundaryMap, GridMetadata, VolumeMeshData, TetrahedralCells

        # bl_nodes carries every extruded layer (BL + transition
        # stage), but bl_prisms only indexes the BL-stage prefix of
        # them (see the convert_layers_to_prisms call above, sliced
        # to bl_split_offset + nodes_per_layer) - keep only that
        # prefix here too, or the export ends up with a trailing
        # block of orphan GRID nodes no CPENTA references.
        used_node_count = bl_split_offset + nodes_per_layer
        export_nodes = bl_nodes[:used_node_count]
        nodes_obj = NodeArray(
            x=export_nodes[:, 0].copy(), y=export_nodes[:, 1].copy(), z=export_nodes[:, 2].copy()
        )

        # Create dummy tet cells to satisfy VolumeMeshData structure
        dummy_tets = np.empty((0, 4), dtype=np.int32)
        dummy_vols = np.empty(0, dtype=np.float64)
        cells_obj = TetrahedralCells(connectivity=dummy_tets, volumes=dummy_vols)

        prism_volumes = PrismCells.compute_volumes(nodes_obj, bl_prisms.astype(np.int32))
        prisms_obj = PrismCells(connectivity=bl_prisms.astype(np.int32), volumes=prism_volumes)

        # A BL-only export has no core tet mesh past the outermost
        # layer, so BOTH the true wall (layer 0) and the BL/core
        # interface (the last layer generated here) are "exterior"
        # faces of this prism block - lumping every prism into one
        # group made _extract_boundary_faces_by_group export both
        # surfaces under the same WALL tag (two overlapping shells).
        # Split them the same way the non-bl-only path already
        # distinguishes layer 0 (see is_layer0_prism a few lines
        # below this block): a prism's v0 is always its own bottom
        # layer's node, and layer L's nodes always occupy
        # bl_nodes[L*nodes_per_layer : (L+1)*nodes_per_layer].
        cell_layer = bl_prisms[:, 0] // nodes_per_layer
        wall_mask = cell_layer == 0
        interface_mask = cell_layer == (_effective_bl_layers - 1)

        dummy_groups = {
            'BL_Wall': np.flatnonzero(wall_mask).astype(np.int32),
            'BL_Interface': np.flatnonzero(interface_mask).astype(np.int32),
        }
        dummy_bc = {'BL_Wall': 'WALL', 'BL_Interface': 'INTERFACE'}
        boundaries_obj = BoundaryMap(groups=dummy_groups, bc_types=dummy_bc)

        metadata = GridMetadata(
            node_count=len(export_nodes),
            cell_count=n_bl_cells,
            boundary_groups=list(dummy_groups.keys()),
            file_format="nas"
        )

        vol_mesh = VolumeMeshData(
            nodes=nodes_obj,
            cells=cells_obj,
            boundaries=boundaries_obj,
            metadata=metadata,
            prism_cells=prisms_obj,
        )

        # Note: export_volume_mesh_to_nas expects meters and converts to mm (scale_factor=1000)
        export_volume_mesh_to_nas(vol_mesh, export_bl_only_path, scale_factor=1000.0)
        logger.success(f"BL-only mesh exported successfully.")
    except Exception as e:
        logger.error(f"Failed to export BL mesh: {e}")
        import traceback
        traceback.print_exc()

    import sys
    sys.exit(0)
