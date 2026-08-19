"""`--bl-only` 调试导出：只把已挤出的边界层棱柱网格写成 NAS 文件并退出。

从 mesh_background_merge.py 的 _build_merged_mesh_with_bl 分支里拆出来
（原文件超过 400 行上限）——这段逻辑只在 export_bl_only=True 时触发，是
一个自包含的"构造导出对象 -> 写文件 -> sys.exit(0)"流程，和周围的核心
网格合并逻辑没有共享状态，适合单独成一个函数。逐字节保留原逻辑，未做任
何改动。
"""

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

    对应 mesh_background_merge._build_merged_mesh 原来含 BL 分支里的
    `if export_bl_only:` 代码块，逐字搬运。调用方只需在 export_bl_only
    为真时调用本函数——本函数自身会在成功或失败后都以 sys.exit(0) 结束
    进程，调用方之后的代码不会再执行到。
    """
    if not export_bl_only_path:
        raise ValueError("export_bl_only=True requires export_bl_only_path to be set")
    logger.success(f"Exporting BL-only mesh to: {export_bl_only_path}")

    try:
        from ...nas_io.nas_export import export_volume_mesh_to_nas
        from ...structures import NodeArray, PrismCells, BoundaryMap, GridMetadata, VolumeMeshData, TetrahedralCells

        # bl_nodes 包含所有挤出层（BL + 过渡层），但 bl_prisms 只索引
        # 其中的 BL 阶段前缀（参见上方的 convert_layers_to_prisms 调用，
        # 切片到 bl_split_offset + nodes_per_layer）——此处也只保留该
        # 前缀，否则导出会带有一堆没有被 CPENTA 引用的孤立 GRID 节点。
        used_node_count = bl_split_offset + nodes_per_layer
        export_nodes = bl_nodes[:used_node_count]
        nodes_obj = NodeArray.from_array(export_nodes)

        # 创建 dummy 四面体单元以满足 VolumeMeshData 结构要求
        dummy_tets = np.empty((0, 4), dtype=np.int32)
        dummy_vols = np.empty(0, dtype=np.float64)
        cells_obj = TetrahedralCells(connectivity=dummy_tets, volumes=dummy_vols)

        prism_volumes = PrismCells.compute_volumes(nodes_obj, bl_prisms.astype(np.int32))
        prisms_obj = PrismCells(connectivity=bl_prisms.astype(np.int32), volumes=prism_volumes)

        # 仅 BL 导出在最外层之后没有核心四面体，因此真正的壁面层（layer 0）
        # 和 BL/核心界面（此处生成的最后一层）都是这个棱柱块的" exterior"面
        # ——把所有棱柱归入一个组会使 _extract_boundary_faces_by_group 在
        # 同一个 WALL 标签下导出两个重叠壳面。
        # 按与非 bl-only 路径区分 layer 0 的相同方式Splitting它们（参见
        # 此代码块下方几行的 is_layer0_prism）：棱柱的 v0 始终是其自身底层
        # 的节点，而第 L 层的节点始终占据
        # bl_nodes[L*nodes_per_layer : (L+1)*nodes_per_layer]。
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

        # 注意: export_volume_mesh_to_nas 期望输入为米制，内部会转换为毫米（scale_factor=1000）
        export_volume_mesh_to_nas(vol_mesh, export_bl_only_path, scale_factor=1000.0)
        logger.success("BL-only mesh exported successfully.")
    except Exception as e:
        logger.error(f"Failed to export BL mesh: {e}")
        import traceback
        traceback.print_exc()

    import sys
    sys.exit(0)
