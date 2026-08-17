"""NASParser 体网格生成委托 (从 parser_core.py 拆分)。

从 parser_core.py 拆出来（该文件原有 428 行，超过 400 行硬性拆分
阈值）：`generate_volume_mesh_from_surface` 不读写任何 NASParser 实例
状态（不引用任何 self.*），是一个可以独立成模块函数的自包含逻辑块，
`NASParser` 上保留同名薄委托方法，调用方式不变。纯代码搬移，不改变
任何行为。
"""

from typing import Dict, Optional

import numpy as np
from loguru import logger

from ..structures import GridData, VolumeMeshData


def generate_volume_mesh_from_surface(
    surface_grid: GridData,
    volume_mesh_params: Optional[Dict] = None,
) -> 'VolumeMeshData':
    """从已解析的表面 GridData 生成体网格。

    从 parse() 的 generate_volume_mesh=True 路径中提取出来，
    使已有解析好的 surface_grid 的调用方（例如对其运行了
    GridValidator 做生成前质量检查后）可以直接送入体网格
    生成，不需要第二次冗余的原始 NAS 文件重新解析——
    parse() 本身现在只构建表面 GridData 并委托到这里。

    Args:
        surface_grid: 已解析的表面网格（nodes/cells/
            boundaries/metadata.bounding_box）
        volume_mesh_params: 体网格生成参数（见 parse() 的
            volume_mesh_params）

    Returns:
        VolumeMeshData
    """
    from ..mesh_gen.volume_mesh_generator import VolumeMeshGenerator

    logger.info("Generating volume mesh from surface geometry...")

    params = volume_mesh_params or {}

    # Hybrid mesh strategy:
    # Stage 1: Boundary Layer (fixed layer count, fine resolution for y+ control)
    # Stage 2: Core fill - tetgen fills the remaining volume directly from
    #   the BL's own outer surface, using its own unstructured grading out
    #   to max_cell_size (see mesh_background_merge._build_merged_mesh;
    #   ProjectFiles Part13 P49 - no separate structured transition stage)
    optimized_params = {
        'growth_rate': params.get('growth_rate', 1.2),
        'min_cell_size': params.get('min_cell_size', 0.01),
        'target_cells': params.get('target_cells', 400000),  # 平衡目标
        'max_cell_size': params.get('max_cell_size'),
        'bl_layers': params.get('bl_layers'),
        'bl_only': params.get('bl_only', False),
        'bl_only_output': params.get('output'),
        'core_only': params.get('core_only', False),
    }

    # 反映实际解析的参数，而不是固定的占位符数字
    # ——以前即使 --growth-rate/--bl-layers 被覆盖，也总是打印
    # "8 layers, growth_rate=1.2 / 4 layers, growth_rate=1.5"，
    # 误导任何试图将此日志与实际生成的内容关联的人（或 agent）。
    resolved_bl_layers = optimized_params['bl_layers'] or 8
    logger.info(
        f"Using hybrid mesh strategy:\n"
        f"  Stage 1 (BL): {resolved_bl_layers} layers, "
        f"growth_rate={optimized_params['growth_rate']}\n"
        f"  Stage 2 (Core fill): tetgen, graded out to "
        f"max_cell_size={optimized_params['max_cell_size']}\n"
        f"  Target total cells: ~{optimized_params['target_cells']:,}"
    )

    generator = VolumeMeshGenerator(**optimized_params)

    nodes = surface_grid.nodes
    surface_nodes_np = np.column_stack([nodes.x, nodes.y, nodes.z])
    bounding_box = surface_grid.metadata.bounding_box

    volume_mesh = generator.generate_from_surface(
        surface_nodes=surface_nodes_np,
        surface_faces=surface_grid.cells.connectivity,
        bounding_box={
            'min': np.array([bounding_box[0], bounding_box[2], bounding_box[4]]),
            'max': np.array([bounding_box[1], bounding_box[3], bounding_box[5]])
        },
        surface_boundaries=surface_grid.boundaries,
    )

    # 保存原始表面网格数据
    volume_mesh.surface_mesh = {
        'nodes': surface_nodes_np,
        'faces': surface_grid.cells.connectivity,
        'boundaries': surface_grid.boundaries
    }

    logger.success(
        f"Volume mesh generated: {volume_mesh.node_count} nodes, "
        f"{volume_mesh.cell_count} cells, "
        f"total volume: {volume_mesh.total_volume:.6e} m^3"
    )

    return volume_mesh
