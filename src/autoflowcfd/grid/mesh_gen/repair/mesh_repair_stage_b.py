"""阶段 B：结合 BL 厚度封顶与 cavity 重新铺网的定向再生成。

从 mesh_background.py 拆分出来以控制行数。
"""

import numpy as np
from typing import List, Optional, Tuple, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from ...schema.grid_faces import FaceData
    from ...validation.quality_validator import MeshQualityValidator


def run_stage_b_repair(
    merged_nodes: np.ndarray,
    merged_cells: np.ndarray,
    cell_groups: np.ndarray,
    n_bl_cells: int,
    pre_repair_faces: 'FaceData',
    bad_mask: np.ndarray,
    validator: 'MeshQualityValidator',
    min_cell_size: float,
    bl_source_vertex: np.ndarray,
    bl_extrude_faces: np.ndarray,
    surface_nodes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Optional[np.ndarray], Optional[np.ndarray]]:
    """执行阶段 B：局部空腔重铺网和/或 BL 厚度封顶。

    Args:
        merged_nodes: 节点坐标数组。
        merged_cells: 单元连接数组。
        cell_groups: 单元组标签。
        n_bl_cells: BL 单元数量。
        pre_repair_faces: 预提取的面数据。
        bad_mask: 阶段 A 输出的坏单元掩码。
        validator: 质量验证器实例。
        min_cell_size: 最小单元尺寸参数。
        bl_source_vertex: BL 节点到表面顶点的映射。
        bl_extrude_faces: 用于 BL 拉伸的面。
        surface_nodes: 原始表面节点。

    Returns:
        (新节点, 新单元, 新单元组, 新坏单元掩码, 修复动作列表,
        extra_limit, bl_verts) 元组 —— 后两者是下方计算的 BL 厚度
        封顶覆盖值（如果阶段 B' 的空腔重铺已清除所有坏单元则为
        (None, None)），返回给调用方 (mesh_background.generate_hybrid_mesh)
        以便在重试时复用，避免用相同参数重新计算完全相同的
        Dijkstra 结果。
    """
    from .mesh_repair import remesh_core_cavity, compute_bl_thickness_limit_override
    from ..extraction.face_extractor import FaceExtractor
    from ...schema.grid_nodes import NodeArray

    repair_actions = []

    if not np.any(bad_mask):
        return merged_nodes, merged_cells, cell_groups, bad_mask, repair_actions, None, None

    # Stage B': Local cavity remesh
    max_b_prime_attempts = 3
    b_prime_attempt_count = 0

    while np.any(bad_mask) and b_prime_attempt_count < max_b_prime_attempts:
        merged_nodes, merged_cells, cell_groups, bad_mask, cavity_actions = remesh_core_cavity(
            merged_nodes, merged_cells, cell_groups, n_bl_cells, pre_repair_faces, bad_mask, validator,
        )
        repair_actions.extend(cavity_actions)

        b_prime_attempt_count += 1
        if np.any(bad_mask):
            logger.warning(f"Stage B' attempt {b_prime_attempt_count}/{max_b_prime_attempts} completed, "
                           f"{int(np.sum(bad_mask))} bad cells remain.")

        # remesh_core_cavity 会将新单元拼接到被移除的空腔单元位置——
        # merged_cells 的大小/内容发生了变化（甚至可能长度不变：空腔
        # 可以被 1:1 替换为不同的铺排方式，所以仅靠长度检查无法可靠
        # 判断是否真的变了），但 pre_repair_faces（其 owner/neighbour
        # 单元索引）仍然反映调用前的拓扑。remesh_core_cavity 内部使用
        # faces 来查找物理边界单元和遍历内部 owner/neighbour 对时，
        # 需要索引到当前的 cells 数组——复用旧的会导致真实的崩溃
        # （已确认：第二次空腔重铺迭代时 owner 索引超出了当时的单元
        # 数量，在 remesh_core_cavity 内部触发 IndexError
        # (touches_physical_boundary[...] = True)），在修复管线有机会
        # 报告或导出任何东西之前就中断了。每次迭代都无条件重新
        # 提取（而不仅仅是检测到变化时才提取），因为下面的
        # compute_bl_thickness_limit_override 调用也需要当前的面数据，
        # 而"是否变化"的判断出错是比一次冗余提取更严重的故障模式。
        node_arr = NodeArray.from_array(merged_nodes)
        pre_repair_faces = FaceExtractor.extract_faces(merged_cells.astype(np.int32), node_arr)

    if np.any(bad_mask):
        logger.warning(f"Stage B' reached max attempts ({max_b_prime_attempts}), "
                       f"{int(np.sum(bad_mask))} bad cells remain.")

    # Stage B: BL thickness capping retry
    extra_limit, bl_verts = None, None
    if np.any(bad_mask):
        n_bad = int(np.sum(bad_mask))
        cap_thickness = min_cell_size * 3.0
        extra_limit, bl_verts = compute_bl_thickness_limit_override(
            bad_mask, n_bl_cells, merged_cells, len(surface_nodes), cap_thickness,
            nodes_per_layer=len(bl_source_vertex), node_original_vertex=bl_source_vertex,
            local_surface_faces=bl_extrude_faces,
        )

        if extra_limit is not None:
            logger.warning(
                f"Stage A/B' left {n_bad} cells still bad ({len(bl_verts)} BL vertices "
                f"implicated) - triggering Stage B: targeted local BL thickness cap."
            )
            # 注意：实际的重试逻辑（再次调用 generate_hybrid_mesh）
            # 由 mesh_background.py 中的编排器处理，使用下面返回的
            # extra_limit/bl_verts。
            repair_actions.append(f"Stage B: computed thickness limit for {len(bl_verts)} vertices")

    return merged_nodes, merged_cells, cell_groups, bad_mask, repair_actions, extra_limit, bl_verts
