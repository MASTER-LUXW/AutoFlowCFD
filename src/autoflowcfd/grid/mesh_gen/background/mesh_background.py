"""域适配混合网格装配（边界层挤出 + tetgen 核心填充）。

生成恰好填充输入面网格所围封闭腔体的体网格：边界层（BL）棱柱仅从
壁面类型表面挤出（mesh_domain_classify 通过拓扑而非边界名称识别这些
表面），剩余内部体积由精确外边界（BL 外表面加上未修改的非壁面表面——
入口/出口/隧道/对称类边界）的约束四面体化（tetgen）填充。由于 tetgen
填充受真实封闭表面而非填充边界盒约束，结果永远不会超出输入面网格
实际描述的区域。

每次装配尝试的实际工作（分类 -> 挤出 -> tetgen 填充 -> 拼接）在
mesh_background_merge._build_merged_mesh 中实现；本模块是其外层的
重试/修复编排（阶段 A 平滑、阶段 B/B' 定向修复、阶段 C-相邻回退重试
递归）——见 generate_hybrid_mesh 的文档字符串。
"""

import traceback
import numpy as np
from typing import Dict, Optional, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from ...structures import BoundaryMap, VolumeMeshData

from ..tetgen.mesh_prism_to_tet import orient_tetrahedra
from ..tetgen.mesh_tetgen_core import _dedupe_coincident_points
from .mesh_background_merge import _build_merged_mesh
from .mesh_background_repair import repair_nonmanifold_tets_with_escalation

# 导入重构后的修复阶段
from ..repair.mesh_overlap_handler import compute_extra_bad_mask
from ..repair.mesh_repair_stage_a import run_stage_a_repair
from ..repair.mesh_repair_stage_b import run_stage_b_repair

# 阶段 A/B/B' 之后、最终装配之前的混合网格收尾修补阶段，拆到了
# mesh_background_mixed_repair.py（本文件超过 400 行上限，见该模块自己的
# 文档字符串）。
from .mesh_background_mixed_repair import _repair_mixed_mesh_post_stage_c

# 阶段 D：BL/core 界面相邻体积比定向修复（V2.0 专家组三次评审新发现的
# 根因修复，见 mesh_repair_interface.py 模块文档——阶段 A/B/B' 的坏单元
# 判据从不覆盖棱柱一侧，看不到 BL/core 界面）。
from ..repair.mesh_repair_interface import run_stage_d_interface_repair


def generate_hybrid_mesh(
    surface_nodes: np.ndarray,
    surface_faces: np.ndarray,
    bounding_box: Dict[str, np.ndarray],
    growth_rate: float = 1.2,
    min_cell_size: float = 0.001,
    target_cells: int = 500000,
    surface_boundaries: Optional['BoundaryMap'] = None,
    max_cell_size: Optional[float] = None,
    extra_thickness_limit: Optional[np.ndarray] = None,
    bl_layers: Optional[int] = None,
    _is_stage_b_retry: bool = False,
    export_bl_only: bool = False,
    export_bl_only_path: Optional[str] = None,
    export_core_only: bool = False,
    export_core_only_path: Optional[str] = None,
) -> 'VolumeMeshData':
    """生成精确贴合封闭输入表面的体网格。"""
    try:
        if surface_boundaries is None or not surface_boundaries.groups:
            raise ValueError(
                "generate_hybrid_mesh requires surface_boundaries with at least "
                "one boundary group, used to classify wall-type surfaces for BL "
                "extrusion versus the outer domain shell"
            )

        logger.info("Starting domain-conforming hybrid mesh generation...")

        (merged_nodes, prism_cells, merged_cells, cell_groups, n_bl_prisms,
         bl_source_vertex, bl_extrude_faces, bl_cell_groups, n_bl_cells) = _build_merged_mesh(
            surface_nodes, surface_faces, bounding_box, surface_boundaries,
            growth_rate, min_cell_size, max_cell_size,
            extra_thickness_limit, bl_layers,
            export_bl_only=export_bl_only,
            export_bl_only_path=export_bl_only_path,
            export_core_only=export_core_only,
            export_core_only_path=export_core_only_path,
        )

        # 如果处于 *-only 导出模式，_build_merged_mesh 已保存并退出
        # 或返回特殊信号。目前假设它正常返回但我们跳过 TetGen 逻辑。
        if export_bl_only or export_core_only:
            logger.success("Partial-pipeline export completed. Exiting.")
            import sys
            sys.exit(0)

        from ...structures import NodeArray, TetrahedralCells, PrismCells, GridMetadata, VolumeMeshData

        logger.info("Step 4/4: Re-orienting and computing tetrahedral volumes...")
        merged_cells = orient_tetrahedra(merged_nodes, merged_cells.astype(np.int64))
        _nodes_obj_tmp = NodeArray.from_array(merged_nodes)
        volumes = TetrahedralCells.compute_volumes(_nodes_obj_tmp, merged_cells.astype(np.int32))

        # 丢弃退化单元
        degenerate_threshold = (min_cell_size ** 3) * 1e-6
        valid_mask = volumes > degenerate_threshold
        n_invalid = np.sum(~valid_mask)
        if n_invalid > 0:
            logger.warning(f"Found {n_invalid} degenerate cells, removing them...")
            n_bl_cells = int(np.sum(valid_mask[:n_bl_cells]))
            merged_cells = merged_cells[valid_mask]
            volumes = volumes[valid_mask]
            cell_groups = cell_groups[valid_mask]

        # 修复非流形面（参见 repair_nonmanifold_tets_with_escalation 自身
        # 文档字符串了解局部重铺/升级/兜底删除的原理）。
        merged_nodes, merged_cells, cell_groups, n_bl_cells, _nm_changed = (
            repair_nonmanifold_tets_with_escalation(merged_nodes, merged_cells, cell_groups, n_bl_cells)
        )
        if _nm_changed:
            _tmp_nodes_obj_nm = NodeArray.from_array(merged_nodes)
            volumes = TetrahedralCells.compute_volumes(_tmp_nodes_obj_nm, merged_cells.astype(np.int32))

        # 合并重合点（seam 合并）
        n_nodes_before_seam_merge = len(merged_nodes)
        merged_nodes, merged_cells, _seam_remap = _dedupe_coincident_points(merged_nodes, merged_cells)
        if len(merged_nodes) != n_nodes_before_seam_merge:
            merged_cells = merged_cells.astype(np.int32)
            prism_cells = _seam_remap[prism_cells]
            _tmp_nodes_obj_seam = NodeArray.from_array(merged_nodes)
            post_merge_volumes_seam = TetrahedralCells.compute_volumes(_tmp_nodes_obj_seam, merged_cells)
            degenerate_threshold_seam = (min_cell_size ** 3) * 1e-6
            valid_mask_seam = post_merge_volumes_seam > degenerate_threshold_seam
            if int(np.sum(~valid_mask_seam)) > 0:
                logger.warning(f"Seam merge left {int(np.sum(~valid_mask_seam))} newly-degenerate cells, removing them...")
                n_bl_cells = int(np.sum(valid_mask_seam[:n_bl_cells]))
                merged_cells = merged_cells[valid_mask_seam]
                volumes = volumes[valid_mask_seam]
                cell_groups = cell_groups[valid_mask_seam]
            merged_nodes, merged_cells, cell_groups, n_bl_cells, _nm_changed_seam = (
                repair_nonmanifold_tets_with_escalation(
                    merged_nodes, merged_cells, cell_groups, n_bl_cells,
                    context_suffix=" (post seam-merge)",
                )
            )
            if _nm_changed_seam:
                _tmp_nodes_obj_nm2 = NodeArray.from_array(merged_nodes)
                volumes = TetrahedralCells.compute_volumes(_tmp_nodes_obj_nm2, merged_cells.astype(np.int32))

        # ------------------------------------------------------------------
        # 网格质量修复管线
        # ------------------------------------------------------------------
        from ...validation.quality_validator import MeshQualityValidator
        from ..extraction.face_extractor import FaceExtractor

        merged_cells = merged_cells.astype(np.int32)
        validator = MeshQualityValidator()
        logger.info("Checking volume mesh quality (pre-repair)...")
        
        _pre_repair_node_arr = NodeArray.from_array(merged_nodes)
        pre_repair_faces = FaceExtractor.extract_faces(merged_cells, _pre_repair_node_arr)
        initial_report = validator.validate(
            merged_nodes, merged_cells, cell_type="tetrahedron", faces=pre_repair_faces
        )

        # 计算额外坏掩码（重叠检测）
        overlap_bad_mask = compute_extra_bad_mask(validator, initial_report, merged_nodes, prism_cells, merged_cells)

        # 运行阶段 A 修复
        nodes_before_repair = merged_nodes
        merged_nodes, bad_mask, repair_actions = run_stage_a_repair(
            merged_nodes, merged_cells, validator, pre_repair_faces, 
            overlap_bad_mask, n_bl_cells
        )
        mesh_changed_by_repair = not np.array_equal(nodes_before_repair, merged_nodes)

        # 运行阶段 B 修复（空腔重划 + BL 厚度限制）
        if np.any(bad_mask):
            (merged_nodes, merged_cells, cell_groups, bad_mask, stage_b_actions,
             extra_limit, bl_verts) = run_stage_b_repair(
                merged_nodes, merged_cells, cell_groups, n_bl_cells, pre_repair_faces,
                bad_mask, validator, min_cell_size, bl_source_vertex, bl_extrude_faces, surface_nodes
            )
            repair_actions.extend(stage_b_actions)

            # 处理阶段 B 重试逻辑（递归调用）——复用 run_stage_b_repair
            # 已计算的 extra_limit/bl_verts（基于 dijkstra，并非免费）
            # 而非在此用相同参数重新计算。
            if np.any(bad_mask) and not _is_stage_b_retry:
                if extra_limit is not None:
                    logger.warning("Stage B: Retrying generation with targeted local BL thickness cap...")
                    del merged_nodes, merged_cells, volumes, cell_groups, bad_mask, initial_report
                    del prism_cells, bl_cell_groups
                    import gc
                    gc.collect()
                    return generate_hybrid_mesh(
                        surface_nodes, surface_faces, bounding_box,
                        growth_rate=growth_rate, min_cell_size=min_cell_size,
                        target_cells=target_cells, surface_boundaries=surface_boundaries,
                        max_cell_size=max_cell_size,
                        extra_thickness_limit=extra_limit,
                        bl_layers=bl_layers,
                        _is_stage_b_retry=True,
                    )

        # 最终防御遍：合并重合点并修复非流形
        n_nodes_before_merge = len(merged_nodes)
        merged_nodes, merged_cells, _remap = _dedupe_coincident_points(merged_nodes, merged_cells)
        if len(merged_nodes) != n_nodes_before_merge:
            mesh_changed_by_repair = True
            merged_cells = merged_cells.astype(np.int32)
            prism_cells = _remap[prism_cells]
            
            # 合并后检查新的退化
            _tmp_nodes_obj = NodeArray.from_array(merged_nodes)
            post_merge_volumes = TetrahedralCells.compute_volumes(_tmp_nodes_obj, merged_cells)
            degenerate_threshold = (min_cell_size ** 3) * 1e-6
            valid_mask = post_merge_volumes > degenerate_threshold
            if int(np.sum(~valid_mask)) > 0:
                logger.warning(f"Final merge left {int(np.sum(~valid_mask))} newly-degenerate cells, removing them...")
                merged_cells = merged_cells[valid_mask]
                cell_groups = cell_groups[valid_mask]

        # 构建最终对象
        nodes_obj = NodeArray.from_array(merged_nodes)

        # 阶段 A/B/B' 之后、最终装配之前的混合网格收尾修补阶段（跨类型
        # 非流形面拼接、BL 棱柱长细比局部重铺、collapsed-corner 棱柱
        # 降级为四面体）拆到了 mesh_background_mixed_repair.py（本文件超
        # 过 400 行上限）——三个子步骤共享同一组滚动状态，作为一个整体
        # 一起搬运，逐字对应原来这里的代码，未改动任何数值逻辑。
        (merged_nodes, prism_cells, merged_cells, bl_cell_groups, cell_groups,
         nodes_obj, mesh_changed_by_repair) = _repair_mixed_mesh_post_stage_c(
            merged_nodes, prism_cells, merged_cells, bl_cell_groups, cell_groups,
            nodes_obj, mesh_changed_by_repair, min_cell_size,
        )

        # 阶段 D：见 run_stage_d_interface_repair 模块文档——在完整混合
        # 面图（棱柱+四面体）上定向修复 BL/core 界面相邻体积比违规，只
        # 局部重铺四面体一侧，不改动任何棱柱/BL 几何。
        (merged_nodes, merged_cells, cell_groups, nodes_obj,
         _stage_d_changed, stage_d_actions) = run_stage_d_interface_repair(
            merged_nodes, prism_cells, merged_cells, cell_groups, nodes_obj, validator,
        )
        if stage_d_actions:
            for _action in stage_d_actions:
                logger.info(_action)
        mesh_changed_by_repair = mesh_changed_by_repair or _stage_d_changed

        # TetrahedralCells 严格执行 int32 连接关系；上方的 patch
        # 路径全程使用 int64（匹配送入的 .astype(np.int64) 转换，
        # 因为 patch_nonmanifold_cavity_mixed 自身的重映射算术在
        # 超大网格上可能在构造期间暂时产生超出 int32 范围的索引）
        # 在构建最终对象前必须转回。
        merged_cells = merged_cells.astype(np.int32)
        volumes = TetrahedralCells.compute_volumes(nodes_obj, merged_cells)
        cells_obj = TetrahedralCells(connectivity=merged_cells, volumes=volumes)

        prism_cells = prism_cells.astype(np.int32)
        n_prism = len(prism_cells)
        prism_cells_obj = None
        if n_prism > 0:
            prism_volumes = PrismCells.compute_volumes(nodes_obj, prism_cells)
            prism_cells_obj = PrismCells(connectivity=prism_cells, volumes=prism_volumes)

        from ..utils.mesh_boundary import identify_boundaries_from_surface
        tet_boundaries = identify_boundaries_from_surface(
            merged_cells, surface_faces, surface_boundaries, direct_cell_groups=cell_groups
        )

        # 合并边界组
        groups: Dict[str, np.ndarray] = {}
        bc_types: Dict[str, str] = {}
        if n_prism > 0:
            for name in np.unique(bl_cell_groups):
                if not name: continue
                idx = np.flatnonzero(bl_cell_groups == name).astype(np.int32)
                groups[name] = idx
                bc_types[name] = surface_boundaries.bc_types.get(name, 'WALL')

        for name, idx in tet_boundaries.groups.items():
            shifted = (idx.astype(np.int64) + n_prism).astype(np.int32)
            if name in groups:
                groups[name] = np.union1d(groups[name], shifted).astype(np.int32)
            else:
                groups[name] = shifted
                bc_types[name] = tet_boundaries.bc_types.get(name, 'WALL')

        from ...schema.grid_boundaries import BoundaryMap
        boundaries_obj = BoundaryMap(groups=groups, bc_types=bc_types)

        metadata = GridMetadata(
            node_count=len(merged_nodes),
            cell_count=n_prism + len(merged_cells),
            boundary_groups=list(boundaries_obj.groups.keys()),
            file_format="hybrid"
        )

        return VolumeMeshData(
            nodes=nodes_obj,
            cells=cells_obj,
            boundaries=boundaries_obj,
            metadata=metadata,
            prism_cells=prism_cells_obj,
        )
    except Exception as e:
        logger.error(f"Error in generate_hybrid_mesh: {e}")
        traceback.print_exc()
        raise
