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
    from ...schema import BoundaryMap, VolumeMeshData

from ..tetgen.mesh_prism_to_tet import orient_tetrahedra
from ..tetgen.mesh_tetgen_core import dedupe_coincident_points
from ..tetgen.mesh_tetgen_postprocess import drop_degenerate_tets
from .mesh_background_merge import _build_merged_mesh
from .mesh_background_merge_utils import PartialMeshReady
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
    growth_rate: float,
    min_cell_size: float,
    surface_boundaries: Optional['BoundaryMap'] = None,
    max_cell_size: Optional[float] = None,
    bl_layers: Optional[int] = None,
    export_bl_only: bool = False,
    export_core_only: bool = False,
    first_layer_height: Optional[float] = None,
) -> 'VolumeMeshData':
    """生成精确贴合封闭输入表面的体网格（`first_layer_height` 见 `extrusion/mesh_extrusion.py::extrude_layers`）。

    输出前做硬检查，任一项不满足即报错、不输出网格：单元不退化（`validation/cell_validity.py`）、每个面最多两个
    单元共享（流形）、面不退化、外表面恰好覆盖输入面网格（不带洞，`map_generated_boundaries`）。中间各步的修补
    （删除退化单元、非流形面保留最大的单元、空腔重铺失败时回退删除）由这道检查兜住结果。

    参数默认值只在 `VolumeMeshGenerator` 一处（本函数与下游都要求显式给出）。

    Raises:
        ValueError: 输出前的硬检查未通过
        RuntimeError: 有面被超过两个单元共享
        PartialMeshReady: `export_bl_only` / `export_core_only` 时在该阶段结束，局部网格随信号带回（不是错误）
    """
    try:
        if surface_boundaries is None or not surface_boundaries.groups:
            raise ValueError(
                "generate_hybrid_mesh requires surface_boundaries with at least "
                "one boundary group, used to classify wall-type surfaces for BL "
                "extrusion versus the outer domain shell"
            )

        logger.info("Starting domain-conforming hybrid mesh generation...")

        merged_nodes, prism_cells, merged_cells = _build_merged_mesh(
            surface_nodes, surface_faces, bounding_box, surface_boundaries,
            growth_rate, min_cell_size, max_cell_size, bl_layers,
            export_bl_only=export_bl_only,
            export_core_only=export_core_only,
            first_layer_height=first_layer_height,
        )
        if export_bl_only or export_core_only:
            raise RuntimeError("调试导出路径没有带回局部网格就返回了（_build_merged_mesh 的分支漏调 _stop_with_partial_mesh）")

        from ...schema import NodeArray, TetrahedralCells, PrismCells, GridMetadata, VolumeMeshData

        logger.info("Step 4/4: Re-orienting and computing tetrahedral volumes...")
        merged_cells = orient_tetrahedra(merged_nodes, merged_cells.astype(np.int64))
        merged_cells, volumes = drop_degenerate_tets(merged_nodes, merged_cells, min_cell_size, "Merged mesh")

        # 修复非流形面（参见 repair_nonmanifold_tets_with_escalation 自身
        # 文档字符串了解局部重铺/升级/兜底删除的原理）。
        merged_nodes, merged_cells, _nm_changed = (
            repair_nonmanifold_tets_with_escalation(merged_nodes, merged_cells)
        )
        if _nm_changed:
            _tmp_nodes_obj_nm = NodeArray.from_array(merged_nodes)
            volumes = TetrahedralCells.compute_volumes(_tmp_nodes_obj_nm, merged_cells.astype(np.int32))

        # 合并重合点（seam 合并）
        n_nodes_before_seam_merge = len(merged_nodes)
        merged_nodes, merged_cells, _seam_remap = dedupe_coincident_points(merged_nodes, merged_cells)
        if len(merged_nodes) != n_nodes_before_seam_merge:
            merged_cells = merged_cells.astype(np.int32)
            prism_cells = _seam_remap[prism_cells]
            merged_cells, volumes = drop_degenerate_tets(merged_nodes, merged_cells, min_cell_size, "Seam merge")
            merged_nodes, merged_cells, _nm_changed_seam = (
                repair_nonmanifold_tets_with_escalation(
                    merged_nodes, merged_cells,
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
            merged_nodes, merged_cells, validator, pre_repair_faces, overlap_bad_mask,
        )
        mesh_changed_by_repair = not np.array_equal(nodes_before_repair, merged_nodes)

        # 运行阶段 B' 修复（局部空腔重铺）
        if np.any(bad_mask):
            merged_nodes, merged_cells, bad_mask, stage_b_actions = run_stage_b_repair(
                merged_nodes, merged_cells, pre_repair_faces, bad_mask, validator,
            )
            repair_actions.extend(stage_b_actions)

        # 最终防御遍：合并重合点并修复非流形
        n_nodes_before_merge = len(merged_nodes)
        merged_nodes, merged_cells, _remap = dedupe_coincident_points(merged_nodes, merged_cells)
        if len(merged_nodes) != n_nodes_before_merge:
            mesh_changed_by_repair = True
            merged_cells = merged_cells.astype(np.int32)
            prism_cells = _remap[prism_cells]

            merged_cells, _ = drop_degenerate_tets(merged_nodes, merged_cells, min_cell_size, "Final merge")

        # 构建最终对象
        nodes_obj = NodeArray.from_array(merged_nodes)

        # 阶段 A/B/B' 之后、最终装配之前的混合网格收尾修补阶段（跨类型
        # 非流形面拼接、BL 棱柱长细比局部重铺、collapsed-corner 棱柱
        # 降级为四面体）拆到了 mesh_background_mixed_repair.py（本文件超
        # 过 400 行上限）——三个子步骤共享同一组滚动状态，作为一个整体
        # 一起搬运，逐字对应原来这里的代码，未改动任何数值逻辑。
        (merged_nodes, prism_cells, merged_cells,
         nodes_obj, mesh_changed_by_repair) = _repair_mixed_mesh_post_stage_c(
            merged_nodes, prism_cells, merged_cells,
            nodes_obj, mesh_changed_by_repair, min_cell_size,
        )

        # 阶段 D：见 run_stage_d_interface_repair 模块文档——在完整混合
        # 面图（棱柱+四面体）上定向修复 BL/core 界面相邻体积比违规，只
        # 局部重铺四面体一侧，不改动任何棱柱/BL 几何。
        (merged_nodes, merged_cells, nodes_obj,
         _stage_d_changed, stage_d_actions) = run_stage_d_interface_repair(
            merged_nodes, prism_cells, merged_cells, nodes_obj, validator,
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
        # 输出前硬检查（流形、面退化、带洞由下方 map_generated_boundaries 判定）
        from ...validation.cell_validity import require_nondegenerate_cells
        require_nondegenerate_cells(merged_nodes, prism_cells, merged_cells)
        volumes = TetrahedralCells.compute_volumes(nodes_obj, merged_cells)
        cells_obj = TetrahedralCells(connectivity=merged_cells, volumes=volumes)

        prism_cells = prism_cells.astype(np.int32)
        n_prism = len(prism_cells)
        prism_cells_obj = None
        if n_prism > 0:
            prism_volumes = PrismCells.compute_volumes(nodes_obj, prism_cells)
            prism_cells_obj = PrismCells(connectivity=prism_cells, volumes=prism_volumes)

        # 边界分组：外部面按节点坐标与输入面网格逐面对应（同时校验流形、面不退化、体网格外表面与输入表面逐面相同）
        from ..utils.mesh_boundary import map_generated_boundaries
        boundaries_obj = map_generated_boundaries(
            merged_nodes, prism_cells, merged_cells, surface_nodes, surface_faces, surface_boundaries,
        )

        metadata = GridMetadata(
            node_count=len(merged_nodes),
            cell_count=n_prism + len(merged_cells),
            boundary_groups=list(boundaries_obj.groups.keys()),
            file_format="hybrid"
        )

        # 输入面网格随体网格保存（与外部导入路径同一格式）：参考面积估算用；边界组已逐面给出（boundaries.faces）
        return VolumeMeshData(
            nodes=nodes_obj,
            cells=cells_obj,
            boundaries=boundaries_obj,
            metadata=metadata,
            prism_cells=prism_cells_obj,
            surface_mesh={'nodes': surface_nodes, 'faces': surface_faces, 'boundaries': surface_boundaries},
        )
    except PartialMeshReady:
        raise
    except Exception as e:
        logger.error(f"Error in generate_hybrid_mesh: {e}")
        traceback.print_exc()
        raise
