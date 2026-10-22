"""阶段 D：BL / 核心界面相邻体积比定向修复（`generate_hybrid_mesh` 的收尾阶段，在阶段 A/B/B' 与最终装配之间运行）。

## 做什么

在**完整混合面图**（棱柱 + 四面体，`FaceExtractor.extract_faces_mixed` 的全局单元索引：棱柱 `[0, n_prism)`、
四面体 `[n_prism, ...)`）上找跨 BL / 核心界面、相邻体积比超过质量门阈值（`max_adjacent_volume_ratio`，与最终
质量报告同一判据）的面，把四面体一侧标为坏单元，交给阶段 B' 的局部空腔重铺（`remesh_core_cavity`）。棱柱一侧
（BL 层结构、厚度、节点）全程不动。最多 `max_rounds` 轮：一批重铺可能在新邻接关系上暴露新的违规。

## 为什么

阶段 A/B/B' 的坏单元判据只作用于核心 tetgen 四面体（`merged_cells`，不含棱柱），看不到最后一层棱柱与紧邻
四面体之间的体积级差；完整混合面图上的体积比原先只在最终质量报告里算一次、不反馈给任何修复阶段。本阶段是
唯一能"看见"这条界面并对其采取行动的修复阶段。

## 约束

* `remesh_core_cavity` 的接受门是通用逐单元形状指标（偏斜度 / 长宽比），不是"体积压到相邻棱柱的 N 倍以内"：
  形状合格但体积未达标的过渡情形可能重铺后仍超阈值。
* 近重合点对造成的近零体积 sliver（BL 逐顶点冻结撕裂柱高、再被跨类型非流形缝合桥接）不在这里处理：缝合阶段
  先按空腔局部尺度焊接近重合点（`mesh_repair_cavity_shared.py::_weld_near_coincident_boundary_points`）并做
  事后质量门（`patch_nonmanifold_cavity_mixed`）。约束 Delaunay 必须尊重给定边界点集，不焊接的话任何重铺都
  绕不开这两点间的退化单元。
* 排查过程（含证伪的方向：质心细分、挤出源头的三角形高度一致性冻结、坍缩角棱柱专用拆分）见
  `ProjectFiles/V2.0/40_网格阶段D-BL与核心界面体积比修复-排查记录.md`。
"""

import numpy as np
from typing import List, Tuple, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from ...schema.grid_nodes import NodeArray
    from ...validation.quality_validator import MeshQualityValidator


def run_stage_d_interface_repair(
    merged_nodes: np.ndarray,
    prism_cells: np.ndarray,
    merged_cells: np.ndarray,
    nodes_obj: 'NodeArray',
    validator: 'MeshQualityValidator',
    max_rounds: int = 2,
) -> Tuple[np.ndarray, np.ndarray, 'NodeArray', bool, List[str]]:
    """阶段 D：基于完整混合面图的 BL/core 界面相邻体积比定向修复。

    Args:
        merged_nodes: 节点坐标数组（阶段 A/B/B' 之后）。
        prism_cells: 棱柱连接关系（本函数只读，从不修改/重铺）。
        merged_cells: 四面体连接关系（阶段 A/B/B' 之后，本函数唯一可能
            修改的单元数组）。
        nodes_obj: merged_nodes 对应的 NodeArray（供面提取/体积计算用，
            调用方已构造好，避免本函数重复构造）。
        validator: 质量校验器实例，复用其 `max_adjacent_volume_ratio`
            阈值，与最终质量报告的判据完全一致（不引入新的独立阈值）。
        max_rounds: 最多重复几轮（修复一批界面单元可能在其新生成的邻接
            关系上暴露新的、原本被旧单元形状掩盖的违规——用有限轮次
            收敛，而非一次性假设修复完全，也避免在退化输入上死循环）。

    Returns:
        (merged_nodes, merged_cells, nodes_obj, changed, actions)
        —— merged_nodes/nodes_obj 原样透传（本阶段从不新增/移动节点属于
        棱柱一侧；`remesh_core_cavity` 可能为四面体一侧插入新的内部点，
        因而 merged_nodes/nodes_obj 在有实际修复发生时会被替换为更新
        后的版本）。

    局限见模块文档"约束"一节。
    """
    from ...schema.grid_cells import TetrahedralCells, PrismCells
    from ...schema.grid_nodes import NodeArray
    from ..extraction.face_extractor import FaceExtractor
    from .mesh_repair_cavity import remesh_core_cavity

    actions: List[str] = []
    changed = False
    n_prism = len(prism_cells)
    threshold = validator.thresholds.get('max_adjacent_volume_ratio', 5.0)

    if n_prism == 0 or len(merged_cells) == 0:
        return merged_nodes, merged_cells, nodes_obj, changed, actions

    for round_idx in range(max_rounds):
        tet_volumes = TetrahedralCells.compute_volumes(nodes_obj, merged_cells.astype(np.int32))
        prism_volumes = PrismCells.compute_volumes(nodes_obj, prism_cells.astype(np.int32))
        cell_volumes = np.concatenate([prism_volumes, tet_volumes])

        mixed_faces = FaceExtractor.extract_faces_mixed(
            prism_cells.astype(np.int64), merged_cells.astype(np.int64), nodes_obj
        )
        conn = mixed_faces.connectivity
        interior_mask = conn[:, 1] >= 0
        owner = conn[interior_mask, 0]
        neighbor = conn[interior_mask, 1]

        # 只关心跨越 BL/core 边界的面：一侧是棱柱（全局索引 < n_prism），
        # 另一侧是四面体（全局索引 >= n_prism）。
        owner_is_prism = owner < n_prism
        neighbor_is_prism = neighbor < n_prism
        crosses_interface = owner_is_prism != neighbor_is_prism
        if not np.any(crosses_interface):
            break

        io = owner[crosses_interface]
        jn = neighbor[crosses_interface]
        v_i = cell_volumes[io]
        v_j = cell_volumes[jn]
        ratio = np.maximum(v_i, v_j) / np.maximum(np.minimum(v_i, v_j), 1e-300)
        bad_face = ratio > threshold
        if not np.any(bad_face):
            break

        # 取每条违规界面面上的四面体一侧全局索引，转回局部（四面体数组
        # 内）索引：全局索引减 n_prism。
        tet_side_global = np.where(owner_is_prism[crosses_interface][bad_face],
                                   jn[bad_face], io[bad_face])
        bad_tet_local = np.unique(tet_side_global - n_prism)
        bad_tet_local = bad_tet_local[(bad_tet_local >= 0) & (bad_tet_local < len(merged_cells))]
        if len(bad_tet_local) == 0:
            break

        bad_cell_mask = np.zeros(len(merged_cells), dtype=bool)
        bad_cell_mask[bad_tet_local] = True

        logger.info(
            f"Stage D (round {round_idx + 1}/{max_rounds}): {len(bad_tet_local)} core tet(s) "
            f"exceed adjacent-volume-ratio threshold {threshold:.2f} against a neighbouring "
            f"BL prism (full mixed-mesh face graph, not visible to stage A/B/B') - "
            f"attempting local cavity retiling on the tet side only"
        )

        tet_faces = FaceExtractor.extract_faces(merged_cells.astype(np.int32), nodes_obj)
        new_nodes, new_cells, _new_bad_mask, cavity_actions = remesh_core_cavity(
            merged_nodes, merged_cells, tet_faces, bad_cell_mask, validator,
        )
        actions.extend(f"Stage D: {a}" for a in cavity_actions)

        if new_cells is merged_cells:
            # remesh_core_cavity 没有接受任何候选空腔时原样返回同一个 cells 对象（不拷贝），身份相等是唯一精确的
            # "这一轮什么都没接受"判据。按单元数与节点数组判断是错的：多数被接受的重铺是"N 个单元 -> N 个单元、
            # 不新增内部点"，只有连接关系变了（此前因此把真实生效的重铺整体丢弃，阶段 D 从未生效）。
            actions.append("Stage D: no cavity candidate improved quality - stopping")
            break

        merged_nodes = new_nodes
        merged_cells = new_cells
        nodes_obj = NodeArray.from_array(merged_nodes)
        changed = True

    return merged_nodes, merged_cells, nodes_obj, changed, actions
