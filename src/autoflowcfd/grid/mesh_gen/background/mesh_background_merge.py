"""合并 BL + tetgen 核心网格装配（单次生成尝试）。

_build_merged_mesh 完成 generate_hybrid_mesh（mesh_background.py）编排的
每次实际装配工作：分类边界组、挤出 BL 层、tetgen 填充剩余核心体积、
然后将两者拼接为一组带单元级源组标记的合并 (nodes, cells)。拆分到独立
模块纯粹是为了控制 mesh_background.py 的文件大小——此函数在
generate_hybrid_mesh 的重试循环（阶段 B）之外没有独立复用，因此保持为
私有函数（前导下划线）并位于其唯一调用者的模块旁边。

两个和 _build_merged_mesh 自身逻辑无关、只是被它调用的独立工具函数
（边界面的最大边长细分、`--*-only` 调试导出）拆到了同目录
mesh_background_merge_utils.py。

_build_merged_mesh 本身的两条分支（没有任何曲面组适合挤出边界层 / 至少
一个曲面组适合挤出边界层）进一步拆到了 mesh_background_merge_no_bl.py
和 mesh_background_merge_with_bl.py（原文件超过 400 行上限）——本文件
只保留两条分支共用的前置步骤（曲面分类、facet marker 映射表构建）和分
发逻辑。两个常量（OVERSIZED_TET_FACTOR、CORE_FILL_VOLUME_CAP_FRACTION）
被两个分支模块延迟导入使用，因此仍留在本文件里。
"""

import numpy as np
from typing import Dict, Optional, Tuple, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from ...schema.grid_boundaries import BoundaryMap

from ..utils.mesh_domain_classify import classify_boundary_groups

# tetgen 自身缺乏按体积细化 oversized 单元的可靠机制时的确定性兜底阈值
# （参见 subdivide_oversized_tetrahedra 的文档字符串了解原因以及
# 为何使用重心细分）。乘以各调用点自身的区域目标 maxvol（并非精确
# 按该目标施加），因此正常的粗而合理的梯度过渡不会被误触发——只有
# 真正的离群值（实测达目标的 100x-16000x）才会被Splitting。
OVERSIZED_TET_FACTOR = 5.0

# 核心四面体允许增长到的最大尺寸（以 max_cell_size**3 的分数表示），
# 用于"直接从 BL 真实外表面填充"的主分支——这是施加在整个核心区域上的
# 单一平坦上限（tetgen 自身的距离梯度背景网格/度量尺寸在此环境下会
# 段错误，参见 fill_core_volume 的 `regions` 文档），因此这是控制
# BL 外表面到核心尺寸跳变陡度的唯一杠杆。
# 刻意独立于 mesh_tetgen_core.CORE_VOLUME_CAP_FRACTION (0.08)——
# 后者针对阶段 B 的小局部空腔重划，是不同负载和独立原理
# （参见该常量的文档字符串），最初在此复用 0.08 使得过渡明显过慢/过细
# （核心单元数过多）；0.2 仍偏慢/偏细，0.3 偏快/偏粗。0.25 是当前
# 折中值——如果仍不合适可直接在此调整。
CORE_FILL_VOLUME_CAP_FRACTION = 0.25


def _build_merged_mesh(
    surface_nodes: np.ndarray,
    surface_faces: np.ndarray,
    bounding_box: Dict[str, np.ndarray],
    surface_boundaries: 'BoundaryMap',
    growth_rate: float = 1.2,
    min_cell_size: float = 0.001,
    max_cell_size: Optional[float] = None,
    extra_thickness_limit: Optional[np.ndarray] = None,
    bl_layers: Optional[int] = None,
    export_bl_only: bool = False,
    export_bl_only_path: Optional[str] = None,
    export_core_only: bool = False,
    export_core_only_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, int]:
    """构建合并网格（BL 棱柱 + TetGen 核心四面体）。"""
    bbox_min = np.asarray(bounding_box['min'], dtype=np.float64)
    bbox_max = np.asarray(bounding_box['max'], dtype=np.float64)

    # 注意: 此处 surface_nodes 已经是米制单位（由 NASParser.parse 转换）。
    # CLI 传入的 max_cell_size 也是米制。无需额外缩放。

    logger.info("Step 1/4: Classifying boundary groups (extrude vs. core-only)...")
    (extrude_faces, core_faces, extruded_groups, extrude_face_groups,
     hole_points, core_face_groups, _is_closed_solid_face) = classify_boundary_groups(
        surface_nodes, surface_faces, surface_boundaries, bbox_min, bbox_max
    )

    # tetgen facet-marker 机制的 marker ID 映射表
    # （attribute_cells_from_trifaces）——仅在 max_cell_size 梯度生效时需要
    # （它是唯一会关闭 fill_core_volume 的 nobisect 的选项，而正是这会导致
    # 简单的节点索引匹配边界归属在细分面上失效）。
    # 0 被 tetgen 保留为"无标记"（未标记/内部面）。
    group_name_to_marker = {name: i + 1 for i, name in enumerate(surface_boundaries.groups.keys())}
    marker_to_name = {v: k for k, v in group_name_to_marker.items()}

    if len(extrude_faces) == 0:
        from .mesh_background_merge_no_bl import _build_merged_mesh_no_bl
        return _build_merged_mesh_no_bl(
            surface_nodes, surface_faces, surface_boundaries, extrude_faces,
            hole_points, max_cell_size, group_name_to_marker, marker_to_name,
            export_core_only, export_core_only_path,
        )
    else:
        from .mesh_background_merge_with_bl import _build_merged_mesh_with_bl
        return _build_merged_mesh_with_bl(
            surface_nodes, surface_boundaries, bbox_min, bbox_max,
            extrude_faces, core_faces, extruded_groups, extrude_face_groups,
            hole_points, core_face_groups, group_name_to_marker, marker_to_name,
            growth_rate, min_cell_size, max_cell_size, extra_thickness_limit, bl_layers,
            export_bl_only, export_bl_only_path, export_core_only, export_core_only_path,
        )
