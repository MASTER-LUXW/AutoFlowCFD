"""_build_merged_mesh 的"含 BL"主分支：挤出边界层棱柱，再用 tetgen
从 BL 的真实外表面直接填充剩余体积。

从 mesh_background_merge.py 拆分出来（原文件超过 400 行上限），纯粹是
代码搬运——逻辑与 _build_merged_mesh 里原来的 `else:` 分支（对应
`if len(extrude_faces) == 0:` 的否定分支）完全一致，只是把它变成一个独
立的模块级函数，由 mesh_background_merge._build_merged_mesh 在该分支下
直接调用并原样返回其结果。`--bl-only` 调试导出那一小段进一步拆到了
mesh_background_merge_bl_export.py（自包含的"构造导出对象 -> 写文件 ->
sys.exit(0)"流程，与本文件其余逻辑没有共享状态）。
"""

import numpy as np
from typing import Optional, Tuple, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from ...schema.grid_boundaries import BoundaryMap

from ..extrusion.mesh_extrusion import extrude_layers
from ..tetgen.mesh_prism_to_tet import convert_layers_to_prisms
from ..utils.mesh_utils import compute_face_normals
from ..utils.mesh_corner_split import split_sharp_corners
from .mesh_background_merge_bl_export import _export_bl_only_and_exit
from .mesh_background_merge_utils import _export_partial_mesh_and_exit
from ..tetgen.mesh_tetgen_core import (
    build_seam_taper_scale, fill_core_volume,
    compute_local_thickness_limit,
    attribute_cells_from_trifaces, generate_core_background_points,
    subdivide_oversized_tetrahedra,
    CORE_TETGEN_MINRATIO, CORE_TETGEN_MINDIHEDRAL,
)


def _build_merged_mesh_with_bl(
    surface_nodes: np.ndarray,
    surface_boundaries: 'BoundaryMap',
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    extrude_faces: np.ndarray,
    core_faces: np.ndarray,
    extruded_groups,
    extrude_face_groups: np.ndarray,
    hole_points,
    core_face_groups: np.ndarray,
    group_name_to_marker: dict,
    marker_to_name: dict,
    growth_rate: float,
    min_cell_size: float,
    max_cell_size: Optional[float],
    extra_thickness_limit: Optional[np.ndarray],
    bl_layers: Optional[int],
    export_bl_only: bool,
    export_bl_only_path: Optional[str],
    export_core_only: bool,
    export_core_only_path: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, int]:
    """至少有一个曲面组适合挤出边界层时的主路径：挤出 BL 棱柱，再用
    tetgen 从 BL 的真实外表面直接填充剩余体积（不再有独立的"过渡"阶段）。

    对应 mesh_background_merge._build_merged_mesh 里原来的 `else:` 分
    支，逐字搬运，未改动任何数值逻辑。

    Returns:
        与 _build_merged_mesh 自身完全相同的 9 元组：
        (merged_nodes, prism_cells, tet_cells, cell_groups, n_bl_cells,
        source_vertex, topology_faces, bl_cell_groups, n_transition_cells)
    """
    # OVERSIZED_TET_FACTOR / CORE_FILL_VOLUME_CAP_FRACTION 定义在
    # mesh_background_merge.py（本函数唯一的调用者所在文件），延迟导入
    # 以避免循环导入——本模块被 mesh_background_merge.py 导入，不能在
    # 模块顶层反向导入它。
    from .mesh_background_merge import OVERSIZED_TET_FACTOR, CORE_FILL_VOLUME_CAP_FRACTION

    logger.info(
        f"Step 2/4: Extruding BL layers from {len(extrude_faces)} faces "
        f"(groups: {extruded_groups})..."
    )
    n_surface_nodes = len(surface_nodes)
    taper_scale = build_seam_taper_scale(n_surface_nodes, extrude_faces, core_faces)

    # 限制每个节点的累积 BL 厚度，防止紧邻对向面（如车身底部靠近地面）
    # 的两侧在交叉前就冻结，而不是完全依赖 repair_nonmanifold_cells
    # 事后清理产生的重叠（参见 compute_local_thickness_limit 的文档字符串
    # 了解为何这是强缓解而非形式保证）。
    domain_size = float(np.linalg.norm(bbox_max - bbox_min))
    thickness_limit = compute_local_thickness_limit(
        surface_nodes, extrude_faces, np.unique(extrude_faces), domain_size
    )
    if extra_thickness_limit is not None:
        thickness_limit = np.minimum(thickness_limit, extra_thickness_limit)

    # 在挤出之前，将每个锐角/硬边顶点的挤出合格子网格Splitting为每个
    # 光滑面片一份副本——参见 mesh_corner_split 的模块文档字符串了解
    # 为何单个平均法向/节点偏移无法表示真正的价-3+角点而不产生自交
    # （已在 cube_demo 上直接确认：级联碰撞冻结
    # mesh_front_collision.freeze_self_colliding_nodes 从第一个 BL 层
    # 开始就触发，正好在车身自身的锐边/角点处，在少数几层内影响大部分
    # 表面）。
    # taper_scale/thickness_limit/extrude_face_groups 是按原始顶点/面
    # 的数组——用相同方式扩展它们（副本继承源的值/组），再送入
    # extrude_layers/下游单元归属。
    # min_feature_radius=min_cell_size: 自身几何暗示的曲率半径达到或超过
    # BL 目标近壁单元尺寸的边被视为普通曲面（无论网格多粗）而非锐折痕
    # 来Splitting——参见 split_sharp_corners 的自身文档字符串。低于该
    # 尺度时，更密的网格分辨率也不会显著改变 BL 如何看待该特征，因此它
    # 仍被归类为硬边。
    split_nodes, topology_faces, real_face_mask, orig_of_node, bevel_source_face = (
        split_sharp_corners(
            surface_nodes, extrude_faces, min_feature_radius=min_cell_size
        )
    )
    taper_scale = taper_scale[orig_of_node]
    thickness_limit = thickness_limit[orig_of_node]
    extrude_face_groups = np.concatenate(
        [extrude_face_groups, extrude_face_groups[bevel_source_face]]
    )

    # source_vertex 将Splitting后的局部（取模后）节点索引映射回其代表的
    # 原始表面顶点——在 n_surface_nodes 以下时是恒等映射（未被Splitting
    # 触及），在以上时映射到该副本所复制自的顶点。阶段 B 自身的
    # 节点到顶点簿记（mesh_repair_bl_thickness.
    # compute_bl_thickness_limit_override）已通过其
    # node_original_vertex/local_surface_faces 参数支持非恒等映射——
    # 甚至在Splitting存在之前就已为此可能性而构建。
    source_vertex = orig_of_node

    normal_faces = topology_faces[real_face_mask]
    normals = compute_face_normals(split_nodes, normal_faces)
    # 几何挤出在 BL 阶段结束时停止——剩余体积从 BL 自身的真实外表面
    # 在一次非结构化、梯度的 tetgen 填充中直接覆盖（参见本函数自身的
    # "直接从 BL 真实外表面填充"段落，位于 BL 棱柱/导出块之后，了解
    # 完整原理——以及导致此设计的 ProjectFiles Part12 P45/P46 和架构
    # 历史）。
    bl_nodes, bl_layer_conn = extrude_layers(
        split_nodes, topology_faces, normals,
        bounding_box={'min': bbox_min, 'max': bbox_max},
        growth_rate=growth_rate, min_cell_size=min_cell_size,
        taper_scale=taper_scale, thickness_limit=thickness_limit,
        max_cell_size=max_cell_size,
        bl_layers=bl_layers,
        normal_faces=normal_faces,
    )
    # extrude_layers 自身的 BL 层数——与实际生成数的裁剪无论如何都保留：
    # BL 阶段本身仍可能在达到请求的 bl_layers 之前提前停止（域边界/
    # 自碰撞冻结）。
    # bl_layer_conn 每个挤出步有一个条目（以 extrude_layers 自身的
    # n_layers_generated 术语），但 bl_nodes（从 extrude_layers 的
    # all_layer_nodes 经 np.vstack）持有 n_layers_generated + 1 个节点
    # 块——起始的 layer-0 块加上每步追加的一个（参见 extrude_layers 自身
    # 的 all_layer_nodes = [current_nodes] 然后每步 .append(new_nodes)）。
    # 直接使用 len(bl_layer_conn) 作为节点层数会差一，并破坏由此导出的
    # 每个步长（nodes_per_layer、bl_split_offset、outer_offset 等）：
    # 已在 cube_demo 上直接确认，它使节点索引算术将 BL "layer 1" 的
    # 体壁节点副本放在完全不相关的远场（隧道出口尺度）坐标上，而非几
    # 毫米之外，产生看似破碎/自交的 BL 几何（并且几乎可以肯定就是
    # 由此产生的损坏 BL 外表面 PLC 向下游 TetGen 的 "Recovering
    # segments" 挂起提供的输入）。
    n_layers = len(bl_layer_conn) + 1
    _effective_bl_layers = bl_layers if bl_layers is not None else 8
    _effective_bl_layers = int(np.clip(_effective_bl_layers, 0, n_layers - 1))

    nodes_per_layer = len(bl_nodes) // n_layers
    outer_offset = (n_layers - 1) * nodes_per_layer
    bl_split_offset = _effective_bl_layers * nodes_per_layer

    # bl_layer_conn[:_effective_bl_layers]（不是 +1）：convert_layers_to_
    # prisms 现在内部已考虑 layer_connectivity 每步持有一个条目（参见
    # 其自身文档字符串/修复）——此处也传入 +1 会重复计数并静默破坏此
    # 调用点，就像下方过渡四面体调用点在该修复之前一样（参见本项目的
    # 自身调查：一个横跨域的 ~14 m^3 过渡四面体，而非 tetgen 缺陷）。
    bl_prisms, bl_face_of_cell = convert_layers_to_prisms(
        bl_nodes[:bl_split_offset + nodes_per_layer],
        bl_layer_conn[:_effective_bl_layers],
        topology_faces,
        min_cell_size=min_cell_size,
    )
    n_bl_cells = len(bl_prisms)
    logger.info(f"  BL mesh: {len(bl_nodes)} nodes, {len(bl_prisms)} prism cells")

    if export_bl_only:
        _export_bl_only_and_exit(
            export_bl_only_path, bl_nodes, bl_prisms, n_bl_cells,
            bl_split_offset, nodes_per_layer, _effective_bl_layers,
        )

    # 将每个 BL 单元直接归属回其源边界分组——通过位置而非节点索引匹配
    # 挤出前的表面：convert_layers_to_prisms 自身的 bl_face_of_cell 将
    # 每个存活的单元直接映射回其 extrude_faces 行（一个简单的平铺——
    # 每层 len(extrude_faces) 个棱柱的连续块——不再精确成立，因为该
    # 函数现在可以丢弃解析零体积的折叠层棱柱，参见其自身文档字符串）。
    # 这对每个存活的 BL 单元都是精确的，包括 body/ground 自身外表面
    # 的绝大部分——节点索引匹配永远无法到达那里（参见 mesh_boundary.py
    # ——那些节点在被挤出真正位移后获得全新的偏移索引，因此其挤出后的
    # 面无法匹配从原始挤出前节点索引构建的查找表中的任何内容）。
    #
    # 只有 LAYER 0 自身的棱柱（底盖为实际物理壁面的那些）被标记源组
    # 名称——其他所有层得到 '' 代替，即使 bl_face_of_cell 会乐意告诉我
    # 们它们的源面。这在具体上很重要，不仅是美观：BL 柱可能在锐利/复杂
    # 几何特征处提前终止（触发局部厚度上限——参见
    # compute_local_thickness_limit），最后一个存活棱柱的顶盖随后成为
    # 合法、不可避免的终端边界面——真实的面，而非面提取中的 bug——但
    # 它不是物理壁面，而是该特定柱恰好停止位置的人工产物。将所有层
    # 标记为相同（之前的行为，自真正棱柱存在之前就一直未变）会将该
    # 终端面归属到与真实壁面相同的 "body"/WALL 组，这会错误地在原本
    # 应该是开放内部空间的地方施加无滑移条件。已在真实案例上确认为
    # 真实（非理论）效应（ProjectFiles Part6/7 P21）：33,448 个这样的
    # 面，集中在立方体锐边处，分布在层 1-3——不在 BL/过渡接缝处（如
    # 最初怀疑），证实这是预先存在的 BL 挤出特性，与棱柱/四面体Splitting
    # 无关，只是之前从未从它静默合并进的壁面组中分离出来。
    #
    # Layer-0 检测是简单的节点索引范围检查，不是 convert_layers_to_prisms
    # 的返回值：第 L 层的节点始终占据
    # bl_nodes[L*nodes_per_layer : (L+1)*nodes_per_layer]
    # （extrude_layers 自身的节点布局，自本会话之前就未变），因此棱柱
    # 的底盖（v0）< nodes_per_layer 是"此棱柱底面为 layer 0"的充要
    # 条件——无需为了重新推导其返回的节点索引中已隐含的信息而通过
    # convert_layers_to_prisms 引入新的返回值。
    is_layer0_prism = bl_prisms[:, 0] < nodes_per_layer
    # 确保 bl_face_of_cell 为整数类型以用于索引
    bl_face_of_cell = bl_face_of_cell.astype(np.int64) if not np.issubdtype(bl_face_of_cell.dtype, np.integer) else bl_face_of_cell
    bl_cell_groups = np.where(is_layer0_prism, extrude_face_groups[bl_face_of_cell], '')

    # Layer 0 保留裸表面节点索引不变；BL 自身的真正最后一层（现在始终
    # 是 extrude_layers 实际生成的最后一层，因为 bl_only=True）占据
    # bl_nodes 自身的最后一个块。core_faces 自身的节点索引只对
    # outer_nodes 有效，因为与 core_faces 共享的接缝节点 taper_scale==0
    # 因此从未离开其原始（layer-0）位置。
    outer_nodes = bl_nodes[outer_offset:outer_offset + nodes_per_layer]
    bl_outer_surface = bl_layer_conn[-1]
    if not np.issubdtype(bl_outer_surface.dtype, np.integer):
        logger.warning(f"bl_outer_surface dtype is {bl_outer_surface.dtype}, converting to int64")
        bl_outer_surface = bl_outer_surface.astype(np.int64)

    # --- 直接从 BL 自身的真实外表面填充，完全没有独立的过渡阶段
    # （既不挤出也不估计）。曾尝试针对估计的核心侧边界构建独立的
    # 过渡区域填充（一个看似合理的设计：独立保护两个界面）——该估计
    # 表面在 cube_demo 自身的锐利 90 度角点上被证明是真正困难的
    # 计算几何问题（盒子到处都有价-3+角点）：六种不同的缓解策略
    # （简单平均法向偏移、真实 BL 挤出使用的相同最小二乘斜接方向、
    # 使用 mesh_front_collision.py 自身经验证的逐步冻结机制的多步
    # 增量挤出、事后收缩/拉回/局部平滑修复循环，以及最后让 tetgen
    # 自身的边界恢复鲁棒性处理尝试修复仍不完善的估计）都留下了某些
    # 残留自交，被 tetgen 自身的硬性、与 nobisect 无关的输入有效性
    # 前提条件直接拒绝（已直接确认：无论 nobisect 如何都成立——该
    # 开关仅控制已有效的输入是否可进一步细分以提高质量，而非是否
    # 容忍真正自交的输入）。这个更简单的替代方案完全规避了整个问题：
    # 根本不需要估计或构建任何表面，因为 outer_nodes 是真实的、已经
    # 挤出的 BL 表面，并在同一次真实运行中被独立确认无自交（
    # mesh_front_collision.find_self_colliding_faces 命中 0 次）。
    # 一次非结构化、梯度的 tetgen 填充现在覆盖整个剩余体积（以前的
    # "过渡"只是这同一个梯度填充的近壁部分，不再是结构上不同的区域）。
    logger.info(
        f"Step 3/4: Tetrahedralizing core volume "
        f"({len(core_faces)} core-only faces + BL outer surface)..."
    )
    core_plc_points = outer_nodes.copy()
    core_plc_faces = np.vstack([topology_faces, core_faces])

    face_markers = None
    regions = None
    background_points = None
    if max_cell_size is not None:
        # bl_outer_surface 自身的部分也用其源组标记（extrude_face_groups）
        # ——通常与 bl_cell_groups 冗余（BL/核心界面面本身从不暴露给域
        # 外部），但完全被接缝锥度钉住（折叠为零 BL 厚度）的柱的"外表面"
        # 变成了真实暴露的壁面——参见 attribute_cells_from_trifaces 自身
        # 的调用方文档。顶点混合了真正生长和早期冻结节点的面被留为未标记
        # 而非猜测，回退到 mesh_boundary.py 自身的 UNCLASSIFIED 兜底，
        # 而非被静默错误归属到物理壁面组。
        bl_outer_markers = np.array(
            [group_name_to_marker.get(n, 0) for n in extrude_face_groups], dtype=np.int32
        )
        core_markers = np.array(
            [group_name_to_marker.get(n, 0) for n in core_face_groups], dtype=np.int32
        )
        face_markers = np.concatenate([bl_outer_markers, core_markers])
        target_edge_length = max_cell_size
        # 参见本模块自身的 CORE_FILL_VOLUME_CAP_FRACTION 注释
        # （文件顶部）了解此值的调优历史。
        volume_cap_fraction = CORE_FILL_VOLUME_CAP_FRACTION
        regions = [(core_plc_points.mean(axis=0), 1, target_edge_length ** 3 * volume_cap_fraction)]
        background_points = generate_core_background_points(
            core_plc_points, core_plc_faces, target_edge_length
        )
        logger.info(f"TetGen constraint: target_edge_length={target_edge_length:.4f}m, volume_cap={volume_cap_fraction}")

    core_nodes, core_tets, trifaces, triface_markers = fill_core_volume(
        core_plc_points, core_plc_faces, holes=hole_points, regions=regions, face_markers=face_markers,
        background_points=background_points,
        minratio=CORE_TETGEN_MINRATIO, mindihedral=CORE_TETGEN_MINDIHEDRAL,
        force_preserve_boundary=True,
    )
    n_core_boundary = len(core_plc_points)
    if not (len(core_nodes) >= n_core_boundary and np.array_equal(core_nodes[:n_core_boundary], core_plc_points)):
        raise RuntimeError(
            "Core tetgen fill did not preserve its own fixed (real BL "
            "outer surface) boundary verbatim despite "
            "force_preserve_boundary=True - the BL/core splice below "
            "assumes point-for-point preservation and cannot proceed "
            "safely"
        )
    if regions:
        oversized_max_volume = regions[0][2] * OVERSIZED_TET_FACTOR
        core_nodes, core_tets = subdivide_oversized_tetrahedra(
            core_nodes, core_tets, oversized_max_volume
        )
    core_cell_groups = (
        attribute_cells_from_trifaces(core_tets, trifaces, triface_markers, marker_to_name)
        if face_markers is not None
        else np.full(len(core_tets), '', dtype=object)
    )

    if export_core_only:
        path = export_core_only_path
        if not path:
            raise ValueError("export_core_only=True requires export_core_only_path to be set")
        _export_partial_mesh_and_exit(
            core_nodes, np.empty((0, 6), dtype=core_tets.dtype), np.empty(0, dtype=object),
            core_tets, core_cell_groups,
            path, "core-only (tetgen core fill from the real BL outer surface)",
        )

    # 最终拼接：bl_nodes（BL 棱柱，不变，已在自身全局空间中）+ core 自身
    # 的新内部点（core_nodes[:n_core_boundary] 重复 outer_nodes，已存在于
    # bl_nodes 的 outer_offset 处——不重新追加）。
    core_remap = np.empty(len(core_nodes), dtype=np.int64)
    core_remap[:n_core_boundary] = np.arange(outer_offset, outer_offset + n_core_boundary)
    core_remap[n_core_boundary:] = len(bl_nodes) + np.arange(len(core_nodes) - n_core_boundary)
    merged_nodes = np.vstack([bl_nodes, core_nodes[n_core_boundary:]])
    core_tets_remapped = core_remap[core_tets]

    # 棱柱和四面体保持为两个独立的连接数组（参见本函数的文档字符串）
    # 而非一个 vstacked 数组——棱柱的 (n,6) 形状无论如何无法与四面体的
    # (n,4) 共享行布局。不再有独立的"过渡"单元块（参见本段开头的注释）
    # ——n_transition_cells 保持为 0 仅因为 generate_hybrid_mesh 自身的
    # 返回签名仍期望"多少个合并单元源自近壁"的计数；此处的每个四面体
    # 现在都是核心填充来源。
    prism_cells = bl_prisms
    tet_cells = core_tets_remapped
    cell_groups = core_cell_groups
    n_transition_cells = 0
    logger.info(
        f"  Merged mesh: {len(merged_nodes)} nodes, "
        f"{len(prism_cells) + len(tet_cells)} cells "
        f"({len(prism_cells)} BL prisms + {len(tet_cells)} core tets)"
    )

    return (
        merged_nodes, prism_cells, tet_cells, cell_groups, n_bl_cells,
        source_vertex, topology_faces, bl_cell_groups, n_transition_cells,
    )
