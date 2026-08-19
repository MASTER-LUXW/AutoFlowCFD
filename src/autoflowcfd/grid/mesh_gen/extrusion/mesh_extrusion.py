"""边界层生成的网格挤出模块。

实现沿法向的表面挤出以创建层状网格，适用于 CFD 仿真中的边界层
分辨率。单层几何步骤（法向平均、尖角补偿）在 mesh_layer_step.extrude_single_layer
中实现；将层状棱柱堆栈转换为四面体在 mesh_prism_to_tet 中实现；
两个尖角特征衰减启发式方法在 mesh_extrusion_attenuation.py 中实现——
均从本文件拆分以满足项目 450 行/文件的规范。
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger

from ..utils.mesh_utils import check_reached_boundary
from .mesh_layer_step import extrude_single_layer
from ..utils.mesh_front_collision import clamp_budget_for_convergence, freeze_self_colliding_nodes
from .mesh_bl_growth import _MAX_SAFETY_LAYERS, compute_layer_thickness
from .mesh_extrusion_attenuation import (
    _compute_sharp_angle_attenuation,
    _compute_edge_distance_field,
)

# 限制每层自身的厚度，使其单元体积永远不会比前一层的跳跃超过这个倍数——
# 棱柱层的体积随其厚度缩放（底面积通过平移偏移几乎不变），所以限制连续层
# 之间的厚度比是保持实际相邻单元体积比质量门控（quality_validator.py 自己的
# max_adjacent_volume_ratio=5.0，“STAR-CCM+ 对齐的体积变化指导”）通过构造满足
# 的直接、廉价代理，而不是依赖它仅作为后生成质量报告失败以后再修复。近壁
# 增长率已经很保守（~1.05-1.3 用于 y+ 控制），所以这在实际中很少真正绑定——
# 作为安全底线保留而不是移除。
MAX_ADJACENT_VOLUME_RATIO = 5.0


def extrude_layers(
    surface_nodes: np.ndarray,
    surface_faces: np.ndarray,
    normals: np.ndarray,
    bounding_box: Dict[str, np.ndarray],
    growth_rate: float = 1.2,
    min_cell_size: float = 0.001,
    taper_scale: 'Optional[np.ndarray]' = None,
    thickness_limit: 'Optional[np.ndarray]' = None,
    max_cell_size: 'Optional[float]' = None,
    bl_layers: 'Optional[int]' = None,
    normal_faces: 'Optional[np.ndarray]' = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """沿法向挤出表面以创建边界层 (BL) 网格。

    对 `bl_layers` 层进行精细、几何分级的挤出（增长率 `growth_rate`，
    为近壁 y+ 控制调整）。挤出在那里停止——剩余体积直接从 BL 自身的
    真实外表面通过 mesh_background_merge._build_merged_mesh 的单次
    tetgen 核心填充调用填充，使用 TetGen 自身的非结构化分级而不是继续
    结构化层挤出（ProjectFiles Part13 P49；单独挤出的“过渡”阶段桥接 BL
    到核心填充已被尝试并放弃——在尖角物体上是一个真正困难的计算几何
    鲁棒性问题，在六次不同的缓解尝试后仍未解决）。

    Args:
        surface_nodes: 基础表面节点，形状=(n_nodes, 3)
        surface_faces: 用于层拓扑的表面连接（convert_layers_to_prisms/
            convert_layers_to_tetrahedra 从什么构建单元）——形状=(n_faces, 3)。
            当 mesh_corner_split.split_sharp_corners 在上游运行时，这是它的
            `topology_faces`（真实三角形 + 斜角/帽三角形）。
        normals: 面法向，形状=(len(normal_faces), 3)——注意：大小为
            `normal_faces`，不是 `surface_faces`，当两者不同时（见下方
            `normal_faces`）。
        bounding_box: 域限制以防止过冲
        growth_rate: 第 1 阶段 (BL) 层厚度的几何增长率
        min_cell_size: 最小允许单元尺寸（米）
        taper_scale: 可选 float 数组，[0, 1]，形状=(n_nodes,)。
            缩放每个节点的每层位移（1 = 完全挤出，0 = 每层精确保持在
            原始位置）。用于在与非挤出边界组共享的 seam 处将 BL 表面
            平滑过渡到零，而不是移动 seam（撕裂网格）或硬钉住它（这会
            将 seam 自身的三角形坍缩到零面积碎片）。
        thickness_limit: 可选 float 数组，米，形状=(n_nodes,)（未约束处
            为 np.inf），来自 mesh_tetgen_core.compute_local_thickness_limit。
            限制每个节点跨所有层的*累积*位移，使得在紧密局部特征（例如
            车身底板离地面很近）上会聚的 BL 前沿在交叉前冻结，而不是以
            均匀速率生长然后重叠。
        max_cell_size: 可选目标层厚度（米），仅用作 ANSA 风格上限
            （`0.5 * max_cell_size`），限制这个手动/结构化挤出允许
            总共增长多远——安全底线，给定 `bl_layers` 已经限制层数，
            很少真正绑定；核心 tetgen 填充通过自身的非结构化分级处理
            输出到 `max_cell_size` 的实际大小范围。
        bl_layers: 可选覆盖挤出多少层后停止。None（默认）使用 8。
        normal_faces: 可选 `surface_faces` 子集，用于确定每个节点偏移
            方向的每节点法向平均（mesh_layer_step.extrude_single_layer
            自己的平均）——None（默认，也是每个现有调用方）使用
            `surface_faces` 自身，行为不变。传递 mesh_corner_split.
            split_sharp_corners 的 `real_face_mask` 过滤面到这里（与从
            相同子集计算的 `normals` 配对）以从法向平均中排除其斜角/
            帽行：斜角三角形自身的角节点已经从其真实面有修正的、单补丁
            法向——在平均中包含斜角面本身会用第三个任意“连接”方向重新
            污染它，违背分割角的整体目的。`surface_faces` 中任何地方引用
            的每个节点保证也出现在 `normal_faces` 中（斜角/帽三角形从不
            引入未使用的节点），所以这永远不会留下未位移的节点。

    Returns:
        all_nodes: 所有层的连接节点，形状=(total_nodes, 3)
        layer_connectivity: 每层的面索引列表
    """
    # 计算特征长度和初始厚度
    domain_size = np.linalg.norm(
        bounding_box['max'] - bounding_box['min']
    )
    normal_faces = surface_faces if normal_faces is None else normal_faces

    # 对于汽车 CFD（Re ~ 1e6 - 1e7），第一层高度应目标 y+ ~ 1-30
    # 使用经验公式：delta_y1 ≈ L * Re^(-0.5) / 100
    # 保守估计：第一层 0.002 * L_char

    # 关键修复：min_cell_size 应该是第一层厚度的主要控制
    # 之前的逻辑使用 domain_size * 0.002，对于紧密几何可能太大
    # 现在我们使用 min_cell_size 作为基础，仅用 domain_size 作为上限
    base_thickness_from_domain = domain_size * 0.002  # 域尺寸的 0.2%
    base_thickness = min(min_cell_size, base_thickness_from_domain)

    # 确保最小厚度合理（但尊重用户的 min_cell_size）
    # 仅当 min_cell_size 不合理地小（< 0.1mm）时才应用硬地板
    if base_thickness < 0.0001:
        logger.warning(
            f"min_cell_size={min_cell_size}m is extremely small, using 0.0001m as safety floor"
        )
        base_thickness = 0.0001

    bl_layers = 8 if bl_layers is None else max(0, int(bl_layers))
    bl_growth_rate = growth_rate
    
    # 两个独立的衰减启发式，通过最小值组合（两者中更保守的获胜）而不是
    # 一个默默覆盖另一个（之前的行为：这个本地计算并记录，然后在下面的
    # edge_attenuation 重新赋值时立即丢弃——真实的、浪费的工作，从不影响实际
    # 挤出）。_compute_sharp_angle_attenuation 直接响应节点自身最尖锐的相邻
    # 二面角（< 90 度时地板为 0.1）；_compute_edge_distance_field 响应到最近
    # 识别的尖锐顶点的欧氏距离（在该顶点处精确为 0 地板，在 2 倍网格自身
    # 尖锐边长度尺度上攀升）。单独任何一个都不足以在稀疏网格化的圆角处可靠地
    # 衰减（物理上平滑的曲线只跨越 1-2 个元素，对两者都仍然读作尖锐边），
    # 所以组合它们让任何一个先响应。
    logger.info("Computing sharp-angle attenuation for BL...")
    angle_attenuation = _compute_sharp_angle_attenuation(
        surface_nodes, surface_faces, normals,
        normal_faces=normal_faces
    )
    n_attenuated = int(np.sum(angle_attenuation < 0.9))
    if n_attenuated > 0:
        logger.info(f"Sharp-angle attenuation active for {n_attenuated} nodes")

    # 优化：计算边距离场以在尖角附近衰减 BL 厚度
    # 这防止边处的严重几何畸变和自交叉
    #
    # min_feature_radius=nominal_bl_thickness：V2.0 专项攻关记录（cube_demo
    # BL 质量campaign 第八轮）：这个判据本来完全没有曲率半径过滤，纯角度
    # 阈值（45°）把车身圆角（cube_demo 实测真实半径 7.6mm）密集三角化后的
    # 普通面片边全部误判成尖锐折痕，导致衰减场沿整条棱边被压到地板值——
    # 参见 _compute_edge_distance_field 自己的 min_feature_radius 文档了解
    # 完整根因链条。这里传入的名义边界层累积厚度（几何级数和，与
    # split_sharp_corners 在别处用 min_cell_size 作为半径判据不同——那里
    # 判断的是"网格分辨率够不够细来正确描述这条边"，这里判断的是"边界层
    # 会不会长得比这个圆角的曲率半径还厚"，是两个不同的问题，用同一个
    # min_cell_size 量级的值在这里量纲上就不对）。
    nominal_bl_thickness = (
        base_thickness * (bl_growth_rate ** bl_layers - 1) / (bl_growth_rate - 1)
        if bl_layers > 0 else 0.0
    )
    logger.info("Computing sharp-edge distance field for BL attenuation...")
    distance_attenuation = _compute_edge_distance_field(
        surface_nodes, surface_faces, normals,
        normal_faces=normal_faces,
        min_feature_radius=nominal_bl_thickness,
    )
    n_attenuated = int(np.sum(distance_attenuation < 0.9))
    if n_attenuated > 0:
        logger.info(f"Sharp-edge attenuation active for {n_attenuated} nodes")

    edge_attenuation = np.minimum(angle_attenuation, distance_attenuation)

    # 手动（结构化）挤出也在约 0.5 * max_cell_size 停止（ANSA 风格）
    # 作为安全底线——见上方 Args 文档中这个上限自己的说明了解为什么
    # 给定 bl_layers 已经限制层数，它很少真正绑定。
    effective_max_thickness = max_cell_size * 0.5 if max_cell_size else np.inf

    logger.info(
        f"BL extrusion: {bl_layers} layers, growth_rate={bl_growth_rate}, "
        f"max adjacent volume ratio={MAX_ADJACENT_VOLUME_RATIO:.1f}x\n"
        f"  Max manual thickness: {effective_max_thickness:.4f}m\n"
        f"  Initial thickness: {base_thickness:.6f}m"
    )

    # 防止退化输入
    if growth_rate <= 1.0:
        raise ValueError(f"growth_rate must be > 1.0, got {growth_rate}")

    n_nodes = len(surface_nodes)
    current_nodes = surface_nodes.copy()
    all_layer_nodes: List[np.ndarray] = [current_nodes]

    # 跟踪累积厚度
    current_thickness = 0.0
    n_layers_generated = 0

    # 前一层的实际（上限后）厚度，用于下面的 MAX_ADJACENT_VOLUME_RATIO——
    # 在第一层提交之前为 None。
    previous_layer_thickness: Optional[float] = None

    # 分配剩余预算用于厚度限制
    remaining_budget = (
        thickness_limit.copy() if thickness_limit is not None
        else np.full(len(surface_nodes), np.inf, dtype=np.float64)
    )
    n_limited = int(np.sum(np.isfinite(remaining_budget)))
    if n_limited:
        logger.info(f"Local BL thickness limiting active for {n_limited} nodes")

    for layer_idx in range(_MAX_SAFETY_LAYERS):
        # 检查是否已到达域边界
        if check_reached_boundary(current_nodes, bounding_box):
            logger.info(
                f"Reached domain boundary at layer {layer_idx + 1}, "
                f"stopping extrusion (generated {n_layers_generated} layers)"
            )
            break

        # 一旦达到请求的 BL 层数就停止——剩余体积直接从这个真实外表面
        # 由核心 tetgen 填充（见本函数自己的文档字符串）。
        if n_layers_generated == bl_layers:
            logger.info(
                f"Reached bl_layers={bl_layers}, stopping extrusion "
                f"(cumulative thickness={current_thickness:.6f}m) - remaining "
                f"volume filled directly by the core tetgen fill"
            )
            break

        # 计算本层结束时的目标累积厚度
        next_cumulative_thickness = compute_layer_thickness(
            current_thickness, growth_rate, base_thickness, n_layers_generated,
        )

        # 附加停止条件：ANSA 风格最大厚度上限
        if next_cumulative_thickness > effective_max_thickness:
            logger.info(f"Reached ANSA-style max thickness ({effective_max_thickness:.4f}m). Stopping manual extrusion.")
            break

        # 本层的实际位移
        layer_thickness = next_cumulative_thickness - current_thickness
        if layer_thickness <= 1e-12:
            logger.info(f"Layer thickness too small ({layer_thickness:.6e}m), stopping.")
            break

        # 见 MAX_ADJACENT_VOLUME_RATIO 自己的注释。
        if previous_layer_thickness is not None:
            max_layer_thickness = previous_layer_thickness * MAX_ADJACENT_VOLUME_RATIO
            if layer_thickness > max_layer_thickness:
                logger.info(
                    f"Layer {layer_idx + 1}: capped thickness {layer_thickness:.6f}m -> "
                    f"{max_layer_thickness:.6f}m to keep the adjacent-cell volume ratio "
                    f"at or below {MAX_ADJACENT_VOLUME_RATIO:.1f}x the previous layer"
                )
                layer_thickness = max_layer_thickness
                next_cumulative_thickness = current_thickness + layer_thickness

        # Reactive convergence clamp
        clamp_budget_for_convergence(current_nodes, surface_faces, remaining_budget)

        # 合并边衰减与 taper_scale 以在尖角附近获得更平滑的 BL
        effective_taper = taper_scale * edge_attenuation if taper_scale is not None else edge_attenuation

        # 沿平均法向挤出节点
        new_nodes = extrude_single_layer(
            current_nodes, normal_faces, normals, layer_thickness,
            taper_scale=effective_taper, remaining_budget=remaining_budget,
        )

        # 反应式局部碰撞冻结
        frozen_now = freeze_self_colliding_nodes(
            new_nodes, current_nodes, surface_faces, remaining_budget,
        )
        if len(frozen_now):
            logger.warning(
                f"Layer {layer_idx + 1}: locally froze {len(frozen_now)} node(s) "
                f"where the advancing front would self-intersect; "
                f"extrusion continues elsewhere"
            )

        all_layer_nodes.append(new_nodes)
        current_nodes = new_nodes
        current_thickness = next_cumulative_thickness
        previous_layer_thickness = layer_thickness
        n_layers_generated += 1

    logger.info(
        f"Extrusion completed: {n_layers_generated} layers generated, "
        f"total nodes: {len(all_layer_nodes) * n_nodes}, "
        f"final cumulative height: {current_thickness:.4f}m"
    )

    # 返回连接的节点和层连接。
    # layer_connectivity 是每层一个面数组的列表。每层使用相同的拓扑
    # （surface_faces），所以我们只是复制它。
    layer_connectivity = [surface_faces.copy() for _ in range(n_layers_generated)]
    return np.vstack(all_layer_nodes), layer_connectivity
