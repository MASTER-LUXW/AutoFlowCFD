"""体网格生成的边界组分类。

决定哪些边界组面应进行边界层（BL）挤出，哪些应作为外部域壳的一部分
原样送入约束四面体化器，并修正每个合格子壳的缠绕方向，使 BL 挤出
向流体域内部生长，而非信任输入网格未经验证的原始面缠绕方向。

分类按*命名边界组*进行，而非按原始全局连通分量：真实汽车表面网格
通常会在小接触面处将壁面组（如车底）与相邻组（地面/隧道）焊接在一起，
因此对所有候选面的简单"连通分量"遍历会将车身+地面+隧道融合为一个
整体，失去区分能力。将分析范围限定在每个组自己的面子集上可避免此问题。

底层几何原语（连通分量、射线-三角形相交、封闭壳体内部点、带符号体积、
包围盒接触面判定）拆到了同目录 mesh_domain_classify_geometry.py，本文件
只保留 classify_boundary_groups 这个上层分类编排逻辑。
"""

from typing import List, NamedTuple, Tuple, TYPE_CHECKING

import numpy as np
from loguru import logger

from .mesh_domain_classify_geometry import (
    _face_edges,
    _connected_components,
    find_point_inside_closed_shell,
    _signed_volume,
    _bbox_touch_fraction,
)

if TYPE_CHECKING:
    from ...schema.grid_boundaries import BoundaryMap

# 这些边界类型永远是开放流动边界或无摩擦（滑移）壁面，因此无论几何形状
# 如何，它们的面都不会被挤出边界层：自由滑移/对称面上不存在需要解析的
# 近壁速度梯度，真正的开放边界上更是完全没有壁面。SLIP_WALL 覆盖例如
# "tunnel"/"farfield" 命名的边界（见 nas_parser_boundary.py 的关键词表和
# bc_handler.py 的 _classify）——此前这里遗漏了它，导致隧道壁（在关键词表
# 修复之前会落到默认的 'WALL' bc_type）仍可能被挤出边界层，而对于横跨
# 整个计算域的壁面，这几乎立刻就会坍缩（1-2 层内就撞到对面的壁面/车身）。
# PERIODIC 同理——周期面是一个数学配对构造（见
# grid/face_connectivity.py::pair_periodic_boundary_faces），不是物理壁面，
# 那里没有边界层可挤出。
NEVER_EXTRUDE_BC_TYPES = {'VELOCITY_INLET', 'PRESSURE_OUTLET', 'SYMMETRY', 'SLIP_WALL', 'PERIODIC'}

# 自身开放边比例低于此值的子壳体，在判定朝向时按闭合（嵌入）实体处理，
# 即使存在一个很小的真实开口（例如车身在与地面小接触面处焊接）。
_CLOSED_OPEN_EDGE_FRACTION = 0.01

# 判定一个节点是否"贴合"某个包围盒面所用的相对容差（相对于计算域特征长度），
# 与 mesh_utils.check_reached_boundary 已有的 1e-6 约定保持一致。
_BBOX_TOUCH_RTOL = 1e-6


class SubShell(NamedTuple):
    """一个已分类、绕向已修正的边界分组子片段。"""
    faces: np.ndarray          # (n, 3) int，共享节点数组中的索引
    extrude: bool
    group_name: str


def classify_boundary_groups(
    nodes: np.ndarray,
    surface_faces: np.ndarray,
    boundaries: 'BoundaryMap',
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[np.ndarray]]:
    """把每个边界分组的面分割为可挤出边界层 vs. 仅用于核心区域两类，
    可挤出的面已做绕向修正以保证边界层生长方向正确。

    Args:
        nodes: (n_nodes, 3) 表面节点坐标
        surface_faces: (n_faces, 3) 表面连接关系
        boundaries: 含分组（单元/面索引）与 bc_types 的 BoundaryMap
        bbox_min, bbox_max: 整体（未加padding）计算域范围，形状 (3,)

    Returns:
        extrude_faces: (m, 3) 绕向已修正、可挤出边界层的面
        core_faces: (k, 3) 原样用作外壳 PLC 输入的面
            （m + k == n_faces；每个输入面恰好出现在其中一类）
        extruded_group_names: 至少有部分面被挤出边界层的边界分组名称
            （供日志/诊断用）
        hole_points: 每个发现的封闭嵌入实体子部件（例如与计算域外壳
            隔离的车身）各取一个点——必须传给
            mesh_tetgen_core.fill_core_volume 作为 tetgen 的 hole 种子点，
            否则 tetgen 会用虚假的四面体去填充该实体自身内部（以及其
            边界层区块围成的空腔），这些四面体会与已经占据该空间的
            边界层棱柱重叠，而不是正确地把这部分区域排除在外。贴着
            计算域边界包围盒的壁面（地面/隧道）永远不是 hole——它是
            终止于计算域自身外边界的一张开放曲面，没有需要排除的
            封闭内部空间。
    """
    L_char = float(np.max(bbox_max - bbox_min))
    tol = L_char * _BBOX_TOUCH_RTOL

    extrude_face_rows: List[np.ndarray] = []
    core_face_rows: List[np.ndarray] = []
    extruded_group_names: List[str] = []
    hole_points: List[np.ndarray] = []

    for name, cell_idx in boundaries.groups.items():
        bc_type = boundaries.bc_types.get(name)
        group_faces = surface_faces[cell_idx].copy()

        if bc_type in NEVER_EXTRUDE_BC_TYPES:
            core_face_rows.append(group_faces)
            continue

        inverse, counts, face_of_edge = _face_edges(group_faces)
        labels = _connected_components(group_faces, inverse, face_of_edge)

        any_extruded_in_group = False

        for comp_id in np.unique(labels):
            comp_face_mask = labels == comp_id
            comp_faces = group_faces[comp_face_mask]

            # 先检查是否贴合包围盒（bounding-box）某一面，再看下面的开放边
            # 比例判据。这个开放边比例不是拓扑不变量：一个大而平的薄片
            # （地面/隧道壁）网格划分足够细后，内部边数会远超过周边边数，
            # 仅凭网格密度就可能落入"闭合"阈值以内——已实测确认：一个
            # >=150x150 划分的平面会被误判成"闭合的嵌入实体"，进而其法向
            # 朝向由一个接近零（数值噪声级别）的带号体积决定，而不是专门
            # 为这种形状设计的 bbox 方向判据，导致本该只参与核心区域填充的
            # 平面被错误地做了边界层挤出。真正的嵌入实体（车身）即使在与
            # 地面小接触面处焊接，也不会主要贴合单一 bbox 面
            # （_BBOX_TOUCH_MAJORITY=0.9 是它自身节点的占比阈值），所以先做
            # 这项检查不会改变这种情形的判定结果。
            comp_node_idx = np.unique(comp_faces)
            direction = _bbox_touch_fraction(nodes, comp_node_idx, bbox_min, bbox_max, tol)

            if direction is not None:
                # 主要贴合某一个 bbox 面：说明这是地面/侧壁一类的薄片，属于
                # 计算域外壳的一部分。法向朝向由该 bbox 方向决定，而不是面
                # 绕序（面绕序对带有真实自由边界的薄片并不可靠）。
                from .mesh_utils import compute_face_normals
                comp_normals = compute_face_normals(nodes, comp_faces)
                mean_normal = comp_normals.mean(axis=0)
                if np.dot(mean_normal, direction) < 0:
                    comp_faces = comp_faces[:, [1, 0, 2]]  # 翻转绕序
                extrude_face_rows.append(comp_faces)
                any_extruded_in_group = True
                continue

            # 不主要贴合单一 bbox 面。重新在该子分量范围内单独统计边信息，
            # 让开放边比例只反映它自身的边界，而不是整个分组的边界。
            _, sub_counts, _ = _face_edges(comp_faces)
            n_unique_edges = len(sub_counts)
            n_open_edges = int(np.count_nonzero(sub_counts == 1))
            open_fraction = n_open_edges / max(n_unique_edges, 1)

            if open_fraction < _CLOSED_OPEN_EDGE_FRACTION:
                # 近似闭合（嵌入实体，例如车身）：按自身包围体积的正负号
                # 定朝向，不直接信任输入的面绕序。
                volume = _signed_volume(nodes, comp_faces)
                if volume < 0:
                    comp_faces = comp_faces[:, [1, 0, 2]]  # 翻转绕序
                extrude_face_rows.append(comp_faces)
                any_extruded_in_group = True

                hole_pt = find_point_inside_closed_shell(nodes, comp_faces)
                if hole_pt is not None:
                    hole_points.append(hole_pt)
                else:
                    logger.warning(
                        f"Could not find a reliable interior point for closed "
                        f"solid '{name}' (component with {len(comp_faces)} "
                        f"faces) - skipping its tetgen hole marker. The core "
                        f"fill may include spurious tetrahedra inside this "
                        f"solid's own BL block."
                    )
            else:
                # 开放且不贴合 bbox：属于外壳壁面（入口/出口/隧道一类），
                # 在别处有真正的自由边界。原样保留、作为核心 PLC 的一部分
                # 使用。
                core_face_rows.append(comp_faces)

        if any_extruded_in_group:
            extruded_group_names.append(name)

    extrude_faces = (
        np.vstack(extrude_face_rows) if extrude_face_rows
        else np.empty((0, 3), dtype=surface_faces.dtype)
    )
    core_faces = (
        np.vstack(core_face_rows) if core_face_rows
        else np.empty((0, 3), dtype=surface_faces.dtype)
    )
    logger.info(
        f"Boundary classification: {len(extrude_faces)} faces eligible for "
        f"BL extrusion (groups: {extruded_group_names}), "
        f"{len(core_faces)} faces used as-is for the outer domain shell, "
        f"{len(hole_points)} isolated embedded solid(s) marked as tetgen holes"
    )

    return extrude_faces, core_faces, extruded_group_names, hole_points
