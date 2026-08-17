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

# Boundary types 那个 are always 打开-flow boundaries 或 frictionless
# (slip) walls, so their faces are never BL-extruded regardless of
# geometry: there is no near-wall velocity gradient to resolve at a
# free-slip/symmetry surface, and no wall at all at a genuine open
# boundary. SLIP_WALL covers e.g. "tunnel"/"farfield"-named boundaries
# (see nas_parser_boundary.py's keyword table and bc_handler.py's
# _classify) - previously missing here, so a tunnel wall (falling through
# to the 'WALL' bc_type default before that keyword-table fix) could still
# get BL-extruded, which collapses almost immediately for a domain-
# spanning wall (hits the opposite wall/body within 1-2 layers). PERIODIC
# is the same story - a periodic plane is a mathematical pairing
# construct (see grid/face_connectivity.py::pair_periodic_boundary_faces),
# not a physical wall; there is no boundary layer to extrude there.
NEVER_EXTRUDE_BC_TYPES = {'VELOCITY_INLET', 'PRESSURE_OUTLET', 'SYMMETRY', 'SLIP_WALL', 'PERIODIC'}

# A sub-shell whose own open-edge fraction is below this is treated as a
# closed (embedded) solid for orientation purposes, even with a small real
# opening (e.g. a body welded to the ground at a small contact patch).
_CLOSED_OPEN_EDGE_FRACTION = 0.01

# Relative tolerance (of the domain's characteristic length) for deciding a
# node lies "on" a bounding-box face, matching mesh_utils.check_reached_boundary's
# existing 1e-6 convention.
_BBOX_TOUCH_RTOL = 1e-6


class SubShell(NamedTuple):
    """One classified, winding-corrected 片段 的 a boundary 分组."""
    faces: np.ndarray          # (n, 3) int, indices into the shared node array
    extrude: bool
    group_name: str


def classify_boundary_groups(
    nodes: np.ndarray,
    surface_faces: np.ndarray,
    boundaries: 'BoundaryMap',
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
    """分割 every boundary 分组's 面 到 extrude-eligible vs. core-仅,
    与 extrude-eligible 面 winding-corrected for 修正 BL growth
    方向.

    Args:
        nodes: (n_nodes, 3) surface node coordinates
        surface_faces: (n_faces, 3) surface connectivity
        boundaries: BoundaryMap with groups (cell/face indices) and bc_types
        bbox_min, bbox_max: overall (unpadded) domain extent, shape (3,)

    Returns:
        extrude_faces: (m, 3) winding-corrected faces eligible for BL extrusion
        core_faces: (k, 3) faces to use unmodified as outer-shell PLC input
            (m + k == n_faces; every input face appears in exactly one)
        extruded_group_names: names of boundary groups that got at least
            some faces extruded (for logging/diagnostics)
        extrude_face_groups: (m,) str array, the original boundary-group name
            for each row of extrude_faces (same order/length) - lets the
            caller attribute BL-extruded tets back 到 它们的 源 分组
            directly 通过 面 位置, 代替 的 matching 节点 indices
            against the pre-extrusion surface (which cannot work for
            genuinely-displaced BL nodes; see mesh_boundary.py).
        hole_points: one point per closed embedded-solid sub-component found
            (e.g. a car body, isolated from the domain's outer shell) - must
            be passed 到 mesh_tetgen_core.fill_core_volume as tetgen hole
            seeds, or tetgen fills that solid's own interior (and its BL
            block's enclosed cavity) with spurious tetrahedra that overlap
             BL prisms 已经 occupying 那个 空间, 代替 的
            correctly excluding it. A bbox-touching wall (ground/tunnel) is
            never a hole - it's an 打开 sheet terminating at  域's
            自身 outer boundary, 与 no enclosed interior 到 排除.
        core_face_groups: (k,) str array, the original boundary-group name
            for each row of core_faces (same order/length) - lets the
            caller attribute core (tetgen-filled) boundary tets back to
            它们的 源 分组 通过 tetgen facet markers, 哪个 存活
            boundary subdivision unlike node-index matching (see
            mesh_tetgen_core.fill_core_volume's `face_markers`/nobisect=False
            路径, needed for graded 最大-单元-大小 regions 到 actually
            refine cells near a coarse far-field wall).
        is_closed_solid_face: (m,) bool array, parallel to extrude_faces -
            True for rows from a closed embedded solid (the `hole_points`
            branch, e.g. a car body), False for a bbox-touching wall sheet
            (ground/tunnel-like). Currently unused by mesh_background.py
            (received as `_is_closed_solid_face`) - it was meant to let the
            caller 构建 最大-单元-大小 grading spheres centered 在...上 仅 
            isolated solid's 自身 geometry, distinct 从 a bbox-touching
            wall sheet 那个 can 跨越 nearly  整体 域 footprint, but
            that per-solid grading-sphere approach was abandoned (see
            mesh_tetgen_core.py's note where those functions used to live)
            in favor 的 one flat core-填充 区域. Kept 此处 自从 it's a
            cheap, 已经-computed byproduct 那个 a future per-solid
            grading scheme could reuse.
    """
    L_char = float(np.max(bbox_max - bbox_min))
    tol = L_char * _BBOX_TOUCH_RTOL

    extrude_face_rows: List[np.ndarray] = []
    extrude_face_group_rows: List[np.ndarray] = []
    is_closed_solid_rows: List[np.ndarray] = []
    core_face_rows: List[np.ndarray] = []
    core_face_group_rows: List[np.ndarray] = []
    extruded_group_names: List[str] = []
    hole_points: List[np.ndarray] = []

    for name, cell_idx in boundaries.groups.items():
        bc_type = boundaries.bc_types.get(name)
        group_faces = surface_faces[cell_idx].copy()

        if bc_type in NEVER_EXTRUDE_BC_TYPES:
            core_face_rows.append(group_faces)
            core_face_group_rows.append(np.full(len(group_faces), name))
            continue

        inverse, counts, face_of_edge = _face_edges(group_faces)
        labels = _connected_components(group_faces, inverse, face_of_edge)

        any_extruded_in_group = False

        for comp_id in np.unique(labels):
            comp_face_mask = labels == comp_id
            comp_faces = group_faces[comp_face_mask]

            # 检查 bounding-box 接触 第一, 之前  打开-边-fraction
            # test below. That fraction is not a topological invariant: a
            # large flat sheet (ground/tunnel wall) has far more internal
            # edges than perimeter edges once meshed finely enough, so it
            # can fall under the "closed" threshold by mesh density alone -
            # empirically confirmed to misclassify a >=150x150-division
            # flat plane as a "closed embedded solid", which then gets its
            # orientation decided by a near-zero (numerically-noisy) signed
            # volume instead of the bbox-direction check meant for exactly
            # this shape, and gets BL-extruded when it should stay core-only.
            # A real embedded solid (car body) never predominantly touches a
            # single bbox face even when welded to the ground at a small
            # contact patch (_BBOX_TOUCH_MAJORITY=0.9 of its own nodes), so
            # checking this first doesn't change that case's outcome.
            comp_node_idx = np.unique(comp_faces)
            direction = _bbox_touch_fraction(nodes, comp_node_idx, bbox_min, bbox_max, tol)

            if direction is not None:
                # Predominantly sits on one bbox face: a floor/wall-like
                # sheet 那个's 部分 的  域's outer shell. Orientation
                # comes from that bbox direction, not face winding (which is
                # unreliable for a sheet with a real free boundary).
                from .mesh_utils import compute_face_normals
                comp_normals = compute_face_normals(nodes, comp_faces)
                mean_normal = comp_normals.mean(axis=0)
                if np.dot(mean_normal, direction) < 0:
                    comp_faces = comp_faces[:, [1, 0, 2]]  # flip winding
                extrude_face_rows.append(comp_faces)
                extrude_face_group_rows.append(np.full(len(comp_faces), name))
                is_closed_solid_rows.append(np.zeros(len(comp_faces), dtype=bool))
                any_extruded_in_group = True
                continue

            # Doesn't predominantly sit 在...上 a 单个 bbox 面. Recompute 边
            # stats scoped to this sub-component alone so the open-edge
            # fraction reflects only its own boundary, not the whole group's.
            _, sub_counts, _ = _face_edges(comp_faces)
            n_unique_edges = len(sub_counts)
            n_open_edges = int(np.count_nonzero(sub_counts == 1))
            open_fraction = n_open_edges / max(n_unique_edges, 1)

            if open_fraction < _CLOSED_OPEN_EDGE_FRACTION:
                # Closed-like (embedded solid, e.g. car body): orient by the
                # sign of its own enclosed volume, not by trusting input
                # winding directly.
                volume = _signed_volume(nodes, comp_faces)
                if volume < 0:
                    comp_faces = comp_faces[:, [1, 0, 2]]  # flip winding
                extrude_face_rows.append(comp_faces)
                extrude_face_group_rows.append(np.full(len(comp_faces), name))
                is_closed_solid_rows.append(np.ones(len(comp_faces), dtype=bool))
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
                # Open and not bbox-touching: an outer-shell wall
                # (inlet/outlet/tunnel-like) with a genuine free boundary
                # elsewhere. 使用 unmodified as 部分 的  core PLC.
                core_face_rows.append(comp_faces)
                core_face_group_rows.append(np.full(len(comp_faces), name))

        if any_extruded_in_group:
            extruded_group_names.append(name)

    extrude_faces = (
        np.vstack(extrude_face_rows) if extrude_face_rows
        else np.empty((0, 3), dtype=surface_faces.dtype)
    )
    extrude_face_groups = (
        np.concatenate(extrude_face_group_rows) if extrude_face_group_rows
        else np.empty((0,), dtype=object)
    )
    core_faces = (
        np.vstack(core_face_rows) if core_face_rows
        else np.empty((0, 3), dtype=surface_faces.dtype)
    )
    core_face_groups = (
        np.concatenate(core_face_group_rows) if core_face_group_rows
        else np.empty((0,), dtype=object)
    )
    is_closed_solid_face = (
        np.concatenate(is_closed_solid_rows) if is_closed_solid_rows
        else np.empty((0,), dtype=bool)
    )

    logger.info(
        f"Boundary classification: {len(extrude_faces)} faces eligible for "
        f"BL extrusion (groups: {extruded_group_names}), "
        f"{len(core_faces)} faces used as-is for the outer domain shell, "
        f"{len(hole_points)} isolated embedded solid(s) marked as tetgen holes"
    )

    return (
        extrude_faces, core_faces, extruded_group_names, extrude_face_groups,
        hole_points, core_face_groups, is_closed_solid_face,
    )
