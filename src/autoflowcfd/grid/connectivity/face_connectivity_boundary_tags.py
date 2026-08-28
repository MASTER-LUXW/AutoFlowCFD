"""
AutoFlowCFD - 边界面组标签打标 (从 face_connectivity.py 拆出，控制单文件行数)

本模块提供三个把 FRFaceConnectivity 的边界面按所属边界组
（WALL/INLET/OUTLET/...）打标签的函数，供 fr_solver/boundary.py 的 BC
幽灵态构建、core/utils/solver_helpers.py 的 WMLES 壁面剪应力、
postprocess/fr_coefficients.py 的气动系数积分等调用点统一复用。
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from .face_connectivity import FRFaceConnectivity


def tag_boundary_groups(
    face_conn: FRFaceConnectivity, boundary_groups: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, Dict[str, int]]:
    """将边界面按所属边界组（WALL/INLET/OUTLET/...）打标签。

    Args:
        face_conn: build_face_connectivity 的输出
        boundary_groups: BoundaryMap.groups，name -> owner 单元全局索引数组
            （沿用 grid/schema/grid_boundaries.py 的既有约定：棱柱在前、
            四面体在后的同一套全局单元索引空间）

    Returns:
        (group_code: (n_faces,) int32，边界面对应组的整数编码，内部面/未匹配为 -1；
         name_to_code: Dict[str,int]，组名到编码的映射)

    真实 bug（已确认，未修复，2026-08-23 用户直接追问"tag_boundary_groups
    修复了吗"促成排查）：按 owner **单元**索引匹配 boundary_groups——若某个
    单元的两个不同边界面分别属于两个不同的组（角单元/边界 patch 交界处），
    这里对该单元只可能命中*一个*组（`np.isin` 按整个 cell_id 集合匹配，
    最后一次循环覆盖前面的），那个单元全部边界面都会被打上同一个组标签，
    即使其中某个面物理上明明属于另一个组。cube_demo 网格的四个边界组
    （tunnel/outlet/inlet/body）互不重叠，未触发此问题，因此当前不是那次
    真实网格 P1 发散的原因（已用实际数据核实排除），但这仍是一个真实、
    未修复的正确性缺陷，触发条件是任意存在角单元的网格。真正的修复是
    `tag_boundary_groups_by_geometry`（按面本身的物理位置几何最近邻匹配，
    不经过单元级别聚合）+ `tag_boundary_groups_for_mesh`（自动选择：有
    `mesh.boundary_surface_mesh` 时用前者，否则回退到本函数）——本函数
    保留作为没有原始面网格数据可用时（例如从 .pkl 加载、或本项目自己
    `grid generate-volume` 的自包含体网格路径）的后备实现，不删除。
    """
    n_faces = face_conn.n_faces
    group_code = np.full(n_faces, -1, dtype=np.int32)
    name_to_code: Dict[str, int] = {}

    boundary_idx = face_conn.get_boundary_face_indices()
    boundary_owners = face_conn.owner_cell[boundary_idx]

    # 真实 bug 修复（2026-08-23，用户明确要求处理这个此前已确认、未修复
    # 的缺陷）：这个函数按 owner *单元* 匹配，对角单元（同一个单元的两个
    # 不同边界面分属两个不同组）本质上没有足够信息真正修复——单元级别的
    # `boundary_groups: name->cell_ids` 输入本身就丢失了"这个单元具体
    # 哪个面属于哪个组"这个信息，唯一真正修复过的路径是
    # `tag_boundary_groups_by_geometry`（有原始面网格几何数据时优先走
    # 这条，见 `tag_boundary_groups_for_mesh`），本函数只在没有原始面
    # 网格数据时作为后备。既然真正修复不可行，这里做能做到的最好事情：
    # 在赋值*之前*先检测哪些单元被多个组同时声称拥有，把这个此前完全
    # 静默的数据缺陷变成一个明确、可诊断的告警（列出具体单元和冲突的
    # 组名），而不是像之前那样"最后一个循环到的组覆盖前面的"、不留任何
    # 痕迹地错标角单元的某个边界面。
    group_items = list(boundary_groups.items())
    boundary_owner_set = set(np.unique(boundary_owners).tolist())
    ambiguous: Dict[int, list] = {}
    for i in range(len(group_items)):
        name_i, cell_ids_i = group_items[i]
        set_i = set(np.asarray(cell_ids_i).tolist()) & boundary_owner_set
        for j in range(i + 1, len(group_items)):
            name_j, cell_ids_j = group_items[j]
            set_j = set(np.asarray(cell_ids_j).tolist())
            for cid in (set_i & set_j):
                names = ambiguous.setdefault(cid, [])
                if name_i not in names:
                    names.append(name_i)
                if name_j not in names:
                    names.append(name_j)

    if ambiguous:
        sample = list(ambiguous.items())[:10]
        sample_str = "; ".join(f"cell {cid} in groups {names}" for cid, names in sample)
        logger.warning(
            f"tag_boundary_groups: {len(ambiguous)} boundary owner cell(s) are claimed "
            f"by more than one boundary_groups entry (corner/edge cells touching two "
            f"different boundary patches) — this cell-granular fallback cannot "
            f"correctly disambiguate which of that cell's boundary faces belongs to "
            f"which group (only tag_boundary_groups_by_geometry, which needs the "
            f"original surface mesh, can); for these cells, group assignment falls "
            f"back to whichever group is iterated last, so some faces WILL be "
            f"mistagged. Provide the original surface mesh (e.g. via --surface-mesh "
            f"or a .pkl that carries VolumeMeshData.surface_mesh) to use the correct "
            f"geometric path instead. Examples: {sample_str}"
            + (f" (+{len(ambiguous) - 10} more)" if len(ambiguous) > 10 else "")
        )

    for code, (name, cell_ids) in enumerate(boundary_groups.items()):
        name_to_code[name] = code
        cell_id_set = np.asarray(cell_ids)
        mask = np.isin(boundary_owners, cell_id_set)
        group_code[boundary_idx[mask]] = code

    n_unmatched = int(np.sum(group_code[boundary_idx] < 0))
    if n_unmatched > 0:
        logger.warning(
            f"{n_unmatched}/{len(boundary_idx)} boundary faces did not match any "
            f"boundary_groups entry (owner cell not found in any group's cell-index "
            f"list) - these faces will not receive a weak BC penalty term unless "
            f"handled by a default/fallback boundary condition."
        )

    return group_code, name_to_code


def tag_boundary_groups_by_geometry(
    face_conn: FRFaceConnectivity,
    surface_mesh: Dict[str, object],
    distance_tolerance_factor: float = 0.75,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """按真实几何最近邻，逐**面**（不经过任何单元级别聚合）匹配边界组标签。

    修复 `tag_boundary_groups` 的单元级别聚合缺陷（见该函数文档）：直接
    对 `face_conn` 自己的每个边界面中心 `center`，用与
    `mesh_gen/utils/mesh_boundary.py::map_boundaries_by_geometry` 完全
    相同的算法（KD-tree 最近邻 + 按匹配到的表面面自身外接半径缩放的
    容差）在原始表面网格的边界组三角面片里找最近邻——同一个单元的两个
    边界面各自独立判定自己的归属，不会互相覆盖。

    Args:
        face_conn: build_face_connectivity 的输出（要打标签的真正对象）
        surface_mesh: VolumeMeshData.surface_mesh 字典（
            `{'nodes': (n,3), 'faces': (m,3) 三角形连接, 'boundaries': BoundaryMap}`，
            见 mesh_external_import.import_external_volume_mesh 的写入处，
            'boundaries'.groups 是原始表面网格的组名->三角面片索引，与
            `map_boundaries_by_geometry` 用的是同一份数据）
        distance_tolerance_factor: 与 map_boundaries_by_geometry 同名参数
            语义一致

    Returns:
        (group_code, name_to_code)，与 tag_boundary_groups 完全相同的
        返回值约定，可直接替换该函数在所有调用点的返回值
    """
    from scipy.spatial import cKDTree

    n_faces = face_conn.n_faces
    group_code = np.full(n_faces, -1, dtype=np.int32)
    name_to_code: Dict[str, int] = {}

    boundary_idx = face_conn.get_boundary_face_indices()
    if len(boundary_idx) == 0:
        return group_code, name_to_code

    surf_nodes = surface_mesh["nodes"]
    surf_faces = surface_mesh["faces"]
    surf_boundaries = surface_mesh["boundaries"]

    surf_centroids_list: List[np.ndarray] = []
    surf_radius_list: List[np.ndarray] = []
    surf_group_code_list: List[np.ndarray] = []
    for code, (name, face_idx) in enumerate(surf_boundaries.groups.items()):
        name_to_code[name] = code
        face_idx = np.asarray(face_idx)
        face_idx = face_idx[face_idx < len(surf_faces)]
        if len(face_idx) == 0:
            continue
        verts = surf_faces[face_idx]
        pts = surf_nodes[verts]
        centroids = pts.mean(axis=1)
        # 外接半径代理量：质心到自己 3 个顶点的最大距离——跟
        # map_boundaries_by_geometry 用同一个近似，不需要精确外接半径，
        # 只需要一个跟"这个面有多大"成正比的局部尺度。
        radius = np.linalg.norm(pts - centroids[:, None, :], axis=2).max(axis=1)
        surf_centroids_list.append(centroids)
        surf_radius_list.append(radius)
        surf_group_code_list.append(np.full(len(face_idx), code, dtype=np.int32))

    if not surf_centroids_list:
        return group_code, name_to_code

    surf_centroids = np.vstack(surf_centroids_list)
    surf_radius = np.concatenate(surf_radius_list)
    surf_group_code = np.concatenate(surf_group_code_list)

    face_centroids = face_conn.center[boundary_idx]
    tree = cKDTree(surf_centroids)
    dist, nearest_idx = tree.query(face_centroids)
    tolerance = np.maximum(surf_radius[nearest_idx] * distance_tolerance_factor, 1e-12)
    matched = dist <= tolerance

    group_code[boundary_idx[matched]] = surf_group_code[nearest_idx[matched]]

    n_unmatched = int(np.sum(~matched))
    if n_unmatched > 0:
        logger.warning(
            f"{n_unmatched}/{len(boundary_idx)} boundary faces (face-level geometric "
            f"match) did not match any surface boundary group within tolerance - these "
            f"faces will not receive a weak BC penalty term unless handled by a "
            f"default/fallback boundary condition."
        )

    return group_code, name_to_code


def tag_boundary_groups_for_mesh(mesh, face_conn: Optional[FRFaceConnectivity] = None) -> Tuple[np.ndarray, Dict[str, int]]:
    """所有调用点的统一入口：优先用逐面几何匹配，缺少原始面网格数据时
    回退到单元级别匹配。

    真实网格数据（cube_demo）已验证：`mesh.boundary_surface_mesh`（
    HighOrderMesh.load_from_volume_mesh 从 VolumeMeshData.surface_mesh
    透传，input_file 是 .nas 体网格 + --surface-mesh 反推边界的路径才有）
    可用时走 `tag_boundary_groups_by_geometry`；不可用时（例如从 .pkl
    加载——边界分组已经在 pkl 里但原始面网格数据没有一起序列化）回退到
    `tag_boundary_groups`。四处调用点（fr_solver/boundary.py 的 BC 幽灵态
    构建、core/utils/solver_helpers.py 的 WMLES 壁面剪应力、
    postprocess/fr_coefficients.py 的两处气动系数积分）此前各自重复同一段
    `tag_boundary_groups(fc, mesh.boundary_groups or {})`，都受同一个
    单元级别聚合缺陷影响（气动系数积分场景下，被错误纳入/排除的 WALL
    面直接污染 Cd/Cl）——统一到这一个入口，行为对四处调用点完全一致。

    Args:
        mesh: HighOrderMesh 实例
        face_conn: 默认使用 mesh.face_connectivity；单独传入用于尚未把
            结果挂回 mesh 的构造中途场景

    Returns:
        (group_code, name_to_code)，与 tag_boundary_groups 完全相同的
        返回值约定
    """
    fc = face_conn if face_conn is not None else mesh.face_connectivity
    surface_mesh = getattr(mesh, "boundary_surface_mesh", None)
    if surface_mesh is not None:
        return tag_boundary_groups_by_geometry(fc, surface_mesh)
    return tag_boundary_groups(fc, mesh.boundary_groups or {})
