"""
AutoFlowCFD - 边界面组标签打标 (从 face_connectivity.py 拆出，控制单文件行数)

本模块提供三个把 FRFaceConnectivity 的边界面按所属边界组
（WALL/INLET/OUTLET/...）打标签的函数，供 fr_solver/boundary.py 的 BC
幽灵态构建等调用点统一复用（WMLES 与气动力积分经幽灵态提供者的组编码取壁面）。
"""

from typing import Dict, Optional, Tuple

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

    按 owner **单元**匹配：只在每个边界单元恰好属于一个组时成立（合成网格按单元定义的组即如此）。一个单元被
    多个组同时声称拥有（角/棱单元的两个边界面分属两个组）时，单元粒度的输入已经丢掉了"哪个面属于哪个组"，
    无从判定——报错，要求提供原始面网格走逐面几何匹配（`tag_boundary_groups_by_geometry`）。本项目的体网格
    生成、导入与 .nas 体网格求解路径都带原始面网格（`VolumeMeshData.surface_mesh`，随 .pkl 一起保存），走的是
    逐面匹配。2026-10-17 以前这里只给 warning、按最后遍历到的组把该单元的全部边界面打成同一个组（I-125）。

    Raises:
        ValueError: 有边界单元被多个组同时声称拥有
    """
    n_faces = face_conn.n_faces
    group_code = np.full(n_faces, -1, dtype=np.int32)
    name_to_code: Dict[str, int] = {}

    boundary_idx = face_conn.get_boundary_face_indices()
    boundary_owners = face_conn.owner_cell[boundary_idx]

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
        raise ValueError(
            f"{len(ambiguous)} 个边界单元被多个边界组同时声称拥有（角/棱单元的边界面分属不同的组），按单元匹配"
            f"无法判定每个面属于哪个组。请提供原始面网格（--surface-mesh，或用 grid generate-volume / "
            f"import-volume 生成的带面网格的 .pkl）走逐面几何匹配。例：{sample_str}"
            + (f"（另有 {len(ambiguous) - 10} 个）" if len(ambiguous) > 10 else ""))

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
            f"list) - build_boundary_ghost_provider rejects untagged boundary faces "
            f"(no boundary condition can be assigned to them)."
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
    from autoflowcfd.grid.mesh_gen.utils.mesh_boundary import match_points_to_surface_groups

    n_faces = face_conn.n_faces
    group_code = np.full(n_faces, -1, dtype=np.int32)

    boundary_idx = face_conn.get_boundary_face_indices()
    surf_groups = surface_mesh["boundaries"].groups
    name_to_code: Dict[str, int] = {name: code for code, name in enumerate(surf_groups)}
    if len(boundary_idx) == 0:
        return group_code, name_to_code

    _names, group_index = match_points_to_surface_groups(
        face_conn.center[boundary_idx], surface_mesh["nodes"], surface_mesh["faces"],
        surf_groups, distance_tolerance_factor,
    )
    matched = group_index >= 0
    group_code[boundary_idx[matched]] = group_index[matched]

    n_unmatched = int(np.sum(~matched))
    if n_unmatched > 0:
        logger.warning(
            f"{n_unmatched}/{len(boundary_idx)} boundary faces (face-level geometric "
            f"match) did not match any surface boundary group within tolerance - "
            f"build_boundary_ghost_provider rejects untagged boundary faces "
            f"(no boundary condition can be assigned to them)."
        )

    return group_code, name_to_code


def tag_boundary_groups_for_mesh(mesh, face_conn: Optional[FRFaceConnectivity] = None) -> Tuple[np.ndarray, Dict[str,
                                                                                                                 int]]:
    """所有调用点的统一入口：优先用逐面几何匹配，缺少原始面网格数据时
    回退到单元级别匹配。

    `mesh.boundary_surface_mesh`（HighOrderMesh.load_from_volume_mesh 从 VolumeMeshData.surface_mesh 透传；体网格
    生成、导入与 .nas 体网格求解路径都带它，随 .pkl 一起保存）可用时走 `tag_boundary_groups_by_geometry`；
    不可用时（按单元定义组的合成网格、旧版 .pkl）走 `tag_boundary_groups`，有歧义的单元报错。当时的四处调用点（BC 幽灵态构建、WMLES 壁面剪应力、
    两处气动系数积分）此前各自重复同一段
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
    # 周期组不参与单元级匹配：周期面配对之后是内部面（`face_connectivity_periodic.py`），这些组在边界上已经
    # 没有面；留在候选里的话，同时贴着周期面与别的边界的角点单元，其余边界面会被标成周期组
    bc_types = getattr(mesh, "boundary_bc_types", None) or {}
    groups = {name: cells for name, cells in (mesh.boundary_groups or {}).items() if bc_types.get(name) != "PERIODIC"}
    return tag_boundary_groups(fc, groups)
