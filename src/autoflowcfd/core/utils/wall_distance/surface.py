"""AutoFlowCFD V2.0 - 从体网格取出壁面三角形（壁面距离的几何来源）。

**为什么用边界面而不是"WALL 组单元的全部节点"**（2026-09-15 修复的一阶物理错误，此前是
"把单元索引当节点索引"）：`BoundaryMap` 存的是单元索引；边界层棱柱有 3 个节点在壁面上、
3 个在第一层之外，按单元取会让壁面虚胖一层。这里取壁面组的**边界面**本身。

**逐面分组**（2026-10-17，I-92）：体网格带原始面网格（`surface_mesh`，生成、导入与 .nas 体网格求解路径都有）时，
每个边界面按自身位置归入面网格的组（`match_points_to_surface_groups`，与边界条件的逐面标记同一个函数）；此前按
"边界面的 owner 单元属于某个壁面组"选取，组交界单元（物面与对称面、出口相接处）的全部边界面都被算成壁面，对称面/
出口上那几个面的壁距被置为 0 附近。没有面网格时（按单元定义组的合成网格）按单元选取，壁面单元同时属于别的组就
无法判定、报错。壁面组的判据是"物理无滑移固壁"（`boundary/wall_kind.py`，认 `bc_overrides`）。
"""

import numpy as np

#: 四边形壁面面片的非平面度容差（相对面片尺度）：判断"四点共面"的浮点重合判据。
_QUAD_PLANARITY_RTOL = 1e-10

#: 边界面中心到面网格三角形的匹配容差（与 `tag_boundary_groups_by_geometry` 同一取值）
_SURFACE_MATCH_TOLERANCE = 0.75


def wall_face_nodes(volume_data, bc_overrides=None) -> np.ndarray:
    """物理无滑移固壁组的边界面节点表 `(n_faces, 3)`。

    Raises:
        ValueError: BoundaryMap 与体网格不匹配、面数据缺少节点连接，或（没有面网格时）壁面单元同时属于别的组。
    """
    from autoflowcfd.boundary.wall_kind import physical_no_slip_wall_groups

    bm = volume_data.boundaries
    wall_names = physical_no_slip_wall_groups(bm.bc_types, bc_overrides)
    if not wall_names:
        return np.empty((0, 3), dtype=np.int64)
    faces = volume_data.ensure_faces_exist()
    if faces.node_connectivity is None:
        raise ValueError(
            "体网格的面数据缺少 node_connectivity，无法取壁面面片——不能退回"
            "'把单元全部节点当壁面'（那会让近壁壁距虚胖一层）。")
    bidx = faces.get_boundary_face_indices()
    if len(bidx) == 0:
        return np.empty((0, 3), dtype=np.int64)
    face_nodes = np.asarray(faces.node_connectivity[bidx], dtype=np.int64)
    surface = getattr(volume_data, "surface_mesh", None)
    if surface is not None:
        keep = _wall_faces_by_surface(volume_data, surface, face_nodes, wall_names)
    else:
        keep = _wall_faces_by_cells(volume_data, bm, faces.connectivity[bidx, 0], wall_names)
    return face_nodes[keep]


def _wall_faces_by_surface(volume_data, surface, face_nodes, wall_names) -> np.ndarray:
    """逐面：边界面中心归入面网格的组，取壁面组。"""
    from autoflowcfd.grid.mesh_gen.utils.mesh_boundary import match_points_to_surface_groups

    centers = volume_data.nodes.get_coordinates()[face_nodes].mean(axis=1)
    names, group_index = match_points_to_surface_groups(
        centers, surface["nodes"], surface["faces"], surface["boundaries"].groups, _SURFACE_MATCH_TOLERANCE)
    is_wall = np.array([name in wall_names for name in names] + [False])
    return is_wall[group_index]                       # 未匹配（-1）取末尾的 False


def _wall_faces_by_cells(volume_data, bm, owner, wall_names) -> np.ndarray:
    """按单元：owner 单元属于壁面组的边界面；壁面单元同时属于别的组时无法判定每个面属于哪个组，报错。"""
    n_cells = volume_data.cell_count
    wall_cells, other_cells = set(), set()
    for bc_name in bm.bc_types:
        if not bm.has_boundary(bc_name):
            continue
        idx = np.asarray(bm.get_cell_indices(bc_name))
        if idx.size and int(idx.max()) >= n_cells:
            raise ValueError(
                f"边界组 '{bc_name}' 的单元索引最大值 {int(idx.max())} 超出体网格单元数 "
                f"{n_cells}——BoundaryMap 与体网格不匹配（壁面距离场会整体错位）。")
        (wall_cells if bc_name in wall_names else other_cells).update(int(c) for c in idx)
    shared = sorted(wall_cells & other_cells)
    if shared:
        raise ValueError(
            f"{len(shared)} 个壁面单元同时属于别的边界组（例如单元 {shared[:5]}），按单元无法判定其哪些边界面是壁面；"
            "请提供原始面网格（体网格生成/导入的 .pkl 都带）。")
    return np.fromiter((int(o) in wall_cells for o in owner), dtype=bool, count=len(owner))


def triangles_from_faces(nodes: np.ndarray, face_nodes: np.ndarray) -> np.ndarray:
    """面片节点表 -> 三角形顶点坐标 `(n_tri, 3, 3)`；四边形沿对角线拆成两个三角形。

    四边形必须共面：直边映射下四边形面上的解点落在双线性曲面上，非共面四边形拆成的两个
    三角形与之不重合，距离会带上非平面度量级的误差——直接报错，不静默近似（边界层网格里
    壁面是棱柱的三角形底面或四面体的面，四边形侧面不会落在壁面上）。

    Raises:
        ValueError: 存在非共面的四边形壁面面片。
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    face_nodes = np.asarray(face_nodes, dtype=np.int64)
    tris = [face_nodes[:, :3]]
    if face_nodes.shape[1] > 3:
        quad = face_nodes[face_nodes[:, 3] >= 0]
        if quad.shape[0]:
            p = nodes[quad]
            normal = np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1])
            area2 = np.linalg.norm(normal, axis=1)
            off = np.abs(np.einsum("ij,ij->i", p[:, 3] - p[:, 0], normal)) / np.maximum(area2, 1e-300)
            size = np.sqrt(area2)
            bad = off > _QUAD_PLANARITY_RTOL * np.maximum(size, 1e-300)
            if np.any(bad):
                raise ValueError(
                    f"{int(bad.sum())} 个四边形壁面面片不共面（最大相对非平面度 "
                    f"{float((off / np.maximum(size, 1e-300)).max()):.3e}）：壁面距离按三角形精确求解，"
                    "非共面四边形无法精确表示，请检查壁面网格。")
            tris.append(quad[:, [0, 2, 3]])
    tri_nodes = np.concatenate(tris, axis=0)
    return np.ascontiguousarray(nodes[tri_nodes])


def wall_triangles(volume_data, bc_overrides=None) -> np.ndarray:
    """体网格物理无滑移固壁组的边界面 -> 壁面三角形 `(n_tri, 3, 3)`。"""
    face_nodes = wall_face_nodes(volume_data, bc_overrides)
    if face_nodes.shape[0] == 0:
        return np.empty((0, 3, 3))
    return triangles_from_faces(volume_data.nodes.get_coordinates(), face_nodes)
