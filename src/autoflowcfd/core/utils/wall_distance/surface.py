"""AutoFlowCFD V2.0 - 从体网格取出壁面三角形（壁面距离的几何来源）。

**为什么用边界面而不是"WALL 组单元的全部节点"**（2026-09-15 修复的一阶物理错误，此前是
"把单元索引当节点索引"）：`BoundaryMap` 存的是单元索引；边界层棱柱有 3 个节点在壁面上、
3 个在第一层之外，按单元取会让壁面虚胖一层。这里取 WALL 组拥有的**边界面**本身。

已知边际情形：`map_boundaries_by_geometry` 把逐面匹配结果折叠成 `{owner_cell: group}`，
同一单元若同时拥有属于不同组的边界面（只发生在组交界棱上），会把它的全部边界面都计入
——这些面与 WALL 面共享棱，过包含只影响交界棱附近、且偏向更短的距离。
"""

import numpy as np

#: 四边形壁面面片的非平面度容差（相对面片尺度）：判断"四点共面"的浮点重合判据。
_QUAD_PLANARITY_RTOL = 1e-10


def wall_face_nodes(volume_data) -> np.ndarray:
    """WALL 组的边界面节点表 `(n_faces, 3 或 4)`（-1 补位）。

    Raises:
        ValueError: BoundaryMap 与体网格不匹配，或面数据缺少节点连接。
    """
    bm = volume_data.boundaries
    n_cells = volume_data.cell_count
    wall_cells = set()
    for bc_name, bc_type in bm.bc_types.items():
        if bc_type != 'WALL' or not bm.has_boundary(bc_name):
            continue
        idx = np.asarray(bm.get_cell_indices(bc_name))
        if idx.size == 0:
            continue
        if int(idx.max()) >= n_cells:
            raise ValueError(
                f"边界组 '{bc_name}' 的单元索引最大值 {int(idx.max())} 超出体网格单元数 "
                f"{n_cells}——BoundaryMap 与体网格不匹配（壁面距离场会整体错位）。")
        wall_cells.update(int(c) for c in idx)
    if not wall_cells:
        return np.empty((0, 3), dtype=np.int64)
    faces = volume_data.ensure_faces_exist()
    if faces.node_connectivity is None:
        raise ValueError(
            "体网格的面数据缺少 node_connectivity，无法取壁面面片——不能退回"
            "'把单元全部节点当壁面'（那会让近壁壁距虚胖一层）。")
    bidx = faces.get_boundary_face_indices()
    if len(bidx) == 0:
        return np.empty((0, 3), dtype=np.int64)
    owner = faces.connectivity[bidx, 0]
    keep = np.fromiter((int(o) in wall_cells for o in owner), dtype=bool, count=len(owner))
    return np.asarray(faces.node_connectivity[bidx[keep]], dtype=np.int64)


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


def wall_triangles(volume_data) -> np.ndarray:
    """体网格 WALL 组边界面 -> 壁面三角形 `(n_tri, 3, 3)`。"""
    face_nodes = wall_face_nodes(volume_data)
    if face_nodes.shape[0] == 0:
        return np.empty((0, 3, 3))
    return triangles_from_faces(volume_data.nodes.get_coordinates(), face_nodes)
