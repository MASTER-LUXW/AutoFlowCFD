"""AutoFlowCFD V2.0 - 从体网格取出壁面三角形（壁面距离的几何来源）。

壁面三角形就是物理无滑移固壁组的逐面边界组（`BoundaryMap.faces`，`boundary/wall_kind.py` 判定哪些组是物理固壁，
认 `bc_overrides`）。历史：2026-09-15 以前把单元下标当节点下标（一阶物理错误）；2026-10-17（I-92）以前按"边界面的
owner 单元属于壁面组"选面，组交界单元的全部边界面都被算成壁面；2026-10-22（I-130）起边界组本身逐面存储。
"""

import numpy as np

#: 四边形壁面面片的非平面度容差（相对面片尺度）：判断"四点共面"的浮点重合判据。
_QUAD_PLANARITY_RTOL = 1e-10


def wall_face_nodes(volume_data, bc_overrides=None) -> np.ndarray:
    """物理无滑移固壁组的逐面边界组拼成的面节点表 `(n_faces, 3)`（体网格节点编号）。"""
    from autoflowcfd.boundary.wall_kind import physical_no_slip_wall_groups

    bm = volume_data.boundaries
    wall_names = physical_no_slip_wall_groups(bm.bc_types, bc_overrides)
    parts = [bm.faces[name] for name in wall_names if name in bm.faces]
    return np.concatenate(parts, axis=0) if parts else np.empty((0, 3), dtype=np.int64)


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
