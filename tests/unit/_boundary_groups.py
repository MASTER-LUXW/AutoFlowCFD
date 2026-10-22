"""测试网格的边界分组辅助：边界组逐面给出（与生产同一格式 `HighOrderMesh.boundary_faces`，见 `BoundaryMap` 文档），
每个边界面都必须属于一个组（`build_boundary_ghost_provider` 拒绝没有组的边界面）。"""

import numpy as np


def set_boundary_faces(mesh, face_group, names, bc_types) -> None:
    """已建面连接关系的网格：边界面按 `face_group`（与 `fc.get_boundary_face_indices()` 同序，`names` 的下标）分组，
    写入逐面边界组与派生的单元组。"""
    fc = mesh.face_connectivity
    bidx = fc.get_boundary_face_indices()
    face_group = np.asarray(face_group)
    mesh.boundary_faces = {name: np.asarray(fc.face_node_ids[bidx[face_group == gi]], dtype=np.int64)
                           for gi, name in enumerate(names) if (face_group == gi).any()}
    mesh.boundary_groups = {name: np.unique(fc.owner_cell[bidx[face_group == gi]]).astype(np.int64)
                            for gi, name in enumerate(names) if (face_group == gi).any()}
    mesh.boundary_bc_types = {name: bc_types[name] for name in mesh.boundary_faces}


def tag_all_boundary_faces(mesh, name: str = "farfield", bc_type: str = "FARFIELD") -> None:
    """全部边界面归一个组。"""
    n = mesh.face_connectivity.get_boundary_face_indices().size
    set_boundary_faces(mesh, np.zeros(n, dtype=np.int64), [name], {name: bc_type})


def tag_wall_cells(mesh, wall_cells, wall_type: str = "WALL", rest_type: str = "FARFIELD") -> None:
    """`wall_cells` 的全部边界面归 'wall_group'（类型 `wall_type`），其余边界面归 'farfield'（类型 `rest_type`）。"""
    fc = mesh.face_connectivity
    owners = fc.owner_cell[fc.get_boundary_face_indices()]
    group = np.where(np.isin(owners, np.asarray(wall_cells, dtype=np.int64)), 1, 0)
    set_boundary_faces(mesh, group, ["farfield", "wall_group"], {"farfield": rest_type, "wall_group": wall_type})
