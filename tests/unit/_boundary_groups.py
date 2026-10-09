"""测试网格的边界分组辅助：每个边界面都必须属于一个组（`build_boundary_ghost_provider` 拒绝没有组的边界面）。"""

import numpy as np


def tag_wall_cells(mesh, wall_cells, wall_type: str = "WALL", rest_type: str = "FARFIELD") -> None:
    """`wall_cells` 的全部边界面归 'wall_group'（类型 `wall_type`），其余边界面归 'farfield'（类型 `rest_type`）。"""
    fc = mesh.face_connectivity
    owners = np.unique(fc.owner_cell[fc.is_boundary])
    wall = np.asarray(wall_cells, dtype=np.int64)
    mesh.boundary_groups = {"farfield": np.setdiff1d(owners, wall).astype(np.int64), "wall_group": wall}
    mesh.boundary_bc_types = {"farfield": rest_type, "wall_group": wall_type}
