"""边界面打组标签：逐面边界组（`BoundaryMap.faces` / `HighOrderMesh.boundary_faces`）按节点三元组精确对号
（`face_connectivity_boundary_tags.py`，2026-10-22 I-130）。

历史：2026-08-23 发现按 owner 单元打标签时，同时拥有两个组外部面的角点单元的全部边界面被打成最后遍历到的那个组；
此后改为对每个边界面中心做几何最近邻匹配（依赖原始面网格），按单元的路径在有歧义时报错（I-125）。现在边界组本身
逐面存储，打标签是整数对号。
"""

from types import SimpleNamespace

import numpy as np
import pytest

from autoflowcfd.grid.connectivity.face_connectivity import FRFaceConnectivity
from autoflowcfd.grid.connectivity.face_connectivity_boundary_tags import (
    face_group_codes,
    tag_boundary_groups_for_mesh,
    triangle_keys,
)


def _corner_cell_face_conn():
    """一个 owner 单元（id=0）带两个边界面（节点 0-1-2 与 0-2-3），另有一个内部面：典型的角点单元。"""
    return FRFaceConnectivity(
        owner_cell=np.array([0, 0, 0], dtype=np.int32),
        neighbor_cell=np.array([-1, -1, 1], dtype=np.int32),
        owner_cube_face=np.array([6, 7, 8], dtype=np.int32),
        neighbor_cube_face=np.array([-1, -1, 6], dtype=np.int32),
        normal=np.array([[-1.0, 0, 0], [0.0, -1.0, 0], [0.0, 0, 1.0]]),
        area=np.ones(3),
        center=np.zeros((3, 3)),
        face_node_ids=np.array([[0, 1, 2], [0, 2, 3], [1, 2, 3]], dtype=np.int32),
        is_boundary=np.array([True, True, False]),
        face_translation=np.zeros((3, 3)),
    )


def _mesh(faces, bc_types):
    return SimpleNamespace(face_connectivity=_corner_cell_face_conn(), boundary_faces=faces, boundary_bc_types=bc_types)


def test_two_faces_of_one_corner_cell_land_in_their_own_groups():
    mesh = _mesh({"WALL": np.array([[2, 1, 0]]), "FARFIELD": np.array([[3, 0, 2]])},
                 {"WALL": "WALL", "FARFIELD": "FARFIELD"})
    group_code, name_to_code = tag_boundary_groups_for_mesh(mesh)
    assert group_code.tolist() == [name_to_code["WALL"], name_to_code["FARFIELD"], -1]   # 顶点顺序无关


def test_group_face_missing_from_the_mesh_is_an_error():
    mesh = _mesh({"WALL": np.array([[0, 1, 2]]), "FARFIELD": np.array([[0, 2, 9]])},
                 {"WALL": "WALL", "FARFIELD": "FARFIELD"})
    with pytest.raises(ValueError, match="找不到"):
        tag_boundary_groups_for_mesh(mesh)


def test_face_claimed_by_two_groups_is_an_error():
    mesh = _mesh({"WALL": np.array([[0, 1, 2]]), "FARFIELD": np.array([[1, 2, 0], [0, 2, 3]])},
                 {"WALL": "WALL", "FARFIELD": "FARFIELD"})
    with pytest.raises(ValueError, match="多个边界组"):
        tag_boundary_groups_for_mesh(mesh)


def test_mesh_without_face_groups_is_rejected():
    with pytest.raises(ValueError, match="逐面"):
        tag_boundary_groups_for_mesh(_mesh(None, {}))


def test_periodic_groups_are_not_tagged():
    """周期面配对之后是内部面，周期组不参与边界面打标签（否则它们的面在边界面里找不到）。"""
    mesh = _mesh({"WALL": np.array([[0, 1, 2], [0, 2, 3]]), "P": np.array([[5, 6, 7]])},
                 {"WALL": "WALL", "P": "PERIODIC"})
    group_code, name_to_code = tag_boundary_groups_for_mesh(mesh)
    assert set(name_to_code) == {"WALL"} and group_code[:2].tolist() == [0, 0]


def test_keys_do_not_overflow_on_large_node_ids():
    """(a n + b) n + c 在 n > 2.1e6 时溢出 int64；按字节拼接的键与节点数无关。"""
    big = 3_000_000
    tri = np.array([[big, big + 1, big + 2], [big + 2, big + 1, big], [big, big + 1, big + 3]])
    keys = triangle_keys(tri)
    assert keys[0] == keys[1] and keys[0] != keys[2]
    codes, names = face_group_codes(tri[[0, 2]], {"a": tri[[2]], "b": tri[[1]]})
    assert codes.tolist() == [1, 0] and names == ["a", "b"]
