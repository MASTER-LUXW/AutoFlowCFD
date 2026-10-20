"""边界组分类不再静默降级（I-150，`mesh_domain_classify.classify_boundary_groups`）。

此前：需要边界层的 WALL 分量若既不闭合、也不贴合包围盒单一面（零厚度板、与地面大口焊接的车身），被静默归入核心
PLC、不挤出边界层；闭合实体找不到内部点时只警告并跳过 tetgen 孔洞标记（核心填充会在实体内部生成与边界层重叠的
四面体）。现在两者都报错并给出分量信息。
"""

import numpy as np
import pytest

from autoflowcfd.grid.mesh_gen.utils import mesh_domain_classify
from autoflowcfd.grid.mesh_gen.utils.mesh_domain_classify import classify_boundary_groups
from autoflowcfd.grid.schema.grid_boundaries import BoundaryMap

_BOX = (np.zeros(3), np.full(3, 10.0))


def _cube(lo, hi, offset):
    p = np.array([[lo, lo, lo], [hi, lo, lo], [hi, hi, lo], [lo, hi, lo],
                  [lo, lo, hi], [hi, lo, hi], [hi, hi, hi], [lo, hi, hi]], float)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]) + offset
    return p, f


def _surface(body_points, body_faces):
    box_p, box_f = _cube(0.0, 10.0, len(body_points))
    nodes = np.vstack([body_points, box_p])
    faces = np.vstack([body_faces, box_f])
    groups = {"body": np.arange(len(body_faces)), "tunnel": len(body_faces) + np.arange(12)}
    return nodes, faces, BoundaryMap(groups=groups, bc_types={"body": "WALL", "tunnel": "SLIP_WALL"})


def test_closed_body_is_extruded_with_a_hole_seed():
    nodes, faces, bm = _surface(*_cube(4.0, 6.0, 0))
    ext, core, names, holes = classify_boundary_groups(nodes, faces, bm, *_BOX)
    assert names == ["body"] and len(ext) == 12 and len(core) == 12 and len(holes) == 1


def test_open_wall_sheet_inside_the_domain_is_an_error():
    """零厚度板（WALL）悬在域中：不闭合、不贴包围盒——此前静默不挤出边界层。"""
    plate = np.array([[3, 3, 5], [7, 3, 5], [7, 7, 5], [3, 7, 5]], float)
    nodes, faces, bm = _surface(plate, np.array([[0, 1, 2], [0, 2, 3]]))
    with pytest.raises(ValueError, match="'body'（WALL）.*既不闭合、也不贴合"):
        classify_boundary_groups(nodes, faces, bm, *_BOX)


def test_closed_body_without_an_interior_point_is_an_error(monkeypatch):
    monkeypatch.setattr(mesh_domain_classify, "find_point_inside_closed_shell", lambda nodes, faces: None)
    nodes, faces, bm = _surface(*_cube(4.0, 6.0, 0))
    with pytest.raises(ValueError, match="找不到可靠的内部点"):
        classify_boundary_groups(nodes, faces, bm, *_BOX)
