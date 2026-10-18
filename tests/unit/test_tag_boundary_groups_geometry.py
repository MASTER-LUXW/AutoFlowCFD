"""Regression test for a real, confirmed, previously-unfixed bug found
2026-08-23 (user directly asked "tag_boundary_groups 修复了吗" after an
earlier P1-divergence investigation flagged it as a live suspect, then
ruled it out for cube_demo specifically because that mesh's four boundary
groups happen not to overlap - but the underlying defect is real and
independent of any specific mesh):

`tag_boundary_groups` matches boundary faces to a named group by their
*owner cell* index (`np.isin(boundary_owners, cell_id_set)`), using
`BoundaryMap.groups` which is itself a cell-granular aggregate (see
`map_boundaries_by_geometry`'s own `volume_cell_to_boundary: Dict[int,
str]` - a plain dict keyed by cell index that already collapses
per-face matches). A "corner cell" that legitimately owns boundary faces
in two different groups (e.g. one face on WALL, another on FARFIELD)
gets ALL of its boundary faces tagged with whichever group is iterated
last - silently applying the wrong boundary condition to the face that
belongs to the other group.

Fix: `tag_boundary_groups_by_geometry` re-does the nearest-neighbour
geometric match directly against the *face's own centroid* (not its
owner cell), using the same KD-tree + per-face-radius-tolerance approach
`map_boundaries_by_geometry` already uses - each face decides its own
group independently, so two faces sharing an owner cell can legitimately
end up in different groups. `tag_boundary_groups_for_mesh` is the single
entry point all four call sites (fr_solver/boundary.py,
core/utils/solver_helpers.py, postprocess/fr_coefficients.py x2) now use;
it prefers the geometric method when `mesh.boundary_surface_mesh` is
available and falls back to the old cell-based method otherwise.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from autoflowcfd.grid.connectivity.face_connectivity import FRFaceConnectivity
from autoflowcfd.grid.connectivity.face_connectivity_boundary_tags import (
    tag_boundary_groups,
    tag_boundary_groups_by_geometry,
    tag_boundary_groups_for_mesh,
)


def _corner_cell_face_conn():
    """一个 owner 单元（id=0）带两个边界面：一个中心在 x=0（真实组 'WALL'），
    一个在 x=10（真实组 'FARFIELD'）——典型的角点单元情形。
    """
    n_faces = 2
    return FRFaceConnectivity(
        owner_cell=np.array([0, 0], dtype=np.int32),
        neighbor_cell=np.array([-1, -1], dtype=np.int32),
        owner_cube_face=np.array([0, 1], dtype=np.int32),
        neighbor_cube_face=np.array([-1, -1], dtype=np.int32),
        normal=np.array([[-1.0, 0, 0], [1.0, 0, 0]]),
        area=np.array([1.0, 1.0]),
        center=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        face_node_ids=np.zeros((n_faces, 3), dtype=np.int32),
        is_boundary=np.array([True, True]),
    )


def _corner_cell_surface_mesh():
    """两个小三角形，一个在 x=0 附近（组 'WALL'），一个在 x=10 附近
    （组 'FARFIELD'）——体网格的每个边界面各自应当匹配到的真值。
    """
    nodes = np.array([
        [-0.1, 0.0, 0.0], [0.1, 0.1, 0.0], [0.1, -0.1, 0.0],  # tri 0: near x=0
        [9.9, 0.0, 0.0], [10.1, 0.1, 0.0], [10.1, -0.1, 0.0],  # tri 1: near x=10
    ])
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    boundaries = SimpleNamespace(groups={
        "WALL": np.array([0], dtype=np.int32),
        "FARFIELD": np.array([1], dtype=np.int32),
    })
    return {"nodes": nodes, "faces": faces, "boundaries": boundaries}


class TestCellBasedTaggingMisattributesCornerCellFaces:
    def test_ambiguous_corner_cell_is_an_error(self):
        """同一个单元被两个组声称拥有（角点单元）：按单元匹配无法判定每个面属于哪个组，报错并点出具体单元与
        冲突的组名（I-125；此前只给 warning、按最后遍历到的组把两个面都打成同一个组）。"""
        fc = _corner_cell_face_conn()
        boundary_groups = {
            "WALL": np.array([0], dtype=np.int32),
            "FARFIELD": np.array([0], dtype=np.int32),
        }
        with pytest.raises(ValueError, match="cell 0") as info:
            tag_boundary_groups(fc, boundary_groups)
        assert "WALL" in str(info.value) and "FARFIELD" in str(info.value)

    def test_no_ambiguity_warning_when_groups_do_not_overlap(self):
        """互不重叠的组（常见情形，例如 cube_demo 的四个边界组）不能触发这条
        新警告——它只应对真正有歧义的单元触发。
        """
        from loguru import logger

        fc = _corner_cell_face_conn()
        boundary_groups = {
            "WALL": np.array([0], dtype=np.int32),
            "FARFIELD": np.array([1], dtype=np.int32),
        }
        messages = []
        handler_id = logger.add(lambda msg: messages.append(msg.record["message"]), level="WARNING")
        try:
            tag_boundary_groups(fc, boundary_groups)
        finally:
            logger.remove(handler_id)

        ambiguity_warnings = [m for m in messages if "claimed by more than one" in m]
        assert ambiguity_warnings == []


class TestGeometricTaggingResolvesCornerCellCorrectly:
    def test_each_face_gets_its_own_true_group(self):
        fc = _corner_cell_face_conn()
        surface_mesh = _corner_cell_surface_mesh()

        group_code, name_to_code = tag_boundary_groups_by_geometry(fc, surface_mesh)

        assert group_code[0] == name_to_code["WALL"]
        assert group_code[1] == name_to_code["FARFIELD"]
        assert group_code[0] != group_code[1]


class TestUnifiedEntryPointPicksStrategyByMeshAttribute:
    def test_uses_geometric_method_when_surface_mesh_present(self):
        fc = _corner_cell_face_conn()
        mesh = SimpleNamespace(
            face_connectivity=fc,
            boundary_groups={"WALL": np.array([0], dtype=np.int32),
                              "FARFIELD": np.array([0], dtype=np.int32)},
            boundary_surface_mesh=_corner_cell_surface_mesh(),
        )
        group_code, name_to_code = tag_boundary_groups_for_mesh(mesh)
        assert group_code[0] != group_code[1]  # geometric method used - correctly resolved

    def test_falls_back_to_cell_based_method_when_surface_mesh_absent(self):
        fc = _corner_cell_face_conn()
        mesh = SimpleNamespace(
            face_connectivity=fc,
            boundary_groups={"WALL": np.array([0], dtype=np.int32),
                              "FARFIELD": np.array([0], dtype=np.int32)},
            boundary_surface_mesh=None,
        )
        with pytest.raises(ValueError, match="原始面网格"):          # 按单元匹配：有歧义的单元报错
            tag_boundary_groups_for_mesh(mesh)
        mesh.boundary_groups = {"WALL": np.array([0], dtype=np.int32), "FARFIELD": np.array([1], dtype=np.int32)}
        group_code, name_to_code = tag_boundary_groups_for_mesh(mesh)     # 无歧义时按单元匹配照常成立
        assert group_code[0] == group_code[1] == name_to_code["WALL"]
