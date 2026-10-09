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
    def test_both_faces_get_the_same_wrong_group_when_owner_cell_is_shared(self):
        """钉住实际的缺陷：boundary_groups 说单元 0 同时在 'WALL' 与 'FARFIELD'
        里（角点单元）——按单元的匹配器只能选一个，并把它用到*两个*面上。
        """
        fc = _corner_cell_face_conn()
        boundary_groups = {
            "WALL": np.array([0], dtype=np.int32),
            "FARFIELD": np.array([0], dtype=np.int32),
        }
        group_code, name_to_code = tag_boundary_groups(fc, boundary_groups)

        # 两个面得到**同一个**编码（最后遍历到的那个组），尽管它们在物理上
        # 属于不同的组。
        assert group_code[0] == group_code[1]
        assert group_code[0] == name_to_code["FARFIELD"]  # last-iterated wins

    def test_warns_loudly_about_the_specific_ambiguous_cell(self):
        """2026-08-23 修复：只凭单元粒度的输入不可能做到真正的逐面修复
        （组->单元编号的映射已经丢掉了哪个面属于哪个组）——所以兜底路径至少
        必须把静默的错标变成响亮、可诊断的警告，点出具体的单元与冲突的组名，
        而不是此前的完全沉默。
        """
        from loguru import logger

        fc = _corner_cell_face_conn()
        boundary_groups = {
            "WALL": np.array([0], dtype=np.int32),
            "FARFIELD": np.array([0], dtype=np.int32),
        }
        messages = []
        handler_id = logger.add(lambda msg: messages.append(msg.record["message"]), level="WARNING")
        try:
            tag_boundary_groups(fc, boundary_groups)
        finally:
            logger.remove(handler_id)

        ambiguity_warnings = [m for m in messages if "claimed by more than one" in m]
        assert len(ambiguity_warnings) == 1, f"expected exactly one ambiguity warning, got: {messages}"
        assert "cell 0" in ambiguity_warnings[0]
        assert "WALL" in ambiguity_warnings[0] and "FARFIELD" in ambiguity_warnings[0]

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
        group_code, name_to_code = tag_boundary_groups_for_mesh(mesh)
        assert group_code[0] == group_code[1]  # 兜底：旧的按单元的行为，仍有歧义
