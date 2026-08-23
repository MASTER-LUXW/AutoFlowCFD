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

from autoflowcfd.grid.connectivity.face_connectivity import (
    FRFaceConnectivity,
    tag_boundary_groups,
    tag_boundary_groups_by_geometry,
    tag_boundary_groups_for_mesh,
)


def _corner_cell_face_conn():
    """A single owner cell (id=0) with two boundary faces: one centred at
    x=0 (true group 'WALL'), one at x=10 (true group 'FARFIELD') - the
    classic corner-cell scenario."""
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
    """Two tiny triangles, one near x=0 (group 'WALL'), one near x=10
    (group 'FARFIELD') - the ground truth the volume mesh's boundary
    faces should each independently match against."""
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
        """Pins the actual bug: boundary_groups says cell 0 is in both
        'WALL' and 'FARFIELD' (a corner cell) - the cell-based matcher
        can only pick one, and applies it to *both* faces."""
        fc = _corner_cell_face_conn()
        boundary_groups = {
            "WALL": np.array([0], dtype=np.int32),
            "FARFIELD": np.array([0], dtype=np.int32),
        }
        group_code, name_to_code = tag_boundary_groups(fc, boundary_groups)

        # Both faces get the SAME code (whichever group was iterated
        # last) even though they physically belong to different groups.
        assert group_code[0] == group_code[1]
        assert group_code[0] == name_to_code["FARFIELD"]  # last-iterated wins


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
        assert group_code[0] == group_code[1]  # fallback: old cell-based behaviour, still ambiguous
