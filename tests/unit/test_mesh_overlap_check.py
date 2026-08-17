"""Unit tests for validation/mesh_overlap_check.py.

The CANDIDATE_CAP_PER_FACE tests are a regression for a real hang: a
single outlier-huge boundary face (one of cube_demo's coarse farfield/
domain-shell panels) gets a broad-phase search radius scaled to its own
huge size and can return hundreds of thousands of
candidates from ONE query - 142,944 in the measured case, blowing one
500-face chunk out to 5.58M candidate pairs and making the whole check
take 6+ minutes and several GB of RAM. The fix caps any one face's
candidate set at its CAP nearest neighbours. These tests confirm the cap
doesn't cause a genuine small-vs-small overlap to be missed just because
an unrelated huge face is also present in the mesh.
"""

import numpy as np
import pytest

from autoflowcfd.grid.validation import mesh_overlap_check
from autoflowcfd.grid.validation.mesh_overlap_check import check_face_overlap_and_proximity
from autoflowcfd.grid.schema.grid_nodes import NodeArray
from autoflowcfd.grid.mesh_gen.extraction.face_extractor import FaceExtractor

# Two triangles that genuinely cross in 3D, sharing no vertices (same
# fixture used in test_overlap_geometry.py / test_mesh_front_collision.py).
A0, A1, A2 = np.array([-2., -2., 0.]), np.array([2., -2., 0.]), np.array([0., 2., 0.])
B0, B1, B2 = np.array([0., 0., -2.]), np.array([0., 0., 2.]), np.array([0., 3., 0.])


def _cap_tet(p0, p1, p2, eps=1e-3):
    """A thin tetrahedron with one face exactly (p0, p1, p2) - the other 3
    faces are thin slivers, irrelevant to the test other than existing."""
    centroid = (p0 + p1 + p2) / 3.0
    normal = np.cross(p1 - p0, p2 - p0)
    normal = normal / np.linalg.norm(normal)
    p3 = centroid + eps * normal
    return np.array([p0, p1, p2, p3])


def _tiny_tet(center, scale=0.01):
    return center + np.array([
        [0.0, 0.0, 0.0],
        [scale, 0.0, 0.0],
        [0.0, scale, 0.0],
        [0.0, 0.0, scale],
    ])


def _huge_tet(center, scale=200.0):
    return _tiny_tet(center, scale=scale)


# Placed far below the distractor line along z (which the distractors
# never occupy) so its own extent can never geometrically reach any other
# tet - only its search radius (which scales with its own huge size) does.
# Tuned empirically (see scratchpad/tune_huge_tet.py): scale=200,
# z-offset=-600 makes its broad-phase candidate count far exceed a small
# patched cap while producing zero real triangle-triangle intersections.
HUGE_TET_CENTER = np.array([40.0, 0.0, -600.0])
HUGE_TET_SCALE = 200.0


def _build_mesh(tets):
    """tets: list of (4,3) node arrays, each an isolated tetrahedron (no
    shared nodes across tets - keeps every face a boundary face)."""
    all_nodes = np.concatenate(tets, axis=0)
    cells = np.arange(len(all_nodes), dtype=np.int64).reshape(-1, 4)
    return all_nodes, cells


class TestCandidateCapPreservesCorrectness:
    def test_small_overlap_still_found_next_to_a_huge_distractor_face(self, monkeypatch):
        """A huge face (far away, not touching anything) forces one of its
        own broad-phase queries to be capped; a genuine small-vs-small
        overlap elsewhere in the mesh must still be detected."""
        monkeypatch.setattr(mesh_overlap_check, "CANDIDATE_CAP_PER_FACE", 3)

        tet_a = _cap_tet(A0, A1, A2)
        tet_b = _cap_tet(B0, B1, B2)

        # Several well-separated, mutually non-overlapping small tets to
        # pad the candidate count for the huge face's own query.
        distractors = [_tiny_tet(np.array([10.0 * i, 0.0, 0.0])) for i in range(1, 9)]

        # Placed near the "cluster center" (within the huge face's own
        # oversized search radius of everything else) but far enough from
        # any other single tet that it never geometrically intersects one.
        huge = _huge_tet(HUGE_TET_CENTER, scale=HUGE_TET_SCALE)

        tets = [tet_a, tet_b] + distractors + [huge]
        nodes, cells = _build_mesh(tets)

        node_arr = NodeArray(x=nodes[:, 0].copy(), y=nodes[:, 1].copy(), z=nodes[:, 2].copy())
        faces = FaceExtractor.extract_faces(cells.astype(np.int32), node_arr)

        report = check_face_overlap_and_proximity(nodes, cells, faces=faces)

        assert report.has_overlaps
        # tet_a is cell 0, tet_b is cell 1 (in `cells` construction order).
        assert 0 in report.overlapping_cell_ids
        assert 1 in report.overlapping_cell_ids
        # The huge distractor never touches anything and must not be
        # implicated by capping-induced false positives.
        huge_cell_id = len(tets) - 1
        assert huge_cell_id not in report.overlapping_cell_ids

    def test_cap_actually_engages_for_the_huge_face(self, monkeypatch):
        """Sanity check on the test fixture itself: without capping, the
        huge face's own query really does exceed a small cap (otherwise
        the test above wouldn't be exercising the capping path at all)."""
        monkeypatch.setattr(mesh_overlap_check, "CANDIDATE_CAP_PER_FACE", 3)

        tet_a = _cap_tet(A0, A1, A2)
        tet_b = _cap_tet(B0, B1, B2)
        distractors = [_tiny_tet(np.array([10.0 * i, 0.0, 0.0])) for i in range(1, 9)]
        huge = _huge_tet(HUGE_TET_CENTER, scale=HUGE_TET_SCALE)

        tets = [tet_a, tet_b] + distractors + [huge]
        nodes, cells = _build_mesh(tets)
        node_arr = NodeArray(x=nodes[:, 0].copy(), y=nodes[:, 1].copy(), z=nodes[:, 2].copy())
        faces = FaceExtractor.extract_faces(cells.astype(np.int32), node_arr)

        boundary_idx = faces.get_boundary_face_indices()
        centroids = faces.center[boundary_idx]
        face_size = np.sqrt(np.maximum(faces.area[boundary_idx], 1e-300))
        from scipy.spatial import cKDTree
        tree = cKDTree(centroids)
        search_radius = 3.0 * face_size
        counts = np.array([
            len(tree.query_ball_point(centroids[i], r=search_radius[i]))
            for i in range(len(boundary_idx))
        ])
        assert counts.max() > 3, "fixture must produce a face whose uncapped candidate count exceeds the patched cap"


class TestNoOverlapCleanMesh:
    def test_well_separated_tets_report_no_overlaps(self):
        distractors = [_tiny_tet(np.array([10.0 * i, 0.0, 0.0])) for i in range(8)]
        nodes, cells = _build_mesh(distractors)
        node_arr = NodeArray(x=nodes[:, 0].copy(), y=nodes[:, 1].copy(), z=nodes[:, 2].copy())
        faces = FaceExtractor.extract_faces(cells.astype(np.int32), node_arr)

        report = check_face_overlap_and_proximity(nodes, cells, faces=faces)

        assert not report.has_overlaps
        assert len(report.overlapping_cell_ids) == 0
