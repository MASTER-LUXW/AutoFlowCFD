"""Regression test for a real bug found 2026-08-23 during a live resume
investigation on a 791k-cell cube_demo run: every prism quadrilateral side
face is triangulated into 2 face-connectivity records by the mesh
generator (face_connectivity.py). Before this session's "exact per-FP
normal" fix (face_flux_points/exact_normal.py), `true_normal`/
`true_area_weight` were each record's own genuine triangulated half-face
geometry, so two records naturally summed to the whole quad's area/normal.
After that fix, `true_normal`/`true_area_weight` became a pure function of
`(owner_cell, owner_axis, owner_side)` — both of a quad's 2 records now
carry the *identical whole-quad* value instead of a genuine half.

`inviscid_p0.py`'s P0 finite-volume kernel processes every face record
unfiltered (by design, to preserve closure under the *old* scheme) — with
the new scheme this silently double-counts the flux/area for ~100% of
prism quad side faces (not just the ~5% that are genuinely multi-source
across two different real tet neighbors). Directly measured on a real
mesh: cell 17790 (a boundary-layer prism) had a 6.76% relative
closed-surface-normal-sum error (sum of area-weighted face normals should
be exactly zero for any valid closed cell), and this alone quantitatively
explained ~100% of an otherwise-unexplained 1.6e7-magnitude spurious
momentum residual (confirmed by zeroing all velocities and finding the
residual unchanged - it's a pure absolute-pressure * unclosed-normal
artifact, unrelated to low-Mach/CFL stiffness).

Fix (`inviscid_p0.py::_extract_p0_face_geometry`): for each
(owner_cell, owner_cube_face) / (neighbor_cell, neighbor_cube_face) group
with >=2 records - if all records share the same real neighbor (or owner),
keep only the primary record's exact whole-face value and zero the area
of the rest (duplicate, not multi-source - can't just sum both without
double-counting). If records point to genuinely different real cells
(true multi-source, ~5%), fall back to `face_connectivity`'s original
triangulated half-face `normal`/`area` for every record (P0 has only 1
Flux Point per face, so it cannot blend multiple real neighbors at the
FP level the way the P>=1 kernel does via `nb_extra_mat`/`ow_extra_mat` -
falling back to the genuine triangulated half is the only geometrically
correct option, and it naturally sums to the whole face exactly once).
"""

from types import SimpleNamespace

import numpy as np

from autoflowcfd.core.fr_residual.inviscid_p0 import _extract_p0_face_geometry
from autoflowcfd.fr.face_flux_points.data import _KernelFaceData


def _make_ffp(n_faces, true_normal, true_area_weight, owner_is_primary,
              neighbor_is_primary, owner_groups, neighbor_groups):
    return _KernelFaceData(
        n_faces=n_faces, n_fp=1,
        true_normal=true_normal, true_area_weight=true_area_weight,
        owner_is_primary=owner_is_primary, neighbor_is_primary=neighbor_is_primary,
        _owner_groups=owner_groups, _neighbor_groups=neighbor_groups,
    )


class TestDuplicateRecordsSameNeighborAreDeduplicated:
    def test_only_primary_record_keeps_area_rest_zeroed(self):
        # f0, f1: same (owner=0, code) group, BOTH pointing at neighbor=1
        # (the ~95% "duplicate, not multi-source" case) - both currently
        # carry the identical whole-quad true_normal/true_area_weight
        # (exactly what Part 1 now produces), which must be deduplicated.
        n_faces = 2
        true_normal = np.tile(np.array([1.0, 0.0, 0.0]), (n_faces, 1, 1))
        true_area_weight = np.full((n_faces, 1), 5.0)
        owner_is_primary = np.array([True, False])
        neighbor_is_primary = np.array([True, True])
        owner_groups = {(0, 7): [0, 1]}
        neighbor_groups = {}

        ffp = _make_ffp(n_faces, true_normal, true_area_weight,
                         owner_is_primary, neighbor_is_primary,
                         owner_groups, neighbor_groups)
        fc = SimpleNamespace(
            normal=np.zeros((n_faces, 3)), area=np.zeros(n_faces),
            owner_cell=np.array([0, 0]), neighbor_cell=np.array([1, 1]),
            is_boundary=np.array([False, False]),
        )

        unit_normals, area_weights = _extract_p0_face_geometry(ffp, fc, n_faces)

        assert area_weights[0] == 5.0  # primary keeps the exact whole-face value
        assert area_weights[1] == 0.0  # duplicate zeroed, not double-counted
        np.testing.assert_array_equal(unit_normals[0], [1.0, 0.0, 0.0])

    def test_boundary_duplicate_records_are_deduplicated(self):
        """Boundary faces never appear in _owner_groups/_neighbor_groups
        (build_face_flux_points groups them separately as
        boundary_owner_groups, not stored on _KernelFaceData) but
        owner_is_primary is still correctly set for them - the dedup must
        catch this case via the plain is_boundary & ~owner_is_primary mask,
        independent of group-dict membership."""
        n_faces = 2
        true_normal = np.tile(np.array([0.0, 1.0, 0.0]), (n_faces, 1, 1))
        true_area_weight = np.full((n_faces, 1), 3.0)
        owner_is_primary = np.array([True, False])
        neighbor_is_primary = np.array([True, True])

        ffp = _make_ffp(n_faces, true_normal, true_area_weight,
                         owner_is_primary, neighbor_is_primary,
                         owner_groups={}, neighbor_groups={})
        fc = SimpleNamespace(
            normal=np.zeros((n_faces, 3)), area=np.zeros(n_faces),
            owner_cell=np.array([2, 2]), neighbor_cell=np.array([-1, -1]),
            is_boundary=np.array([True, True]),
        )

        _, area_weights = _extract_p0_face_geometry(ffp, fc, n_faces)

        assert area_weights[0] == 3.0
        assert area_weights[1] == 0.0


class TestGenuineMultiSourceFallsBackToTriangulatedHalfFace:
    def test_both_records_keep_area_using_old_triangulated_geometry(self):
        # f0, f1: same (owner=0, code) group, but pointing at TWO
        # DIFFERENT real neighbors (2, 3) - genuine multi-source (~5%
        # case). Both records must stay active (can't drop either real
        # neighbor's contribution), but must use face_connectivity's
        # original triangulated half-face normal/area, NOT the new
        # exact-but-identical whole-quad true_normal/true_area_weight
        # (using the whole-quad value for both would double the area
        # exactly like the duplicate case, just against two different
        # neighbors instead of the same one).
        n_faces = 2
        true_normal = np.tile(np.array([1.0, 0.0, 0.0]), (n_faces, 1, 1))
        true_area_weight = np.full((n_faces, 1), 5.0)  # whole-quad value, must NOT be used as-is
        owner_is_primary = np.array([True, False])
        neighbor_is_primary = np.array([True, True])
        owner_groups = {(0, 7): [0, 1]}

        ffp = _make_ffp(n_faces, true_normal, true_area_weight,
                         owner_is_primary, neighbor_is_primary,
                         owner_groups, neighbor_groups={})
        old_normal = np.array([[0.9, 0.1, 0.0], [0.8, -0.2, 0.0]])
        old_area = np.array([2.1, 2.4])  # genuine triangulated halves, sum to the whole quad
        fc = SimpleNamespace(
            normal=old_normal, area=old_area,
            owner_cell=np.array([0, 0]), neighbor_cell=np.array([2, 3]),
            is_boundary=np.array([False, False]),
        )

        unit_normals, area_weights = _extract_p0_face_geometry(ffp, fc, n_faces)

        np.testing.assert_array_equal(area_weights, old_area)
        np.testing.assert_array_equal(unit_normals, old_normal)


class TestUngroupedFacesAreUntouched:
    def test_plain_tet_tet_face_keeps_exact_value(self):
        n_faces = 1
        true_normal = np.array([[[0.0, 0.0, 1.0]]])
        true_area_weight = np.array([[7.0]])
        owner_is_primary = np.array([True])
        neighbor_is_primary = np.array([True])

        ffp = _make_ffp(n_faces, true_normal, true_area_weight,
                         owner_is_primary, neighbor_is_primary,
                         owner_groups={}, neighbor_groups={})
        fc = SimpleNamespace(
            normal=np.zeros((n_faces, 3)), area=np.zeros(n_faces),
            owner_cell=np.array([9]), neighbor_cell=np.array([10]),
            is_boundary=np.array([False]),
        )

        unit_normals, area_weights = _extract_p0_face_geometry(ffp, fc, n_faces)

        assert area_weights[0] == 7.0
        np.testing.assert_array_equal(unit_normals[0], [0.0, 0.0, 1.0])
