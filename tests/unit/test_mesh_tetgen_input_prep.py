"""Unit tests for mesh_tetgen_input_prep.prepare_plc_input - pure PLC
input validation/cleanup extracted out of mesh_tetgen_core.fill_core_volume
(background-point concatenation, out-of-bounds face index check, and
degenerate-face removal), isolated here from tetgen itself which this
module never invokes."""

import numpy as np
import pytest

from autoflowcfd.grid.mesh_gen.mesh_tetgen_input_prep import prepare_plc_input

_CUBE_POINTS = np.array([
    [0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
    [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.],
], dtype=np.float64)
# Two triangles of the cube's bottom face - enough for these tests, which
# never actually call tetgen.
_VALID_FACES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)


class TestPreparePlcInput:
    def test_valid_input_passes_through_unchanged(self):
        points, faces, markers = prepare_plc_input(_CUBE_POINTS, _VALID_FACES)
        assert np.array_equal(points, _CUBE_POINTS)
        assert np.array_equal(faces, _VALID_FACES)
        assert markers is None

    def test_background_points_are_appended_after_original_points(self):
        background = np.array([[0.5, 0.5, 0.5]])
        points, faces, _ = prepare_plc_input(_CUBE_POINTS, _VALID_FACES, background_points=background)
        assert len(points) == len(_CUBE_POINTS) + 1
        assert np.array_equal(points[:len(_CUBE_POINTS)], _CUBE_POINTS)
        assert np.array_equal(points[-1], background[0])
        # Face indices are unaffected - they only ever referenced the
        # original points, appended-after-the-fact background points don't
        # shift anything.
        assert np.array_equal(faces, _VALID_FACES)

    def test_empty_background_points_is_a_no_op(self):
        points, _, _ = prepare_plc_input(_CUBE_POINTS, _VALID_FACES, background_points=np.zeros((0, 3)))
        assert len(points) == len(_CUBE_POINTS)

    def test_out_of_bounds_face_index_raises(self):
        bad_faces = np.array([[0, 1, 99]], dtype=np.int32)
        with pytest.raises(RuntimeError, match="Invalid face indices"):
            prepare_plc_input(_CUBE_POINTS, bad_faces)

    def test_negative_face_index_raises(self):
        bad_faces = np.array([[0, 1, -1]], dtype=np.int32)
        with pytest.raises(RuntimeError, match="Invalid face indices"):
            prepare_plc_input(_CUBE_POINTS, bad_faces)

    def test_degenerate_face_is_removed(self):
        faces = np.array([[0, 1, 2], [0, 0, 3]], dtype=np.int32)  # 2nd face: repeated vertex 0
        _, kept_faces, _ = prepare_plc_input(_CUBE_POINTS, faces)
        assert len(kept_faces) == 1
        assert np.array_equal(kept_faces[0], [0, 1, 2])

    def test_degenerate_face_removal_keeps_face_markers_in_sync(self):
        faces = np.array([[0, 1, 2], [0, 0, 3], [1, 2, 3]], dtype=np.int32)
        markers = np.array([10, 20, 30], dtype=np.int32)
        _, kept_faces, kept_markers = prepare_plc_input(_CUBE_POINTS, faces, face_markers=markers)
        assert len(kept_faces) == 2
        assert kept_markers.tolist() == [10, 30]

    def test_background_points_do_not_trigger_bounds_check_against_faces(self):
        """Background points are appended AFTER every point `faces` can
        reference, so a valid faces array must stay valid regardless of
        how many background points are added - the bounds check must run
        against the post-concatenation point count, not the pre-
        concatenation one (points referenced by faces are always a strict
        prefix)."""
        background = np.zeros((5, 3))
        points, faces, _ = prepare_plc_input(_CUBE_POINTS, _VALID_FACES, background_points=background)
        assert faces.max() < len(points)
