"""mesh_tetgen_input_prep.prepare_plc_input 的单元测试——从
mesh_tetgen_core.fill_core_volume 里抽出来的纯 PLC 输入校验/清理（拼接
背景点、面索引越界检查、退化面剔除），在这里与 tetgen 本身隔离，本模块
从不调用它。
"""

import numpy as np
import pytest

from autoflowcfd.grid.mesh_gen.tetgen.mesh_tetgen_input_prep import prepare_plc_input

_CUBE_POINTS = np.array([
    [0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
    [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.],
], dtype=np.float64)
# 立方体底面的两个三角形——对这些测试足够，它们从不真的调用 tetgen。
_VALID_FACES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)


class TestPreparePlcInput:
    def test_valid_input_passes_through_unchanged(self):
        points, faces = prepare_plc_input(_CUBE_POINTS, _VALID_FACES)
        assert np.array_equal(points, _CUBE_POINTS)
        assert np.array_equal(faces, _VALID_FACES)

    def test_background_points_are_appended_after_original_points(self):
        background = np.array([[0.5, 0.5, 0.5]])
        points, faces = prepare_plc_input(_CUBE_POINTS, _VALID_FACES, background_points=background)
        assert len(points) == len(_CUBE_POINTS) + 1
        assert np.array_equal(points[:len(_CUBE_POINTS)], _CUBE_POINTS)
        assert np.array_equal(points[-1], background[0])
        # 面索引不受影响——它们只引用原始的点，事后追加的背景点不会让任何
        # 索引移位。
        assert np.array_equal(faces, _VALID_FACES)

    def test_empty_background_points_is_a_no_op(self):
        points, _ = prepare_plc_input(_CUBE_POINTS, _VALID_FACES, background_points=np.zeros((0, 3)))
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
        _, kept_faces = prepare_plc_input(_CUBE_POINTS, faces)
        assert len(kept_faces) == 1
        assert np.array_equal(kept_faces[0], [0, 1, 2])

    def test_background_points_do_not_trigger_bounds_check_against_faces(self):
        """背景点追加在 `faces` 可能引用的所有点**之后**，所以不论加多少背景点，
        合法的 faces 数组都必须保持合法——越界检查必须针对拼接之后的点数，而不是
        拼接之前的（faces 引用的点总是严格的前缀）。
        """
        background = np.zeros((5, 3))
        points, faces = prepare_plc_input(_CUBE_POINTS, _VALID_FACES, background_points=background)
        assert faces.max() < len(points)
