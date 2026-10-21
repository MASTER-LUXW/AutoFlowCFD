"""网格生成的三角形面法向（`mesh_utils.compute_face_normals`，唯一实现，I-206）。

此前退化判据是绝对阈值（叉积模长 1e-9 m^2）、退化面给固定方向 +z：边长约 0.03 mm 的正常面被当成退化面，
法向被换成 +z，混进 BL 挤出的逐节点平均法向；另有两份私有实现各自处理退化面。
"""

import numpy as np

from autoflowcfd.grid.mesh_gen.utils.mesh_utils import compute_face_normals


def test_small_but_regular_face_keeps_its_true_normal():
    h = 1.0e-5                                     # 边长 10 um：BL 近壁面元的量级
    nodes = np.array([[0.0, 0.0, 0.0], [0.0, h, 0.0], [0.0, 0.0, h]])
    np.testing.assert_allclose(compute_face_normals(nodes, np.array([[0, 1, 2]])), [[1.0, 0.0, 0.0]], atol=1e-12)


def test_degenerate_face_contributes_no_direction():
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    normals = compute_face_normals(nodes, np.array([[0, 1, 2], [0, 1, 3]]))
    np.testing.assert_array_equal(normals[0], [0.0, 0.0, 0.0])          # 共线：零向量，不是任意固定方向
    np.testing.assert_allclose(normals[1], [0.0, 0.0, 1.0])


def test_scale_invariant():
    rng = np.random.default_rng(0)
    nodes = rng.standard_normal((30, 3))
    faces = rng.integers(0, 30, size=(40, 3))
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
    np.testing.assert_allclose(compute_face_normals(1.0e-6 * nodes, faces), compute_face_normals(nodes, faces),
                               atol=1e-9)
