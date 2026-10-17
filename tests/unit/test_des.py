"""DES 网格尺度几何（`core/turbulence/des/grid_scale.py::compute_h_max_and_h_wn`）。

混合长度尺度公式本身见 `test_des_hybrid_length.py`。
"""

import numpy as np

from autoflowcfd.core.turbulence.des import compute_h_max_and_h_wn


class TestIDDESGridScaleGeometry:
    """compute_h_max_and_h_wn 用一个几何已知的合成"网格"（1 棱柱 + 1 四面体）
    钉住 h_max/h_wn 的精确解析值，防止边长->h_max/h_wn 的映射逻辑被误改
    （例如棱柱竖直边的切片索引写错、单元顺序假设弄反）。"""

    class _MockMesh:
        """只提供 compute_h_max_and_h_wn 需要读取的最小属性集
        （n_cells/n_prism_cells/_node_coords/_fixed_prism_conn/
        _fixed_tet_conn），不是真正的 HighOrderMesh。"""

        def __init__(self):
            self.n_cells = 2
            self.n_prism_cells = 1
            # 棱柱：底三角形 v0,v1,v2 边长 1/1/sqrt(2)；顶三角形 w0,w1,w2
            # 由底三角形沿 z 平移 0.01 得到（水平边长不变）；竖直边
            # （近似壁面法向）长度恰好 0.01——典型薄 BL 棱柱形状。
            # 四面体：三条正交棱边长度均为 3，用于验证各向同性单元下
            # h_wn 退化为 h_max。
            self._node_coords = np.array([
                [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.01], [1.0, 0.0, 0.01], [0.0, 1.0, 0.01],
                [0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0],
            ])
            self._fixed_prism_conn = np.array([[0, 1, 2, 3, 4, 5]])
            self._fixed_tet_conn = np.array([[6, 7, 8, 9]])

    def test_prism_h_max_and_h_wn(self):
        """薄棱柱：h_max 取三角形对角边 sqrt(2)，h_wn 取竖直边 0.01——
        验证"最薄方向"被正确识别为法向间距，而不是与 h_max 混淆。"""
        mesh = self._MockMesh()
        h_max, h_wn = compute_h_max_and_h_wn(mesh)
        np.testing.assert_allclose(h_max[0], np.sqrt(2.0), rtol=1e-10)
        np.testing.assert_allclose(h_wn[0], 0.01, rtol=1e-10)

    def test_tet_h_wn_equals_h_max(self):
        """四面体（核心/LES 区）没有明确的"法向"概念，h_wn 应退化为
        h_max（见 compute_h_max_and_h_wn 文档的各向同性简化说明）。"""
        mesh = self._MockMesh()
        h_max, h_wn = compute_h_max_and_h_wn(mesh)
        expected_tet_h_max = 3.0 * np.sqrt(2.0)
        np.testing.assert_allclose(h_max[1], expected_tet_h_max, rtol=1e-10)
        np.testing.assert_allclose(h_wn[1], expected_tet_h_max, rtol=1e-10)
