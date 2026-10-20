"""射线奇偶判内外的稳健性（I-144，`mesh_domain_classify_geometry.point_inside_closed_surface`）。

此前 `_ray_triangle_intersect_count` 用包含式重心坐标判定：射线恰过两三角形公共边 / 公共顶点时重复计数、奇偶翻转；
平行判据 1e-9 是长度平方量纲的绝对量，边长约 3e-5 m 的细小三角形被当成平行漏计；背景点撒点固定 +x 方向，方形
远场面沿对角线三角化时规则网格候选点系统性落在对角线上。
"""

import numpy as np
import pytest

from autoflowcfd.grid.mesh_gen.tetgen.mesh_tetgen_seeding import generate_core_background_points
from autoflowcfd.grid.mesh_gen.utils.mesh_domain_classify_geometry import (
    _ray_parity, find_point_inside_closed_shell, point_inside_closed_surface,
)

# 单位立方体，每个面沿 y=z（或同类）对角线拆成两个三角形
_P = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
_F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
               [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])


def _tri(points, faces=_F):
    return points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]


def test_ray_through_a_shared_diagonal_is_degenerate_not_double_counted():
    v0, v1, v2 = _tri(_P)
    center = np.full(3, 0.5)                     # +x 射线命中 x=1 面的对角线 (1,0,0)-(1,1,1) 中点
    assert _ray_parity(center, np.array([1.0, 0, 0]), v0, v1, v2, 1.0) is None
    assert point_inside_closed_surface(center, v0, v1, v2) is True
    assert point_inside_closed_surface(np.array([1.5, 0.5, 0.5]), v0, v1, v2) is False


@pytest.mark.parametrize("size", [3e-5, 1.0, 3e4])
def test_criteria_are_scale_free(size):
    v0, v1, v2 = _tri(_P * size)
    assert point_inside_closed_surface(np.array([0.3, 0.6, 0.45]) * size, v0, v1, v2, n_agree=5) is True
    assert point_inside_closed_surface(np.array([1.3, 0.6, 0.45]) * size, v0, v1, v2, n_agree=5) is False
    found = find_point_inside_closed_shell(_P * size, _F)
    assert found is not None and np.all((found > 0) & (found < size))


def test_grid_seeding_keeps_candidates_on_face_diagonals():
    """规则网格候选点 (y, z) 同序号时落在 x 向各面的对角线上：都在立方体内，必须全部保留。"""
    pts = generate_core_background_points(_P * 10.0, _F, target_edge_length=1.0, grid_spacing_factor=2.5,
                                          clearance_factor=1.0)
    axis = np.arange(1.25, 10.0, 2.5)
    expected = [(x, y, z) for x in axis for y in axis for z in axis
                if min(x, y, z, 10 - x, 10 - y, 10 - z) >= 1.0]
    assert len(pts) == len(expected)
