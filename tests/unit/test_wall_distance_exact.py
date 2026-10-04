# -*- coding: utf-8 -*-
"""壁面距离：到壁面三角形的精确最近距离（`core/utils/wall_distance`）。

2026-10-04 之前来源是"到最近壁面**节点**的距离"：高阶解点不在节点上，贴壁解点被高估为
`sqrt(d^2 + s^2)`（`s` 为切向到最近顶点的偏移）。plate_demo 实测 P2 最贴壁 5% 的解点被
高估中位数 4.1 倍、最多 76 倍。下面 `test_node_distance_overestimates_near_wall_points`
在同一构造上复现旧做法的结果，其余用例钉住新做法：与暴力求解逐位一致、壁面上的点精确为 0、
跨进程序列化后结果不变。
"""

import pickle

import numpy as np
import pytest
from scipy.spatial import cKDTree

from autoflowcfd.core.utils.wall_distance import (
    WALL_COINCIDENCE_ULPS, WallDistanceSource, triangles_from_faces,
)
from autoflowcfd.core.utils.wall_distance.aabb_tree import point_triangle_d2


def _brute_point_triangle(p, a, b, c, n=401):
    """在三角形上取稠密重心网格 + 三条边的稠密采样求最小距离（参考值，只用于粗验区域判别）。"""
    u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    m = (u + v) <= 1.0
    q = a + u[m, None] * (b - a) + v[m, None] * (c - a)
    return np.sqrt(((q - p) ** 2).sum(axis=1)).min()


class TestPointTriangle:
    A, B, C = np.array([0.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])

    @pytest.mark.parametrize("p,expected", [
        ([0.5, 0.25, 0.3], 0.3),                         # 面区域：到平面的距离
        ([-1.0, -1.0, 0.0], np.sqrt(2.0)),                # 顶点 A 区域
        ([3.0, -1.0, 0.0], np.sqrt(2.0)),                 # 顶点 B 区域
        ([1.0, -2.0, 0.0], 2.0),                          # 边 AB 区域
        ([-0.5, 0.5, 0.0], 0.5),                          # 边 AC 区域
        ([2.0, 1.0, 0.0], 2.0 / np.sqrt(5.0)),            # 边 BC 区域（到直线 x + 2y = 2）
    ])
    def test_voronoi_regions_exact(self, p, expected):
        d = np.sqrt(point_triangle_d2(np.array(p, dtype=float), self.A, self.B, self.C))
        assert d == pytest.approx(expected, rel=1e-14, abs=1e-15)

    def test_random_points_match_dense_sampling(self):
        rng = np.random.default_rng(0)
        for _ in range(30):
            a, b, c = rng.standard_normal((3, 3))
            p = 2.0 * rng.standard_normal(3)
            d = np.sqrt(point_triangle_d2(p, a, b, c))
            ref = _brute_point_triangle(p, a, b, c)
            # 采样参考值只会偏大（采样点是三角形的子集），且偏大量不超过采样间距
            assert d <= ref + 1e-12
            assert ref - d < 5e-3 * max(1.0, np.linalg.norm(b - a) + np.linalg.norm(c - a))

    def test_degenerate_triangle_uses_edges(self):
        a, b, c = np.zeros(3), np.array([1.0, 0, 0]), np.array([2.0, 0, 0])   # 共线
        d = np.sqrt(point_triangle_d2(np.array([1.5, 1.0, 0.0]), a, b, c))
        assert d == pytest.approx(1.0, rel=1e-14)


def _triangle_soup(rng, n):
    centers = rng.uniform(-1.0, 1.0, (n, 1, 3))
    return centers + 0.05 * rng.standard_normal((n, 3, 3))


class TestTreeQuery:
    def test_matches_brute_force_exactly(self):
        rng = np.random.default_rng(1)
        tri = _triangle_soup(rng, 700)
        pts = rng.uniform(-1.5, 1.5, (500, 3))
        src = WallDistanceSource(tri)
        d = src.query(pts)
        ref = np.array([min(np.sqrt(point_triangle_d2(p, t[0], t[1], t[2])) for t in tri) for p in pts])
        np.testing.assert_array_equal(d, ref)

    def test_query_keeps_leading_shape(self):
        src = WallDistanceSource(np.array([[[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]]))
        assert src.query(np.ones((4, 6, 3))).shape == (4, 6)

    def test_empty_wall_is_an_error(self):
        with pytest.raises(ValueError):
            WallDistanceSource(np.empty((0, 3, 3)))

    def test_pickle_roundtrip_gives_identical_distances(self):
        """分布式把来源发给各 rank：只传三角形、接收端重建树，结果逐位相同。"""
        rng = np.random.default_rng(2)
        src = WallDistanceSource(_triangle_soup(rng, 200))
        pts = rng.uniform(-1.5, 1.5, (300, 3))
        clone = pickle.loads(pickle.dumps(src))
        np.testing.assert_array_equal(clone.query(pts), src.query(pts))


class TestWallPoints:
    """`d == 0` 当且仅当点位于壁面上（SA-neg 的强 Dirichlet 点据此识别）。"""

    def _plate(self, offset=0.0):
        # 远离原点的平板：检验重合判据随坐标量级缩放
        o = np.array([offset, offset, offset])
        nodes = o + np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        return WallDistanceSource(triangles_from_faces(nodes, np.array([[0, 1, 2], [0, 2, 3]])))

    @pytest.mark.parametrize("offset", [0.0, 37.0, 1.0e3])
    def test_points_on_wall_are_exactly_zero(self, offset):
        src = self._plate(offset)
        rng = np.random.default_rng(3)
        uv = rng.uniform(0.0, 1.0, (200, 2))
        o = np.array([offset, offset, offset])
        # 用"顶点坐标的线性组合"生成（与直边映射算解点坐标同一种舍入）
        p00, p10, p11, p01 = o + np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        pts = ((1 - uv[:, :1]) * (1 - uv[:, 1:]) * p00 + uv[:, :1] * (1 - uv[:, 1:]) * p10
               + uv[:, :1] * uv[:, 1:] * p11 + (1 - uv[:, :1]) * uv[:, 1:] * p01)
        assert np.all(src.query(pts) == 0.0)

    @pytest.mark.parametrize("offset", [0.0, 1.0e3])
    def test_points_just_off_wall_are_not_snapped(self, offset):
        src = self._plate(offset)
        scale = offset + 1.0 + np.sqrt(2.0)
        h = 4.0 * WALL_COINCIDENCE_ULPS * np.finfo(float).eps * scale
        p = np.array([[offset + 0.3, offset + 0.6, offset + h]])
        d = src.query(p)[0]
        assert d > 0.0 and d == pytest.approx(h, rel=1e-3)

    def test_first_bl_solution_point_distance_is_kept(self):
        """壁面解析网格上 P3 第一排解点（~7e-8 m）远在重合判据之上。"""
        src = self._plate()
        assert src.query(np.array([[0.5, 0.5, 7e-8]]))[0] == pytest.approx(7e-8, rel=1e-12)


def test_node_distance_overestimates_near_wall_points():
    """fail 半边：旧来源（到最近壁面节点）在贴壁、远离节点的解点上高估成 `sqrt(d^2 + s^2)`。

    壁面网格尺度 1e-2、解点高 1e-5 位于三角形内部：旧做法给出 ~3.3e-3（高估 ~330 倍），
    新做法给出 1e-5。
    """
    nodes = np.array([[0.0, 0, 0], [1e-2, 0, 0], [0, 1e-2, 0]])
    p = np.array([[1e-2 / 3, 1e-2 / 3, 1e-5]])
    old = cKDTree(nodes).query(p)[0][0]
    new = WallDistanceSource(nodes[None]).query(p)[0]
    assert old / 1e-5 > 300.0
    assert new == pytest.approx(1e-5, rel=1e-12)


class TestQuadFaces:
    def test_planar_quad_is_split_exactly(self):
        nodes = np.array([[0.0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]])
        tri = triangles_from_faces(nodes, np.array([[0, 1, 2, 3]]))
        assert tri.shape == (2, 3, 3)
        src = WallDistanceSource(tri)
        assert src.query(np.array([[0.2, 0.9, 0.5]]))[0] == pytest.approx(0.5, rel=1e-14)

    def test_non_planar_quad_is_an_error_not_an_approximation(self):
        nodes = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0.1], [0, 1, 0]])
        with pytest.raises(ValueError, match="不共面"):
            triangles_from_faces(nodes, np.array([[0, 1, 2, 3]]))

    def test_triangle_rows_with_padding_are_triangles(self):
        nodes = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
        tri = triangles_from_faces(nodes, np.array([[0, 1, 2, -1]]))
        assert tri.shape == (1, 3, 3)
