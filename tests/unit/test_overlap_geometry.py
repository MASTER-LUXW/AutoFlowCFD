"""validation/overlap_geometry.py 三角形-三角形相交判定的单元测试——这个
模块此前没有常驻的测试文件（ProjectFiles Part5 P2 提到的"15 个手工边界
情形 + 3000 例压力测试"是一次性的验证脚本，从未提交）。

这里的薄片三角形回归算例不是人造的最坏情形——它们直接取自 cube_demo 的
一次真实边界层挤出（来自不同原始表面的两个三角形，各自形状正确，尖角
斜接补偿让它们变薄且朝向相近），在修复 triangle_triangle_intersect 之前
经独立的暴力点采样确认是误报。
"""

import numpy as np
import pytest

from autoflowcfd.grid.validation.overlap_geometry import (
    triangle_triangle_intersect,
    triangle_triangle_min_distance,
)

# 两个在三维里真正相交、不共享顶点的三角形（本项目 mesh_front_collision
# 测试通用的同一个夹具）。
A0, A1, A2 = np.array([-2., -2., 0.]), np.array([2., -2., 0.]), np.array([0., 2., 0.])
B0, B1, B2 = np.array([0., 0., -2.]), np.array([0., 0., 2.]), np.array([0., 3., 0.])

# cube_demo 上真实的薄片误报对：来自不同原始表面的两个薄"翅"三角形，
# 沿 z 有一个真实、明确的 0.01m 间隙（经独立的暴力采样验证：真实最小距离
# 约 0.01，远不在 eps 附近）——修复之前却被判为相交，因为它们近乎退化的
# 形状让各自的平面对沿三角形长轴方向的真实偏移只有很弱的敏感性（完整机制
# 见 triangle_triangle_intersect 里"薄片三角形修正"的注释）。
SLIVER_A = np.array([
    [0.503, 0.241369, -0.055],
    [0.503, 0.24137, -0.045],
    [0.5025455844122716, 0.2525455844122716, -0.05],
])
SLIVER_B = np.array([
    [0.503, 0.241371, -0.035],
    [0.503, 0.24137200000000003, -0.025],
    [0.5025455844122716, 0.2525455844122716, -0.03],
])


def _intersects(p, q):
    return bool(triangle_triangle_intersect(
        p[0][None], p[1][None], p[2][None], q[0][None], q[1][None], q[2][None],
    )[0])


class TestTriangleTriangleIntersect:
    def test_crossing_triangles_are_detected(self):
        assert _intersects(np.array([A0, A1, A2]), np.array([B0, B1, B2]))

    def test_well_separated_triangles_are_not_flagged(self):
        far = np.array([B0, B1, B2]) + 100.0
        assert not _intersects(np.array([A0, A1, A2]), far)

    def test_coplanar_overlapping_triangles_are_detected(self):
        a = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
        b = np.array([[0.5, 0.5, 0.], [2.5, 0.5, 0.], [0.5, 2.5, 0.]])
        assert _intersects(a, b)

    def test_coplanar_non_overlapping_triangles_are_not_flagged(self):
        a = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        b = np.array([[10., 10., 0.], [11., 10., 0.], [10., 11., 0.]])
        assert not _intersects(a, b)

    def test_thin_sliver_triangles_with_a_real_gap_are_not_flagged(self):
        """cube_demo 上发现的真实误报的回归测试。两个三角形的 z 范围相隔一个明确
        的 0.01m 间隙（比任何 float64 噪声大一千万倍）——修复之前经
        triangle_triangle_min_distance 与独立的暴力点采样确认过；未修复的函数仍
        把这一对报成相交。
        """
        assert not _intersects(SLIVER_A, SLIVER_B)

        dist = triangle_triangle_min_distance(
            SLIVER_A[0][None], SLIVER_A[1][None], SLIVER_A[2][None],
            SLIVER_B[0][None], SLIVER_B[1][None], SLIVER_B[2][None],
        )[0]
        assert dist == pytest.approx(0.01, abs=1e-4)

    def test_genuine_intersection_still_detected_regardless_of_correction(self):
        """薄片修正（用 triangle_triangle_min_distance 做第二意见）只能把误报变成
        正确的否定——真正、明确的相交（最小距离 0）仍必须报告为 True。
        """
        assert _intersects(np.array([A0, A1, A2]), np.array([B0, B1, B2]))
        dist = triangle_triangle_min_distance(
            A0[None], A1[None], A2[None], B0[None], B1[None], B2[None],
        )
        # 违反前提的调用（这一对**确实**相交）——这里只用来确认它不会大得
        # 离谱；对真正的重叠它不是有意义的"距离"（见该函数的文档）。
        assert dist[0] < 1.0

    def test_shared_vertex_is_not_reported_as_intersecting(self):
        """只在一个共享顶点处相接的两个三角形，重叠区间的测度为零——按这个函数
        的构造不报告为相交（调用方为网格用途另有一道共享节点的预过滤，但几何
        原语本身在这个临界情形上也必须如此）。
        """
        shared = np.array([0., 0., 0.])
        a = np.array([shared, [1., 0., 0.], [0., 1., 0.]])
        b = np.array([shared, [-1., 0., 0.], [0., -1., 0.]])
        assert not _intersects(a, b)

    def test_vectorized_batch_matches_per_row_results(self):
        a_batch = np.stack([A0, SLIVER_A[0], np.array([0., 0., 0.])])
        a1_batch = np.stack([A1, SLIVER_A[1], np.array([1., 0., 0.])])
        a2_batch = np.stack([A2, SLIVER_A[2], np.array([0., 1., 0.])])
        b_batch = np.stack([B0, SLIVER_B[0], np.array([10., 10., 0.])])
        b1_batch = np.stack([B1, SLIVER_B[1], np.array([11., 10., 0.])])
        b2_batch = np.stack([B2, SLIVER_B[2], np.array([10., 11., 0.])])

        result = triangle_triangle_intersect(a_batch, a1_batch, a2_batch, b_batch, b1_batch, b2_batch)

        assert list(result) == [True, False, False]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
