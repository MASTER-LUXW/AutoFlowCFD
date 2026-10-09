"""mesh_gen/mesh_tetgen_core.py 的 Steiner 点预算估计
（estimate_steinerleft）的单元测试——与 tetgen 本身隔离，这些测试从不
调用它。
"""

import numpy as np
import pytest

from autoflowcfd.grid.mesh_gen.tetgen.mesh_tetgen_core import estimate_steinerleft


def _box_points(dx, dy, dz):
    """张成 dx x dy x dz 包围盒的最小点集。"""
    return np.array([[0.0, 0.0, 0.0], [dx, dy, dz]])


class TestEstimateSteinerleft:
    def test_no_regions_uses_tetgen_default(self):
        assert estimate_steinerleft(_box_points(1, 1, 1), None) == 100_000
        assert estimate_steinerleft(_box_points(1, 1, 1), []) == 100_000

    def test_single_region_matches_bbox_over_maxvol_formula(self):
        points = _box_points(2.0, 3.0, 5.0)  # bbox_volume = 30
        maxvol = 0.01
        regions = [(np.array([1.0, 1.0, 1.0]), 1, maxvol)]

        result = estimate_steinerleft(points, regions)

        expected_estimated_tets = 30.0 / maxvol  # 3000
        expected = int(np.clip(expected_estimated_tets * 3.0, 300_000, 20_000_000))
        assert result == expected == 300_000  # clipped to the floor here

    def test_extra_small_regions_do_not_explode_the_estimate(self):
        """真实缺陷的回归测试：Stage B 的小局部修复区域（由 min_cell_size 得到的
        细 maxvol）不能拿去除**整个**包围盒——真实算例上（包围盒约 72 m^3、
        一个 0.003m 的 Stage B 区域）那样估出约 178 亿个四面体，把 steinerleft
        顶到 2000 万上限，让 tetgen 把实际的核心填充放大了 5 倍（120 万 -> 610 万
        个四面体）。
        """
        points = _box_points(8.0, 3.0, 3.0)  # bbox_volume = 72，与真实算例一致
        main_maxvol = 0.1 ** 3 * 0.15  # 与 _build_merged_mesh 自己的公式一致
        stage_b_maxvol = 0.003 ** 3 * 0.15  # tiny relative to main_maxvol
        regions = [
            (np.array([4.0, 1.5, 1.5]), 1, main_maxvol),
        ] + [
            (np.array([float(i), 1.0, 1.0]), 1000 + i, stage_b_maxvol) for i in range(8)
        ]

        result = estimate_steinerleft(points, regions)

        # 旧的（有缺陷的）公式——bbox_volume / min(maxvol)——在这组输入下
        # 的值：
        old_buggy_estimate = 72.0 / stage_b_maxvol
        assert old_buggy_estimate > 1e10  # 确认它确实会爆

        # 修复后的公式必须远低于它，并且低于新旧公式共用的 2000 万上限——
        # 不应两种算法只是碰巧都撞到同一个上限。
        assert result < 10_000_000
        assert result < 20_000_000

        # 并且量级应当正确：由主区域自己的全域估计主导（目标 48 万个四面体，
        # 与真实日志报告的一致），每个额外区域加一个有界的余量，而不是由最细区域的
        # 目标分辨率主导。
        main_region_estimate = 72.0 / main_maxvol
        assert main_region_estimate == pytest.approx(480_000.0)
        expected = int(np.clip((main_region_estimate + 8 * 200_000) * 3.0, 300_000, 20_000_000))
        assert result == expected

    def test_only_small_regions_no_main_region(self):
        """完全没有全域的 max_cell_size 区域（只有 Stage B 补丁）——仍不能爆，
        应取各小区域里最粗的那个，而不是在最细的分辨率上用整个包围盒。
        """
        points = _box_points(8.0, 3.0, 3.0)
        maxvol_a = 0.01
        maxvol_b = 0.02  # coarsest of the two
        regions = [
            (np.array([1.0, 1.0, 1.0]), 1000, maxvol_a),
            (np.array([2.0, 1.0, 1.0]), 1001, maxvol_b),
        ]

        result = estimate_steinerleft(points, regions)

        expected_estimated_tets = 72.0 / maxvol_b  # coarsest, not finest
        expected = int(np.clip((expected_estimated_tets + 1 * 200_000) * 3.0, 300_000, 20_000_000))
        assert result == expected

    def test_result_always_within_bounds(self):
        points = _box_points(100.0, 100.0, 100.0)
        regions = [(np.array([0.0, 0.0, 0.0]), 1, 1e-12)]  # 细得离谱，会冲过上限
        result = estimate_steinerleft(points, regions)
        assert 300_000 <= result <= 20_000_000
