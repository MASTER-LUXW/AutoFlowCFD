"""Unit tests for mesh_gen/mesh_tetgen_core.py's Steiner-point budget
estimation (estimate_steinerleft) - isolated from tetgen itself, which
these tests never invoke."""

import numpy as np
import pytest

from autoflowcfd.grid.mesh_gen.mesh_tetgen_core import estimate_steinerleft


def _box_points(dx, dy, dz):
    """Minimal point set spanning a dx x dy x dz bounding box."""
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
        """Regression test for the real bug: Stage B's small local repair
        regions (fine maxvol, from min_cell_size) must not get divided
        into the FULL bounding box - that produced an estimate of ~17.8
        billion tets on a real case (bbox ~72 m^3, a 0.003m Stage B
        region), which inflated steinerleft to the 20M ceiling and let
        tetgen balloon the actual core fill 5x (1.2M -> 6.1M tets)."""
        points = _box_points(8.0, 3.0, 3.0)  # bbox_volume = 72, matches the real case
        main_maxvol = 0.1 ** 3 * 0.15  # matches _build_merged_mesh's own formula
        stage_b_maxvol = 0.003 ** 3 * 0.15  # tiny relative to main_maxvol
        regions = [
            (np.array([4.0, 1.5, 1.5]), 1, main_maxvol),
        ] + [
            (np.array([float(i), 1.0, 1.0]), 1000 + i, stage_b_maxvol) for i in range(8)
        ]

        result = estimate_steinerleft(points, regions)

        # The old (buggy) formula - bbox_volume / min(maxvol) - for these
        # exact inputs:
        old_buggy_estimate = 72.0 / stage_b_maxvol
        assert old_buggy_estimate > 1e10  # confirms this really would have exploded

        # The fixed formula must stay far below that, and below the 20M
        # ceiling both old and new formulas share - it should not just
        # coincidentally hit the same ceiling both ways.
        assert result < 10_000_000
        assert result < 20_000_000

        # And it should be in the right ballpark: dominated by the main
        # region's own domain-wide estimate (480,000 target tets, matching
        # what the real log reported) plus a bounded allowance per extra
        # region, not by the finest region's target resolution.
        main_region_estimate = 72.0 / main_maxvol
        assert main_region_estimate == pytest.approx(480_000.0)
        expected = int(np.clip((main_region_estimate + 8 * 200_000) * 3.0, 300_000, 20_000_000))
        assert result == expected

    def test_only_small_regions_no_main_region(self):
        """No domain-wide max_cell_size region at all (only Stage B
        patches) - still must not blow up, using the coarsest of the small
        regions rather than the full bbox at the finest one."""
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
        regions = [(np.array([0.0, 0.0, 0.0]), 1, 1e-12)]  # absurdly fine, would blow past ceiling
        result = estimate_steinerleft(points, regions)
        assert 300_000 <= result <= 20_000_000
