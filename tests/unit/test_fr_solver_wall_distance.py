"""壁面距离的单机施加与换阶重查（`core/fr_solver/turbulence/wall_distance.py`）；来源对象本身的
精确性见 `test_wall_distance_exact.py`。

历史回归点：换阶后必须按新 SP 坐标重新查询，不能沿用插值/广播出的值（2026-09-05/06）；
没有 sps_coords 时报错，不退回单元中心 / 全域平均值（2026-09-25）。
"""

from types import SimpleNamespace
import numpy as np
import pytest

from autoflowcfd.core.fr_solver.turbulence import (
    apply_wall_distance_source,
    recompute_wall_distance_for_current_order,
)
from autoflowcfd.core.utils.wall_distance import WallDistanceSource

#: x = 0 平面上覆盖 (y, z) 原点附近的一块壁面：点 (x, 0, 0) 的壁距就是 |x|
_WALL_X0 = WallDistanceSource(np.array([[[0.0, -10.0, -10.0], [0.0, 10.0, -10.0], [0.0, 0.0, 10.0]]]))


def _make_solver(turb_model_name, n_cells, n_sps, sps_coords=None):
    mesh = SimpleNamespace(sps_coords=sps_coords, n_cells=n_cells, n_sps_per_cell=n_sps)
    state = SimpleNamespace(U=np.zeros((n_cells, n_sps, 5)))
    return SimpleNamespace(turb_model_name=turb_model_name, mesh=mesh, state=state,
                           wall_distance=None, current_order=0)


class TestApplyWallDistanceSource:
    def test_maps_to_every_sp(self):
        solver = _make_solver("SST", 1, 2, sps_coords=np.array([[[1.5, 0., 0.], [0.25, 0., 0.]]]))
        apply_wall_distance_source(solver, _WALL_X0)
        np.testing.assert_allclose(solver.wall_distance, [[1.5, 0.25]], rtol=1e-15)

    def test_missing_sps_coords_is_an_error_not_a_fallback(self):
        solver = _make_solver("SST", 1, 2, sps_coords=None)
        with pytest.raises(RuntimeError):
            apply_wall_distance_source(solver, _WALL_X0)


class TestRecomputeWallDistanceForCurrentOrder:
    def test_recomputes_true_per_sp_resolution_after_order_upgrade(self):
        solver = _make_solver("SST", 1, 1, sps_coords=np.array([[[1.0, 0., 0.]]]))
        apply_wall_distance_source(solver, _WALL_X0)
        np.testing.assert_allclose(solver.wall_distance, [[1.0]])

        # P0 -> P1：插值/广播会把两个 SP 都赋成 P0 的 1.0；重查必须恢复 0.2/1.8
        solver.mesh.sps_coords = np.array([[[0.2, 0., 0.], [1.8, 0., 0.]]])
        solver.mesh.n_sps_per_cell = 2
        solver.state.U = np.zeros((1, 2, 5))
        solver.wall_distance = np.array([[1.0, 1.0]])
        solver.current_order = 1
        assert recompute_wall_distance_for_current_order(solver) is True
        np.testing.assert_allclose(solver.wall_distance, [[0.2, 1.8]])

    def test_wall_distance_without_a_source_is_an_error(self):
        """壁距是直接赋值的（没有来源）：换阶后无法在新解点上得到正确值，报错而不是保留旧阶数的值
        （此前返回 False、旧值被静默沿用）。"""
        solver = _make_solver("SST", 1, 2, sps_coords=np.array([[[0.2, 0., 0.], [1.8, 0., 0.]]]))
        solver.wall_distance = np.array([[1.0, 1.0]])
        with pytest.raises(RuntimeError, match="没有来源"):
            recompute_wall_distance_for_current_order(solver)

    def test_returns_false_when_wall_distance_is_none(self):
        solver = _make_solver("SST", 1, 1, sps_coords=np.array([[[1.0, 0., 0.]]]))
        assert recompute_wall_distance_for_current_order(solver) is False
