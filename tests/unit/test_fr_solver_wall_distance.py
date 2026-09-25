"""壁面距离：来源对象（`core/utils/wall_distance_source.py`）与单机施加/换阶重查
（`core/fr_solver/turbulence/wall_distance.py`）。

历史回归点都保留：
* `--use-eikonal` 曾经没有任何可观测效果（映射步骤丢掉刚算出的节点级
  Eikonal 距离、直接对壁面节点重做 KD-Tree）——Eikonal 来源必须取最近网格
  节点上的 Eikonal 值，KD-Tree 来源必须是直接欧氏距离，两者不能串；
* 换阶后必须按新 SP 坐标重新查询，不能沿用插值/广播出的值（2026-09-05/06）。

2026-09-25 起删除了"没有 sps_coords 时退回单元中心 / 全域平均值"两级兜底：
单机 CLI 入口本来就在没有壁面时报错，兜底只会让近壁湍流行为静默变坏，
现在一律报错（见 `apply_wall_distance_source`）。
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from autoflowcfd.core.fr_solver.turbulence import (
    compute_wall_distance_field,
    recompute_wall_distance_for_current_order,
)
from autoflowcfd.core.utils.wall_distance_source import WallDistanceSource


def _make_solver(turb_model_name, n_cells, n_sps, sps_coords=None):
    mesh = SimpleNamespace(sps_coords=sps_coords, n_cells=n_cells, n_sps_per_cell=n_sps)
    state = SimpleNamespace(U=np.zeros((n_cells, n_sps, 5)))
    return SimpleNamespace(turb_model_name=turb_model_name, mesh=mesh, state=state,
                           wall_distance=None, current_order=0)


class TestWallDistanceSource:
    def test_kdtree_is_direct_euclidean_distance(self):
        src = WallDistanceSource.kdtree(np.array([[0., 0., 0.], [10., 0., 0.]]))
        np.testing.assert_allclose(src.query(np.array([[1.5, 0., 0.], [9., 0., 0.]])), [1.5, 1.0])

    def test_eikonal_takes_nearest_nodes_value_not_an_interpolation(self):
        src = WallDistanceSource.eikonal(np.array([[0., 0., 0.], [10., 0., 0.]]),
                                         np.array([0.0, 100.0]), n_wall_nodes=1)
        assert src.query(np.array([[0.1, 0., 0.]]))[0] == 0.0

    def test_query_keeps_leading_shape(self):
        src = WallDistanceSource.kdtree(np.zeros((1, 3)))
        assert src.query(np.ones((4, 6, 3))).shape == (4, 6)

    @patch("autoflowcfd.core.utils.wall_distance.compute_wall_distance")
    def test_eikonal_parameters_are_forwarded(self, mock_compute):
        mesh_nodes = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        mock_compute.return_value = np.array([0.0, 1.0, 2.0])
        conn = np.array([[1, -1], [0, 2], [1, -1]])
        src = WallDistanceSource.from_nodes(mesh_nodes, [0], connectivity=conn, use_eikonal=True)
        _, kwargs = mock_compute.call_args
        assert kwargs["use_eikonal"] is True and kwargs["connectivity"] is conn
        assert src.kind == "eikonal"
        # 一个正好落在节点 1 上的点：Eikonal 值 1.0（不是到壁面节点 0 的欧氏距离重算）
        assert src.query(np.array([[1., 0., 0.]]))[0] == pytest.approx(1.0)

    def test_empty_wall_is_an_error_not_an_estimate(self):
        with pytest.raises(ValueError):
            WallDistanceSource.from_nodes(np.zeros((3, 3)), np.array([], dtype=np.int64))

    def test_eikonal_without_connectivity_is_an_error(self):
        with pytest.raises(ValueError):
            WallDistanceSource.from_nodes(np.zeros((3, 3)), [0], use_eikonal=True)


class TestComputeWallDistanceField:
    def test_non_turbulent_model_is_a_no_op(self):
        solver = _make_solver("NONE", n_cells=2, n_sps=1)
        compute_wall_distance_field(solver, np.zeros((3, 3)), np.array([0]))
        assert solver.wall_distance is None

    def test_kdtree_maps_to_every_sp(self):
        solver = _make_solver("SST", 1, 2, sps_coords=np.array([[[1.5, 0., 0.], [0.25, 0., 0.]]]))
        compute_wall_distance_field(solver, np.array([[0., 0., 0.]]), np.array([0]))
        np.testing.assert_allclose(solver.wall_distance, [[1.5, 0.25]])

    def test_missing_sps_coords_is_an_error_not_a_fallback(self):
        solver = _make_solver("SST", 1, 2, sps_coords=None)
        with pytest.raises(RuntimeError):
            compute_wall_distance_field(solver, np.array([[0., 0., 0.]]), np.array([0]))


class TestRecomputeWallDistanceForCurrentOrder:
    def test_recomputes_true_per_sp_resolution_after_order_upgrade(self):
        solver = _make_solver("SST", 1, 1, sps_coords=np.array([[[1.0, 0., 0.]]]))
        compute_wall_distance_field(solver, np.array([[0., 0., 0.]]), np.array([0]))
        np.testing.assert_allclose(solver.wall_distance, [[1.0]])

        # P0 -> P1：插值/广播会把两个 SP 都赋成 P0 的 1.0；重查必须恢复 0.2/1.8
        solver.mesh.sps_coords = np.array([[[0.2, 0., 0.], [1.8, 0., 0.]]])
        solver.mesh.n_sps_per_cell = 2
        solver.state.U = np.zeros((1, 2, 5))
        solver.wall_distance = np.array([[1.0, 1.0]])
        solver.current_order = 1
        assert recompute_wall_distance_for_current_order(solver) is True
        np.testing.assert_allclose(solver.wall_distance, [[0.2, 1.8]])

    def test_returns_false_without_a_source(self):
        solver = _make_solver("SST", 1, 2, sps_coords=np.array([[[0.2, 0., 0.], [1.8, 0., 0.]]]))
        solver.wall_distance = np.array([[1.0, 1.0]])
        assert recompute_wall_distance_for_current_order(solver) is False
        np.testing.assert_allclose(solver.wall_distance, [[1.0, 1.0]])

    def test_returns_false_when_wall_distance_is_none(self):
        solver = _make_solver("SST", 1, 1, sps_coords=np.array([[[1.0, 0., 0.]]]))
        assert recompute_wall_distance_for_current_order(solver) is False
