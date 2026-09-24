"""Unit tests for core/fr_solver/turbulence.py's compute_wall_distance_field
use_eikonal handling and _map_node_distances_to_points - the mapping step
that used to silently discard the just-computed node-level distance field
and re-query mesh_nodes/wall_coords directly via KD-Tree regardless of
which method (KD-Tree or Eikonal) computed it, making --use-eikonal have
no observable effect at all. These tests use a lightweight stand-in for
FRSolver (constructing a real one is out of scope here) and mock
compute_wall_distance itself, so what's under test is purely
compute_wall_distance_field's OWN branch selection and mapping logic.

## patch 目标必须指向**子模块**（2026-09-24）

`fr_solver.turbulence` 已拆成子包，`compute_wall_distance` 的**调用点**在
`turbulence/wall_distance.py` 里。patch 包的 `__init__` 属性**不会**改变
子模块内部的调用 —— 那样 patch 静默失效、测试照样"通过"，比没有测试更糟
（本会话第 5 次撞上同类假通过）。所以 patch 目标是
`...turbulence.wall_distance.compute_wall_distance`。
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from autoflowcfd.core.fr_solver.turbulence import (
    _map_node_distances_to_points,
    compute_wall_distance_field,
    recompute_wall_distance_for_current_order,
)


def _make_solver(turb_model_name, n_cells, n_sps, sps_coords=None, cell_centers=None):
    mesh = SimpleNamespace(sps_coords=sps_coords, cell_centers=cell_centers)
    state = SimpleNamespace(U=np.zeros((n_cells, n_sps, 5)))
    return SimpleNamespace(turb_model_name=turb_model_name, mesh=mesh, state=state, wall_distance=None)


class TestMapNodeDistancesToPoints:
    def test_query_point_exactly_at_a_node_returns_that_nodes_distance(self):
        mesh_nodes = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        node_distances = np.array([0.0, 1.0, 2.0])
        result = _map_node_distances_to_points(mesh_nodes, node_distances, mesh_nodes.copy())
        assert np.allclose(result, node_distances)

    def test_query_point_takes_nearest_nodes_value_not_an_interpolation(self):
        mesh_nodes = np.array([[0., 0., 0.], [10., 0., 0.]])
        node_distances = np.array([0.0, 100.0])
        # Closer to node 0 (distance 0.1) than to node 1 (distance 9.9).
        query = np.array([[0.1, 0., 0.]])
        result = _map_node_distances_to_points(mesh_nodes, node_distances, query)
        assert result[0] == 0.0


class TestComputeWallDistanceFieldBranching:
    def test_non_turbulent_model_is_a_no_op(self):
        solver = _make_solver("NONE", n_cells=2, n_sps=1)
        compute_wall_distance_field(solver, np.zeros((3, 3)), np.array([0]))
        assert solver.wall_distance is None

    @patch("autoflowcfd.core.fr_solver.turbulence.wall_distance.compute_wall_distance")
    def test_eikonal_with_sps_coords_maps_via_nearest_node(self, mock_compute):
        # 3 mesh nodes on a line; node_distances mocked as if Eikonal had
        # already solved them.
        mesh_nodes = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        mock_compute.return_value = np.array([0.0, 1.0, 2.0])

        n_cells, n_sps = 1, 1
        # A single SP sitting exactly on node 1 (distance should be 1.0).
        sps_coords = np.array([[[1., 0., 0.]]])
        solver = _make_solver("DDES", n_cells, n_sps, sps_coords=sps_coords)

        connectivity = np.array([[1, -1], [0, 2], [1, -1]])
        compute_wall_distance_field(
            solver, mesh_nodes, wall_indices=np.array([0]),
            connectivity=connectivity, use_eikonal=True,
        )

        mock_compute.assert_called_once()
        # use_eikonal/connectivity must have been forwarded to
        # core.wall_distance.compute_wall_distance, not silently dropped.
        _, kwargs = mock_compute.call_args
        assert kwargs["use_eikonal"] is True
        assert kwargs["connectivity"] is connectivity

        assert solver.wall_distance.shape == (n_cells, n_sps)
        assert solver.wall_distance[0, 0] == pytest.approx(1.0)

    @patch("autoflowcfd.core.fr_solver.turbulence.wall_distance.compute_wall_distance")
    def test_eikonal_without_sps_coords_falls_back_to_cell_centers(self, mock_compute):
        mesh_nodes = np.array([[0., 0., 0.], [5., 0., 0.]])
        mock_compute.return_value = np.array([0.0, 5.0])

        n_cells, n_sps = 1, 2
        cell_centers = np.array([[5., 0., 0.]])  # right on node 1 -> distance 5.0
        solver = _make_solver("SST", n_cells, n_sps, sps_coords=None, cell_centers=cell_centers)

        compute_wall_distance_field(
            solver, mesh_nodes, wall_indices=np.array([0]), use_eikonal=True,
        )

        assert solver.wall_distance.shape == (n_cells, n_sps)
        assert np.allclose(solver.wall_distance, 5.0)

    @patch("autoflowcfd.core.fr_solver.turbulence.wall_distance.compute_wall_distance")
    def test_eikonal_without_any_query_points_falls_back_to_mean(self, mock_compute):
        mesh_nodes = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        mock_compute.return_value = np.array([0.0, 1.0, 2.0])

        n_cells, n_sps = 2, 1
        solver = _make_solver("WMLES", n_cells, n_sps, sps_coords=None, cell_centers=None)

        compute_wall_distance_field(
            solver, mesh_nodes, wall_indices=np.array([0]), use_eikonal=True,
        )

        assert solver.wall_distance.shape == (n_cells, n_sps)
        assert np.allclose(solver.wall_distance, 1.0)  # mean([0,1,2])

    @patch("autoflowcfd.core.fr_solver.turbulence.wall_distance.compute_wall_distance")
    def test_default_kdtree_path_is_unaffected_by_eikonal_changes(self, mock_compute):
        """Regression guard: use_eikonal=False (the default, pre-existing
        behaviour) must still do its own direct SP-to-wall KD-Tree query,
        not route through the new nearest-mesh-node mapping."""
        mesh_nodes = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        mock_compute.return_value = np.array([0.0, 1.0, 2.0])  # only used for logging on this path

        n_cells, n_sps = 1, 1
        sps_coords = np.array([[[1.5, 0., 0.]]])  # 0.5 from wall node 0 at x=1... actually nearest wall (only node 0 at x=0) is 1.5
        solver = _make_solver("SST", n_cells, n_sps, sps_coords=sps_coords)

        compute_wall_distance_field(
            solver, mesh_nodes, wall_indices=np.array([0]), use_eikonal=False,
        )

        # Direct Euclidean distance from the SP (1.5,0,0) to the only wall
        # node (0,0,0) is 1.5 - not the nearest-mesh-node value (which
        # would incorrectly be node_distances[1] == 1.0 if this had been
        # routed through the Eikonal mapping path by mistake).
        assert solver.wall_distance[0, 0] == pytest.approx(1.5)


class TestRecomputeWallDistanceForCurrentOrder:
    """真实 bug 回归测试（2026-09-06，cube_demo 真实网格 Order
    Continuation P0->P1 升阶后 k_mean 持续增长排查发现）：阶数切换此前
    对 wall_distance 做 Lagrange 插值（升阶）/`np.mean` 压缩（resume 重置
    到 P0），P0 单 SP 的常数基函数会把"单元内离墙最近的 SP"这个空间
    分辨率信息广播抹平——见 `recompute_wall_distance_for_current_order`
    文档完整推导。这里验证：阶数切换后调用这个函数，能用缓存的纯几何量
    重新查询出真正的逐 SP 精确值，而不是继续沿用被插值/广播污染的
    近似值。"""

    def _make_mesh_solver(self, n_cells, n_sps, sps_coords, cell_centers=None):
        mesh = SimpleNamespace(
            sps_coords=sps_coords, cell_centers=cell_centers,
            n_cells=n_cells, n_sps_per_cell=n_sps,
        )
        state = SimpleNamespace(U=np.zeros((n_cells, n_sps, 5)))
        return SimpleNamespace(
            turb_model_name="SST", mesh=mesh, state=state, wall_distance=None,
            current_order=0,
        )

    def test_recomputes_true_per_sp_resolution_after_order_upgrade(self):
        mesh_nodes = np.array([[0., 0., 0.]])
        wall_indices = np.array([0])

        # P0：1 个单元、1 个 SP，位于 x=1.0（到墙节点距离=1.0）。
        p0_sps = np.array([[[1.0, 0., 0.]]])
        solver = self._make_mesh_solver(n_cells=1, n_sps=1, sps_coords=p0_sps)
        compute_wall_distance_field(solver, mesh_nodes, wall_indices, use_eikonal=False)
        np.testing.assert_allclose(solver.wall_distance, [[1.0]])
        assert hasattr(solver, "_wall_coords_for_recompute")

        # 模拟 P0->P1 阶数切换：单元内新增一个近壁 SP（x=0.2，真实距离
        # 应为 0.2）和一个远壁 SP（x=1.8，真实距离应为 1.8）——真实的
        # 单元内空间变化远比 P0 那 1 个 SP 更大，如果沿用旧的插值/广播
        # 机制，这两个 SP 会被错误地都赋成 P0 那个常数 1.0。这里先手动
        # 把 wall_distance 设成"插值广播"会产生的（错误）结果，模拟
        # `interpolate_to_new_order` 已经跑过、set_order 也已切到 P1 之后
        # 的状态，再调用本函数验证它能纠正过来。
        solver.mesh.sps_coords = np.array([[[0.2, 0., 0.], [1.8, 0., 0.]]])
        solver.mesh.n_sps_per_cell = 2
        solver.state.U = np.zeros((1, 2, 5))
        solver.wall_distance = np.array([[1.0, 1.0]])  # 插值广播出的（错误）占位值
        solver.current_order = 1

        result = recompute_wall_distance_for_current_order(solver)

        assert result is True
        # 必须是真正的逐 SP 精确值（0.2/1.8），不是继续沿用广播的 1.0，
        # 且两个 SP 的值必须不同——证明空间分辨率被真正恢复。
        np.testing.assert_allclose(solver.wall_distance, [[0.2, 1.8]])
        assert solver.wall_distance[0, 0] != solver.wall_distance[0, 1]

    def test_returns_false_and_leaves_untouched_when_no_cache_available(self):
        """没有调用过 compute_wall_distance_field（没有缓存）时必须安全
        返回 False，不能崩溃，也不能把 wall_distance 改成别的东西——
        调用方（order_continuation.py）依赖这个信号来决定是否保留原有
        的插值/平均结果作为退化后备。"""
        solver = self._make_mesh_solver(
            n_cells=1, n_sps=2, sps_coords=np.array([[[0.2, 0., 0.], [1.8, 0., 0.]]]),
        )
        solver.wall_distance = np.array([[1.0, 1.0]])  # 假设是插值结果

        result = recompute_wall_distance_for_current_order(solver)

        assert result is False
        np.testing.assert_allclose(solver.wall_distance, [[1.0, 1.0]])

    def test_returns_false_when_wall_distance_is_none(self):
        """湍流模型不需要壁面距离（wall_distance 恒为 None）时的既有
        no-op 行为必须保持——不能因为新增的重新查询逻辑而意外报错。"""
        solver = self._make_mesh_solver(n_cells=1, n_sps=1, sps_coords=np.array([[[1.0, 0., 0.]]]))
        solver.wall_distance = None

        assert recompute_wall_distance_for_current_order(solver) is False

    def test_cell_center_fallback_cache_also_recomputes(self):
        """没有 sps_coords（走单元中心回退路径）时缓存的 wall_coords 同样
        能在阶数切换后重新查询——虽然这条路径本来就是逐单元一个值（不是
        逐 SP），但至少要证明"阶数切换后重新查询"这个统一机制对这条
        分支同样生效、不会因为分支不同而崩溃或被跳过。"""
        mesh_nodes = np.array([[0., 0., 0.]])
        wall_indices = np.array([0])
        centers = np.array([[2.0, 0., 0.]])
        solver = self._make_mesh_solver(n_cells=1, n_sps=1, sps_coords=None, cell_centers=centers)

        compute_wall_distance_field(solver, mesh_nodes, wall_indices, use_eikonal=False)
        np.testing.assert_allclose(solver.wall_distance, [[2.0]])

        # 阶数切换：单元中心本身不随阶数变化，但形状(n_sps)变了。
        solver.mesh.n_sps_per_cell = 3
        solver.wall_distance = np.full((1, 3), 2.0)

        result = recompute_wall_distance_for_current_order(solver)

        assert result is True
        np.testing.assert_allclose(solver.wall_distance, [[2.0, 2.0, 2.0]])
