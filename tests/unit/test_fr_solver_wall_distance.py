"""Unit tests for core/fr_solver/turbulence.py's compute_wall_distance_field
use_eikonal handling and _map_node_distances_to_points - the mapping step
that used to silently discard the just-computed node-level distance field
and re-query mesh_nodes/wall_coords directly via KD-Tree regardless of
which method (KD-Tree or Eikonal) computed it, making --use-eikonal have
no observable effect at all. These tests use a lightweight stand-in for
FRSolver (constructing a real one is out of scope here) and mock
compute_wall_distance itself, so what's under test is purely
compute_wall_distance_field's OWN branch selection and mapping logic.
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from autoflowcfd.core.fr_solver.turbulence import (
    _map_node_distances_to_points,
    compute_wall_distance_field,
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

    @patch("autoflowcfd.core.fr_solver.turbulence.compute_wall_distance")
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

    @patch("autoflowcfd.core.fr_solver.turbulence.compute_wall_distance")
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

    @patch("autoflowcfd.core.fr_solver.turbulence.compute_wall_distance")
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

    @patch("autoflowcfd.core.fr_solver.turbulence.compute_wall_distance")
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
