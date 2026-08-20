"""Unit tests for cli/solve_helpers.compute_wall_distance_for_solver's
use_eikonal handling - specifically that it now actually builds a node
adjacency graph (grid.node_connectivity.build_node_adjacency) and forwards
it to solver.compute_wall_distance_field, instead of the previous dead
branch that printed a "falling back to KD-Tree" warning and did exactly
that regardless of the flag.
"""

from unittest.mock import MagicMock, patch

import numpy as np

from autoflowcfd.cli.solve_helpers import compute_wall_distance_for_solver
from autoflowcfd.grid.structures import (
    BoundaryMap, GridMetadata, NodeArray, TetrahedralCells, VolumeMeshData,
)


def _volume_mesh_with_wall(n_nodes=4):
    nodes = NodeArray(x=np.arange(n_nodes, dtype=float), y=np.zeros(n_nodes), z=np.zeros(n_nodes))
    cells = TetrahedralCells(connectivity=np.array([[0, 1, 2, 3]], dtype=np.int32), volumes=np.array([1.0]))
    boundaries = BoundaryMap(groups={'wall': np.array([0, 1], dtype=np.int32)}, bc_types={'wall': 'WALL'})
    metadata = GridMetadata(node_count=n_nodes, cell_count=1, boundary_groups=['wall'], file_format='test')
    return VolumeMeshData(nodes=nodes, cells=cells, boundaries=boundaries, metadata=metadata)


class TestComputeWallDistanceForSolverEikonalWiring:
    def test_non_turbulent_model_skips_everything(self):
        solver = MagicMock()
        solver.turb_model_name = 'NONE'
        compute_wall_distance_for_solver(solver, _volume_mesh_with_wall(), use_eikonal=True)
        solver.compute_wall_distance_field.assert_not_called()

    def test_use_eikonal_false_does_not_build_connectivity(self):
        """The adjacency graph has a real construction cost - it must only
        be built when actually needed."""
        solver = MagicMock()
        solver.turb_model_name = 'SST'
        with patch("autoflowcfd.grid.connectivity.node_connectivity.build_node_adjacency") as mock_build:
            compute_wall_distance_for_solver(solver, _volume_mesh_with_wall(), use_eikonal=False)
        mock_build.assert_not_called()
        solver.compute_wall_distance_field.assert_called_once()
        _, kwargs = solver.compute_wall_distance_field.call_args
        assert kwargs == {}  # positional-only call, no connectivity/use_eikonal kwargs on this path

    def test_use_eikonal_true_builds_and_forwards_connectivity(self):
        solver = MagicMock()
        solver.turb_model_name = 'DDES'
        compute_wall_distance_for_solver(solver, _volume_mesh_with_wall(), use_eikonal=True)

        solver.compute_wall_distance_field.assert_called_once()
        args, kwargs = solver.compute_wall_distance_field.call_args
        assert kwargs["use_eikonal"] is True
        connectivity = kwargs["connectivity"]
        assert connectivity.shape[0] == 4  # one row per node
        # node 0 and node 1 share the single tet - must be neighbors.
        assert 1 in connectivity[0]
        assert 0 in connectivity[1]
