"""Unit tests for grid/node_connectivity.build_node_adjacency - the node
adjacency graph the Eikonal (Fast Marching) wall-distance solver needs
(core/wall_distance.solve_eikonal_approximation's own `connectivity`
parameter), built from tetrahedron/prism cell connectivity."""

import numpy as np

from autoflowcfd.grid.node_connectivity import build_node_adjacency


def _neighbors_of(adjacency: np.ndarray, node: int) -> set:
    return {int(n) for n in adjacency[node] if n != -1}


class TestBuildNodeAdjacencyTets:
    def test_single_tet_all_four_nodes_mutually_connected(self):
        tets = np.array([[0, 1, 2, 3]])
        adjacency = build_node_adjacency(4, tet_connectivity=tets)
        for node in range(4):
            assert _neighbors_of(adjacency, node) == {0, 1, 2, 3} - {node}

    def test_shared_face_nodes_connect_to_both_tets_own_apex(self):
        """Two tets sharing the face (0,1,2): node 3 is only tet A's own
        apex, node 4 only tet B's - they must NOT be connected to each
        other directly (no edge between them exists in either tet)."""
        tets = np.array([[0, 1, 2, 3], [0, 1, 2, 4]])
        adjacency = build_node_adjacency(5, tet_connectivity=tets)
        assert _neighbors_of(adjacency, 3) == {0, 1, 2}
        assert _neighbors_of(adjacency, 4) == {0, 1, 2}
        assert 4 not in _neighbors_of(adjacency, 3)
        assert 3 not in _neighbors_of(adjacency, 4)
        # shared-face nodes see everything
        assert _neighbors_of(adjacency, 0) == {1, 2, 3, 4}

    def test_isolated_node_has_no_neighbors(self):
        """n_nodes can exceed the highest node id any cell references -
        an isolated node's row must be all -1, not raise or get skipped."""
        tets = np.array([[0, 1, 2, 3]])
        adjacency = build_node_adjacency(6, tet_connectivity=tets)  # node 4, 5 unused
        assert _neighbors_of(adjacency, 4) == set()
        assert _neighbors_of(adjacency, 5) == set()


class TestBuildNodeAdjacencyPrisms:
    def test_prism_edges_exclude_face_diagonals(self):
        """A prism (v0,v1,v2,w0,w1,w2) has exactly 9 real edges (3 bottom +
        3 top + 3 vertical) - v0 must connect to v1,v2,w0 but NOT to w1/w2
        (those would be face diagonals, not real mesh edges)."""
        prisms = np.array([[0, 1, 2, 3, 4, 5]])
        adjacency = build_node_adjacency(6, prism_connectivity=prisms)
        assert _neighbors_of(adjacency, 0) == {1, 2, 3}
        assert _neighbors_of(adjacency, 3) == {4, 5, 0}


class TestBuildNodeAdjacencyMixedAndEdgeCases:
    def test_mixed_tet_and_prism_cells_combine(self):
        tets = np.array([[0, 1, 2, 3]])
        prisms = np.array([[3, 4, 5, 6, 7, 8]])
        adjacency = build_node_adjacency(9, tet_connectivity=tets, prism_connectivity=prisms)
        # node 3 is the tet's own apex AND the prism's own v0 - neighbors
        # from both cells must both show up.
        assert _neighbors_of(adjacency, 3) == {0, 1, 2, 4, 5, 6}

    def test_no_cells_returns_all_negative_one_columns(self):
        adjacency = build_node_adjacency(5)
        assert adjacency.shape == (5, 0)

    def test_max_degree_is_data_driven_not_a_fixed_guess(self):
        """A fan of 5 tets all sharing one central node gives that node
        degree > any single tet's own 3 - the adjacency table's column
        count must grow to fit the actual data, not truncate it."""
        # Central node 0, plus 5 outer triangles each forming a tet with
        # node 0 and two more unique outer nodes.
        tets = []
        next_node = 1
        for _ in range(5):
            tets.append([0, next_node, next_node + 1, next_node + 2])
            next_node += 3
        tets = np.array(tets)
        n_nodes = next_node
        adjacency = build_node_adjacency(n_nodes, tet_connectivity=tets)
        neighbors_of_0 = _neighbors_of(adjacency, 0)
        assert len(neighbors_of_0) == 15  # 5 tets * 3 other nodes each, all distinct
        assert adjacency.shape[1] >= 15  # column width must fit the real max degree
