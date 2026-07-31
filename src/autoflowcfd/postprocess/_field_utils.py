"""Shared field-interpolation helpers for post-processing modules.

Used by both vtk_export.py (writing POINT_DATA from cell-centered FVM
solution data) and transient_stats.py (accumulating node-resolution
time statistics from the same cell-centered data) - both need the exact
same cell-to-node conversion, so it lives here once rather than twice.
"""

import numpy as np


def cell_to_node(
    connectivity: np.ndarray,
    cell_values: np.ndarray,
    n_points: int,
    volumes: np.ndarray = None,
    fallback: float = 0.0,
) -> np.ndarray:
    """Interpolate a per-cell scalar field to per-node values.

    Volume-weighted average over each node's connected cells (a vectorized
    bincount scatter), not a plain unweighted mean - weighting by cell
    volume keeps a node's value from being pulled toward whichever
    neighboring cells happen to be largest/smallest, which matters near a
    boundary layer where cell size can vary by orders of magnitude between
    adjacent tets.

    Args:
        connectivity: (n_cells, nodes_per_cell) int cell-to-node array
        cell_values: (n_cells,) per-cell field values
        n_points: number of mesh nodes
        volumes: (n_cells,) cell volumes for weighting; None = unweighted
            (equal-weight) average
        fallback: value assigned to any node with no connected cells
            (shouldn't happen for a proper volume mesh, but avoids a
            divide-by-zero if it does)

    Returns:
        (n_points,) per-node interpolated values
    """
    conn = np.asarray(connectivity)
    n_cells, nodes_per_cell = conn.shape
    if volumes is None:
        weights = np.ones(n_cells, dtype=np.float64)
    else:
        weights = np.maximum(np.asarray(volumes, dtype=np.float64), 1e-30)

    node_ids = conn.ravel()
    cell_ids = np.repeat(np.arange(n_cells), nodes_per_cell)
    w = weights[cell_ids]

    weighted_sum = np.bincount(node_ids, weights=cell_values[cell_ids] * w, minlength=n_points)
    weight_sum = np.bincount(node_ids, weights=w, minlength=n_points)

    node_values = np.full(n_points, fallback, dtype=np.float64)
    has_data = weight_sum > 0
    node_values[has_data] = weighted_sum[has_data] / weight_sum[has_data]
    return node_values
