"""后处理共用：单元场 -> 节点场的体积加权平均（混合网格：棱柱在前、四面体在后）。

低阶 VTK 导出的 POINT_DATA 与瞬态统计（时均/RMS）的节点场共用这一份实现。
"""

import numpy as np
from loguru import logger


def cell_node_incidence(grid_data):
    """全部"单元-节点"关联：`(node_ids, cell_ids)`，单元编号按求解器约定（棱柱在前、四面体在后）。

    `grid_data.cells` 是四面体（体网格）或三角形（面网格）；体网格的棱柱在 `grid_data.prism_cells`。
    """
    node_parts, cell_parts = [], []
    offset = 0
    prism = getattr(grid_data, "prism_cells", None)
    if prism is not None and len(prism.connectivity):
        conn = np.asarray(prism.connectivity, dtype=np.int64)
        node_parts.append(conn.ravel())
        cell_parts.append(np.repeat(np.arange(len(conn)), conn.shape[1]))
        offset = len(conn)
    conn = np.asarray(grid_data.cells.connectivity, dtype=np.int64)
    node_parts.append(conn.ravel())
    cell_parts.append(offset + np.repeat(np.arange(len(conn)), conn.shape[1]))
    return np.concatenate(node_parts), np.concatenate(cell_parts)


def cell_weights(grid_data) -> np.ndarray:
    """单元权重：体网格用单元体积（与 `get_cell_volumes` 同一单元顺序），面网格用等权。"""
    if hasattr(grid_data, "get_cell_volumes"):
        return np.asarray(grid_data.get_cell_volumes(), dtype=np.float64)
    return np.ones(int(grid_data.cell_count))


def cell_to_node(grid_data, cell_values: np.ndarray) -> np.ndarray:
    """单元场按体积加权平均到节点：`node = sum_c V_c f_c / sum_c V_c`（c 遍历含该节点的单元）。

    Args:
        cell_values: `(n_cells,)` 或 `(n_cells, k)`，单元顺序为棱柱在前、四面体在后
    Returns:
        `(n_nodes,)` 或 `(n_nodes, k)`；不属于任何单元的节点为 NaN（不填伪造值）
    """
    values = np.asarray(cell_values, dtype=np.float64)
    node_ids, cell_ids = cell_node_incidence(grid_data)
    n_cells = int(cell_ids.max()) + 1 if cell_ids.size else 0
    if values.shape[0] != n_cells:
        raise ValueError(f"单元场长度 {values.shape[0]} 与网格单元数 {n_cells} 不符")
    w = cell_weights(grid_data)[cell_ids]
    n_nodes = int(grid_data.node_count)
    weight_sum = np.bincount(node_ids, weights=w, minlength=n_nodes)
    orphan = weight_sum <= 0.0
    if orphan.any():
        logger.warning(f"{int(orphan.sum())} 个节点不属于任何单元，节点场在这些节点上写 NaN")
    denom = np.where(orphan, np.nan, weight_sum)
    if values.ndim == 1:
        return np.bincount(node_ids, weights=values[cell_ids] * w, minlength=n_nodes) / denom
    return np.column_stack([
        np.bincount(node_ids, weights=values[cell_ids, j] * w, minlength=n_nodes) / denom
        for j in range(values.shape[1])
    ])
