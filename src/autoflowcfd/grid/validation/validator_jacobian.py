"""GridValidator 的雅可比行列式 / 三角形绕向一致性检查。

从 validator.py 中拆分出来（原文件超过 400 行的项目约定上限）：
GridValidator._check_jacobian 及其私有辅助方法 _count_flipped_triangles
是一个自成一体的算法阶段——检测相邻三角形绕向是否互相矛盾（曲面网格
翻转单元的判据），除了读取 self.grid_data 之外不与类的其它检查共享任何
状态，因此按"取 grid_data 为参数的模块级函数"整体搬移，validator.py 里
只保留转调用的薄包装方法，逻辑/数值结果完全不变。
"""

from typing import Dict

import numpy as np
from loguru import logger

from ..structures import GridData


def check_jacobian(grid_data: GridData) -> Dict[str, float]:
    """检查雅可比行列式

    Computes Jacobian determinant for each cell. The Jacobian measures
    the local scaling factor of the coordinate transformation.

    Returns:
        Dict: Statistics with 'max', 'avg', 'min' keys

    Note:
        Positive Jacobian indicates valid cell orientation.
        Negative or near-zero Jacobian indicates inverted or degenerate cells.
    """
    logger.debug("Computing Jacobian determinants...")

    connectivity = grid_data.cells.connectivity
    nodes = grid_data.nodes

    # Get node coordinates
    cell_coords = np.stack([
        nodes.get_coordinates(connectivity[:, i])
        for i in range(3)
    ], axis=1)

    # Compute vectors from node 0 to nodes 1 and 2
    v0 = cell_coords[:, 1] - cell_coords[:, 0]
    v1 = cell_coords[:, 2] - cell_coords[:, 0]

    # Compute cross product (gives normal vector)
    cross = np.cross(v0, v1)

    # Jacobian determinant is magnitude of cross product
    jacobians = np.linalg.norm(cross, axis=1)

    # A single triangle's winding has no absolute sign without an
    # external reference frame, so "jacobians < 0" here (np.linalg.norm
    # is never negative) could never fire for any mesh - inverted
    # triangles were silently invisible to this check regardless of
    # input. What *is* well-defined without an external reference is
    # whether neighboring triangles agree with each other: on a
    # consistently-oriented manifold surface, two triangles sharing an
    # edge must traverse that shared edge in opposite directions. If
    # they traverse it in the same direction, one of the pair is
    # flipped relative to its neighbor - detect that via directed-edge
    # sign accumulation instead of the magnitude's sign.
    negative_count = _count_flipped_triangles(grid_data, connectivity)

    # Compute statistics
    stats = {
        'max': float(np.max(jacobians)),
        'avg': float(np.mean(jacobians)),
        'min': float(np.min(jacobians)),
        'std': float(np.std(jacobians)),
        'negative_count': negative_count
    }

    logger.debug(
        f"Jacobian - Max: {stats['max']:.6f}, "
        f"Avg: {stats['avg']:.6f}, Min: {stats['min']:.6f}, "
        f"Negative: {stats['negative_count']}"
    )

    return stats


def _count_flipped_triangles(grid_data: GridData, connectivity: np.ndarray) -> int:
    """Count triangles whose winding is inconsistent with a neighbor.

    For every shared (manifold-interior) edge - one that borders
    exactly two triangles - a consistently-oriented surface must
    traverse it in opposite directions from each side. Two triangles
    that instead traverse their shared edge in the same direction
    cannot both be correctly oriented; both are flagged. Edges shared
    by a triangle count other than 2 (open boundary or non-manifold)
    aren't orientation-checkable this way and are skipped here - that
    is a distinct mesh-integrity issue, not this metric's concern.
    """
    n1, n2, n3 = connectivity[:, 0], connectivity[:, 1], connectivity[:, 2]
    directed_edges = np.concatenate([
        np.stack([n1, n2], axis=1),
        np.stack([n2, n3], axis=1),
        np.stack([n3, n1], axis=1),
    ], axis=0)
    owner_cells = np.tile(np.arange(len(connectivity)), 3)

    n_nodes = grid_data.nodes.count
    lo = np.minimum(directed_edges[:, 0], directed_edges[:, 1]).astype(np.int64)
    hi = np.maximum(directed_edges[:, 0], directed_edges[:, 1]).astype(np.int64)
    edge_key = lo * n_nodes + hi
    sign = np.where(directed_edges[:, 0] < directed_edges[:, 1], 1, -1)

    order = np.argsort(edge_key, kind='stable')
    sorted_keys = edge_key[order]
    sorted_signs = sign[order]
    sorted_owners = owner_cells[order]

    group_end = np.flatnonzero(
        np.concatenate([sorted_keys[1:] != sorted_keys[:-1], [True]])
    )
    group_start = np.concatenate([[0], group_end[:-1] + 1])
    group_size = group_end - group_start + 1

    pair_mask = group_size == 2
    pair_first = group_start[pair_mask]
    same_direction = sorted_signs[pair_first] == sorted_signs[pair_first + 1]
    bad_first = pair_first[same_direction]

    flipped_cells = np.unique(np.concatenate([
        sorted_owners[bad_first], sorted_owners[bad_first + 1]
    ])) if bad_first.size else np.array([], dtype=owner_cells.dtype)

    return int(flipped_cells.size)
