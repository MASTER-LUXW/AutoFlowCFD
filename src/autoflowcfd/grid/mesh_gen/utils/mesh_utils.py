"""体网格生成用的网格工具函数。

提供网格校验、边界探测等无状态辅助函数，均可独立调用。
"""

import numpy as np
from typing import Dict, Optional
from loguru import logger

#: 退化三角形面的相对判据：叉积模长（= 2 x 面积）不超过最长边平方的这个倍数（`compute_face_normals`）
DEGENERATE_FACE_RELATIVE_AREA = 1e-12


def validate_surface_mesh(
    nodes: np.ndarray,
    faces: np.ndarray
) -> None:
    """验证表面网格输入。

    Args:
        nodes: 节点坐标, shape=(n_nodes, 3)
        faces: 面连接关系, shape=(n_faces, 3)

    Raises:
        ValueError: 网格无效时抛出
    """
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError(f"Nodes must be (n, 3), got {nodes.shape}")

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Faces must be (n, 3), got {faces.shape}")

    if faces.max() >= len(nodes):
        raise ValueError(
            f"Face indices out of range: max={faces.max()}, "
            f"n_nodes={len(nodes)}"
        )

    logger.info(
        f"Surface mesh validated: {len(nodes)} nodes, {len(faces)} faces"
    )


def validate_bounding_box(
    bounding_box: Dict[str, np.ndarray]
) -> None:
    """验证包围盒定义。

    Args:
        bounding_box: {min: [x,y,z], max: [x,y,z]}

    Raises:
        ValueError: 包围盒无效时抛出
    """
    if 'min' not in bounding_box or 'max' not in bounding_box:
        raise ValueError("Bounding box must have 'min' and 'max' keys")

    if len(bounding_box['min']) != 3 or len(bounding_box['max']) != 3:
        raise ValueError("Bounding box coordinates must be 3D")

    if np.any(bounding_box['max'] <= bounding_box['min']):
        raise ValueError("Bounding box max must be > min in all dimensions")


def compute_face_normals(
    nodes: np.ndarray,
    faces: np.ndarray
) -> np.ndarray:
    """三角形面的单位法向（右手定则），`(n_faces, 3)`；网格生成各处的唯一实现。

    退化面（叉积模长不超过最长边平方的 `DEGENERATE_FACE_RELATIVE_AREA` 倍，即相对自身尺度面积为零）给零向量：
    零向量在逐节点平均法向里不贡献方向，任何固定方向都会把平均带偏。2026-10-21（I-206）以前这里给 `[0, 0, 1]`、
    按绝对阈值（叉积模长 1e-9 m^2，边长约 0.03 mm 的正常面就会被判为退化）判定，另有两份私有实现各自处理退化面。

    Args:
        nodes: 节点坐标, shape=(n_nodes, 3)
        faces: 面连接关系, shape=(n_faces, 3)
    """
    tri = nodes[faces]
    edges = (tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 1], tri[:, 0] - tri[:, 2])
    cross = np.cross(edges[0], -edges[2])
    norm = np.linalg.norm(cross, axis=1)
    longest_sq = np.max([np.einsum("ij,ij->i", e, e) for e in edges], axis=0)
    degenerate = norm <= DEGENERATE_FACE_RELATIVE_AREA * longest_sq
    normals = np.zeros_like(cross)
    ok = ~degenerate
    normals[ok] = cross[ok] / norm[ok, None]
    n_degenerate = int(np.count_nonzero(degenerate))
    if n_degenerate:
        logger.warning(f"{n_degenerate} 个退化三角形面（面积相对自身尺度为零），法向记为零向量")
    return normals


def face_edge_incidence(faces: np.ndarray):
    """三角形面网格的唯一无向边与边-面关联（网格生成各处的唯一实现）。

    "槽位"是每个面的每条边：槽位 `j * n_faces + f` 是面 `f` 的第 `j` 条边（`j` = 0, 1, 2 依次为顶点对
    (0,1)、(1,2)、(2,0)）。

    Returns:
        edge_vertices: (n_edges, 2) 每条边的两个顶点（升序），边按 (v0, v1) 字典序编号
        edge_of_slot: (3 n_faces,) 每个槽位所属的边编号
        face_of_slot: (3 n_faces,) 每个槽位所属的面
        counts: (n_edges,) 每条边的相邻面数（1 = 开放边，2 = 流形内部边，> 2 = 非流形）
    """
    faces = np.asarray(faces)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0), axis=1)
    face_of_slot = np.tile(np.arange(len(faces), dtype=np.int64), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    sorted_edges = edges[order]
    is_new = np.ones(len(sorted_edges), dtype=bool)
    is_new[1:] = np.any(sorted_edges[1:] != sorted_edges[:-1], axis=1)
    edge_id_sorted = np.cumsum(is_new) - 1
    edge_of_slot = np.empty(len(order), dtype=np.int64)
    edge_of_slot[order] = edge_id_sorted
    counts = np.bincount(edge_id_sorted, minlength=int(edge_id_sorted[-1]) + 1 if len(order) else 0)
    return sorted_edges[is_new], edge_of_slot, face_of_slot, counts


def two_sided_edges(edge_of_slot: np.ndarray, face_of_slot: np.ndarray, counts: np.ndarray):
    """恰好两个相邻面的边：返回 `(edge_ids, face_a, face_b)`，`face_a` / `face_b` 按槽位顺序（`face_edge_incidence`）。"""
    order = np.argsort(edge_of_slot, kind="stable")
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    edge_ids = np.flatnonzero(counts == 2)
    first = order[starts[edge_ids]]
    second = order[starts[edge_ids] + 1]
    return edge_ids, face_of_slot[first], face_of_slot[second]


def face_components(n_faces: int, edge_of_slot: np.ndarray, face_of_slot: np.ndarray,
                    edge_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """按共享边相邻给每个面标上连通分量编号 `(n_faces,)`（编号按分量内最小面号的顺序）。`edge_mask`
    `(n_edges,)` 给出时只经过为 True 的边相邻。"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    order = np.argsort(edge_of_slot, kind="stable")
    e_sorted = edge_of_slot[order]
    f_sorted = face_of_slot[order]
    start = np.ones(len(order), dtype=bool)
    start[1:] = e_sorted[1:] != e_sorted[:-1]
    anchor = f_sorted[np.flatnonzero(start)[np.cumsum(start) - 1]]     # 每个槽位所属边的第一个面
    link = ~start
    if edge_mask is not None:
        link &= np.asarray(edge_mask, dtype=bool)[e_sorted]
    graph = coo_matrix((np.ones(int(np.count_nonzero(link)), dtype=bool), (anchor[link], f_sorted[link])),
                       shape=(n_faces, n_faces))
    return connected_components(graph, directed=False)[1]


def check_reached_boundary(
    nodes: np.ndarray,
    bounding_box: Dict[str, np.ndarray]
) -> bool:
    """检查挤出层是否已超出计算域边界。

    Args:
        nodes: 当前层节点, shape=(n_nodes, 3)
        bounding_box: 计算域边界 {min: [x,y,z], max: [x,y,z]}

    Returns:
        若有任何节点超出包围盒则返回 True
    """
    bbox_min = bounding_box['min']
    bbox_max = bounding_box['max']

    # 检查是否有节点在包围盒外部
    if np.any(nodes < bbox_min - 1e-6) or np.any(nodes > bbox_max + 1e-6):
        return True

    return False
