"""体网格生成用的网格工具函数。

提供网格校验、边界探测等无状态辅助函数，均可独立调用。
"""

import numpy as np
from typing import Dict
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
