"""体网格生成用的网格工具函数。

提供网格校验、边界探测等无状态辅助函数，均可独立调用。
"""

import numpy as np
from typing import Dict
from loguru import logger


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
    """计算表面面的单位法向量。
    
    使用边向量的叉乘（右手定则）。
    向量化实现以提升性能。
    
    Args:
        nodes: 节点坐标, shape=(n_nodes, 3)
        faces: 面连接关系, shape=(n_faces, 3)
        
    Returns:
        单位法向量, shape=(n_faces, 3)
    """
    logger.info("Computing face normals (vectorized)...")
    n_faces = len(faces)
    
    # 向量化计算——比循环快得多
    v0 = nodes[faces[:, 0]]  # shape=(n_faces, 3)
    v1 = nodes[faces[:, 1]]
    v2 = nodes[faces[:, 2]]
    
    # 边向量
    e1 = v1 - v0  # shape=(n_faces, 3)
    e2 = v2 - v0
    
    # 所有面的叉乘
    normals = np.cross(e1, e2)  # shape=(n_faces, 3)
    
    # 计算模长
    norms = np.linalg.norm(normals, axis=1, keepdims=True)  # shape=(n_faces, 1)
    
    # 避免除以零
    norms = np.maximum(norms, 1e-10)
    
    # 归一化所有法向量
    normals = normals / norms
    
    # 检查退化面
    degenerate_count = np.sum(norms.flatten() < 1e-9)
    if degenerate_count > 0:
        logger.warning(f"Found {degenerate_count} degenerate faces with near-zero area")
        # 为退化面设置默认法向量
        normals[norms.flatten() < 1e-9] = [0, 0, 1]
    
    logger.info(f"Computed {n_faces} face normals")
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
