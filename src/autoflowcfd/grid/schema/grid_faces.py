"""有限体积法通量计算的网格面数据结构。

提供 FaceData 类，存储面连接关系和几何属性，
用于有限体积法计算。
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class FaceData:
    """面数据结构（用于 FVM 通量计算）
    
    存储面的连接关系和几何属性，用于有限体积法计算。
    
    属性:
        connectivity: 面连接关系, int32, shape=(N_faces, 2)
                     每行包含构成边/面的 2 个节点索引
        area: 面面积数组, float64, shape=(N_faces,)
             每个面的面积（m^2）
        normal: 面法向量数组, float64, shape=(N_faces, 3)
                每个面的单位法向量
        center: 面中心坐标, float64, shape=(N_faces, 3)
                每个面的中心坐标
    
    示例:
        >>> faces = FaceData(
        ...     connectivity=np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int32),
        ...     area=np.array([1.0, 1.0, 1.0]),
        ...     normal=np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]]),
        ...     center=np.array([[0.5, 0, 0], [1, 0.5, 0], [0.5, 0.5, 0]])
        ... )
        >>> print(f"Number of faces: {faces.count}")
        >>> print(f"Total area: {faces.area.sum():.6f} m^2")
    """
    connectivity: np.ndarray  # int32, shape=(N_faces, 2) - 内部面的左右单元, 边界面右=-1
    area: np.ndarray  # float64, shape=(N_faces,) - 标量面面积 (m^2)
    normal: np.ndarray  # float64, shape=(N_faces, 3) - 单位法向量
    center: np.ndarray  # float64, shape=(N_faces, 3) - 面中心坐标
    node_connectivity: Optional[np.ndarray] = None  # int32, shape=(N_faces, 3) - 三角面角点节点索引（见 FaceExtractor.extract_faces）；未设置时为 None
    _area_vectors: Optional[np.ndarray] = None  # float64, shape=(N_faces, 3) - 面积向量 (normal * area)，内部使用
    
    def __post_init__(self):
        """验证面数据一致性
        
        抛出异常:
            ValueError: 如果数组形状或数据类型无效
        """
        n_faces = self.connectivity.shape[0]
        
        # 检查连接关系形状
        if len(self.connectivity.shape) != 2:
            raise ValueError(
                f"Connectivity must be 2D array, got shape {self.connectivity.shape}"
            )
        if self.connectivity.shape[1] != 2:
            raise ValueError(
                f"Face connectivity must have 2 columns, "
                f"got {self.connectivity.shape[1]}"
            )
        
        # 检查其他数组的形状
        if self.area.shape != (n_faces,):
            raise ValueError(
                f"Area shape {self.area.shape} doesn't match "
                f"connectivity count {n_faces}"
            )
        if self.normal.shape != (n_faces, 3):
            raise ValueError(
                f"Normal shape {self.normal.shape} doesn't match "
                f"connectivity count {n_faces}"
            )
        if self.center.shape != (n_faces, 3):
            raise ValueError(
                f"Center shape {self.center.shape} doesn't match "
                f"connectivity count {n_faces}"
            )
        if self._area_vectors is not None and self._area_vectors.shape != (n_faces, 3):
            raise ValueError(
                f"Internal area_vectors shape {self._area_vectors.shape} doesn't match "
                f"connectivity count {n_faces}"
            )
        if self.node_connectivity is not None and self.node_connectivity.shape != (n_faces, 3):
            raise ValueError(
                f"node_connectivity shape {self.node_connectivity.shape} doesn't match "
                f"connectivity count {n_faces}"
            )
        
        # 检查数据类型
        if self.connectivity.dtype != np.int32:
            raise ValueError(f"Connectivity must be int32, got {self.connectivity.dtype}")
        if self.area.dtype != np.float64:
            raise ValueError(f"Area must be float64, got {self.area.dtype}")
        if self.normal.dtype != np.float64:
            raise ValueError(f"Normal must be float64, got {self.normal.dtype}")
        if self.center.dtype != np.float64:
            raise ValueError(f"Center must be float64, got {self.center.dtype}")
        
        # 验证面积为正值
        if np.any(self.area <= 0):
            n_negative = np.sum(self.area <= 0)
            logger.warning(
                f"Found {n_negative} non-positive face areas. Filtering them out..."
            )
            # 过滤输出无效面
            valid_mask = self.area > 0
            self.connectivity = self.connectivity[valid_mask]
            self.area = self.area[valid_mask]
            self.normal = self.normal[valid_mask]
            self.center = self.center[valid_mask]
            if self.node_connectivity is not None:
                self.node_connectivity = self.node_connectivity[valid_mask]
        
        # 确保连续内存布局
        if not self.connectivity.flags['C_CONTIGUOUS']:
            self.connectivity = np.ascontiguousarray(self.connectivity)
        if not self.area.flags['C_CONTIGUOUS']:
            self.area = np.ascontiguousarray(self.area)
        if not self.normal.flags['C_CONTIGUOUS']:
            self.normal = np.ascontiguousarray(self.normal)
        if not self.center.flags['C_CONTIGUOUS']:
            self.center = np.ascontiguousarray(self.center)
        if self.node_connectivity is not None and not self.node_connectivity.flags['C_CONTIGUOUS']:
            self.node_connectivity = np.ascontiguousarray(self.node_connectivity)
    
    @property
    def count(self) -> int:
        """获取面数量"""
        return self.connectivity.shape[0]
        
    @property
    def n_interior_faces(self) -> int:
        """获取内部面数量（连接两个单元的面）"""
        return int(np.sum(self.connectivity[:, 1] >= 0))
    
    @property
    def n_boundary_faces(self) -> int:
        """获取边界面上数量（只属于一个单元的面）"""
        return int(np.sum(self.connectivity[:, 1] < 0))
    
    @property
    def area_vectors(self) -> np.ndarray:
        """获取面积向量数组（法向量×面积），用于通量计算
        
        Returns:
            np.ndarray: 面积向量, shape=(N_faces, 3)
        """
        if self._area_vectors is None:
            # 计算面积向量（缓存）
            self._area_vectors = self.normal * self.area.reshape(-1, 1)
        return self._area_vectors
    
    def get_boundary_face_indices(self) -> np.ndarray:
        """获取边界面的索引数组
        
        Returns:
            np.ndarray: 边界面索引, dtype=int32
        """
        return np.where(self.connectivity[:, 1] < 0)[0].astype(np.int32)
    
    def get_interior_face_indices(self) -> np.ndarray:
        """获取内部面的索引数组
        
        Returns:
            np.ndarray: 内部面索引, dtype=int32
        """
        return np.where(self.connectivity[:, 1] >= 0)[0].astype(np.int32)
