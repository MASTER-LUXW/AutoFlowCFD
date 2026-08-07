"""Grid face data structures for FVM flux computation.

Provides FaceData class for storing face connectivity and geometric properties
used in finite volume method calculations.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class FaceData:
    """面数据结构（用于FVM通量计算）
    
    Stores face connectivity and geometric properties for finite volume methods.
    
    Attributes:
        connectivity: 面连接关系, int32, shape=(N_faces, 2)
                     Each row contains two node indices forming an edge/face
        area: 面面积数组, float64, shape=(N_faces,)
             Pre-computed areas for each face (in m^2)
        normal: 面法向量数组, float64, shape=(N_faces, 3)
                Unit normal vectors for each face
        center: 面中心坐标, float64, shape=(N_faces, 3)
                Center coordinates of each face
    
    Example:
        >>> faces = FaceData(
        ...     connectivity=np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int32),
        ...     area=np.array([1.0, 1.0, 1.0]),
        ...     normal=np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]]),
        ...     center=np.array([[0.5, 0, 0], [1, 0.5, 0], [0.5, 0.5, 0]])
        ... )
        >>> print(f"Number of faces: {faces.count}")
        >>> print(f"Total area: {faces.area.sum():.6f} m^2")
    """
    connectivity: np.ndarray  # int32, shape=(N_faces, 2) - left/right cells for interior faces, right=-1 for boundary
    area: np.ndarray  # float64, shape=(N_faces,) - scalar face areas (m^2)
    normal: np.ndarray  # float64, shape=(N_faces, 3) - unit normal vectors
    center: np.ndarray  # float64, shape=(N_faces, 3) - face center coordinates
    node_connectivity: Optional[np.ndarray] = None  # int32, shape=(N_faces, 3) - triangle corner node indices (see FaceExtractor.extract_faces); None for callers that never populate it
    _area_vectors: Optional[np.ndarray] = None  # float64, shape=(N_faces, 3) - area vectors (normal * area), internal use
    
    def __post_init__(self):
        """验证面数据一致性
        
        Raises:
            ValueError: If arrays have invalid shapes or dtypes
        """
        n_faces = self.connectivity.shape[0]
        
        # Check connectivity shape
        if len(self.connectivity.shape) != 2:
            raise ValueError(
                f"Connectivity must be 2D array, got shape {self.connectivity.shape}"
            )
        if self.connectivity.shape[1] != 2:
            raise ValueError(
                f"Face connectivity must have 2 columns, "
                f"got {self.connectivity.shape[1]}"
            )
        
        # Check other arrays' shapes
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
        
        # Check dtypes
        if self.connectivity.dtype != np.int32:
            raise ValueError(f"Connectivity must be int32, got {self.connectivity.dtype}")
        if self.area.dtype != np.float64:
            raise ValueError(f"Area must be float64, got {self.area.dtype}")
        if self.normal.dtype != np.float64:
            raise ValueError(f"Normal must be float64, got {self.normal.dtype}")
        if self.center.dtype != np.float64:
            raise ValueError(f"Center must be float64, got {self.center.dtype}")
        
        # Validate areas are positive
        if np.any(self.area <= 0):
            n_negative = np.sum(self.area <= 0)
            logger.warning(
                f"Found {n_negative} non-positive face areas. Filtering them out..."
            )
            # Filter out invalid faces
            valid_mask = self.area > 0
            self.connectivity = self.connectivity[valid_mask]
            self.area = self.area[valid_mask]
            self.normal = self.normal[valid_mask]
            self.center = self.center[valid_mask]
            if self.node_connectivity is not None:
                self.node_connectivity = self.node_connectivity[valid_mask]
        
        # Ensure contiguous memory layout
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
