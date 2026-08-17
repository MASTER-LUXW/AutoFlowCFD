"""CFD 网格节点数据结构。

提供 SoA（Structure of Arrays）布局的 CPU 和 GPU 节点坐标存储，
优化数值计算过程中的缓存性能。
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class NodeArray:
    """节点坐标数组(SoA布局)
    
    Stores node coordinates in Structure of Arrays layout for optimal
    cache performance during numerical computations.
    
    Attributes:
        x: X坐标数组, float64, shape=(N_nodes,)
        y: Y坐标数组, float64, shape=(N_nodes,)
        z: Z坐标数组, float64, shape=(N_nodes,)
    
    Example:
        >>> nodes = NodeArray(
        ...     x=np.array([0.0, 1.0, 2.0]),
        ...     y=np.array([0.0, 0.0, 0.0]),
        ...     z=np.array([0.0, 0.0, 0.0])
        ... )
        >>> print(nodes.count)  # 3
        >>> print(nodes.x.shape)  # (3,)
    """
    x: np.ndarray  # float64, shape=(N_nodes,)
    y: np.ndarray  # float64, shape=(N_nodes,)
    z: np.ndarray  # float64, shape=(N_nodes,)
    
    def __post_init__(self):
        """验证数组形状一致性与数据类型
        
        Raises:
            ValueError: If arrays have inconsistent shapes or wrong dtype
        """
        # Check shape consistency
        if not (self.x.shape == self.y.shape == self.z.shape):
            raise ValueError(
                f"Node coordinates shape mismatch: "
                f"x={self.x.shape}, y={self.y.shape}, z={self.z.shape}"
            )
        
        # Check dtype (must be float64 for numerical stability)
        if not (self.x.dtype == self.y.dtype == self.z.dtype == np.float64):
            raise ValueError(
                f"Node coordinates must be float64, got: "
                f"x={self.x.dtype}, y={self.y.dtype}, z={self.z.dtype}"
            )
        
        # Ensure contiguous memory layout for performance
        if not self.x.flags['C_CONTIGUOUS']:
            self.x = np.ascontiguousarray(self.x)
            self.y = np.ascontiguousarray(self.y)
            self.z = np.ascontiguousarray(self.z)
        
        logger.debug(f"NodeArray initialized with {self.count} nodes")
    
    @property
    def count(self) -> int:
        """获取节点数量
        
        Returns:
            int: Number of nodes
        """
        return len(self.x)
    
    @property
    def shape(self) -> Tuple[int]:
        """获取节点数组形状
        
        Returns:
            Tuple[int]: Shape tuple (N_nodes,)
        """
        return self.x.shape
    
    def to_gpu(self) -> 'CupyNodeArray':
        """转换为GPU数据结构
        
        Returns:
            CupyNodeArray: CuPy版本的节点数组
            
        Example:
            >>> gpu_nodes = nodes.to_gpu()
            >>> print(type(gpu_nodes.x))  # <class 'cupy.ndarray'>
        """
        try:
            import cupy as cp
            return CupyNodeArray(
                x=cp.asarray(self.x),
                y=cp.asarray(self.y),
                z=cp.asarray(self.z)
            )
        except ImportError:
            raise ImportError(
                "CuPy is required for GPU operations. "
                "Install it with: pip install cupy-cuda11x"
            )
    
    @classmethod
    def from_gpu(cls, gpu_nodes: 'CupyNodeArray') -> 'NodeArray':
        """从GPU数据结构转换
        
        Args:
            gpu_nodes: CuPy版本的节点数组
            
        Returns:
            NodeArray: NumPy版本的节点数组
        """
        try:
            import cupy as cp
            return cls(
                x=cp.asnumpy(gpu_nodes.x),
                y=cp.asnumpy(gpu_nodes.y),
                z=cp.asnumpy(gpu_nodes.z)
            )
        except ImportError:
            raise ImportError(
                "CuPy is required for GPU operations. "
                "Install it with: pip install cupy-cuda11x"
            )
    
    def get_coordinates(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """获取节点坐标数组
        
        Args:
            indices: 节点索引数组,如果为None则返回所有节点
            
        Returns:
            np.ndarray: 坐标数组, shape=(N, 3) where N is number of selected nodes
        """
        if indices is None:
            return np.stack([self.x, self.y, self.z], axis=-1)
        else:
            return np.stack([self.x[indices], self.y[indices], self.z[indices]], axis=-1)


@dataclass
class CupyNodeArray:
    """GPU节点坐标数组(CuPy版本)
    
    GPU-accelerated version of NodeArray using CuPy arrays.
    
    Attributes:
        x: X坐标数组 (cupy.ndarray)
        y: Y坐标数组 (cupy.ndarray)
        z: Z坐标数组 (cupy.ndarray)
    """
    x: 'cp.ndarray'  # Will be typed at runtime
    y: 'cp.ndarray'
    z: 'cp.ndarray'
    
    @property
    def count(self) -> int:
        """获取节点数量"""
        return len(self.x)
    
    def to_cpu(self) -> NodeArray:
        """转换为CPU数据结构
        
        Returns:
            NodeArray: NumPy版本的节点数组
        """
        import cupy as cp
        return NodeArray(
            x=cp.asnumpy(self.x),
            y=cp.asnumpy(self.y),
            z=cp.asnumpy(self.z)
        )
