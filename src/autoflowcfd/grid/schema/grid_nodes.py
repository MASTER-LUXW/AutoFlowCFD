"""CFD 网格节点数据结构。

提供 SoA（Structure of Arrays）布局的 CPU 和 GPU 节点坐标存储，
优化数值计算过程中的缓存性能。
"""

import numpy as np
from typing import Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from loguru import logger

if TYPE_CHECKING:
    import cupy as cp


@dataclass
class NodeArray:
    """节点坐标数组(SoA布局)
    
    以 SoA（Structure of Arrays）布局存储节点坐标，优化数值计算过程中的缓存性能。
    
    属性:
        x: X坐标数组, float64, shape=(N_nodes,)
        y: Y坐标数组, float64, shape=(N_nodes,)
        z: Z坐标数组, float64, shape=(N_nodes,)
    
    示例:
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
        
        抛出异常:
            ValueError: 如果数组形状不一致或数据类型错误
        """
        # 检查形状一致性
        if not (self.x.shape == self.y.shape == self.z.shape):
            raise ValueError(
                f"Node coordinates shape mismatch: "
                f"x={self.x.shape}, y={self.y.shape}, z={self.z.shape}"
            )
        
        # 检查 dtype（必须为 float64 以保证数值稳定性）
        if not (self.x.dtype == self.y.dtype == self.z.dtype == np.float64):
            raise ValueError(
                f"Node coordinates must be float64, got: "
                f"x={self.x.dtype}, y={self.y.dtype}, z={self.z.dtype}"
            )
        
        # 确保连续内存布局以提升性能
        if not self.x.flags['C_CONTIGUOUS']:
            self.x = np.ascontiguousarray(self.x)
            self.y = np.ascontiguousarray(self.y)
            self.z = np.ascontiguousarray(self.z)
        
        logger.debug(f"NodeArray initialized with {self.count} nodes")
    
    @property
    def count(self) -> int:
        """获取节点数量
        
        Returns:
            int: 节点数量
        """
        return len(self.x)
    
    @property
    def shape(self) -> Tuple[int]:
        """获取节点数组形状
        
        Returns:
            Tuple[int]: 形状元组 (N_nodes,)
        """
        return self.x.shape
    
    def to_gpu(self) -> 'CupyNodeArray':
        """转换为GPU数据结构
        
        Returns:
            CupyNodeArray: CuPy版本的节点数组
            
        示例:
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
    
    @classmethod
    def from_array(cls, nodes: np.ndarray) -> 'NodeArray':
        """从 (N, 3) 形状的 ndarray 快速构造 NodeArray。

        等价于 ``cls(x=nodes[:, 0].copy(), y=nodes[:, 1].copy(), z=nodes[:, 2].copy())``，
        在代码库中多处重复使用此模式，统一为此工厂方法。内部使用
        ``np.ascontiguousarray`` 确保列数据在内存中连续，同时覆盖原来
        用 ``np.ascontiguousarray(nodes[:, col])`` 的调用点。

        Args:
            nodes: 节点坐标数组, shape=(N, 3)

        Returns:
            NodeArray: 新实例
        """
        return cls(
            x=np.ascontiguousarray(nodes[:, 0]),
            y=np.ascontiguousarray(nodes[:, 1]),
            z=np.ascontiguousarray(nodes[:, 2]),
        )

    def get_coordinates(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """获取节点坐标数组
        
        Args:
            indices: 节点索引数组,如果为None则返回所有节点
            
        Returns:
            np.ndarray: 坐标数组, shape=(N, 3)，其中 N 为选中节点数量
        """
        if indices is None:
            return np.stack([self.x, self.y, self.z], axis=-1)
        else:
            return np.stack([self.x[indices], self.y[indices], self.z[indices]], axis=-1)


@dataclass
class CupyNodeArray:
    """GPU节点坐标数组(CuPy版本)
    
    NodeArray 的 GPU 加速版本，使用 CuPy 数组。
    
    属性:
        x: X坐标数组 (cupy.ndarray)
        y: Y坐标数组 (cupy.ndarray)
        z: Z坐标数组 (cupy.ndarray)
    """
    x: 'cp.ndarray'  # noqa: F821  (CuPy 类型，运行时由 cupy.ndarray 实例填充)
    y: 'cp.ndarray'  # noqa: F821
    z: 'cp.ndarray'  # noqa: F821
    
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
