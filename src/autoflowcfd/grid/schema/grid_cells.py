"""CFD 网格单元数据结构。

提供 SoA（Structure of Arrays）布局的 CPU 和 GPU 单元连接关系存储，
包括三角形面单元和四面体/棱柱体单元。
"""

import numpy as np
from typing import Tuple, TYPE_CHECKING
from dataclasses import dataclass
from loguru import logger

if TYPE_CHECKING:
    import cupy as cp
    from .grid_nodes import NodeArray


@dataclass
class CellArray:
    """单元数组(SoA布局)
    
    以 SoA 布局存储单元连接关系和类型信息。
    支持三角形面网格（每个单元3个节点）。
    
    属性:
        connectivity: 单元连接关系, int32, shape=(N_cells, 3)
                     每行包含构成三角形的3个节点索引
        cell_type: 单元类型数组, int32, shape=(N_cells,)
                  0=三角形，预留用于未来类型
    
    示例:
        >>> cells = CellArray(
        ...     connectivity=np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32),
        ...     cell_type=np.array([0, 0], dtype=np.int32)
        ... )
        >>> print(cells.count)  # 2
    """
    connectivity: np.ndarray  # int32, shape=(N_cells, 3)
    cell_type: np.ndarray  # int32, shape=(N_cells,)
    
    def __post_init__(self):
        """验证数组形状与数据类型
        
        抛出异常:
            ValueError: 如果数组形状或数据类型无效
        """
        # 检查连接关系形状
        if len(self.connectivity.shape) != 2:
            raise ValueError(
                f"Connectivity must be 2D array, got shape {self.connectivity.shape}"
            )
        if self.connectivity.shape[1] != 3:
            raise ValueError(
                f"Connectivity must have 3 columns (triangular mesh), "
                f"got {self.connectivity.shape[1]}"
            )
        
        # 检查单元类型形状与连接关系匹配
        if len(self.cell_type.shape) != 1:
            raise ValueError(
                f"Cell type must be 1D array, got shape {self.cell_type.shape}"
            )
        if self.cell_type.shape[0] != self.connectivity.shape[0]:
            raise ValueError(
                f"Cell type count ({self.cell_type.shape[0]}) doesn't match "
                f"connectivity count ({self.connectivity.shape[0]})"
            )
        
        # 检查数据类型
        if self.connectivity.dtype != np.int32:
            raise ValueError(
                f"Connectivity must be int32, got {self.connectivity.dtype}"
            )
        if self.cell_type.dtype != np.int32:
            raise ValueError(
                f"Cell type must be int32, got {self.cell_type.dtype}"
            )
        
        # 确保连续内存布局
        if not self.connectivity.flags['C_CONTIGUOUS']:
            self.connectivity = np.ascontiguousarray(self.connectivity)
        if not self.cell_type.flags['C_CONTIGUOUS']:
            self.cell_type = np.ascontiguousarray(self.cell_type)
        
        logger.debug(f"CellArray initialized with {self.count} cells")
    
    @property
    def count(self) -> int:
        """获取单元数量
        
        Returns:
            int: 单元数量
        """
        return len(self.connectivity)
    
    @property
    def shape(self) -> Tuple[int, int]:
        """获取单元数组形状
        
        Returns:
            Tuple[int, int]: 形状元组 (N_cells, 3)
        """
        return self.connectivity.shape
    
    def to_gpu(self) -> 'CupyCellArray':
        """转换为GPU数据结构
        
        Returns:
            CupyCellArray: CuPy版本的单元数组
        """
        try:
            import cupy as cp
            return CupyCellArray(
                connectivity=cp.asarray(self.connectivity),
                cell_type=cp.asarray(self.cell_type)
            )
        except ImportError:
            raise ImportError(
                "CuPy is required for GPU operations. "
                "Install it with: pip install cupy-cuda11x"
            )
    
    @classmethod
    def from_gpu(cls, gpu_cells: 'CupyCellArray') -> 'CellArray':
        """从GPU数据结构转换
        
        Args:
            gpu_cells: CuPy版本的单元数组
            
        Returns:
            CellArray: NumPy版本的单元数组
        """
        try:
            import cupy as cp
            return cls(
                connectivity=cp.asnumpy(gpu_cells.connectivity),
                cell_type=cp.asnumpy(gpu_cells.cell_type)
            )
        except ImportError:
            raise ImportError(
                "CuPy is required for GPU operations. "
                "Install it with: pip install cupy-cuda11x"
            )


@dataclass
class CupyCellArray:
    """GPU单元数组(CuPy版本)
    
    GPU 加速版本的 CellArray 使用 CuPy arrays.
    
    属性:
        connectivity: 单元连接关系 (cupy.ndarray)
        cell_type: 单元类型数组 (cupy.ndarray)
    """
    connectivity: 'cp.ndarray'  # noqa: F821
    cell_type: 'cp.ndarray'  # noqa: F821
    
    @property
    def count(self) -> int:
        """获取单元数量"""
        return len(self.connectivity)
    
    def to_cpu(self) -> CellArray:
        """转换为CPU数据结构
        
        Returns:
            CellArray: NumPy版本的单元数组
        """
        import cupy as cp
        return CellArray(
            connectivity=cp.asnumpy(self.connectivity),
            cell_type=cp.asnumpy(self.cell_type)
        )


@dataclass
class TetrahedralCells:
    """四面体单元数组（专为体网格设计）
    
    存储体网格的四面体单元连接关系。
    每个四面体具有 4 个顶点和正体积。
    
    属性:
        connectivity: 单元连接关系, int32, shape=(N_cells, 4)
                     每行包含构成四面体的 4 个节点索引
        volumes: 单元体积数组, float64, shape=(N_cells,)
                每个四面体的体积（m^3）
    
    示例:
        >>> cells = TetrahedralCells(
        ...     connectivity=np.array([[0, 1, 2, 3], [1, 4, 2, 3]], dtype=np.int32),
        ...     volumes=np.array([1e-6, 2e-6])
        ... )
        >>> print(cells.count)  # 2
        >>> print(f"Total volume: {cells.volumes.sum():.6e} m^3")
    """
    connectivity: np.ndarray  # int32, shape=(N_cells, 4)
    volumes: np.ndarray  # float64, shape=(N_cells,)
    
    def __post_init__(self):
        """验证数组形状与数据类型
        
        抛出异常:
            ValueError: 如果数组形状或数据类型无效
        """
        # 检查连接关系形状
        if len(self.connectivity.shape) != 2:
            raise ValueError(
                f"Connectivity must be 2D array, got shape {self.connectivity.shape}"
            )
        if self.connectivity.shape[1] != 4:
            raise ValueError(
                f"Tetrahedral connectivity must have 4 columns, "
                f"got {self.connectivity.shape[1]}"
            )
        
        # 检查体积形状与连接关系匹配
        if len(self.volumes.shape) != 1:
            raise ValueError(
                f"Volumes must be 1D array, got shape {self.volumes.shape}"
            )
        if self.volumes.shape[0] != self.connectivity.shape[0]:
            raise ValueError(
                f"Volumes count ({self.volumes.shape[0]}) doesn't match "
                f"cell count ({self.connectivity.shape[0]})"
            )
        
        # 检查数据类型
        if self.connectivity.dtype != np.int32:
            raise ValueError(f"Connectivity must be int32, got {self.connectivity.dtype}")
        if self.volumes.dtype != np.float64:
            raise ValueError(f"Volumes must be float64, got {self.volumes.dtype}")
        
        # 验证体积为正值
        if np.any(self.volumes <= 0):
            n_negative = np.sum(self.volumes <= 0)
            raise ValueError(
                f"All tetrahedral volumes must be positive, "
                f"found {n_negative} non-positive volumes"
            )
        
        # 确保连续内存布局
        if not self.connectivity.flags['C_CONTIGUOUS']:
            self.connectivity = np.ascontiguousarray(self.connectivity)
        if not self.volumes.flags['C_CONTIGUOUS']:
            self.volumes = np.ascontiguousarray(self.volumes)
    
    @property
    def count(self) -> int:
        """获取单元数量"""
        return self.connectivity.shape[0]
    
    @staticmethod
    def compute_volumes(nodes: 'NodeArray', connectivity: np.ndarray) -> np.ndarray:
        """计算四面体体积（使用绝对值确保体积为正）
        
        四面体体积公式（顶点 A,B,C,D）：
            V = |det(B-A, C-A, D-A)| / 6
        
        注意: 使用绝对值处理节点顺序不一致的情况，确保返回正值。
        对于生产级网格，建议在网格生成阶段确保一致的定向。
        
        Args:
            nodes: 节点坐标
            connectivity: 四面体连接关系, shape=(N_cells, 4)
            
        Returns:
            volumes: 单元体积 (始终为正), shape=(N_cells,), 单位m^3
        """
        # 向量化实现，高性能
        # 一次性提取所有单元的节点索引
        n0 = connectivity[:, 0]
        n1 = connectivity[:, 1]
        n2 = connectivity[:, 2]
        n3 = connectivity[:, 3]
        
        # 使用高级索引获取节点坐标
        x = nodes.x
        y = nodes.y
        z = nodes.z
        
        # 计算从顶点 0 到其他顶点的向量（向量化）
        v1_x = x[n1] - x[n0]
        v1_y = y[n1] - y[n0]
        v1_z = z[n1] - z[n0]
        
        v2_x = x[n2] - x[n0]
        v2_y = y[n2] - y[n0]
        v2_z = z[n2] - z[n0]
        
        v3_x = x[n3] - x[n0]
        v3_y = y[n3] - y[n0]
        v3_z = z[n3] - z[n0]
        
        # 叉乘: v2 × v3（向量化）
        cross_x = v2_y * v3_z - v2_z * v3_y
        cross_y = v2_z * v3_x - v2_x * v3_z
        cross_z = v2_x * v3_y - v2_y * v3_x
        
        # 点乘: v1 · (v2 × v3)（向量化）
        det = v1_x * cross_x + v1_y * cross_y + v1_z * cross_z
        
        # 体积 = |det| / 6
        volumes = np.abs(det) / 6.0

        return volumes


@dataclass
class PrismCells:
    """边界层三棱柱单元数组（BL 区域专用，不再拆分为四面体）

    存储边界层区域的三棱柱连接关系。棱柱的 6 个节点为
    (v0, v1, v2, w0, w1, w2)：v0..v2 为底层（近壁面侧）三角形，
    w0..w2 为顶层三角形，其中 w_i 是 v_i 的挤出方向对应点
    （w_i 直接在 v_i “上方”——不是任意排列；这与
    mesh_extrusion.py / mesh_prism_to_tet.py 中已有的逐层节点对应关系一致）。
    (v0,v1,v2) 和 (w0,w1,w2) 内的列顺序无需按节点索引预排序——
    面提取（face_extractor.extract_faces_mixed）会自行推导规范排序，
    与 mesh_prism_to_tet.convert_layers_to_tetrahedra 的方式相同，
    因此共享边面的两个棱柱无论存储顺序如何，都能在其对角线上保持一致。

    作为 TetrahedralCells 的兄弟类（而非合并到一个填充数组中），
    所有现有的仅处理四面体的 TetrahedralCells 使用者可以完全不变地
    继续处理核心区域单元；只有显式感知 BL/棱柱的代码才需要知道这个类。

    属性:
        connectivity: 单元连接关系, int32, shape=(N_cells, 6)
        volumes: 单元体积数组（无符号，见 compute_volumes）, float64, shape=(N_cells,)
    """
    connectivity: np.ndarray  # int32, shape=(N_cells, 6)
    volumes: np.ndarray  # float64, shape=(N_cells,)

    def __post_init__(self):
        if len(self.connectivity.shape) != 2:
            raise ValueError(
                f"Connectivity must be 2D array, got shape {self.connectivity.shape}"
            )
        if self.connectivity.shape[1] != 6:
            raise ValueError(
                f"Prism connectivity must have 6 columns, "
                f"got {self.connectivity.shape[1]}"
            )

        if len(self.volumes.shape) != 1:
            raise ValueError(
                f"Volumes must be 1D array, got shape {self.volumes.shape}"
            )
        if self.volumes.shape[0] != self.connectivity.shape[0]:
            raise ValueError(
                f"Volumes count ({self.volumes.shape[0]}) doesn't match "
                f"cell count ({self.connectivity.shape[0]})"
            )

        if self.connectivity.dtype != np.int32:
            raise ValueError(f"Connectivity must be int32, got {self.connectivity.dtype}")
        if self.volumes.dtype != np.float64:
            raise ValueError(f"Volumes must be float64, got {self.volumes.dtype}")

        if np.any(self.volumes <= 0):
            n_negative = np.sum(self.volumes <= 0)
            raise ValueError(
                f"All prism volumes must be positive, "
                f"found {n_negative} non-positive volumes"
            )

        if not self.connectivity.flags['C_CONTIGUOUS']:
            self.connectivity = np.ascontiguousarray(self.connectivity)
        if not self.volumes.flags['C_CONTIGUOUS']:
            self.volumes = np.ascontiguousarray(self.volumes)

    @property
    def count(self) -> int:
        return self.connectivity.shape[0]

    @staticmethod
    def compute_volumes(nodes: 'NodeArray', connectivity: np.ndarray) -> np.ndarray:
        """棱柱体积（无符号）：拆成 3 个子四面体分别取 |signed 体积| 后求和。

        委托给 quality_metrics.compute_prism_volumes 获取实际公式
        （集中在一处——参见该函数的文档字符串了解为什么每个子四面体的贡献
        必须取绝对值，而不是用原始索引顺序的符号求和）。
        """
        from ..validation.quality_metrics import compute_prism_volumes
        pts = np.column_stack([nodes.x, nodes.y, nodes.z])
        return compute_prism_volumes(pts, connectivity)
