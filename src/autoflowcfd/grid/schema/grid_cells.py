"""Grid cell data structures for CFD mesh.

Provides CPU and GPU cell connectivity storage in Structure of Arrays (SoA) layout,
including triangular surface cells and tetrahedral volume cells.
"""

import numpy as np
from typing import Tuple, TYPE_CHECKING
from dataclasses import dataclass
from loguru import logger

if TYPE_CHECKING:
    from .grid_nodes import NodeArray


@dataclass
class CellArray:
    """单元数组(SoA布局)
    
    Stores cell connectivity and type information in SoA layout.
    Supports triangular surface meshes (3 nodes per cell).
    
    Attributes:
        connectivity: 单元连接关系, int32, shape=(N_cells, 3)
                     Each row contains 3 node indices forming a triangle
        cell_type: 单元类型数组, int32, shape=(N_cells,)
                  0=triangle, reserved for future types
    
    Example:
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
        
        Raises:
            ValueError: If arrays have invalid shapes or dtypes
        """
        # Check connectivity shape
        if len(self.connectivity.shape) != 2:
            raise ValueError(
                f"Connectivity must be 2D array, got shape {self.connectivity.shape}"
            )
        if self.connectivity.shape[1] != 3:
            raise ValueError(
                f"Connectivity must have 3 columns (triangular mesh), "
                f"got {self.connectivity.shape[1]}"
            )
        
        # Check cell_type shape matches connectivity
        if len(self.cell_type.shape) != 1:
            raise ValueError(
                f"Cell type must be 1D array, got shape {self.cell_type.shape}"
            )
        if self.cell_type.shape[0] != self.connectivity.shape[0]:
            raise ValueError(
                f"Cell type count ({self.cell_type.shape[0]}) doesn't match "
                f"connectivity count ({self.connectivity.shape[0]})"
            )
        
        # Check dtypes
        if self.connectivity.dtype != np.int32:
            raise ValueError(
                f"Connectivity must be int32, got {self.connectivity.dtype}"
            )
        if self.cell_type.dtype != np.int32:
            raise ValueError(
                f"Cell type must be int32, got {self.cell_type.dtype}"
            )
        
        # Ensure contiguous layout
        if not self.connectivity.flags['C_CONTIGUOUS']:
            self.connectivity = np.ascontiguousarray(self.connectivity)
        if not self.cell_type.flags['C_CONTIGUOUS']:
            self.cell_type = np.ascontiguousarray(self.cell_type)
        
        logger.debug(f"CellArray initialized with {self.count} cells")
    
    @property
    def count(self) -> int:
        """获取单元数量
        
        Returns:
            int: Number of cells
        """
        return len(self.connectivity)
    
    @property
    def shape(self) -> Tuple[int, int]:
        """获取单元数组形状
        
        Returns:
            Tuple[int, int]: Shape tuple (N_cells, 3)
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
    
    GPU-accelerated version of CellArray using CuPy arrays.
    
    Attributes:
        connectivity: 单元连接关系 (cupy.ndarray)
        cell_type: 单元类型数组 (cupy.ndarray)
    """
    connectivity: 'cp.ndarray'
    cell_type: 'cp.ndarray'
    
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
    
    Stores tetrahedral cell connectivity for volume meshes.
    Each tetrahedron has 4 vertices and a positive volume.
    
    Attributes:
        connectivity: 单元连接关系, int32, shape=(N_cells, 4)
                     Each row contains 4 node indices forming a tetrahedron
        volumes: 单元体积数组, float64, shape=(N_cells,)
                Pre-computed volumes for each tetrahedron (in m^3)
    
    Example:
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
        
        Raises:
            ValueError: If arrays have invalid shapes or dtypes
        """
        # Check connectivity shape
        if len(self.connectivity.shape) != 2:
            raise ValueError(
                f"Connectivity must be 2D array, got shape {self.connectivity.shape}"
            )
        if self.connectivity.shape[1] != 4:
            raise ValueError(
                f"Tetrahedral connectivity must have 4 columns, "
                f"got {self.connectivity.shape[1]}"
            )
        
        # Check volumes shape matches connectivity
        if len(self.volumes.shape) != 1:
            raise ValueError(
                f"Volumes must be 1D array, got shape {self.volumes.shape}"
            )
        if self.volumes.shape[0] != self.connectivity.shape[0]:
            raise ValueError(
                f"Volumes count ({self.volumes.shape[0]}) doesn't match "
                f"cell count ({self.connectivity.shape[0]})"
            )
        
        # Check dtypes
        if self.connectivity.dtype != np.int32:
            raise ValueError(f"Connectivity must be int32, got {self.connectivity.dtype}")
        if self.volumes.dtype != np.float64:
            raise ValueError(f"Volumes must be float64, got {self.volumes.dtype}")
        
        # Validate volumes are positive
        if np.any(self.volumes <= 0):
            n_negative = np.sum(self.volumes <= 0)
            raise ValueError(
                f"All tetrahedral volumes must be positive, "
                f"found {n_negative} non-positive volumes"
            )
        
        # Ensure contiguous memory layout
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
        
        Volume of tetrahedron with vertices A,B,C,D:
            V = |det(B-A, C-A, D-A)| / 6
        
        Note: 使用绝对值处理节点顺序不一致的情况，确保返回正值。
        对于生产级网格，建议在网格生成阶段确保一致的定向。
        
        Args:
            nodes: 节点坐标
            connectivity: 四面体连接关系, shape=(N_cells, 4)
            
        Returns:
            volumes: 单元体积 (始终为正), shape=(N_cells,), 单位m^3
        """
        # Vectorized implementation for high performance
        # Extract node indices for all cells at once
        n0 = connectivity[:, 0]
        n1 = connectivity[:, 1]
        n2 = connectivity[:, 2]
        n3 = connectivity[:, 3]
        
        # Get node coordinates using advanced indexing
        x = nodes.x
        y = nodes.y
        z = nodes.z
        
        # Compute vectors from vertex 0 to other vertices (vectorized)
        v1_x = x[n1] - x[n0]
        v1_y = y[n1] - y[n0]
        v1_z = z[n1] - z[n0]
        
        v2_x = x[n2] - x[n0]
        v2_y = y[n2] - y[n0]
        v2_z = z[n2] - z[n0]
        
        v3_x = x[n3] - x[n0]
        v3_y = y[n3] - y[n0]
        v3_z = z[n3] - z[n0]
        
        # Cross product: v2 × v3 (vectorized)
        cross_x = v2_y * v3_z - v2_z * v3_y
        cross_y = v2_z * v3_x - v2_x * v3_z
        cross_z = v2_x * v3_y - v2_y * v3_x
        
        # Dot product: v1 · (v2 × v3) (vectorized)
        det = v1_x * cross_x + v1_y * cross_y + v1_z * cross_z
        
        # Volume = |det| / 6
        volumes = np.abs(det) / 6.0

        return volumes


@dataclass
class PrismCells:
    """边界层三棱柱单元数组（BL 区域专用，不再拆分为四面体）

    Stores triangular-prism connectivity for the BL region. A prism's 6
    nodes are (v0, v1, v2, w0, w1, w2): v0..v2 the bottom-layer (near-wall
    side) triangle, w0..w2 the top-layer triangle, with w_i the extrusion
    of v_i (w_i directly "above" v_i - NOT an arbitrary permutation; this
    is the same per-layer node correspondence mesh_extrusion.py/
    mesh_prism_to_tet.py already rely on). Column order within (v0,v1,v2)
    and (w0,w1,w2) need not be pre-sorted by node index - face extraction
    (face_extractor.extract_faces_mixed) re-derives the canonical sort
    itself, the same way mesh_prism_to_tet.convert_layers_to_tetrahedra
    already does, so two prisms sharing a side face agree on its diagonal
    regardless of storage order.

    Kept as a sibling to TetrahedralCells (not merged into one padded
    array) so every existing tet-only consumer of TetrahedralCells keeps
    working completely unchanged on the core-region cells; only code that
    is explicitly BL/prism-aware needs to know this class exists at all.

    Attributes:
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
        """棱柱体积（无符号）：拆成 3 个子四面体分别取 |signed volume| 后求和。

        Delegates to quality_metrics.compute_prism_volumes for the actual
        formula (kept in one place - see that function's docstring for why
        each sub-tet's contribution must be taken as an absolute value,
        not summed with its raw index-order sign).
        """
        from ..validation.quality_metrics import compute_prism_volumes
        pts = np.column_stack([nodes.x, nodes.y, nodes.z])
        return compute_prism_volumes(pts, connectivity)
