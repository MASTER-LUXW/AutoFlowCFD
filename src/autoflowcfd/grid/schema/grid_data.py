"""Grid data container classes.

Provides GridData, CupyGridData, and VolumeMeshData classes for managing
complete grid structures with nodes, cells, boundaries, and metadata.
"""

import numpy as np
from typing import Optional, Dict
from dataclasses import dataclass
from loguru import logger

from .grid_nodes import NodeArray, CupyNodeArray
from .grid_cells import CellArray, CupyCellArray, TetrahedralCells, PrismCells
from .grid_boundaries import BoundaryMap
from .grid_metadata import GridMetadata
from .grid_faces import FaceData


@dataclass
class GridData:
    """网格数据主类
    
    Top-level container for all grid-related data. Aggregates nodes, cells,
    boundaries, and metadata into a single coherent structure.
    
    This is the primary interface for grid operations in AutoFlowCFD.
    
    Attributes:
        nodes: 节点数组
        cells: 单元数组
        boundaries: 边界映射
        metadata: 网格元数据
    """
    nodes: NodeArray
    cells: CellArray
    boundaries: BoundaryMap
    metadata: GridMetadata
    
    def __post_init__(self):
        """验证网格数据一致性"""
        if self.metadata.node_count != self.nodes.count:
            raise ValueError(
                f"Metadata node count ({self.metadata.node_count}) doesn't match "
                f"actual node count ({self.nodes.count})"
            )
        
        if self.metadata.cell_count != self.cells.count:
            raise ValueError(
                f"Metadata cell count ({self.metadata.cell_count}) doesn't match "
                f"actual cell count ({self.cells.count})"
            )
        
        logger.info(
            f"GridData validated: {self.metadata.node_count} nodes, "
            f"{self.metadata.cell_count} cells"
        )
    
    @property
    def node_count(self) -> int:
        """获取节点数量"""
        return self.nodes.count
    
    @property
    def cell_count(self) -> int:
        """获取单元数量"""
        return self.cells.count
    
    def to_gpu(self) -> 'CupyGridData':
        """转换为GPU数据结构"""
        return CupyGridData(
            nodes=self.nodes.to_gpu(),
            cells=self.cells.to_gpu(),
            boundaries=self.boundaries,
            metadata=self.metadata
        )
    
    @classmethod
    def from_gpu(cls, gpu_grid: 'CupyGridData') -> 'GridData':
        """从GPU数据结构转换"""
        return cls(
            nodes=gpu_grid.nodes.to_cpu(),
            cells=gpu_grid.cells.to_cpu(),
            boundaries=gpu_grid.boundaries,
            metadata=gpu_grid.metadata
        )
    
    def save_hdf5(self, filepath: str) -> None:
        """保存网格数据到HDF5文件"""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for HDF5 I/O. Install it with: pip install h5py")
        
        logger.info(f"Saving grid to HDF5: {filepath}")
        
        with h5py.File(filepath, 'w') as f:
            # Save metadata
            meta_group = f.create_group('metadata')
            meta_group.attrs['node_count'] = self.metadata.node_count
            meta_group.attrs['cell_count'] = self.metadata.cell_count
            meta_group.attrs['file_format'] = self.metadata.file_format
            if self.metadata.bounding_box:
                meta_group.attrs['bounding_box'] = self.metadata.bounding_box
            if self.metadata.creation_time:
                meta_group.attrs['creation_time'] = self.metadata.creation_time
            
            # Save nodes
            node_group = f.create_group('nodes')
            node_group.create_dataset('x', data=self.nodes.x)
            node_group.create_dataset('y', data=self.nodes.y)
            node_group.create_dataset('z', data=self.nodes.z)
            
            # Save cells
            cell_group = f.create_group('cells')
            cell_group.create_dataset('connectivity', data=self.cells.connectivity)
            cell_group.create_dataset('cell_type', data=self.cells.cell_type)
            
            # Save boundaries
            self.boundaries.save_hdf5(f)
        
        logger.success(f"Grid saved to {filepath}")
    
    @classmethod
    def load_hdf5(cls, filepath: str) -> 'GridData':
        """从HDF5文件加载网格数据"""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for HDF5 I/O. Install it with: pip install h5py")
        
        logger.info(f"Loading grid from HDF5: {filepath}")
        
        with h5py.File(filepath, 'r') as f:
            # Load metadata
            meta_group = f['metadata']
            metadata = GridMetadata(
                node_count=int(meta_group.attrs['node_count']),
                cell_count=int(meta_group.attrs['cell_count']),
                file_format=str(meta_group.attrs['file_format']),
                boundary_groups=[],
                bounding_box=tuple(meta_group.attrs['bounding_box']) if 'bounding_box' in meta_group.attrs else None,
                creation_time=str(meta_group.attrs['creation_time']) if 'creation_time' in meta_group.attrs else None
            )
            
            # Load nodes
            node_group = f['nodes']
            nodes = NodeArray(
                x=node_group['x'][:],
                y=node_group['y'][:],
                z=node_group['z'][:]
            )
            
            # Load cells
            cell_group = f['cells']
            cells = CellArray(
                connectivity=cell_group['connectivity'][:],
                cell_type=cell_group['cell_type'][:]
            )
            
            # Load boundaries
            boundaries = BoundaryMap.load_hdf5(f)
            metadata.boundary_groups = boundaries.boundary_names
        
        logger.success(f"Grid loaded from {filepath}")
        
        return cls(nodes=nodes, cells=cells, boundaries=boundaries, metadata=metadata)


@dataclass
class CupyGridData:
    """GPU网格数据(CuPy版本)"""
    nodes: CupyNodeArray
    cells: CupyCellArray
    boundaries: BoundaryMap
    metadata: GridMetadata
    
    def to_cpu(self) -> GridData:
        """转换为CPU数据结构"""
        return GridData(
            nodes=self.nodes.to_cpu(),
            cells=self.cells.to_cpu(),
            boundaries=self.boundaries,
            metadata=self.metadata
        )


@dataclass
class VolumeMeshData:
    """体网格数据主类（基于真实三维体积）
    
    Top-level container for volumetric mesh data with true 3D cells.
    Supports tetrahedral, hexahedral, and hybrid meshes.
    
    Attributes:
        nodes: 节点数组
        cells: 四面体单元数组（核心区 tetgen 填充 + 未转换为棱柱的 BL 单元）
        boundaries: 边界映射
        metadata: 网格元数据
        faces: 面数据数组（用于FVM通量计算）
        surface_mesh: 原始表面网格数据，用于参考面积计算
        prism_cells: 边界层三棱柱单元（可选）。存在时，全局单元索引约定为
            [0, prism_cells.count) 是棱柱，[prism_cells.count, cell_count)
            是 cells（四面体）——与既有 n_bl_cells "BL 单元在前、核心单元在后"
            的约定完全一致，只是 n_bl_cells 此时数的是真棱柱个数而不是
            拆分后的四面体个数。None（默认）时行为与之前完全一样，纯四面体。
    """
    nodes: NodeArray
    cells: TetrahedralCells
    boundaries: BoundaryMap
    metadata: GridMetadata
    faces: Optional[FaceData] = None
    surface_mesh: Optional[Dict] = None
    prism_cells: Optional[PrismCells] = None

    def __post_init__(self):
        """验证体网格数据一致性

        Unlike GridData, VolumeMeshData is not a subclass of it (it carries
        TetrahedralCells plus faces/surface_mesh instead), so it never
        inherited GridData's node/cell count check. That left the object
        that actually flows into quality validation and NAS/VTK/HDF5 export
        able to silently carry a metadata.node_count/cell_count that no
        longer matched the real nodes/cells arrays if a repair or generation
        step mutated one without updating the other.
        """
        if self.metadata.node_count != self.nodes.count:
            raise ValueError(
                f"Metadata node count ({self.metadata.node_count}) doesn't match "
                f"actual node count ({self.nodes.count})"
            )

        total_cells = self.cells.count + (self.prism_cells.count if self.prism_cells else 0)
        if self.metadata.cell_count != total_cells:
            raise ValueError(
                f"Metadata cell count ({self.metadata.cell_count}) doesn't match "
                f"actual cell count ({total_cells})"
            )

    @property
    def node_count(self) -> int:
        """获取节点数量"""
        return self.nodes.count

    @property
    def cell_count(self) -> int:
        """获取单元数量（棱柱 + 四面体）"""
        return self.cells.count + (self.prism_cells.count if self.prism_cells else 0)

    @property
    def total_volume(self) -> float:
        """获取总体积（m³）"""
        total = float(np.sum(self.cells.volumes))
        if self.prism_cells is not None:
            total += float(np.sum(self.prism_cells.volumes))
        return total

    def get_cell_volumes(self) -> np.ndarray:
        """获取单元体积数组，按全局单元索引顺序（棱柱在前、四面体在后，
        与 n_bl_cells 约定一致）"""
        if self.prism_cells is None:
            return self.cells.volumes
        return np.concatenate([self.prism_cells.volumes, self.cells.volumes])

    def ensure_faces_exist(self) -> FaceData:
        """确保面数据存在，如不存在则从体网格中提取"""
        if self.faces is None:
            logger.info("Extracting face data from volume mesh...")
            try:
                from ..mesh_gen.face_extractor import FaceExtractor

                boundary_groups = self.boundaries.groups if self.boundaries else None

                # strict=True: this is the genuine solve/export-time gate
                # (called from solver_steady.py, transient_solver_loop.py,
                # nas_export.py, quality_validator.py, postprocessing) - by
                # this point mesh generation's own repair stages have all
                # already run, so a face still shared by >2 cells here is a
                # real, uncorrected topology defect (silent flux loss, see
                # face_geometry_finalize.finalize_face_data), not a transient repair-in-
                # progress state. Intermediate extract_faces/extract_faces_
                # mixed calls DURING generation/repair deliberately stay
                # non-strict (default) instead.
                if self.prism_cells is not None:
                    self.faces = FaceExtractor.extract_faces_mixed(
                        prism_connectivity=self.prism_cells.connectivity,
                        tet_connectivity=self.cells.connectivity,
                        nodes=self.nodes,
                        strict=True,
                    )
                else:
                    self.faces = FaceExtractor.extract_faces(
                        cell_connectivity=self.cells.connectivity,
                        nodes=self.nodes,
                        boundary_groups=boundary_groups,
                        strict=True,
                    )

                logger.info(
                    f"Face extraction completed: {self.faces.count} total faces "
                    f"({self.faces.n_interior_faces} interior, "
                    f"{self.faces.n_boundary_faces} boundary)"
                )
            except Exception as e:
                logger.error(f"Failed to extract face data: {e}")
                raise RuntimeError(f"Face extraction failed: {e}")

        return self.faces
