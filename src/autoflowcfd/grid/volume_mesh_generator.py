"""Volume mesh generator for CFD simulations.

Generates 3D tetrahedral volume meshes from surface triangulations.
Supports extrusion-based and hybrid (BL + background) mesh generation strategies.

This module serves as a coordinator that delegates to specialized submodules:
- mesh_extrusion: Boundary layer extrusion
- mesh_background: Cartesian background grid and hybrid assembly
- mesh_boundary: Boundary identification and mapping
- mesh_utils: Validation and utility functions
"""

import numpy as np
from typing import Dict, Optional
from loguru import logger

from .structures import GridData
from .mesh_utils import validate_surface_mesh, validate_bounding_box
from .mesh_extrusion import extrude_layers, convert_layers_to_tetrahedra
from .mesh_boundary import identify_boundaries_from_surface


class VolumeMeshGenerator:
    """Generate 3D volume meshes from surface geometry.
    
    This class converts surface triangulations (from NAS files) into
    volumetric meshes suitable for FVM solvers. Supports two strategies:
    
    1. Pure Extrusion: Only boundary layer layers (fast, simple)
    2. Hybrid Mesh: BL extrusion + Cartesian background (recommended)
    
    Attributes:
        growth_rate: Mesh growth rate for boundary layers
        max_layers: Maximum number of extrusion layers
        min_cell_size: Minimum cell size constraint
        target_cells: Target number of volume cells
    """
    
    def __init__(
        self,
        growth_rate: float = 1.2,
        max_layers: int = 30,
        min_cell_size: float = 0.001,
        target_cells: int = 500000
    ):
        """Initialize volume mesh generator.
        
        Args:
            growth_rate: Geometric growth rate for layer thickness (1.2 typical)
            max_layers: Maximum number of boundary layers (30 for automotive CFD)
            min_cell_size: Minimum allowable cell size in meters (1mm default)
            target_cells: Target total cell count
        """
        self.growth_rate = growth_rate
        self.max_layers = max_layers
        self.min_cell_size = min_cell_size
        self.target_cells = target_cells
        
        logger.info(
            f"VolumeMeshGenerator initialized: growth_rate={growth_rate}, "
            f"max_layers={max_layers}, min_cell_size={min_cell_size}m, "
            f"target_cells={target_cells}"
        )
    
    def generate_from_surface(
        self,
        surface_nodes: np.ndarray,
        surface_faces: np.ndarray,
        bounding_box: Dict[str, np.ndarray],
        method: str = "extrusion",
        surface_boundaries: Optional['BoundaryMap'] = None,
        use_hybrid_mesh: bool = False
    ) -> 'VolumeMeshData':
        """Generate volume mesh from surface geometry.
        
        Args:
            surface_nodes: Surface node coordinates, shape=(n_nodes, 3)
            surface_faces: Surface triangle connectivity, shape=(n_faces, 3)
            bounding_box: Computational domain bounds {min: [x,y,z], max: [x,y,z]}
            method: Generation method ('extrusion' or 'background')
            surface_boundaries: Optional boundary mapping from surface mesh
            use_hybrid_mesh: If True and method='extrusion', add background mesh
            
        Returns:
            VolumeMeshData: Complete volume mesh with nodes, cells, and boundaries
            
        Raises:
            ValueError: If input geometry is invalid
            RuntimeError: If mesh generation fails
        """
        # Validate inputs
        validate_surface_mesh(surface_nodes, surface_faces)
        validate_bounding_box(bounding_box)
        
        logger.info(
            f"Generating volume mesh from {len(surface_nodes)} nodes, "
            f"{len(surface_faces)} faces using {method} method..."
        )
        
        if method == "extrusion":
            if use_hybrid_mesh:
                # Use hybrid mesh (BL + background)
                from .mesh_background import generate_hybrid_mesh
                return generate_hybrid_mesh(
                    surface_nodes, surface_faces, bounding_box,
                    growth_rate=self.growth_rate,
                    max_layers=self.max_layers,
                    min_cell_size=self.min_cell_size,
                    target_cells=self.target_cells,
                    surface_boundaries=surface_boundaries
                )
            else:
                # Pure extrusion mode
                return self._generate_pure_extrusion(
                    surface_nodes, surface_faces, bounding_box, surface_boundaries
                )
        elif method == "background":
            from .mesh_background import generate_hybrid_mesh
            return generate_hybrid_mesh(
                surface_nodes, surface_faces, bounding_box,
                growth_rate=self.growth_rate,
                max_layers=self.max_layers,
                min_cell_size=self.min_cell_size,
                target_cells=self.target_cells,
                surface_boundaries=surface_boundaries
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _generate_pure_extrusion(
        self,
        surface_nodes: np.ndarray,
        surface_faces: np.ndarray,
        bounding_box: Dict[str, np.ndarray],
        surface_boundaries: Optional['BoundaryMap'] = None
    ) -> 'VolumeMeshData':
        """Generate mesh by pure extrusion (no background mesh).
        
        Strategy:
        1. Compute surface normals
        2. Extrude layers with geometric growth
        3. Connect layers to form tetrahedra
        4. Build VolumeMeshData structure
        
        Args:
            surface_nodes: Surface geometry
            surface_faces: Surface connectivity
            bounding_box: Domain bounds
            surface_boundaries: Optional boundary mapping from surface mesh
            
        Returns:
            VolumeMeshData with volume mesh
        """
        from .mesh_utils import compute_face_normals, check_mesh_quality
        from .structures import NodeArray, TetrahedralCells, GridMetadata, VolumeMeshData
        
        logger.info("Starting pure extrusion-based mesh generation...")
        
        # Step 1: Compute face normals
        normals = compute_face_normals(surface_nodes, surface_faces)
        logger.info(f"Computed {len(normals)} face normals")
        
        # Step 2: Generate layered nodes
        all_nodes, layer_connectivity = extrude_layers(
            surface_nodes, surface_faces, normals, bounding_box,
            growth_rate=self.growth_rate,
            max_layers=self.max_layers,
            min_cell_size=self.min_cell_size
        )
        logger.info(f"Generated {len(all_nodes)} nodes in {len(layer_connectivity)} layers")
        
        # Step 3: Convert layers to tetrahedral cells
        volume_cells = convert_layers_to_tetrahedra(
            all_nodes, layer_connectivity, surface_faces
        )
        logger.info(f"Created {len(volume_cells)} tetrahedral cells")
        
        # Step 4: Build VolumeMeshData structure
        nodes_obj = NodeArray(
            x=all_nodes[:, 0],
            y=all_nodes[:, 1],
            z=all_nodes[:, 2]
        )
        
        logger.info("Computing tetrahedral volumes...")
        volumes = TetrahedralCells.compute_volumes(
            nodes_obj, volume_cells.astype(np.int32)
        )
        logger.info(
            f"Computed {len(volumes)} tetrahedral volumes, "
            f"total volume: {volumes.sum():.6e} m^3"
        )
        
        cells_obj = TetrahedralCells(
            connectivity=volume_cells.astype(np.int32),
            volumes=volumes
        )
        
        # Identify boundaries
        boundaries_obj = identify_boundaries_from_surface(
            volume_cells, surface_faces, surface_boundaries
        )
        
        metadata = GridMetadata(
            node_count=len(all_nodes),
            cell_count=len(volume_cells),
            boundary_groups=list(boundaries_obj.groups.keys()),
            file_format="volume"
        )
        
        volume_mesh = VolumeMeshData(
            nodes=nodes_obj,
            cells=cells_obj,
            boundaries=boundaries_obj,
            metadata=metadata
        )
        
        # Quality check
        check_mesh_quality(volume_mesh)
        
        logger.success(
            f"Extrusion mesh generation complete: "
            f"{volume_mesh.node_count} nodes, {volume_mesh.cell_count} cells, "
            f"total volume: {volume_mesh.total_volume:.6e} m^3"
        )
        
        return volume_mesh
