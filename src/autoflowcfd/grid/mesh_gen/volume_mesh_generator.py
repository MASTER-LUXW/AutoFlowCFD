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
from typing import Dict, Optional, TYPE_CHECKING
from loguru import logger

from ..structures import GridData

if TYPE_CHECKING:
    from ..structures import BoundaryMap, VolumeMeshData
from .mesh_utils import validate_surface_mesh, validate_bounding_box
from .mesh_extrusion import extrude_layers
from .mesh_prism_to_tet import convert_layers_to_tetrahedra
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
        max_layers: int = 12,
        min_cell_size: float = 0.01,
        target_cells: int = 400000,
        max_cell_size: Optional[float] = None,
        bl_layers: Optional[int] = None,
    ):
        """Initialize volume mesh generator.

        Args:
            growth_rate: Geometric growth rate for layer thickness (1.2 typical)
            max_layers: Maximum number of boundary layers (30 for automotive CFD)
            min_cell_size: Minimum allowable cell size in meters (1mm default)
            target_cells: Target total cell count
            max_cell_size: Optional hard cap (meters) on core-region cell size,
                graded outward from the BL's near-wall size
                (mesh_background.generate_hybrid_mesh). None leaves the core
                fill's cell size unbounded (only tetgen's own shape-quality
                bounds apply, cells can grow as large as a coarse far-field
                input facet allows).
            bl_layers: Optional override for how many of max_layers count as
                "Stage 1 (BL)" before switching to the transition growth
                rate (see mesh_extrusion.extrude_layers' own bl_layers
                doc). None (default) keeps the previous hardcoded
                `min(8, max_layers)` split - notably, any max_layers <= 8
                then leaves zero layers for the transition stage, silently
                disabling max_cell_size's BL/core size-matching regardless
                of whether max_cell_size itself is set.
        """
        self.growth_rate = growth_rate
        self.max_layers = max_layers
        self.min_cell_size = min_cell_size
        self.target_cells = target_cells
        self.max_cell_size = max_cell_size
        self.bl_layers = bl_layers

        logger.info(
            f"VolumeMeshGenerator initialized: growth_rate={growth_rate}, "
            f"max_layers={max_layers}, min_cell_size={min_cell_size}m, "
            f"target_cells={target_cells}, max_cell_size={max_cell_size}, "
            f"bl_layers={bl_layers}"
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
                return self._generate_hybrid_with_backoff(
                    surface_nodes, surface_faces, bounding_box, surface_boundaries
                )
            else:
                # Pure extrusion mode
                return self._generate_pure_extrusion(
                    surface_nodes, surface_faces, bounding_box, surface_boundaries
                )
        elif method == "background":
            return self._generate_hybrid_with_backoff(
                surface_nodes, surface_faces, bounding_box, surface_boundaries
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    def _generate_hybrid_with_backoff(
        self,
        surface_nodes: np.ndarray,
        surface_faces: np.ndarray,
        bounding_box: Dict[str, np.ndarray],
        surface_boundaries: Optional['BoundaryMap'],
        max_backoff_attempts: int = 1,
    ) -> 'VolumeMeshData':
        """Stage C of the mesh quality repair loop: if generate_hybrid_mesh
        (which already runs Stage A smoothing and one Stage B targeted
        retry internally - see mesh_gen/mesh_repair.py) still doesn't pass
        MeshQualityValidator, retry the *entire* generation with backed-off
        global parameters (larger min_cell_size, fewer layers) - a coarser
        mesh gives sharp features proportionally more room before hitting
        the same degenerate-tetrahedra failure mode, at the cost of
        resolution. This is the blunt, untargeted fallback for when Stage
        A/B's more targeted repairs aren't enough; each attempt is a full
        (potentially multi-minute) regeneration, so attempts are capped -
        combined with the one Stage B retry generate_hybrid_mesh may run
        internally per attempt, the theoretical worst case is
        (max_backoff_attempts + 1) * 2 full generation passes (default: 2
        Stage C levels * 2 = 4, down from an earlier 3 * 2 = 6 - measured
        directly on a hard real case (a 90-degree sharp-corner body) that
        never actually converges regardless of how many attempts are
        allowed, where every extra level was pure added wall-clock time
        for no quality benefit).
        """
        from .mesh_background import generate_hybrid_mesh
        from ..validation.quality_validator import MeshQualityValidator

        growth_rate = self.growth_rate
        max_layers = self.max_layers
        min_cell_size = self.min_cell_size

        validator = MeshQualityValidator()
        # Tracks the best mesh produced so far even across a later attempt
        # that raises - a later backoff level failing outright shouldn't
        # discard an earlier attempt's still-usable (if quality-failing)
        # mesh.
        best_mesh = None
        best_report = None
        last_error: Optional[Exception] = None

        for attempt in range(max_backoff_attempts + 1):
            if attempt > 0:
                min_cell_size *= 1.5
                max_layers = max(1, max_layers - 1)
                logger.warning(
                    f"Stage C: retrying generation (attempt "
                    f"{attempt}/{max_backoff_attempts}) with backed-off "
                    f"parameters: min_cell_size={min_cell_size:.6f}m, "
                    f"max_layers={max_layers} - "
                    + (
                        f"previous attempt raised: {last_error}"
                        if last_error is not None
                        else "mesh quality gate still failing after Stage A/B"
                    )
                )

            try:
                volume_mesh = generate_hybrid_mesh(
                    surface_nodes, surface_faces, bounding_box,
                    growth_rate=growth_rate,
                    max_layers=max_layers,
                    min_cell_size=min_cell_size,
                    target_cells=self.target_cells,
                    surface_boundaries=surface_boundaries,
                    max_cell_size=self.max_cell_size,
                    bl_layers=self.bl_layers,
                )
            except RuntimeError as e:
                # fill_core_volume (mesh_tetgen_core.py) raises RuntimeError
                # on a self-intersecting BL surface or a tetgen robustness
                # failure - exactly the failure mode backed-off parameters
                # (fewer/thinner layers) are meant to fix. Previously this
                # exception wasn't caught here at all, so it propagated
                # straight past every remaining backoff attempt and aborted
                # generation entirely, defeating Stage C for its most
                # severe failure class while still working for the milder
                # "generated but failed the quality gate" case below.
                last_error = e
                logger.warning(f"Stage C: attempt {attempt} raised during generation: {e}")
                continue

            report = validator.validate_volume_mesh(volume_mesh)
            # Genuinely keep the BEST attempt, not just the most recent one
            # - despite the variable's name, this used to unconditionally
            # overwrite best_mesh/best_report every iteration regardless of
            # whether the new attempt was actually better, so whichever
            # attempt happened to run last silently won even when an
            # earlier one was clearly superior. Confirmed directly on a
            # real case: attempt 0 produced 37 overlapping cells, attempt 1
            # (backed-off params, triggered because attempt 0 still failed
            # the quality gate on OTHER criteria) produced 127 - worse on
            # the one metric this project's own overlap-prevention work is
            # about - yet attempt 1 was the one returned. n_overlapping_
            # cells is used as the ranking key (not overall pass/fail,
            # which every attempt here already lacks by construction, and
            # not a multi-criteria score) because it is this quality
            # report's own CRITICAL-severity field - the other warnings
            # (skewness, non-orthogonality, aspect ratio) are HIGH/MEDIUM.
            if best_report is None or report.n_overlapping_cells < best_report.n_overlapping_cells:
                best_mesh, best_report = volume_mesh, report
            if report.passed:
                if attempt > 0:
                    logger.success(f"Stage C: attempt {attempt} passed the quality gate")
                break
        else:
            if best_mesh is None:
                # Every attempt, including the last, raised - there is no
                # mesh to fall back to, so surface the last failure instead
                # of returning None to the caller.
                raise RuntimeError(
                    f"Stage C: mesh generation failed on all "
                    f"{max_backoff_attempts + 1} attempt(s) (including backed-off "
                    f"parameters); last error: {last_error}"
                ) from last_error
            logger.error(
                f"Stage C: mesh quality gate still failing after "
                f"{max_backoff_attempts} backoff attempt(s) - returning the best "
                f"attempt's mesh anyway (best-effort); see the quality report above "
                f"for which cells/regions are still implicated. The solve-time "
                f"quality gate (autoflowcfd solve run) will catch this before any "
                f"iterations run, unless --skip-quality-check is passed."
            )

        return best_mesh
    
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
        from ..structures import NodeArray, TetrahedralCells, GridMetadata, VolumeMeshData
        
        logger.info("Starting pure extrusion-based mesh generation...")
        
        # Step 1: Compute face normals
        normals = compute_face_normals(surface_nodes, surface_faces)
        logger.info(f"Computed {len(normals)} face normals")
        
        # Step 2: Generate layered nodes
        all_nodes, layer_connectivity = extrude_layers(
            surface_nodes, surface_faces, normals, bounding_box,
            growth_rate=self.growth_rate,
            max_layers=self.max_layers,
            min_cell_size=self.min_cell_size,
            bl_layers=self.bl_layers,
        )
        logger.info(f"Generated {len(all_nodes)} nodes in {len(layer_connectivity)} layers")
        
        # Step 3: Convert layers to tetrahedral cells
        volume_cells, _ = convert_layers_to_tetrahedra(
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
