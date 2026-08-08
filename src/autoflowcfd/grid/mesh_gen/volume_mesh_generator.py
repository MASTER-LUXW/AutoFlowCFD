"""CFD 用体网格生成器。

通过 BL 挤出 + tetgen 核心填充的混合装配方式，从面三角化网格生成三维
四面体体网格。

本模块是一个协调者，把实际工作委托给专门的子模块：
- mesh_background：混合装配编排
- mesh_utils：校验与辅助函数
"""

import numpy as np
from typing import Dict, Optional, TYPE_CHECKING
from loguru import logger

from ..structures import GridData

if TYPE_CHECKING:
    from ..structures import BoundaryMap, VolumeMeshData
from .mesh_utils import validate_surface_mesh, validate_bounding_box


class VolumeMeshGenerator:
    """Generate 3D volume meshes from surface geometry.

    Converts surface triangulations (from NAS files) into volumetric
    meshes suitable for FVM solvers, via BL extrusion + tetgen core-fill
    (mesh_background.generate_hybrid_mesh).

    Attributes:
        growth_rate: Mesh growth rate for boundary layers
        min_cell_size: Minimum cell size constraint
        target_cells: Target number of volume cells
    """

    def __init__(
        self,
        growth_rate: float = 1.2,
        min_cell_size: float = 0.01,
        target_cells: int = 400000,
        max_cell_size: Optional[float] = None,
        bl_layers: Optional[int] = None,
        bl_only: bool = False,
        bl_only_output: Optional[str] = None,
        core_only: bool = False,
    ):
        """Initialize volume mesh generator.

        Args:
            growth_rate: Geometric growth rate for layer thickness (1.2 typical)
            min_cell_size: Minimum allowable cell size in meters (1cm default)
            target_cells: Target total cell count
            max_cell_size: Optional hard cap (meters) on core-region cell size,
                graded outward from the BL's near-wall size
                (mesh_background.generate_hybrid_mesh). None leaves the core
                fill's cell size unbounded (only tetgen's own shape-quality
                bounds apply, cells can grow as large as a coarse far-field
                input facet allows).
            bl_layers: How many layers the BL stage extrudes before the
                remaining volume is filled directly from the BL's own outer
                surface by tetgen (see mesh_extrusion.extrude_layers' own
                bl_layers doc). None (default) uses 8.
            bl_only: If True, only generate and export the BL prism layer mesh.
            bl_only_output: Output .nas path to use when bl_only is True.
                Required together with bl_only.
            core_only: If True, export right after core-region tetgen fill
                (core tets alone, not spliced with BL) and stop - same
                output-path reuse as bl_only_output.
        """
        self.growth_rate = growth_rate
        self.min_cell_size = min_cell_size
        self.target_cells = target_cells
        self.max_cell_size = max_cell_size
        self.bl_layers = bl_layers
        self.bl_only = bl_only
        self.bl_only_output = bl_only_output
        self.core_only = core_only

        logger.info(
            f"VolumeMeshGenerator initialized: growth_rate={growth_rate}, "
            f"min_cell_size={min_cell_size}m, "
            f"target_cells={target_cells}, max_cell_size={max_cell_size}, "
            f"bl_layers={bl_layers}, bl_only={bl_only}"
        )

    def generate_from_surface(
        self,
        surface_nodes: np.ndarray,
        surface_faces: np.ndarray,
        bounding_box: Dict[str, np.ndarray],
        surface_boundaries: Optional['BoundaryMap'] = None,
    ) -> 'VolumeMeshData':
        """Generate volume mesh from surface geometry (BL + tetgen core-fill hybrid).

        Args:
            surface_nodes: Surface node coordinates, shape=(n_nodes, 3)
            surface_faces: Surface triangle connectivity, shape=(n_faces, 3)
            bounding_box: Computational domain bounds {min: [x,y,z], max: [x,y,z]}
            surface_boundaries: Optional boundary mapping from surface mesh

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
            f"{len(surface_faces)} faces..."
        )

        return self._generate_hybrid_with_backoff(
            surface_nodes, surface_faces, bounding_box, surface_boundaries
        )

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
                logger.warning(
                    f"Stage C: retrying generation (attempt "
                    f"{attempt}/{max_backoff_attempts}) with backed-off "
                    f"parameters: min_cell_size={min_cell_size:.6f}m - "
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
                    min_cell_size=min_cell_size,
                    target_cells=self.target_cells,
                    surface_boundaries=surface_boundaries,
                    max_cell_size=self.max_cell_size,
                    bl_layers=self.bl_layers,
                    export_bl_only=self.bl_only,
                    export_bl_only_path=self.bl_only_output,
                    export_core_only=self.core_only,
                    export_core_only_path=self.bl_only_output,
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
