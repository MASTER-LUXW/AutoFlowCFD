"""面向 CFD 网格的质量校验工具。

为四面体和三角形网格提供全面的质量指标，包括体积检查、长宽比分析、
扭曲度评估和正交性评估。

关键指标：
    - 体积质量（负体积）
    - 相邻单元体积比（关系到 Green-Gauss 梯度重构的条件数）
    - 长宽比（单元形状质量，BL 区域和 core 区域用不同阈值）
    - 扭曲度（基于半径比的形状度量）
    - 正交性（面法向与单元质心连线的夹角）

指标的选取是针对本项目具体求解器校准的，不是通用默认值——推导依据：
    - 梯度重构用的是 FR 微分算子（core/fr_operators.py），
      grad ~ D_ij * q_j——通过预计算的微分矩阵进行高阶求导。
      体积比相邻单元小几个数量级的单元，梯度会被同样倍数放大，
      与局部伪时间步长无关（局部时间步长保护的是该单元自身的*稳定性*，
      不是它交给相邻单元的量的*精度*）。这就是为什么这里检查相邻单元
      体积比和非正交性（两者都直接影响离散格式的数值条件数），而不只是
      一个全局最大/最小体积比——BL 网格从近壁到远场的全局体积范围本来
      就会跨越好几个数量级，这本身不是缺陷。
    - 网格只有四面体（没有六面体/棱柱），近壁是 BL 挤出棱柱拆分成的
      四面体，其余是 tetgen 核心填充——长宽比对两个区域分别检查，因为
      BL 单元预期比 core 单元拉伸得多。

参考文献：
    - Knupp, P. "Advances in grid quality metrics", 2000
    - Field, D.A. "Qualitative measures for initial mesh generation", 1988
    - Verdict Geometric Quality Library (Sandia) - TetRadiusRatio metric
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
from loguru import logger

from .quality_report import MeshQualityReport
from . import quality_metrics as _qm
from .quality_evaluation import evaluate_quality, generate_recommendations

if TYPE_CHECKING:
    from ..structures import FaceData, VolumeMeshData


class MeshQualityValidator:
    """Validate mesh quality for CFD simulations.

    Computes various quality metrics to ensure the mesh is suitable for
    accurate and stable CFD simulations.

    Attributes:
        thresholds: Quality metric thresholds for pass/fail criteria
    """

    def __init__(self):
        """Initialize validator with default quality thresholds."""
        self.thresholds = {
            'max_negative_volumes': 0,       # No negative volumes allowed
            'max_volume_ratio': 1e6,         # Global range - informational only, see MeshQualityReport docstring
            'max_aspect_ratio': 100.0,       # fallback when no BL/core split is available
            'bl_max_aspect_ratio': 50.0,     # BL cells: expected to be stretched
            'core_max_aspect_ratio': 10.0,   # core-fill cells: should be close to isotropic
            'max_skewness': 0.95,            # radius-ratio based (Fluent-equivalent severity)
            'max_orthogonality_angle': 70.0, # degrees; OpenFOAM-aligned (Green-Gauss is more
                                              # sensitive to non-orthogonality than surface-normal-
                                              # correction schemes, so this is deliberately tighter
                                              # than Fluent's permissive orthogonal-quality floor)
            'max_adjacent_volume_ratio': 5.0,  # STAR-CCM+-aligned "Volume Change" guidance;
                                                # this is what actually governs Green-Gauss's 1/V
                                                # gradient-amplification conditioning
            'max_overlapping_cells': 0,        # any physically-overlapping cell pair fails -
                                                # see mesh_overlap_check.py; "close but not yet
                                                # overlapping" is informational only, doesn't gate
        }

        logger.info("MeshQualityValidator initialized with default thresholds")

    def validate(
        self,
        nodes: np.ndarray,
        cells: np.ndarray,
        cell_type: str = "tetrahedron",
        faces: Optional['FaceData'] = None,
        bl_cell_mask: Optional[np.ndarray] = None,
        log_summary: bool = True,
        check_overlap: bool = True,
    ) -> MeshQualityReport:
        """Perform comprehensive mesh quality validation.

        Args:
            nodes: Node coordinates, shape=(n_nodes, 3)
            cells: Cell connectivity, shape=(n_cells, n_vertices)
            cell_type: Type of cells ('tetrahedron' or 'triangle')
            faces: Optional precomputed FaceData (owner/neighbour
                connectivity + normals). Orthogonality, adjacent-volume-
                ratio, and overlap/proximity checks need face connectivity;
                if not supplied it is derived internally via
                FaceExtractor.extract_faces (a real but non-trivial cost
                for large meshes - callers that already have this, e.g.
                the mesh generation/repair pipeline, should pass it through
                to avoid redundant work). Ignored for cell_type='triangle'.
            bl_cell_mask: Optional bool array, shape=(n_cells,), True for
                BL-region cells - enables the separate BL-region/core-
                region aspect ratio breakdown. None falls back to a single
                whole-mesh aspect ratio check (previous behaviour).
            log_summary: Log the full formatted report via logger.info.
                False for callers that will print a more complete version
                of this same report themselves right after (e.g. one with
                a before/after comparison attached) - avoids the same
                report text appearing twice in a row.
            check_overlap: Run the cell overlap/proximity check (see
                mesh_overlap_check.py). Unlike every other check here, its
                cost scales with local mesh density (broad-phase spatial
                search + exact geometric tests on survivors), not purely
                cell count - opt-out (not silently skipped) for a caller
                that needs the fastest possible turnaround and is willing
                to accept an overlap going undetected until the next full
                validate() call.

        Returns:
            MeshQualityReport with all quality metrics
        """
        logger.info(f"Validating mesh quality: {len(cells)} {cell_type}s...")

        report = MeshQualityReport(
            n_cells=len(cells),
            n_nodes=len(nodes)
        )

        # Compute all quality metrics
        self._check_volumes(report, nodes, cells, cell_type)
        self._check_aspect_ratios(report, nodes, cells, cell_type, bl_cell_mask)
        self._check_skewness(report, nodes, cells, cell_type)
        if cell_type == "tetrahedron":
            # Extracted (at most) once and shared with both checks below -
            # _check_orthogonality_and_adjacency and _check_overlap_and_proximity
            # would otherwise each independently call self._extract_faces
            # when the caller didn't pre-supply `faces`, paying for a full
            # face extraction over the WHOLE mesh twice in a row for no
            # reason (confirmed directly: "Extracting faces from N
            # tetrahedral cells..." logged twice back-to-back in one
            # validate() call on a real 1.5M-cell mesh, since neither
            # sub-check's own internally-extracted FaceData was cached back
            # here for the other to reuse).
            if faces is None:
                faces = self._extract_faces(nodes, cells)
            self._check_orthogonality_and_adjacency(report, nodes, cells, faces)
            if check_overlap:
                self._check_overlap_and_proximity(report, nodes, cells, faces)

        # Evaluate pass/fail criteria
        evaluate_quality(report, self.thresholds)

        # Generate recommendations
        generate_recommendations(report, self.thresholds)

        # Log summary
        if log_summary:
            logger.info(f"\n{report.summary()}")

        return report

    def validate_volume_mesh(
        self,
        volume_mesh: 'VolumeMeshData',
        faces: Optional['FaceData'] = None,
        bl_cell_mask: Optional[np.ndarray] = None,
        check_overlap: bool = True,
    ) -> MeshQualityReport:
        """Validate VolumeMeshData object (convenience method).

        Args:
            volume_mesh: VolumeMeshData with tetrahedral cells (and,
                optionally, prism_cells - dispatches to validate_mixed()
                when present, see that method)
            faces: Optional precomputed FaceData - if not supplied and
                volume_mesh.faces is already populated (ensure_faces_exist
                was called), that gets reused instead of re-extracting.
            bl_cell_mask: Optional BL/core region split, see validate().
                Ignored when volume_mesh.prism_cells is set - validate_mixed
                derives this itself (prisms ARE the BL region, tets are all
                core, by this project's global cell-index convention).
            check_overlap: see validate()

        Returns:
            MeshQualityReport with all quality metrics
        """
        if faces is None:
            faces = volume_mesh.faces

        if volume_mesh.prism_cells is not None:
            return self.validate_mixed(volume_mesh, faces=faces, check_overlap=check_overlap)

        return self.validate(
            nodes=np.column_stack([
                volume_mesh.nodes.x,
                volume_mesh.nodes.y,
                volume_mesh.nodes.z
            ]),
            cells=volume_mesh.cells.connectivity,
            cell_type="tetrahedron",
            faces=faces,
            bl_cell_mask=bl_cell_mask,
            check_overlap=check_overlap,
        )

    def validate_mixed(
        self,
        volume_mesh: 'VolumeMeshData',
        faces: Optional['FaceData'] = None,
        log_summary: bool = True,
        check_overlap: bool = True,
    ) -> MeshQualityReport:
        """Validate a mixed prism(BL) + tetrahedron(core) VolumeMeshData.

        Mirrors validate()'s structure, but every per-cell metric is
        computed separately per region (prism cells via quality_metrics'
        prism functions, tet cells via the existing tetrahedron functions -
        the two shapes need genuinely different formulas, see quality_
        metrics.py) and then concatenated in the SAME global cell-index
        order every other prism-aware piece of this codebase uses (prisms
        [0, n_prism), tets [n_prism, n_prism+n_tet) - see PrismCells/
        face_extractor.extract_faces_mixed). Orthogonality and adjacent-
        volume-ratio (face-based, so they inherently span the BL/core
        interface) use ONE combined pass over the global face graph, via
        compute_face_diagnostics' cell_centroids/cell_volumes parameters
        (added specifically so it doesn't have to re-derive a per-cell
        centroid/volume from a single uniform connectivity array, which a
        mixed mesh doesn't have).

        bl_cell_mask is not a parameter here (unlike validate()) - it's
        exactly [True]*n_prism + [False]*n_tet by construction, not
        something a caller could meaningfully override.

        Implementation lives in quality_validator_mixed.py (extracted for
        the project's >400-line file-split rule) - lazy-imported here to
        avoid a module-load-time circular import (that module's type hints
        reference MeshQualityValidator from this file).
        """
        from .quality_validator_mixed import validate_mixed_mesh

        return validate_mixed_mesh(
            self, volume_mesh, faces=faces, log_summary=log_summary, check_overlap=check_overlap
        )

    @staticmethod
    def _extract_faces(nodes: np.ndarray, cells: np.ndarray) -> 'FaceData':
        """Derive face connectivity when the caller didn't already have it.
        Lazy-imported (mesh_gen -> validation is a one-way dependency
        elsewhere in this package; importing the other direction here only
        at call time avoids ever needing to reason about import order)."""
        from ..mesh_gen.face_extractor import FaceExtractor
        from ..schema.grid_nodes import NodeArray

        node_arr = NodeArray(
            x=np.ascontiguousarray(nodes[:, 0]),
            y=np.ascontiguousarray(nodes[:, 1]),
            z=np.ascontiguousarray(nodes[:, 2]),
        )
        return FaceExtractor.extract_faces(cells.astype(np.int32), node_arr)

    def _check_volumes(
        self,
        report: MeshQualityReport,
        nodes: np.ndarray,
        cells: np.ndarray,
        cell_type: str
    ) -> None:
        """Check cell volumes for validity (vectorized).

        Implementation lives in quality_validator_metrics.py (extracted
        for the project's >400-line file-split rule; the body never used
        `self`, so it moved as a plain function).
        """
        from .quality_validator_metrics import check_volumes

        check_volumes(report, nodes, cells, cell_type)

    def _compute_tetrahedron_volumes(self, nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """Thin instance-method wrapper over quality_metrics.compute_tetrahedron_volumes
        - kept for external callers (mesh_gen/mesh_repair.py's Stage A) that
        reach into this validator instance directly rather than importing
        the metric function themselves."""
        return _qm.compute_tetrahedron_volumes(nodes, cells)

    def _check_aspect_ratios(
        self,
        report: MeshQualityReport,
        nodes: np.ndarray,
        cells: np.ndarray,
        cell_type: str,
        bl_cell_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Check cell aspect ratios (vectorized), optionally split by
        BL-region vs. core-region (see MeshQualityReport docstring for why
        these need separate thresholds).

        Implementation lives in quality_validator_metrics.py (extracted
        for the project's >400-line file-split rule; the body never used
        `self`, so it moved as a plain function).
        """
        from .quality_validator_metrics import check_aspect_ratios

        check_aspect_ratios(report, nodes, cells, cell_type, bl_cell_mask=bl_cell_mask)

    def _check_skewness(
        self,
        report: MeshQualityReport,
        nodes: np.ndarray,
        cells: np.ndarray,
        cell_type: str
    ) -> None:
        """Check cell skewness (vectorized).

        Implementation lives in quality_validator_metrics.py (extracted
        for the project's >400-line file-split rule; the body never used
        `self`, so it moved as a plain function).
        """
        from .quality_validator_metrics import check_skewness

        check_skewness(report, nodes, cells, cell_type)

    def compute_cell_skewness(self, nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """Public per-cell radius-ratio skewness array, shape=(n_cells,) -
        the raw values behind max_skewness/mean_skewness, for callers (e.g.
        the mesh repair loop in mesh_gen/mesh_repair.py) that need to know
        *which* cells are bad, not just aggregate statistics."""
        from .quality_validator_metrics import compute_cell_skewness as _compute_cell_skewness

        return _compute_cell_skewness(nodes, cells)

    def compute_face_diagnostics(
        self,
        nodes: np.ndarray,
        cells: np.ndarray,
        faces: Optional['FaceData'] = None,
        cell_centroids: Optional[np.ndarray] = None,
        cell_volumes: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Public per-internal-face diagnostics - the raw arrays behind
        orthogonality_max/adjacent_volume_ratio_max, for callers that need
        to know which faces/cells are implicated, not just aggregates.

        Implementation lives in quality_validator_metrics.py (extracted
        for the project's >400-line file-split rule) - see that module's
        compute_face_diagnostics for the full Args/Returns documentation.
        """
        from .quality_validator_metrics import compute_face_diagnostics as _compute_face_diagnostics

        return _compute_face_diagnostics(
            self, nodes, cells, faces=faces, cell_centroids=cell_centroids, cell_volumes=cell_volumes
        )

    def _check_orthogonality_and_adjacency(
        self,
        report: MeshQualityReport,
        nodes: np.ndarray,
        cells: np.ndarray,
        faces: Optional['FaceData'],
        cell_centroids: Optional[np.ndarray] = None,
        cell_volumes: Optional[np.ndarray] = None,
    ) -> None:
        """Check face non-orthogonality and adjacent-cell (face-neighbour)
        volume ratio - the two metrics that directly govern Green-Gauss
        gradient conditioning for this project's solver (see module
        docstring). Both need face owner/neighbour connectivity, so they
        share a single face-extraction pass (compute_face_diagnostics).

        cell_centroids/cell_volumes: see compute_face_diagnostics - pass
        through for a mixed prism+tet mesh, where `cells` alone can't
        describe every cell's shape.
        """
        diag = self.compute_face_diagnostics(
            nodes, cells, faces, cell_centroids=cell_centroids, cell_volumes=cell_volumes
        )
        if len(diag['angle_deg']) == 0:
            return

        report.orthogonality_max = float(np.max(diag['angle_deg']))
        report.orthogonality_mean = float(np.mean(diag['angle_deg']))
        report.adjacent_volume_ratio_max = float(np.max(diag['volume_ratio']))
        report.adjacent_volume_ratio_mean = float(np.mean(diag['volume_ratio']))

    def _check_overlap_and_proximity(
        self,
        report: MeshQualityReport,
        nodes: np.ndarray,
        cells: np.ndarray,
        faces: Optional['FaceData'],
    ) -> None:
        """Detect cells whose faces physically overlap a different, non-
        adjacent cell's faces, or sit close enough to be one parameter
        change away from it - see mesh_overlap_check.py for the exact
        geometric tests and why this is a distinct defect class from
        negative/degenerate volume."""
        from .mesh_overlap_check import check_face_overlap_and_proximity

        overlap_report = check_face_overlap_and_proximity(nodes, cells, faces=faces)
        report.n_overlapping_cells = len(overlap_report.overlapping_cell_ids)
        report.n_close_cell_pairs = overlap_report.n_close_pairs
        report.overlap_min_gap = overlap_report.min_gap_found
        report.overlapping_cell_ids = overlap_report.overlapping_cell_ids

