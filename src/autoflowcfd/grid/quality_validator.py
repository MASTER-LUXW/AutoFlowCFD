"""Mesh quality validation tools for CFD grids.

Provides comprehensive quality metrics for tetrahedral and triangular meshes,
including volume checks, aspect ratio analysis, skewness evaluation, and
orthogonality assessment.

Key Metrics:
    - Volume quality (negative volumes, extreme ratios)
    - Aspect ratio (cell shape quality)
    - Skewness (deviation from ideal shape)
    - Orthogonality (face normal alignment)
    - Smoothness (gradual size transitions)

References:
    - Knupp, P. "Advances in grid quality metrics", 2000
    - Field, D.A. "Qualitative measures for initial mesh generation", 1988
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class MeshQualityReport:
    """Comprehensive mesh quality report.
    
    Attributes:
        n_cells: Total number of cells
        n_nodes: Total number of nodes
        passed: Overall quality check result
        negative_volumes: Count of cells with negative volume
        min_volume: Minimum cell volume
        max_volume: Maximum cell volume
        mean_volume: Mean cell volume
        std_volume: Standard deviation of volumes
        volume_ratio: Max/Min volume ratio
        min_aspect_ratio: Minimum aspect ratio
        max_aspect_ratio: Maximum aspect ratio
        mean_aspect_ratio: Mean aspect ratio
        max_skewness: Maximum skewness value
        mean_skewness: Mean skewness
        orthogonality_min: Minimum orthogonality angle
        warnings: List of quality warnings
        recommendations: List of improvement recommendations
    """
    n_cells: int = 0
    n_nodes: int = 0
    passed: bool = True
    
    # Volume metrics
    negative_volumes: int = 0
    min_volume: float = float('inf')
    max_volume: float = 0.0
    mean_volume: float = 0.0
    std_volume: float = 0.0
    volume_ratio: float = 0.0
    
    # Aspect ratio metrics
    min_aspect_ratio: float = float('inf')
    max_aspect_ratio: float = 0.0
    mean_aspect_ratio: float = 0.0
    
    # Skewness metrics
    max_skewness: float = 0.0
    mean_skewness: float = 0.0
    
    # Orthogonality metrics
    orthogonality_min: float = 180.0
    
    # Qualitative feedback
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def summary(self) -> str:
        """Generate human-readable quality summary."""
        lines = [
            "=" * 70,
            "MESH QUALITY REPORT",
            "=" * 70,
            f"Overall Status: {'PASSED ✓' if self.passed else 'FAILED ✗'}",
            "",
            f"Grid Size:",
            f"  Cells: {self.n_cells:,}",
            f"  Nodes: {self.n_nodes:,}",
            "",
            f"Volume Quality:",
            f"  Negative volumes: {self.negative_volumes}",
            f"  Volume range: [{self.min_volume:.6e}, {self.max_volume:.6e}]",
            f"  Mean ± Std: {self.mean_volume:.6e} ± {self.std_volume:.6e}",
            f"  Max/Min ratio: {self.volume_ratio:.2f}",
            "",
            f"Aspect Ratio:",
            f"  Range: [{self.min_aspect_ratio:.3f}, {self.max_aspect_ratio:.3f}]",
            f"  Mean: {self.mean_aspect_ratio:.3f}",
            "",
            f"Skewness:",
            f"  Max: {self.max_skewness:.4f}",
            f"  Mean: {self.mean_skewness:.4f}",
            "",
            f"Orthogonality:",
            f"  Min angle: {self.orthogonality_min:.2f}°",
        ]
        
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for i, warning in enumerate(self.warnings, 1):
                lines.append(f"  {i}. {warning}")
        
        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"  {i}. {rec}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


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
            'max_negative_volumes': 0,  # No negative volumes allowed
            'max_volume_ratio': 1e6,    # Max/min volume ratio
            'min_aspect_ratio': 0.1,    # Minimum acceptable aspect ratio
            'max_aspect_ratio': 100.0,  # Maximum acceptable aspect ratio
            'max_skewness': 0.95,       # Maximum skewness (0-1 scale)
            'min_orthogonality': 10.0,  # Minimum orthogonality angle (degrees)
        }
        
        logger.info("MeshQualityValidator initialized with default thresholds")
    
    def validate(
        self,
        nodes: np.ndarray,
        cells: np.ndarray,
        cell_type: str = "tetrahedron"
    ) -> MeshQualityReport:
        """Perform comprehensive mesh quality validation.
        
        Args:
            nodes: Node coordinates, shape=(n_nodes, 3)
            cells: Cell connectivity, shape=(n_cells, n_vertices)
            cell_type: Type of cells ('tetrahedron' or 'triangle')
            
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
        self._check_aspect_ratios(report, nodes, cells, cell_type)
        self._check_skewness(report, nodes, cells, cell_type)
        self._check_orthogonality(report, nodes, cells)
        
        # Evaluate pass/fail criteria
        self._evaluate_quality(report)
        
        # Generate recommendations
        self._generate_recommendations(report)
        
        # Log summary
        logger.info(f"\n{report.summary()}")
        
        return report
    
    def validate_volume_mesh(self, volume_mesh: 'VolumeMeshData') -> MeshQualityReport:
        """Validate VolumeMeshData object (convenience method).
        
        Args:
            volume_mesh: VolumeMeshData with tetrahedral cells
            
        Returns:
            MeshQualityReport with all quality metrics
        """
        return self.validate(
            nodes=np.column_stack([
                volume_mesh.nodes.x,
                volume_mesh.nodes.y,
                volume_mesh.nodes.z
            ]),
            cells=volume_mesh.cells.connectivity,
            cell_type="tetrahedron"
        )
    
    def _check_volumes(
        self,
        report: MeshQualityReport,
        nodes: np.ndarray,
        cells: np.ndarray,
        cell_type: str
    ) -> None:
        """Check cell volumes for validity (vectorized).

        A pure-Python per-cell loop here does not scale: automotive volume
        meshes routinely have hundreds of thousands of tetrahedra, and this
        method previously iterated them one at a time in Python.

        Args:
            report: Quality report to update
            nodes: Node coordinates
            cells: Cell connectivity
            cell_type: Type of cells
        """
        if cell_type == "tetrahedron":
            volumes_array = self._compute_tetrahedron_volumes(nodes, cells)
        elif cell_type == "triangle":
            volumes_array = self._compute_triangle_areas(nodes, cells)
        else:
            raise ValueError(f"Unsupported cell type: {cell_type}")

        negative_mask = volumes_array < 0
        report.negative_volumes = int(np.sum(negative_mask))
        if report.negative_volumes > 0:
            bad_indices = np.where(negative_mask)[0]
            preview = ", ".join(str(i) for i in bad_indices[:10].tolist())
            more = f" (+{len(bad_indices) - 10} more)" if len(bad_indices) > 10 else ""
            logger.warning(
                f"Negative volume detected in {report.negative_volumes} cells: "
                f"{preview}{more}"
            )

        if len(volumes_array) > 0:
            positive_volumes = volumes_array[volumes_array > 0]

            if len(positive_volumes) > 0:
                report.min_volume = float(np.min(positive_volumes))
                report.max_volume = float(np.max(positive_volumes))
                report.mean_volume = float(np.mean(positive_volumes))
                report.std_volume = float(np.std(positive_volumes))
                report.volume_ratio = report.max_volume / max(report.min_volume, 1e-12)

    @staticmethod
    def _compute_tetrahedron_volumes(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """Signed volume of every tetrahedron: det(p1-p0, p2-p0, p3-p0) / 6.

        Note: kept signed (not absolute) so a negative-volume/inverted-cell
        check upstream is meaningful; magnitude statistics below use the
        positive subset regardless.
        """
        p0 = nodes[cells[:, 0]]
        p1 = nodes[cells[:, 1]]
        p2 = nodes[cells[:, 2]]
        p3 = nodes[cells[:, 3]]

        v1 = p1 - p0
        v2 = p2 - p0
        v3 = p3 - p0

        return np.einsum('ij,ij->i', v1, np.cross(v2, v3)) / 6.0

    @staticmethod
    def _compute_triangle_areas(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """Area of every triangle: 0.5 * |cross(p1-p0, p2-p0)|."""
        p0 = nodes[cells[:, 0]]
        p1 = nodes[cells[:, 1]]
        p2 = nodes[cells[:, 2]]

        cross = np.cross(p1 - p0, p2 - p0)
        return 0.5 * np.linalg.norm(cross, axis=1)
    
    def _check_aspect_ratios(
        self,
        report: MeshQualityReport,
        nodes: np.ndarray,
        cells: np.ndarray,
        cell_type: str
    ) -> None:
        """Check cell aspect ratios (vectorized).

        Aspect ratio = longest edge / shortest edge, for every cell at once.

        Args:
            report: Quality report to update
            nodes: Node coordinates
            cells: Cell connectivity
            cell_type: Type of cells
        """
        if cell_type == "triangle":
            ar_array = self._compute_triangle_aspect_ratios(nodes, cells)
        elif cell_type == "tetrahedron":
            ar_array = self._compute_tetrahedron_aspect_ratios(nodes, cells)
        else:
            return

        if len(ar_array) > 0:
            report.min_aspect_ratio = float(np.min(ar_array))
            report.max_aspect_ratio = float(np.max(ar_array))
            report.mean_aspect_ratio = float(np.mean(ar_array))

    @staticmethod
    def _triangle_edge_lengths(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """Edge lengths for every triangle, shape=(n_cells, 3)."""
        p0, p1, p2 = nodes[cells[:, 0]], nodes[cells[:, 1]], nodes[cells[:, 2]]
        e1 = np.linalg.norm(p1 - p0, axis=1)
        e2 = np.linalg.norm(p2 - p1, axis=1)
        e3 = np.linalg.norm(p0 - p2, axis=1)
        return np.stack([e1, e2, e3], axis=1)

    @staticmethod
    def _tetrahedron_edge_lengths(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """All 6 edge lengths for every tetrahedron, shape=(n_cells, 6)."""
        pts = nodes[cells]  # (n_cells, 4, 3)
        edges = []
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append(np.linalg.norm(pts[:, i] - pts[:, j], axis=1))
        return np.stack(edges, axis=1)

    def _compute_triangle_aspect_ratios(self, nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """AR = longest_edge / shortest_edge for every triangle (1.0 = equilateral)."""
        edges = self._triangle_edge_lengths(nodes, cells)
        return np.max(edges, axis=1) / (np.min(edges, axis=1) + 1e-12)

    def _compute_tetrahedron_aspect_ratios(self, nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """AR = longest_edge / shortest_edge across all 6 edges of every tet."""
        edges = self._tetrahedron_edge_lengths(nodes, cells)
        return np.max(edges, axis=1) / (np.min(edges, axis=1) + 1e-12)

    def _check_skewness(
        self,
        report: MeshQualityReport,
        nodes: np.ndarray,
        cells: np.ndarray,
        cell_type: str
    ) -> None:
        """Check cell skewness (vectorized).

        Skewness measures deviation from ideal shape.
        For triangles: based on angles deviation from 60°

        Args:
            report: Quality report to update
            nodes: Node coordinates
            cells: Cell connectivity
            cell_type: Type of cells
        """
        if cell_type == "triangle":
            sk_array = self._compute_triangle_skewness_values(nodes, cells)
        elif cell_type == "tetrahedron":
            sk_array = self._compute_tetrahedron_skewness_values(nodes, cells)
        else:
            return

        if len(sk_array) > 0:
            report.max_skewness = float(np.max(sk_array))
            report.mean_skewness = float(np.mean(sk_array))

    def _compute_triangle_skewness_values(self, nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """Skewness for every triangle: max(|angle - 60°|) / 60°, in [0, 1].

        Based on angle deviation from equilateral (60° each), via the law of
        cosines on each triangle's 3 edge lengths.
        """
        p0, p1, p2 = nodes[cells[:, 0]], nodes[cells[:, 1]], nodes[cells[:, 2]]
        a = np.linalg.norm(p1 - p2, axis=1)
        b = np.linalg.norm(p0 - p2, axis=1)
        c = np.linalg.norm(p0 - p1, axis=1)

        degenerate = (a < 1e-12) | (b < 1e-12) | (c < 1e-12)
        # Guard the law-of-cosines division for degenerate triangles; their
        # skewness is overridden to the worst value (1.0) below regardless.
        safe_b = np.where(degenerate, 1.0, b)
        safe_c = np.where(degenerate, 1.0, c)
        safe_a = np.where(degenerate, 1.0, a)

        cos0 = np.clip((safe_b**2 + safe_c**2 - safe_a**2) / (2 * safe_b * safe_c), -1.0, 1.0)
        cos1 = np.clip((safe_a**2 + safe_c**2 - safe_b**2) / (2 * safe_a * safe_c), -1.0, 1.0)
        angle_0 = np.arccos(cos0)
        angle_1 = np.arccos(cos1)
        angle_2 = np.pi - angle_0 - angle_1

        angles_deg = np.degrees(np.stack([angle_0, angle_1, angle_2], axis=1))
        max_dev = np.max(np.abs(angles_deg - 60.0), axis=1)
        skewness = np.minimum(max_dev / 60.0, 1.0)
        skewness[degenerate] = 1.0

        return skewness

    def _compute_tetrahedron_skewness_values(self, nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """Skewness for every tetrahedron (simplified: aspect-ratio proxy).

        AR=1 -> skewness=0, AR>=10 -> skewness=1.
        """
        ar = self._compute_tetrahedron_aspect_ratios(nodes, cells)
        skewness = np.minimum((ar - 1.0) / 9.0, 1.0)
        return np.maximum(skewness, 0.0)
    
    def _check_orthogonality(
        self,
        report: MeshQualityReport,
        nodes: np.ndarray,
        cells: np.ndarray
    ) -> None:
        """Check face orthogonality.
        
        Measures angle between face normal and vector connecting cell centers.
        Perfect orthogonality = 90°.
        
        Args:
            report: Quality report to update
            nodes: Node coordinates
            cells: Cell connectivity
        """
        # This requires connectivity information (which faces belong to which cells)
        # For now, skip this check
        # TODO: Implement when face-to-cell connectivity is available
        pass
    
    def _evaluate_quality(self, report: MeshQualityReport) -> None:
        """Evaluate overall quality based on thresholds.
        
        Args:
            report: Quality report to evaluate
        """
        # Check critical failures
        if report.negative_volumes > self.thresholds['max_negative_volumes']:
            report.passed = False
            report.warnings.append(
                f"CRITICAL: {report.negative_volumes} cells with negative volume"
            )
        
        if report.volume_ratio > self.thresholds['max_volume_ratio']:
            report.passed = False
            report.warnings.append(
                f"HIGH: Volume ratio {report.volume_ratio:.2e} exceeds threshold "
                f"{self.thresholds['max_volume_ratio']:.2e}"
            )
        
        if report.max_aspect_ratio > self.thresholds['max_aspect_ratio']:
            report.warnings.append(
                f"MEDIUM: Max aspect ratio {report.max_aspect_ratio:.2f} exceeds "
                f"threshold {self.thresholds['max_aspect_ratio']:.2f}"
            )
        
        if report.max_skewness > self.thresholds['max_skewness']:
            report.warnings.append(
                f"MEDIUM: Max skewness {report.max_skewness:.4f} exceeds "
                f"threshold {self.thresholds['max_skewness']:.4f}"
            )
    
    def _generate_recommendations(self, report: MeshQualityReport) -> None:
        """Generate improvement recommendations based on quality issues.
        
        Args:
            report: Quality report with identified issues
        """
        if report.negative_volumes > 0:
            report.recommendations.append(
                "Fix negative volumes: Check surface mesh orientation and repair "
                "self-intersecting elements"
            )
        
        if report.volume_ratio > 1e4:
            report.recommendations.append(
                "Reduce volume ratio: Use smoother mesh grading or adaptive "
                "refinement to avoid abrupt size changes"
            )
        
        if report.max_aspect_ratio > 50:
            report.recommendations.append(
                "Improve aspect ratio: Refine highly stretched cells, especially "
                "near curved surfaces and sharp corners"
            )
        
        if report.max_skewness > 0.9:
            report.recommendations.append(
                "Reduce skewness: Improve mesh generation parameters, consider "
                "using different algorithm or smoothing"
            )
        
        if not report.recommendations and report.passed:
            report.recommendations.append("Mesh quality is good - no immediate action needed")
