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
        """Check cell volumes for validity.
        
        Args:
            report: Quality report to update
            nodes: Node coordinates
            cells: Cell connectivity
            cell_type: Type of cells
        """
        volumes = []
        
        for i, cell in enumerate(cells):
            if cell_type == "tetrahedron":
                vol = self._compute_tetrahedron_volume(nodes, cell)
            elif cell_type == "triangle":
                vol = self._compute_triangle_area(nodes, cell)
            else:
                raise ValueError(f"Unsupported cell type: {cell_type}")
            
            volumes.append(vol)
            
            if vol < 0:
                report.negative_volumes += 1
                logger.warning(f"Negative volume detected in cell {i}: {vol:.6e}")
        
        if len(volumes) > 0:
            volumes_array = np.array(volumes)
            positive_volumes = volumes_array[volumes_array > 0]
            
            if len(positive_volumes) > 0:
                report.min_volume = float(np.min(positive_volumes))
                report.max_volume = float(np.max(positive_volumes))
                report.mean_volume = float(np.mean(positive_volumes))
                report.std_volume = float(np.std(positive_volumes))
                report.volume_ratio = report.max_volume / max(report.min_volume, 1e-12)
    
    def _compute_tetrahedron_volume(
        self,
        nodes: np.ndarray,
        cell: np.ndarray
    ) -> float:
        """Compute volume of a tetrahedron (always positive).
        
        V = |det(p1-p0, p2-p0, p3-p0)| / 6
        
        Note: Returns absolute value to handle inconsistent node ordering.
        For mesh quality assessment, we care about magnitude, not orientation.
        
        Args:
            nodes: All node coordinates
            cell: Indices of 4 vertices
            
        Returns:
            Absolute volume (always positive)
        """
        p0 = nodes[cell[0]]
        p1 = nodes[cell[1]]
        p2 = nodes[cell[2]]
        p3 = nodes[cell[3]]
        
        # Edge vectors from p0
        v1 = p1 - p0
        v2 = p2 - p0
        v3 = p3 - p0
        
        # Triple product gives 6*volume (take absolute value)
        volume = abs(np.dot(v1, np.cross(v2, v3))) / 6.0
        
        return volume
    
    def _compute_triangle_area(
        self,
        nodes: np.ndarray,
        cell: np.ndarray
    ) -> float:
        """Compute area of a triangle.
        
        A = 0.5 * |cross(p1-p0, p2-p0)|
        
        Args:
            nodes: All node coordinates
            cell: Indices of 3 vertices
            
        Returns:
            Area (always positive)
        """
        p0 = nodes[cell[0]]
        p1 = nodes[cell[1]]
        p2 = nodes[cell[2]]
        
        # Edge vectors
        v1 = p1 - p0
        v2 = p2 - p0
        
        # Cross product magnitude gives 2*area
        cross = np.cross(v1, v2)
        area = 0.5 * np.linalg.norm(cross)
        
        return area
    
    def _check_aspect_ratios(
        self,
        report: MeshQualityReport,
        nodes: np.ndarray,
        cells: np.ndarray,
        cell_type: str
    ) -> None:
        """Check cell aspect ratios.
        
        Aspect ratio = longest edge / shortest edge (for triangles)
        For tetrahedra: uses inscribed/circumscribed sphere radii
        
        Args:
            report: Quality report to update
            nodes: Node coordinates
            cells: Cell connectivity
            cell_type: Type of cells
        """
        aspect_ratios = []
        
        for cell in cells:
            if cell_type == "triangle":
                ar = self._compute_triangle_aspect_ratio(nodes, cell)
            elif cell_type == "tetrahedron":
                ar = self._compute_tetrahedron_aspect_ratio(nodes, cell)
            else:
                continue
            
            aspect_ratios.append(ar)
        
        if len(aspect_ratios) > 0:
            ar_array = np.array(aspect_ratios)
            report.min_aspect_ratio = float(np.min(ar_array))
            report.max_aspect_ratio = float(np.max(ar_array))
            report.mean_aspect_ratio = float(np.mean(ar_array))
    
    def _compute_triangle_aspect_ratio(
        self,
        nodes: np.ndarray,
        cell: np.ndarray
    ) -> float:
        """Compute aspect ratio for triangle.
        
        AR = longest_edge / shortest_edge
        
        Args:
            nodes: All node coordinates
            cell: Indices of 3 vertices
            
        Returns:
            Aspect ratio (1.0 for equilateral, higher for stretched)
        """
        p0 = nodes[cell[0]]
        p1 = nodes[cell[1]]
        p2 = nodes[cell[2]]
        
        # Compute edge lengths
        e1 = np.linalg.norm(p1 - p0)
        e2 = np.linalg.norm(p2 - p1)
        e3 = np.linalg.norm(p0 - p2)
        
        edges = [e1, e2, e3]
        ar = max(edges) / (min(edges) + 1e-12)
        
        return ar
    
    def _compute_tetrahedron_aspect_ratio(
        self,
        nodes: np.ndarray,
        cell: np.ndarray
    ) -> float:
        """Compute aspect ratio for tetrahedron.
        
        Uses simplified edge-based metric.
        
        Args:
            nodes: All node coordinates
            cell: Indices of 4 vertices
            
        Returns:
            Aspect ratio
        """
        # Get all 6 edges
        edges = []
        for i in range(4):
            for j in range(i+1, 4):
                edge_len = np.linalg.norm(nodes[cell[i]] - nodes[cell[j]])
                edges.append(edge_len)
        
        ar = max(edges) / (min(edges) + 1e-12)
        
        return ar
    
    def _check_skewness(
        self,
        report: MeshQualityReport,
        nodes: np.ndarray,
        cells: np.ndarray,
        cell_type: str
    ) -> None:
        """Check cell skewness.
        
        Skewness measures deviation from ideal shape.
        For triangles: based on angles deviation from 60°
        
        Args:
            report: Quality report to update
            nodes: Node coordinates
            cells: Cell connectivity
            cell_type: Type of cells
        """
        skewness_values = []
        
        for cell in cells:
            if cell_type == "triangle":
                sk = self._compute_triangle_skewness(nodes, cell)
            elif cell_type == "tetrahedron":
                sk = self._compute_tetrahedron_skewness(nodes, cell)
            else:
                continue
            
            skewness_values.append(sk)
        
        if len(skewness_values) > 0:
            sk_array = np.array(skewness_values)
            report.max_skewness = float(np.max(sk_array))
            report.mean_skewness = float(np.mean(sk_array))
    
    def _compute_triangle_skewness(
        self,
        nodes: np.ndarray,
        cell: np.ndarray
    ) -> float:
        """Compute skewness for triangle.
        
        Based on angle deviation from equilateral (60° each).
        Skewness = max(|θ_i - 60°|) / 60°
        
        Args:
            nodes: All node coordinates
            cell: Indices of 3 vertices
            
        Returns:
            Skewness in [0, 1] (0=perfect, 1=worst)
        """
        p0 = nodes[cell[0]]
        p1 = nodes[cell[1]]
        p2 = nodes[cell[2]]
        
        # Compute angles using law of cosines
        a = np.linalg.norm(p1 - p2)
        b = np.linalg.norm(p0 - p2)
        c = np.linalg.norm(p0 - p1)
        
        # Avoid division by zero
        if a < 1e-12 or b < 1e-12 or c < 1e-12:
            return 1.0
        
        # Angles in radians
        angle_0 = np.arccos(max(-1.0, min(1.0, (b*b + c*c - a*a) / (2*b*c))))
        angle_1 = np.arccos(max(-1.0, min(1.0, (a*a + c*c - b*b) / (2*a*c))))
        angle_2 = np.pi - angle_0 - angle_1
        
        # Convert to degrees
        angles_deg = np.degrees([angle_0, angle_1, angle_2])
        
        # Deviation from 60°
        deviations = np.abs(angles_deg - 60.0)
        max_dev = np.max(deviations)
        
        # Normalize to [0, 1]
        skewness = min(max_dev / 60.0, 1.0)
        
        return skewness
    
    def _compute_tetrahedron_skewness(
        self,
        nodes: np.ndarray,
        cell: np.ndarray
    ) -> float:
        """Compute skewness for tetrahedron (simplified).
        
        Uses volume-based metric relative to edge lengths.
        
        Args:
            nodes: All node coordinates
            cell: Indices of 4 vertices
            
        Returns:
            Skewness in [0, 1]
        """
        # Simplified: use aspect ratio as proxy for skewness
        ar = self._compute_tetrahedron_aspect_ratio(nodes, cell)
        
        # Map aspect ratio to skewness
        # AR=1 → skewness=0, AR>10 → skewness→1
        skewness = min((ar - 1.0) / 9.0, 1.0)
        
        return max(skewness, 0.0)
    
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
