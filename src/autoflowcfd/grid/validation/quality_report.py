"""Mesh quality report data structure.

MeshQualityReport is MeshQualityValidator's (quality_validator.py) output
type - split into its own module so quality_validator.py (the check
implementations) doesn't also have to carry this large a dataclass/summary
formatter inline. GridValidator (validator.py) is a separate, simpler
surface-mesh checker with its own plain-dict return shape, unrelated to
this class.
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field


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
        volume_ratio: Global max/min volume ratio - reference/informational
            only (see module docstring - a BL mesh's legitimate near-wall-
            to-far-field range dominates this and swamps any real local
            defect signal); adjacent_volume_ratio_max is the metric that
            actually gates.
        min_aspect_ratio: Minimum aspect ratio
        max_aspect_ratio: Maximum aspect ratio (whole mesh)
        mean_aspect_ratio: Mean aspect ratio (whole mesh)
        bl_max_aspect_ratio: Max aspect ratio within BL-region cells only
            (None unless bl_cell_mask was supplied to validate())
        core_max_aspect_ratio: Max aspect ratio within core-region cells
        max_skewness: Maximum radius-ratio skewness (0=regular tet, 1=sliver)
        mean_skewness: Mean skewness
        orthogonality_max: Worst (largest) face-normal vs. centroid-
            connector angle across all internal faces, degrees (0=ideal)
        orthogonality_mean: Mean of the same
        adjacent_volume_ratio_max: Worst face-neighbour volume ratio
            (max(V)/min(V) across the two cells sharing a face) - the
            metric that actually governs Green-Gauss conditioning
        adjacent_volume_ratio_mean: Mean of the same
        n_overlapping_cells: Cells with at least one face that physically
            overlaps a DIFFERENT, non-adjacent cell's face (see
            mesh_gen/../validation/mesh_overlap_check.py) - a distinct
            defect class from a negative/degenerate cell: both cells
            involved can individually have perfectly valid positive
            volume and reasonable shape, the problem is that they occupy
            overlapping physical space. None unless check_overlap=True was
            passed to validate() (it is the one check here whose cost
            scales with local mesh density, not purely cell count, so it
            is opt-out rather than always-silently-skipped).
        n_close_cell_pairs: Cells whose faces don't yet overlap but are
            closer than a locally-scaled threshold - informational only,
            not a defect by itself (this is deliberately how
            mesh_tetgen_core.compute_local_thickness_limit already tries
            to prevent BL fronts from crossing at generation time; this is
            the post-hoc visibility check for whatever that heuristic
            didn't fully prevent).
        overlap_min_gap: Smallest non-overlapping face-to-face distance
            found among close pairs, meters (None if none found/checked)
        warnings: List of quality warnings
        recommendations: List of improvement recommendations
        repair_stages_applied: Human-readable log of repair actions taken
            before this report was produced (see mesh_gen/mesh_repair.py) -
            empty for a bare validate() call with no repair attempted.
        initial_report: The report *before* any repair was attempted, for
            a before/after comparison in summary() - None if this report
            itself is the pre-repair baseline, or no repair was run.
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
    bl_max_aspect_ratio: Optional[float] = None
    bl_mean_aspect_ratio: Optional[float] = None
    core_max_aspect_ratio: Optional[float] = None
    core_mean_aspect_ratio: Optional[float] = None

    # Skewness metrics (radius-ratio based)
    max_skewness: float = 0.0
    mean_skewness: float = 0.0

    # Orthogonality metrics
    orthogonality_max: float = 0.0
    orthogonality_mean: float = 0.0

    # Adjacent-cell (face-neighbour) volume ratio
    adjacent_volume_ratio_max: float = 0.0
    adjacent_volume_ratio_mean: float = 0.0

    # Cell overlap / near-touching-face metrics (see mesh_overlap_check.py)
    n_overlapping_cells: int = 0
    n_close_cell_pairs: int = 0
    overlap_min_gap: Optional[float] = None
    # (n_overlapping_cells,) int64 - for a caller (e.g. the mesh repair
    # loop) that needs to know WHICH cells, not just the count; avoids
    # re-running the overlap check a second time just to recover this.
    overlapping_cell_ids: Optional[np.ndarray] = None

    # Qualitative feedback
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Repair-loop bookkeeping
    repair_stages_applied: List[str] = field(default_factory=list)
    initial_report: Optional['MeshQualityReport'] = None

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
            f"  Global max/min ratio: {self.volume_ratio:.2e} (informational - BL grading, not a defect by itself)",
            f"  Adjacent-cell (face-neighbour) ratio: max={self.adjacent_volume_ratio_max:.2f}, mean={self.adjacent_volume_ratio_mean:.2f}",
            "",
            f"Aspect Ratio:",
            f"  Overall: [{self.min_aspect_ratio:.3f}, {self.max_aspect_ratio:.3f}], mean={self.mean_aspect_ratio:.3f}",
        ]
        if self.bl_max_aspect_ratio is not None:
            lines.append(f"  BL region:   max={self.bl_max_aspect_ratio:.3f}, mean={self.bl_mean_aspect_ratio:.3f}")
        if self.core_max_aspect_ratio is not None:
            lines.append(f"  Core region: max={self.core_max_aspect_ratio:.3f}, mean={self.core_mean_aspect_ratio:.3f}")
        lines += [
            "",
            f"Skewness (radius-ratio, 0=regular tet .. 1=sliver):",
            f"  Max: {self.max_skewness:.4f}",
            f"  Mean: {self.mean_skewness:.4f}",
            "",
            f"Orthogonality (face-normal vs. centroid-connector angle, 0deg=ideal):",
            f"  Max: {self.orthogonality_max:.2f} deg",
            f"  Mean: {self.orthogonality_mean:.2f} deg",
        ]

        if self.n_overlapping_cells > 0 or self.n_close_cell_pairs > 0:
            lines += [
                "",
                f"Cell Overlap / Proximity:",
                f"  Overlapping cells: {self.n_overlapping_cells}",
                f"  Near-touching cell pairs: {self.n_close_cell_pairs}"
                + (f" (min gap {self.overlap_min_gap:.3e} m)" if self.overlap_min_gap is not None else ""),
            ]

        if self.initial_report is not None:
            ir = self.initial_report
            lines += [
                "",
                "Before/After Repair Comparison:",
                f"  Status:                {'PASSED' if ir.passed else 'FAILED'} -> {'PASSED' if self.passed else 'FAILED'}",
                f"  Max skewness:           {ir.max_skewness:.4f} -> {self.max_skewness:.4f}",
                f"  Max non-orthogonality:  {ir.orthogonality_max:.2f} deg -> {self.orthogonality_max:.2f} deg",
                f"  Max adjacent vol ratio: {ir.adjacent_volume_ratio_max:.2f} -> {self.adjacent_volume_ratio_max:.2f}",
                f"  Negative volumes:       {ir.negative_volumes} -> {self.negative_volumes}",
                f"  Overlapping cells:      {ir.n_overlapping_cells} -> {self.n_overlapping_cells}",
            ]

        if self.repair_stages_applied:
            lines.append("")
            lines.append("Repair Actions Applied:")
            for i, action in enumerate(self.repair_stages_applied, 1):
                lines.append(f"  {i}. {action}")

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


