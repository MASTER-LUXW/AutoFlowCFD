"""质量报告的 pass/fail 判定与改进建议文本生成。

从 quality_validator.py 拆分出来：evaluate_quality 只依据阈值字典判定
MeshQualityReport 是否通过、写警告文本；generate_recommendations 只根据
report 里已有的问题字段生成人类可读的改进建议。两者都只需要
report + thresholds，不依赖 MeshQualityValidator 实例的其它状态，所以
拆成自由函数而不是留在类里。
"""

from .quality_report import MeshQualityReport


def evaluate_quality(report: MeshQualityReport, thresholds: dict) -> None:
    """Evaluate overall quality based on thresholds.

    Args:
        report: Quality report to evaluate
        thresholds: MeshQualityValidator.thresholds
    """
    # Check critical failures
    if report.negative_volumes > thresholds['max_negative_volumes']:
        report.passed = False
        report.warnings.append(
            f"CRITICAL: {report.negative_volumes} cells with negative volume"
        )

    # Global volume ratio is informational only now - see
    # MeshQualityReport docstring for why it doesn't gate.
    if report.volume_ratio > thresholds['max_volume_ratio']:
        report.warnings.append(
            f"INFO: Global volume ratio {report.volume_ratio:.2e} exceeds "
            f"{thresholds['max_volume_ratio']:.2e} - expected for a "
            f"graded BL mesh, not itself a defect. See adjacent-cell "
            f"volume ratio below for the metric that actually matters."
        )

    if report.adjacent_volume_ratio_max > thresholds['max_adjacent_volume_ratio']:
        report.passed = False
        report.warnings.append(
            f"HIGH: Max adjacent-cell volume ratio {report.adjacent_volume_ratio_max:.2f} "
            f"exceeds threshold {thresholds['max_adjacent_volume_ratio']:.2f} - "
            f"Green-Gauss gradient reconstruction (grad ~ 1/V) will be severely "
            f"ill-conditioned at these cells, and will pollute their neighbours' fluxes"
        )

    if report.orthogonality_max > thresholds['max_orthogonality_angle']:
        report.passed = False
        report.warnings.append(
            f"HIGH: Max non-orthogonality {report.orthogonality_max:.1f} deg exceeds "
            f"threshold {thresholds['max_orthogonality_angle']:.1f} deg"
        )

    if report.bl_max_aspect_ratio is not None and report.bl_max_aspect_ratio > thresholds['bl_max_aspect_ratio']:
        report.warnings.append(
            f"MEDIUM: BL-region max aspect ratio {report.bl_max_aspect_ratio:.2f} exceeds "
            f"threshold {thresholds['bl_max_aspect_ratio']:.2f}"
        )
    if report.core_max_aspect_ratio is not None and report.core_max_aspect_ratio > thresholds['core_max_aspect_ratio']:
        report.warnings.append(
            f"MEDIUM: Core-region max aspect ratio {report.core_max_aspect_ratio:.2f} exceeds "
            f"threshold {thresholds['core_max_aspect_ratio']:.2f}"
        )
    if report.bl_max_aspect_ratio is None and report.max_aspect_ratio > thresholds['max_aspect_ratio']:
        report.warnings.append(
            f"MEDIUM: Max aspect ratio {report.max_aspect_ratio:.2f} exceeds "
            f"threshold {thresholds['max_aspect_ratio']:.2f}"
        )

    if report.max_skewness > thresholds['max_skewness']:
        report.passed = False
        report.warnings.append(
            f"HIGH: Max skewness {report.max_skewness:.4f} exceeds threshold "
            f"{thresholds['max_skewness']:.4f}"
        )

    if report.n_overlapping_cells > thresholds['max_overlapping_cells']:
        report.passed = False
        report.warnings.append(
            f"CRITICAL: {report.n_overlapping_cells} cells physically overlap a "
            f"different, non-adjacent cell's faces"
        )
    if report.n_close_cell_pairs > 0:
        report.warnings.append(
            f"INFO: {report.n_close_cell_pairs} cell pair(s) are close enough to "
            f"overlap with a small further parameter change "
            + (f"(min gap {report.overlap_min_gap:.3e} m)" if report.overlap_min_gap is not None else "")
            + " - not a defect by itself, see summary for details"
        )


def generate_recommendations(report: MeshQualityReport, thresholds: dict) -> None:
    """Generate improvement recommendations based on quality issues.

    Args:
        report: Quality report with identified issues
        thresholds: MeshQualityValidator.thresholds
    """
    if report.negative_volumes > 0:
        report.recommendations.append(
            "Fix negative volumes: Check surface mesh orientation and repair "
            "self-intersecting elements"
        )

    if report.n_overlapping_cells > 0:
        report.recommendations.append(
            "Fix overlapping cells: usually a BL extrusion front crossing a "
            "facing surface (tight underbody-to-ground gaps) or a core-fill "
            "artifact at a tight BL seam - see the mesh repair loop "
            "(mesh_gen/mesh_repair.py) for automated local cavity re-tiling"
        )

    if report.adjacent_volume_ratio_max > thresholds['max_adjacent_volume_ratio']:
        report.recommendations.append(
            "Reduce adjacent-cell volume ratio: usually caused by degenerate "
            "(sliver) tetrahedra at sharp convex edges/corners of the body, or "
            "abrupt tetgen size-grading transitions in the core fill - see the "
            "mesh repair loop (mesh_gen/mesh_repair.py) for automated fixes"
        )

    if report.orthogonality_max > thresholds['max_orthogonality_angle']:
        report.recommendations.append(
            "Reduce non-orthogonality: smooth or locally re-mesh the implicated "
            "cells - Green-Gauss gradient accuracy degrades sharply beyond this"
        )

    if (report.bl_max_aspect_ratio or 0) > thresholds['bl_max_aspect_ratio']:
        report.recommendations.append(
            "Improve BL-region aspect ratio: reduce growth_rate or first-layer "
            "min_cell_size"
        )
    if (report.core_max_aspect_ratio or report.max_aspect_ratio) > thresholds.get('core_max_aspect_ratio', thresholds['max_aspect_ratio']):
        report.recommendations.append(
            "Improve core-region aspect ratio: tighten max_cell_size grading"
        )

    if report.max_skewness > 0.9:
        report.recommendations.append(
            "Reduce skewness: improve mesh generation parameters, consider "
            "using different algorithm or smoothing"
        )

    if not report.recommendations and report.passed:
        report.recommendations.append("Mesh quality is good - no immediate action needed")
