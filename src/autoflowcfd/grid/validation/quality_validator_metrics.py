"""MeshQualityValidator 的体积/长宽比/扭曲度单体积指标检查。

从 quality_validator.py 中拆分出来（原文件超过 400 行的项目约定上限）：
_check_volumes / _compute_tetrahedron_volumes / _check_aspect_ratios /
_check_skewness / compute_cell_skewness 这一组方法都只是对
quality_metrics.py 里已有的向量化单元几何指标函数做统计汇总，不访问
MeshQualityValidator 实例的任何状态（原方法体内从未用到 self），逻辑上
是独立于类本身的一组"单元级指标计算"辅助函数，因此按纯函数搬移，不
携带 self/validator 参数。quality_validator.py 里对应方法改为转调用这里
的函数的薄包装，签名和行为完全不变。
"""

from typing import Dict, Optional, TYPE_CHECKING

import numpy as np
from loguru import logger

from . import quality_metrics as _qm
from .quality_report import MeshQualityReport

if TYPE_CHECKING:
    from ..structures import FaceData
    from .quality_validator import MeshQualityValidator


def check_volumes(
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
        volumes_array = _qm.compute_tetrahedron_volumes(nodes, cells)
    elif cell_type == "triangle":
        volumes_array = _qm.compute_triangle_areas(nodes, cells)
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


def compute_tetrahedron_volumes(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Thin wrapper over quality_metrics.compute_tetrahedron_volumes - kept
    for external callers (mesh_gen/mesh_repair.py's Stage A) that reach into
    MeshQualityValidator._compute_tetrahedron_volumes directly rather than
    importing the metric function themselves."""
    return _qm.compute_tetrahedron_volumes(nodes, cells)


def check_aspect_ratios(
    report: MeshQualityReport,
    nodes: np.ndarray,
    cells: np.ndarray,
    cell_type: str,
    bl_cell_mask: Optional[np.ndarray] = None,
) -> None:
    """Check cell aspect ratios (vectorized), optionally split by
    BL-region vs. core-region (see MeshQualityReport docstring for why
    these need separate thresholds).

    Aspect ratio = longest edge / shortest edge, for every cell at once.
    """
    if cell_type == "triangle":
        ar_array = _qm.compute_triangle_aspect_ratios(nodes, cells)
    elif cell_type == "tetrahedron":
        ar_array = _qm.compute_tetrahedron_aspect_ratios(nodes, cells)
    else:
        return

    if len(ar_array) > 0:
        report.min_aspect_ratio = float(np.min(ar_array))
        report.max_aspect_ratio = float(np.max(ar_array))
        report.mean_aspect_ratio = float(np.mean(ar_array))

    if bl_cell_mask is not None and len(bl_cell_mask) == len(ar_array):
        bl_cell_mask = np.asarray(bl_cell_mask, dtype=bool)
        bl_ar = ar_array[bl_cell_mask]
        core_ar = ar_array[~bl_cell_mask]
        if len(bl_ar) > 0:
            report.bl_max_aspect_ratio = float(np.max(bl_ar))
            report.bl_mean_aspect_ratio = float(np.mean(bl_ar))
        if len(core_ar) > 0:
            report.core_max_aspect_ratio = float(np.max(core_ar))
            report.core_mean_aspect_ratio = float(np.mean(core_ar))


def check_skewness(
    report: MeshQualityReport,
    nodes: np.ndarray,
    cells: np.ndarray,
    cell_type: str
) -> None:
    """Check cell skewness (vectorized).

    Triangles: based on angle deviation from 60 deg.
    Tetrahedra: radius-ratio shape measure (see
    _compute_tetrahedron_skewness_values).
    """
    if cell_type == "triangle":
        sk_array = _qm.compute_triangle_skewness_values(nodes, cells)
    elif cell_type == "tetrahedron":
        sk_array = _qm.compute_tetrahedron_skewness_values(nodes, cells)
    else:
        return

    if len(sk_array) > 0:
        report.max_skewness = float(np.max(sk_array))
        report.mean_skewness = float(np.mean(sk_array))


def compute_cell_skewness(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Public per-cell radius-ratio skewness array, shape=(n_cells,) -
    the raw values behind max_skewness/mean_skewness, for callers (e.g.
    the mesh repair loop in mesh_gen/mesh_repair.py) that need to know
    *which* cells are bad, not just aggregate statistics."""
    return _qm.compute_tetrahedron_skewness_values(nodes, cells)


def compute_face_diagnostics(
    validator: 'MeshQualityValidator',
    nodes: np.ndarray,
    cells: np.ndarray,
    faces: 'FaceData' = None,
    cell_centroids: Optional[np.ndarray] = None,
    cell_volumes: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Public per-internal-face diagnostics - the raw arrays behind
    orthogonality_max/adjacent_volume_ratio_max, for callers that need
    to know which faces/cells are implicated, not just aggregates.

    Args:
        validator: the owning MeshQualityValidator instance (only used
            for its `_extract_faces` staticmethod when `faces` isn't
            already supplied).
        cells: single uniform-width (n_cells, k) connectivity array,
            used to derive per-cell centroid/volume via the tet-only
            formula (nodes[cells].mean(axis=1), compute_tetrahedron_
            volumes) - IGNORED if cell_centroids/cell_volumes are both
            given directly instead (only the row count needs to match).
        cell_centroids, cell_volumes: optional pre-computed (n_cells,3)
            / (n_cells,) arrays, in the SAME global cell-index order
            `faces`' owner/neighbour indices use. Required for a mixed
            prism+tet mesh, where cells isn't a single uniform-width
            array at all (a prism's 6-node row and a tet's 4-node row
            can't share one connectivity array) - the caller (e.g.
            MeshQualityValidator.validate_volume_mesh's mixed-mesh
            path) computes these once per region with each region's
            own correct formula (PrismCells.compute_volumes vs.
            TetrahedralCells.compute_volumes) and concatenates.

    Returns a dict with, for every internal face:
        'owner', 'neighbor': cell indices (n_internal_faces,)
        'angle_deg': non-orthogonality angle, degrees (0=ideal)
        'volume_ratio': max(V)/min(V) between the two cells
    """
    if faces is None:
        faces = validator._extract_faces(nodes, cells)

    conn = faces.connectivity  # (n_faces, 2): [owner, neighbour], neighbour=-1 for boundary
    internal_mask = conn[:, 1] >= 0
    if not np.any(internal_mask):
        empty = np.array([], dtype=np.int64)
        return {'owner': empty, 'neighbor': empty, 'angle_deg': np.array([]), 'volume_ratio': np.array([])}

    owner = conn[internal_mask, 0]
    neigh = conn[internal_mask, 1]
    normal = faces.normal[internal_mask]

    centroids = cell_centroids if cell_centroids is not None else nodes[cells].mean(axis=1)
    d = centroids[neigh] - centroids[owner]
    d_norm = np.maximum(np.linalg.norm(d, axis=1), 1e-300)
    cos_angle = np.einsum('ij,ij->i', d, normal) / d_norm
    angle_deg = np.degrees(np.arccos(np.clip(np.abs(cos_angle), 0.0, 1.0)))

    volumes = cell_volumes if cell_volumes is not None else np.abs(_qm.compute_tetrahedron_volumes(nodes, cells))
    v_owner = volumes[owner]
    v_neigh = volumes[neigh]
    vmax = np.maximum(v_owner, v_neigh)
    vmin = np.maximum(np.minimum(v_owner, v_neigh), 1e-300)
    ratio = vmax / vmin

    return {'owner': owner, 'neighbor': neigh, 'angle_deg': angle_deg, 'volume_ratio': ratio}
