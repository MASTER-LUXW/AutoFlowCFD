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
    """检查单元体积的有效性（向量化）。

    纯 Python 的 per-cell 循环在此不可扩展：汽车体网格通常
    具有数十万到数百万个四面体，而此方法之前是在 Python 中
    逐个遍历它们。

    Args:
        report: 要更新的质量报告
        nodes: 节点坐标
        cells: 单元连接关系
        cell_type: 单元类型
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
    """quality_metrics.compute_tetrahedron_volumes 的薄包装
    ——保留给外部调用者（例如 mesh_gen/mesh_repair.py 的 Stage A），
    它们直接访问 MeshQualityValidator._compute_tetrahedron_volumes
    而不是自行导入度量函数。"""
    return _qm.compute_tetrahedron_volumes(nodes, cells)


def check_aspect_ratios(
    report: MeshQualityReport,
    nodes: np.ndarray,
    cells: np.ndarray,
    cell_type: str,
    bl_cell_mask: Optional[np.ndarray] = None,
) -> None:
    """检查单元长宽比（向量化），可选按 BL 区域与核心区域拆分
    （见 MeshQualityReport 文档字符串了解为何需要分离阈值）。

    长宽比 = 最长边 / 最短边，对所有单元一次性计算。
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
    """检查单元偏斜度（向量化）。

    三角形：基于角度与 60 度的偏差。
    四面体：半径比形状度量（见 compute_tetrahedron_skewness_values）。
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
    """公共 per-cell 半径比偏斜度数组, shape=(n_cells,)——
    max_skewness/mean_skewness 背后的原始值，供需要知道
    *哪些*单元有问题（而非仅聚合统计）的调用者使用
    （例如 mesh_gen/mesh_repair.py 中的网格修复循环）。"""
    return _qm.compute_tetrahedron_skewness_values(nodes, cells)


def compute_face_diagnostics(
    validator: 'MeshQualityValidator',
    nodes: np.ndarray,
    cells: np.ndarray,
    faces: 'FaceData' = None,
    cell_centroids: Optional[np.ndarray] = None,
    cell_volumes: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """公共 per-内部面诊断——orthogonality_max/adjacent_volume_ratio_max
    背后的原始数组，供需要知道哪些面/单元涉及（而非仅聚合值）的调用者使用。

    Args:
        validator: 所属的 MeshQualityValidator 实例（仅用于
            其 `_extract_faces` 静态方法，当 `faces` 未预提供时）。
        cells: 单一均匀宽度的 (n_cells, k) 连接关系数组，
            用于通过四面体公式推导 per-cell 质心/体积
            （nodes[cells].mean(axis=1)、compute_tetrahedron_
            volumes）——如果同时直接提供了 cell_centroids/cell_volumes
            则忽略（仅需行数匹配）。
        cell_centroids, cell_volumes: 可选的预计算 (n_cells,3)
            / (n_cells,) 数组，按 `faces` 的 owner/neighbour 索引
            使用的相同全局单元索引顺序。对混合棱柱+四面体网格必需，
            其中 `cells` 根本不是单一均匀宽度的数组（棱柱的 6 节点行
            和四面体的 4 节点行无法共享一个连接关系数组）——
            调用者（例如 MeshQualityValidator.validate_volume_mesh
            的混合网格路径）用每个区域各自的正确公式
            （PrismCells.compute_volumes vs. TetrahedralCells.compute_volumes）
            按区域计算一次并拼接。

    返回字典，对每个内部面包含：
        'owner', 'neighbor': 单元索引 (n_internal_faces,)
        'angle_deg': 非正交角，角度制（0=理想）
        'volume_ratio': 两单元间 max(V)/min(V)
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
