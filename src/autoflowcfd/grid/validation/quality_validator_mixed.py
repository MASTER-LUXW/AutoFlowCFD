"""混合网格（棱柱 BL + 四面体 core）质量校验实现。

从 quality_validator.py 中拆分出来（原文件超过 400 行的项目约定上限）：
MeshQualityValidator.validate_mixed 是该文件中最长的单个方法（原地 ~104
行），且逻辑上自成一段——专门处理"棱柱+四面体混合体网格"这一种输入
形态，与 validate()/validate_volume_mesh() 处理的均匀单一单元类型网格
是并列而非嵌套的关系。拆分只是纯粹的代码搬移：把原方法体逐字迁移为一个
以 validator 实例为第一个参数的模块级函数，quality_validator.py 里只保留
一个转调用的薄包装方法，任何行为/数值逻辑均未改动。

同时一并迁移了两个只被 validate_mixed 使用的模块级几何辅助函数
（_compute_prism_centroids/_compute_tet_centroids），避免它们在原文件里
变成无人问津的孤儿定义。
"""

import numpy as np
from typing import TYPE_CHECKING
from loguru import logger

from . import quality_metrics as _qm
from .quality_evaluation import evaluate_quality, generate_recommendations

if TYPE_CHECKING:
    from ..structures import FaceData, VolumeMeshData
    from .quality_report import MeshQualityReport
    from .quality_validator import MeshQualityValidator


def _compute_tet_centroids(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """每个四面体的顶点平均质心，shape=(n_cells, 3)。"""
    if len(cells) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return nodes[cells].mean(axis=1)


def _compute_prism_centroids(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """每个三棱柱的顶点平均质心，shape=(n_cells, 3)。"""
    if len(cells) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return nodes[cells].mean(axis=1)


def validate_mixed_mesh(
    validator: 'MeshQualityValidator',
    volume_mesh: 'VolumeMeshData',
    faces: 'FaceData' = None,
    log_summary: bool = True,
    check_overlap: bool = True,
) -> 'MeshQualityReport':
    """校验混合棱柱(BL) + 四面体(core) VolumeMeshData。

    结构与 validate() 相同，但每个 per-cell 指标按区域分别计算
    （棱柱单元用 quality_metrics 的棱柱函数，四面体单元用已有的
    四面体函数——两种形状需要完全不同的公式，见 quality_metrics.py），
    然后按所有其他棱柱感知代码使用的全局单元索引顺序拼接
    （棱柱 [0, n_prism)，四面体 [n_prism, n_prism+n_tet)——
    见 PrismCells / face_extractor.extract_faces_mixed）。
    正交性和相邻体积比（基于面，因此天然跨越 BL/core 界面）
    使用全局面图的一次合并遍历，通过 compute_face_diagnostics 的
    cell_centroids/cell_volumes 参数（专门为此添加，避免从混合网格
    没有的单一均匀连接关系数组重新推导 per-cell 质心/体积）。

    bl_cell_mask 不是这里的参数（与 validate() 不同）——按构造它
    恰好是 [True]*n_prism + [False]*n_tet，调用者无法有意义地覆盖。

    Args:
        validator: 所属的 MeshQualityValidator 实例（此函数最初是
            MeshQualityValidator.validate_mixed；现在作为薄包装委托从该方法调用）。
    """
    from .quality_report import MeshQualityReport

    nodes = np.column_stack([volume_mesh.nodes.x, volume_mesh.nodes.y, volume_mesh.nodes.z])
    prism_conn = volume_mesh.prism_cells.connectivity
    tet_conn = volume_mesh.cells.connectivity
    n_prism = len(prism_conn)
    n_tet = len(tet_conn)
    n_cells = n_prism + n_tet

    logger.info(f"Validating mesh quality: {n_prism} prism + {n_tet} tetrahedron cells...")

    report = MeshQualityReport(n_cells=n_cells, n_nodes=len(nodes))

    # --- Volumes ---
    prism_vol = _qm.compute_prism_volumes(nodes, prism_conn)
    tet_vol_signed = _qm.compute_tetrahedron_volumes(nodes, tet_conn)
    negative_mask = tet_vol_signed < 0  # 棱柱体积按构造为无符号——
                                        # 见 compute_prism_volumes 的文档字符串，
                                        # 了解为何它们尚无反转检查
    report.negative_volumes = int(np.sum(negative_mask))
    all_volumes = np.concatenate([prism_vol, np.abs(tet_vol_signed)])
    positive_volumes = all_volumes[all_volumes > 0]
    if len(positive_volumes) > 0:
        report.min_volume = float(np.min(positive_volumes))
        report.max_volume = float(np.max(positive_volumes))
        report.mean_volume = float(np.mean(positive_volumes))
        report.std_volume = float(np.std(positive_volumes))
        report.volume_ratio = report.max_volume / max(report.min_volume, 1e-12)

    # --- Aspect ratio (BL/core split is exact here, not heuristic) ---
    prism_ar = _qm.compute_prism_aspect_ratios(nodes, prism_conn)
    tet_ar = _qm.compute_tetrahedron_aspect_ratios(nodes, tet_conn)
    all_ar = np.concatenate([prism_ar, tet_ar])
    if len(all_ar) > 0:
        report.min_aspect_ratio = float(np.min(all_ar))
        report.max_aspect_ratio = float(np.max(all_ar))
        report.mean_aspect_ratio = float(np.mean(all_ar))
    if n_prism > 0:
        report.bl_max_aspect_ratio = float(np.max(prism_ar))
        report.bl_mean_aspect_ratio = float(np.mean(prism_ar))
    if n_tet > 0:
        report.core_max_aspect_ratio = float(np.max(tet_ar))
        report.core_mean_aspect_ratio = float(np.mean(tet_ar))

    # --- Skewness ---
    prism_sk = _qm.compute_prism_skewness_values(nodes, prism_conn)
    tet_sk = _qm.compute_tetrahedron_skewness_values(nodes, tet_conn)
    all_sk = np.concatenate([prism_sk, tet_sk])
    if len(all_sk) > 0:
        report.max_skewness = float(np.max(all_sk))
        report.mean_skewness = float(np.mean(all_sk))

    # --- Orthogonality / adjacent-volume-ratio / overlap (face-based) ---
    if faces is None:
        faces = volume_mesh.ensure_faces_exist()
    cell_centroids = np.vstack([
        _compute_prism_centroids(nodes, prism_conn),
        _compute_tet_centroids(nodes, tet_conn),
    ]) if n_prism > 0 else _compute_tet_centroids(nodes, tet_conn.astype(np.int64))
    cell_volumes = all_volumes  # already concatenated prism+tet, same global order

    validator._check_orthogonality_and_adjacency(
        report, nodes, tet_conn, faces, cell_centroids=cell_centroids, cell_volumes=cell_volumes
    )
    if check_overlap:
        # `cells` 在 faces 已提供时（这里总是如此）不会被
        # check_face_overlap_and_proximity 使用——见
        # mesh_overlap_check.py，它仅在调用者尚未拥有面时
        # 才读取 `cells` 来推导面。
        validator._check_overlap_and_proximity(report, nodes, tet_conn, faces)

    evaluate_quality(report, validator.thresholds)
    generate_recommendations(report, validator.thresholds)

    if log_summary:
        logger.info(f"\n{report.summary()}")

    return report
