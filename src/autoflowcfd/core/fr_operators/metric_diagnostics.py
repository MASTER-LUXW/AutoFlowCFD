"""
AutoFlowCFD V2.0 - 网格度量诊断（网格加载时打印一次，不修改残差）。

两个与物理尺寸无关的诊断量：

1. **缩放雅可比形状质量** `det(J) / prod_m |dx/dxi_m|`（`compute_scaled_jacobian_quality`）：1 对应正交映射，越接近
   0 越扭曲。正常单元约 0.55~0.92，网格生成产生的畸形单元约 1e-4~1e-2。
2. **面法向失配**（`precompute_cell_face_misalignment`）：每个单元自身映射在面通量点上的外法向（精确 adj(J) 行）与
   该面公共法向 `true_normal`（取自 owner 侧）的最大偏离 `1 - cos(夹角)`。owner 侧恒为零；非零值来自 neighbor 侧，
   典型来源是棱柱四边形侧面（双线性曲面）与相邻四面体平面三角形之间的真实几何不协调。

退化单元在运行期没有任何自动干预——对策是求解前的网格质量门（`grid/validation/solve_gate.py`）。这里的报告用于
判断是否需要改善网格（棱柱-四面体过渡区尺寸梯度约束等）。
"""

from typing import Dict, Optional

import numpy as np
from loguru import logger
from numba import njit

#: 形状质量预警阈值（缩放雅可比的单元内最小值）
SHAPE_QUALITY_WARN_THRESHOLD = 0.1

#: 面法向失配预警阈值（度）
FACE_MISALIGNMENT_WARN_DEG = 1.0
_FACE_MISALIGNMENT_WARN = 1.0 - np.cos(np.radians(FACE_MISALIGNMENT_WARN_DEG))


def compute_scaled_jacobian_quality(jacobians: np.ndarray, det_jacs: np.ndarray) -> np.ndarray:
    """缩放雅可比形状质量，形状与 `det_jacs` 相同。

    Args:
        jacobians: 雅可比矩阵，形状 (...,3,3)，`jacobians[...,i,m] = dx_i/dxi_m`（列 m 是参考方向 m 的物理切向量）
        det_jacs: 形状 (...,)，与 jacobians 对应的行列式
    """
    col_norms = np.linalg.norm(jacobians, axis=-2)  # (...,3): 每个参考方向切向量的模长
    prod = np.prod(col_norms, axis=-1)
    return det_jacs / np.maximum(prod, 1e-300)


@njit(cache=True)
def _cell_face_misalignment_kernel(
    owner_cell: np.ndarray, neighbor_cell: np.ndarray, is_boundary: np.ndarray,
    owner_is_primary: np.ndarray, neighbor_is_primary: np.ndarray,
    true_normal: np.ndarray,
    owner_adj_row_exact: np.ndarray, neighbor_adj_row_exact: np.ndarray,
    n_faces: int, n_fp: int, n_cells: int,
) -> np.ndarray:
    """逐单元 `max(1 - dir(adj_row) . n_outward)`；原生面的 adj 行已外向定向，neighbor 侧外法向为 `-true_normal`。"""
    cell_misalign = np.zeros(n_cells)
    for f in range(n_faces):
        if owner_is_primary[f]:
            oc = owner_cell[f]
            worst = 0.0
            for i in range(n_fp):
                rx = owner_adj_row_exact[f, i, 0]
                ry = owner_adj_row_exact[f, i, 1]
                rz = owner_adj_row_exact[f, i, 2]
                mag = np.sqrt(rx * rx + ry * ry + rz * rz)
                mag = mag if mag > 1e-300 else 1e-300
                dot = (rx * true_normal[f, i, 0] + ry * true_normal[f, i, 1] + rz * true_normal[f, i, 2]) / mag
                m = 1.0 - dot
                if m > worst:
                    worst = m
            if worst > cell_misalign[oc]:
                cell_misalign[oc] = worst
        if (not is_boundary[f]) and neighbor_is_primary[f]:
            nc = neighbor_cell[f]
            worst = 0.0
            for i in range(n_fp):
                rx = neighbor_adj_row_exact[f, i, 0]
                ry = neighbor_adj_row_exact[f, i, 1]
                rz = neighbor_adj_row_exact[f, i, 2]
                mag = np.sqrt(rx * rx + ry * ry + rz * rz)
                mag = mag if mag > 1e-300 else 1e-300
                dot = -(rx * true_normal[f, i, 0] + ry * true_normal[f, i, 1] + rz * true_normal[f, i, 2]) / mag
                m = 1.0 - dot
                if m > worst:
                    worst = m
            if worst > cell_misalign[nc]:
                cell_misalign[nc] = worst
    return cell_misalign


def precompute_cell_face_misalignment(mesh) -> np.ndarray:
    """每个单元自身连接的所有面上，自身外法向相对该面 `true_normal` 的最大偏离量 `1-cos(夹角)`，形状 (n_cells,)。

    每一侧独立计算（不需要跨单元查找），棱柱四边形侧面拆成 2 条子面记录时每条记录各自贡献。直接读展平面几何
    （`get_flat_face_geometry`）的数组交给 numba 核。
    """
    from autoflowcfd.core.fr_operators.face_kernels import get_flat_face_geometry

    flat = get_flat_face_geometry(mesh, mesh.operators)
    return _cell_face_misalignment_kernel(
        flat.owner_cell, flat.neighbor_cell, flat.is_boundary,
        flat.owner_is_primary, flat.neighbor_is_primary,
        flat.true_normal,
        flat.owner_adj_row_exact, flat.neighbor_adj_row_exact,
        flat.n_faces, flat.n_fp, mesh.n_cells,
    )


def log_mesh_metric_report(scaled_quality: np.ndarray,
                           cell_face_misalignment: Optional[np.ndarray] = None) -> Dict[str, int]:
    """打印网格度量诊断并返回计数 `{"n_cells", "n_shape_warn", "n_misaligned"}`（未给失配量时 `n_misaligned` 为 0）。

    Args:
        scaled_quality: (n_cells, n_sps) 逐解点缩放雅可比
        cell_face_misalignment: (n_cells,) `precompute_cell_face_misalignment` 的结果
    """
    n_cells = int(scaled_quality.shape[0])
    n_shape_warn = int(np.sum(scaled_quality.min(axis=1) < SHAPE_QUALITY_WARN_THRESHOLD))
    n_misaligned = (0 if cell_face_misalignment is None
                    else int(np.sum(cell_face_misalignment > _FACE_MISALIGNMENT_WARN)))
    stats = {"n_cells": n_cells, "n_shape_warn": n_shape_warn, "n_misaligned": n_misaligned}
    if n_shape_warn == 0 and n_misaligned == 0:
        logger.info(f"网格度量诊断：{n_cells} 个单元的形状质量与面法向一致性均正常")
        return stats
    logger.warning(
        f"网格度量诊断（仅诊断，运行期不干预）：{n_shape_warn}/{n_cells}（{100 * n_shape_warn / n_cells:.3f}%）个单元的缩放"
        f"雅可比形状质量 < {SHAPE_QUALITY_WARN_THRESHOLD}；{n_misaligned}（{100 * n_misaligned / n_cells:.3f}%）个单元的"
        f"面法向失配 > {FACE_MISALIGNMENT_WARN_DEG:.1f}°（典型来源：棱柱四边形侧面与相邻四面体的几何不协调）。"
        f"占比偏高时应改善网格。")
    return stats
