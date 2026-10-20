"""AutoFlowCFD V2.0 - "解点值 -> VTK 节点值"插值矩阵与 VTK 节点的顶点形函数。

插值矩阵取 `fr/point_interpolation.py`（解多项式在任意参考点上的取值，求解器侧探针用同一份），宽度是**真实自由度
数**（零填充槽位不携带场值，调用方取前 n_real 行）。

几何是直边单元：VTK 节点的物理坐标 = 顶点形函数 x 顶点坐标（四面体为重心坐标，棱柱为三角形重心坐标 x 挤出方向
线性），所有单元共用同一个形函数矩阵。
"""

from typing import Tuple

import numpy as np

from ...fr.point_interpolation import prism_interpolation_matrix, tet_interpolation_matrix
from .node_layout import _tet_vtk_node_barycentrics, _wedge_vtk_node_layout


def native_prism_export_data(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """三棱柱：`(N (n_vtk, 6) 顶点形函数, E (n_vtk, n_native) 插值矩阵)`。

    顶点顺序 (v0, v1, v2, w0, w1, w2)（底、顶），与 `_fixed_prism_conn` 及原生棱柱参考坐标
    `r = 2 l1 - 1, s = 2 l2 - 1, t = 2 z - 1` 一致（`fr/native_prism/basis.py::map_native_prism_to_physical`）。
    """
    from autoflowcfd.fr.native_prism.basis import map_native_prism_to_physical

    tri, z = _wedge_vtk_node_layout(order)
    target_rst = np.column_stack([2.0 * tri[:, 1] - 1.0, 2.0 * tri[:, 2] - 1.0, 2.0 * z - 1.0])
    E = prism_interpolation_matrix(order, target_rst)
    # 顶点形函数取自几何映射本身（映射对顶点坐标线性：第 a 列 = 顶点 a 取单位坐标时的映射值）
    N = np.column_stack([map_native_prism_to_physical(target_rst, np.eye(6)[:, [a, a, a]])[:, 0] for a in range(6)])
    return N, E


def native_tet_export_data(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """四面体：`(N (n_vtk, 4) 顶点形函数（重心坐标）, E (n_vtk, n_native) 插值矩阵)`。

    原生单纯形基的参考坐标 `r = 2 L1 - 1, s = 2 L2 - 1, t = 2 L3 - 1`（与原生面 Vandermonde 同一个线性映射）。
    """
    bary = _tet_vtk_node_barycentrics(order)
    E = tet_interpolation_matrix(order, 2.0 * bary[:, 1:] - 1.0)
    return bary, E
