"""AutoFlowCFD V2.0 - FR 解多项式在单元内任意参考点上的取值矩阵（高阶 VTK 导出与求解器侧探针共用）。

`E = V_target V_sps^{-1}`：`V_sps` / `V_target` 是该单元原生模态基（原生棱柱基、原生单纯形基）在真实解点与目标
参考点处的取值，宽度是**真实自由度数**（零填充槽位不携带场值，调用方只取解的前 n_real 个解点）。必须用解所在的那套
基与解点——2026-09-03（四面体）与 2026-09-20（棱柱）各修过一次把插值矩阵拟合在已删除坍缩基节点上、形状对得上
却静默给错值的缺陷。

参考坐标约定（与几何映射同一套，`fr/native_prism/basis.py::map_native_prism_to_physical`、
`fr/native_tet/basis.py::map_native_tet_to_physical`）：

* 棱柱 `r = 2 l1 - 1, s = 2 l2 - 1, t = 2 z - 1`（l1、l2 为底面三角形顶点 v1、v2 的重心坐标，z 为挤出方向参数）；
* 四面体 `r = 2 L1 - 1, s = 2 L2 - 1, t = 2 L3 - 1`（L1~L3 为顶点 p1~p3 的重心坐标）。
"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve


def prism_interpolation_matrix(order: int, target_rst: np.ndarray) -> np.ndarray:
    """原生棱柱基：`(n_target, n_real)`，第 i 行乘以单元真实解点值得到解多项式在 `target_rst[i]` 处的值。"""
    from autoflowcfd.fr.native_prism.basis import build_native_prism_nodes, build_native_prism_vandermonde

    target_rst = np.atleast_2d(np.asarray(target_rst, dtype=np.float64))
    V_sps = build_native_prism_vandermonde(order, build_native_prism_nodes(order))[0]
    V_target = build_native_prism_vandermonde(order, target_rst)[0]
    return lu_solve(lu_factor(V_sps.T), V_target.T).T


def tet_interpolation_matrix(order: int, target_rst: np.ndarray) -> np.ndarray:
    """原生单纯形基：`(n_target, n_real)`，含义同 `prism_interpolation_matrix`。"""
    from autoflowcfd.fr.native_tet.basis import (
        build_native_tet_operators, restricted_tet_modes, rst_to_abc, simplex3d_value,
    )

    target_rst = np.atleast_2d(np.asarray(target_rst, dtype=np.float64))
    ref_sps, _ = build_native_tet_operators(order)
    modes = restricted_tet_modes(order)
    abc_sps = rst_to_abc(ref_sps[:, 0], ref_sps[:, 1], ref_sps[:, 2])
    abc_t = rst_to_abc(target_rst[:, 0], target_rst[:, 1], target_rst[:, 2])
    V_sps = np.column_stack([simplex3d_value(*abc_sps, *m) for m in modes])
    V_target = np.column_stack([simplex3d_value(*abc_t, *m) for m in modes])
    return lu_solve(lu_factor(V_sps.T), V_target.T).T
