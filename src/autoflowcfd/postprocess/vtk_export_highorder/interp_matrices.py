"""AutoFlowCFD V2.0 - "SPs 节点值 -> VTK 节点值"插值矩阵（按基分派）。

从 `vtk_export_highorder.py` 拆出（2026-09-20）。纯搬家，逻辑未改。
三条基各自为什么必须用自己的模态族与解点，见各构造函数的文档 ——
其中原生棱柱那条是 2026-09-20 修掉的一处**静默给错值**的真实缺陷。
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from .node_layout import (
    _MAX_SUPPORTED_ORDER,
    _MAX_SUPPORTED_ORDER_TET,
    _MAX_SUPPORTED_ORDER_WEDGE,
    _tet_vtk_node_barycentrics,
    _tri_barycentric_to_cube_ab,
    _wedge_vtk_node_layout,
)

def _build_vtk_lagrange_export_data(cell_type: str, order: int, ref_cube_sps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """构造棱柱 VTK Lagrange 节点在参考立方体坐标下的位置与"SPs 节点值
    -> VTK 节点值"插值矩阵（`cell_type="prism"` 专用——四面体已改用
    native 单纯形基专属的 `_build_native_tet_vtk_lagrange_export_data`，
    见该函数文档"2026-09-03 更正"一节）。与所有单元共享（同一 order 的
    所有棱柱用同一份，不需要逐单元重算）。

    Args:
        cell_type: 只接受 "prism"
        order: 多项式阶数（<= _MAX_SUPPORTED_ORDER）
        ref_cube_sps: 现有 SPs 的参考立方体坐标 (n_sps,3)

    Returns:
        (target_cube, E)：target_cube 形状 (n_vtk_nodes,3)，E 形状
        (n_vtk_nodes, n_sps)，`E @ field_at_sps` 给出 field 在 VTK 节点
        处的插值取值。
    """
    from ...fr.collapsed_basis import prism_modal_basis_and_grad
    from scipy.linalg import lu_factor, lu_solve

    if cell_type != "prism":
        raise ValueError(
            f"_build_vtk_lagrange_export_data 只接受 cell_type='prism'（收到 {cell_type!r}）——"
            "四面体已改用 _build_native_tet_vtk_lagrange_export_data，见模块文档。"
        )
    tri_bary, z_frac = _wedge_vtk_node_layout(order)
    ab = _tri_barycentric_to_cube_ab(tri_bary)
    c = 2.0 * z_frac - 1.0
    target_cube = np.column_stack([ab[:, 0], ab[:, 1], c])
    basis_fn = prism_modal_basis_and_grad

    a_sps, b_sps, c_sps = ref_cube_sps[:, 0], ref_cube_sps[:, 1], ref_cube_sps[:, 2]
    V_sps, _, _, _ = basis_fn(a_sps, b_sps, c_sps, order)

    a_t, b_t, c_t = target_cube[:, 0], target_cube[:, 1], target_cube[:, 2]
    V_target, _, _, _ = basis_fn(a_t, b_t, c_t, order)

    lu_piv = lu_factor(V_sps.T)
    E = lu_solve(lu_piv, V_target.T).T
    return target_cube, E


def _build_native_prism_vtk_lagrange_export_data(
        order: int) -> Tuple[np.ndarray, np.ndarray]:
    """棱柱 VTK Lagrange 节点导出插值矩阵 —— **native 棱柱基专属版本**
    （2026-09-20 新增，与四面体 2026-09-03 那次是同一类修正）。

    坍缩档走的路径是"VTK 节点重心坐标 -> 解析求逆到坍缩坐标 (a,b,c) ->
    用坍缩模态基 `prism_modal_basis_and_grad` 构造 Vandermonde"。原生
    棱柱基（`AFCFD_PRISM_BASIS=native`，2026-09-20 起是默认）的 SPs 是
    三角形 Warp&Blend ⊗ 挤出 Gauss 的 `(p+1)^2(p+2)/2` 个原生节点，
    既不在坍缩坐标网格上、模态族也不同 —— 继续走旧路径会把插值矩阵拟合
    在错误的节点位置上，**静默给出错误结果**（矩阵形状仍然对得上，所以
    不会报错）。实测：P2 混合网格上导出节点值与解析场的相对误差
    1.66e-02，而正确路径是机器零。

    原生档下这条路径同样更直接：VTK wedge 节点的三角形重心坐标就是原生
    参考三角形的原生参数化，`r = 2*l2 - 1`、`s = 2*l3 - 1`
    （`tri_barycentric` 的解析逆，与 `native_prism_exact_jacobian` 里
    `dl2/dr = +1/2`、`dl3/ds = +1/2` 是同一组关系），挤出方向
    `t = 2*z_frac - 1`，不需要任何坍缩坐标中间表示。

    Returns:
        `(target_rst, E)`：`target_rst` 形状 `(n_vtk_nodes, 3)`（原生
        参考坐标，供调用方用 `map_native_prism_to_physical` 直接求物理
        坐标），`E` 形状 `(n_vtk_nodes, n_native)` —— **宽度是
        `n_native` 而不是全局 `(order+1)^3`**，理由与四面体那条完全相同
        （零填充行不携带真实场值），调用方必须先按 `[:n_native]` 切片。
    """
    from ...fr.native_prism.basis import (
        build_native_prism_nodes, build_native_prism_vandermonde,
    )
    from scipy.linalg import lu_factor, lu_solve

    tri_bary, z_frac = _wedge_vtk_node_layout(order)
    r_t = 2.0 * tri_bary[:, 1] - 1.0
    s_t = 2.0 * tri_bary[:, 2] - 1.0
    t_t = 2.0 * z_frac - 1.0
    target_rst = np.column_stack([r_t, s_t, t_t])

    nodes = build_native_prism_nodes(order)
    V_sps, _, _, _ = build_native_prism_vandermonde(order, nodes)
    V_target, _, _, _ = build_native_prism_vandermonde(order, target_rst)

    lu_piv = lu_factor(V_sps.T)
    E = lu_solve(lu_piv, V_target.T).T          # (n_vtk_nodes, n_native)
    assert E.shape == (target_rst.shape[0], nodes.shape[0])
    return target_rst, E


def _build_native_tet_vtk_lagrange_export_data(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """四面体 VTK Lagrange 节点导出插值矩阵——native 单纯形基专属版本
    （2026-09-03 更正，删除 collapsed 四面体基后新增）。

    此前（collapsed 时代）走的路径：VTK 节点重心坐标 -> 解析求逆到
    坍缩坐标 (a,b,c) -> 用坍缩坐标模态基 `tet_modal_basis_and_grad`
    构造 Vandermonde、解出插值矩阵，物理坐标另外用
    `curved_mapping.map_tet_to_physical` 对同一个 (a,b,c) 目标点求值。
    坍缩坐标四面体基已删除（见 fr/operators.py 模块文档），四面体 SPs
    现在恒为 native 单纯形基节点（`n_native = (order+1)(order+2)
    (order+3)/6` 个真实自由度 + 零填充块对角到全局 `(order+1)^3` 宽度，
    见 native_padding.py），不再对应坍缩坐标 (a,b,c) 网格——继续
    用旧路径会把插值矩阵拟合在错误的节点位置上，静默给出错误结果
    （不会报错，因为矩阵形状仍然对得上）。

    native 基下这条路径实际上更直接：VTK 节点的重心坐标本身就是四面体
    单纯形基的原生参数化，不需要经过任何坍缩坐标的解析求逆——
    `r=2*L1-1, s=2*L2-1, t=2*L3-1`（`L0,L1,L2,L3` 是重心坐标，
    与 `native_tet/basis.py::_native_face_value_vandermondes`
    构造面 Vandermonde 时用的同一个线性映射，验证方式同源）。物理坐标
    同样直接用重心坐标仿射组合（`map_native_tet_to_physical` 同一个
    公式），不经过任何参考立方体中间表示。

    Returns:
        (target_rst, E)：target_rst 形状 (n_vtk_nodes,3)（(r,s,t) 参考
        单纯形坐标，供调用方按重心坐标直接算物理坐标，不需要另外的
        map_fn），E 形状 (n_vtk_nodes, n_native)——**注意宽度是
        `n_native`，不是全局 `(order+1)^3`**：零填充行不携带真实场值
        （只是被残差组装强制保持初值不变的占位槽位，不是该处物理场的
        真实多项式取值），必须只用体积节点数组的前 `n_native` 行做
        插值，调用方（`export_highorder_vtk`）对四面体单元的场数据要
        先按 `[:n_native]` 切片，不能像棱柱那样直接用完整宽度。
    """
    from ...fr.native_tet.basis import (
        build_native_tet_operators, restricted_tet_modes, simplex3d_value, rst_to_abc,
    )
    from ...grid.curved_mapping.curved_mapping import tet_barycentric
    from scipy.linalg import lu_factor, lu_solve

    bary = _tet_vtk_node_barycentrics(order)  # (n_vtk_nodes, 4), (L0,L1,L2,L3)
    r_t = 2.0 * bary[:, 1] - 1.0
    s_t = 2.0 * bary[:, 2] - 1.0
    t_t = 2.0 * bary[:, 3] - 1.0
    target_rst = np.column_stack([r_t, s_t, t_t])

    ref_rst_sps, _ = build_native_tet_operators(order)
    n_native = ref_rst_sps.shape[0]
    modes = restricted_tet_modes(order)

    a_sps, b_sps, c_sps = rst_to_abc(ref_rst_sps[:, 0], ref_rst_sps[:, 1], ref_rst_sps[:, 2])
    a_t, b_t, c_t = rst_to_abc(r_t, s_t, t_t)

    V_sps = np.column_stack([simplex3d_value(a_sps, b_sps, c_sps, i, j, k) for (i, j, k) in modes])
    V_target = np.column_stack([simplex3d_value(a_t, b_t, c_t, i, j, k) for (i, j, k) in modes])

    lu_piv = lu_factor(V_sps.T)
    E = lu_solve(lu_piv, V_target.T).T  # (n_vtk_nodes, n_native)
    assert E.shape == (target_rst.shape[0], n_native)
    return target_rst, E
