"""AutoFlowCFD V2.0 - Persson-Peraire 传感器的**求值函数**（三条基）。

从 `artificial_viscosity.py` 拆出（2026-09-20）。纯搬家，逻辑未改。
算子构造与缓存在 `sensor_operators.py`。
"""

from typing import Dict, Optional, Tuple

import numpy as np

from autoflowcfd.fr.collapsed_basis import prism_modal_basis_and_grad, tet_modal_basis_and_grad
from autoflowcfd.fr.quadrature_points import gauss_legendre
from autoflowcfd.fr.native_prism.mode import prism_basis_is_native
from autoflowcfd.core.utils.array_module import array_module as _array_module

from .sensor_operators import (
    _build_native_prism_sensor_operators,
    _build_native_tet_sensor_operators,
    _build_sensor_operators,
    _operators_on,
)


def compute_persson_peraire_sensor_native_tet(
    field_nodal: np.ndarray, order: int
) -> np.ndarray:
    """native 四面体专属的 Persson-Peraire 传感器，`s_e = log10(S_e)`。

    `S_e = sum_{i+j+k==order} chat^2 / sum_all chat^2`，其中 `chat` 是在
    正交归一 PKD/Dubiner 基下的模态系数——正交归一保证这个比值**精确
    等于**"只保留最高次模态的重构"与"完整重构"的 L2 能量比，不需要
    任何求积权重。理由与背景见 `_build_native_tet_sensor_operators`。

    Args:
        field_nodal: (n_cells, n_sps) 待探测的场。只使用**前 n_native 列**
            ——其余列是零填充槽位、按约定冻结在初值，不是自由度，
            混进来会引入纯人造的阶跃。
        order: 当前多项式阶数；order==0 时返回 -inf（无最高阶模态可截断）

    Returns:
        s_e: (n_cells,)
    """
    xp = _array_module(field_nodal)
    n_cells = field_nodal.shape[0]
    if order == 0:
        return xp.full(n_cells, -np.inf)

    V_inv, top_mask, n_native = _build_native_tet_sensor_operators(order)
    V_inv, top_mask = _operators_on(
        xp, ("native_tet", order), (V_inv, top_mask))
    if field_nodal.shape[1] < n_native:
        raise ValueError(
            f"field 每单元只有 {field_nodal.shape[1]} 个解点，少于 native "
            f"四面体 order={order} 所需的 {n_native} 个真实自由度")

    real = xp.ascontiguousarray(field_nodal[:, :n_native])
    modal = xp.einsum("ij,cj->ci", V_inv, real)            # (n_cells,n_native)
    energy_all = xp.einsum("ci,ci->c", modal, modal)
    modal_top = xp.where(top_mask[xp.newaxis, :], modal, 0.0)
    energy_top = xp.einsum("ci,ci->c", modal_top, modal_top)

    S_e = energy_top / xp.maximum(energy_all, 1e-300)
    # `np.errstate` 只改 NumPy 自己的浮点错误策略，对 CuPy 数组是无操作，
    # 留着即可（CuPy 不发这类 warning）。
    with np.errstate(divide="ignore"):
        return xp.log10(xp.maximum(S_e, 1e-300))


def compute_persson_peraire_sensor_native_prism(
    field_nodal: np.ndarray, order: int
) -> np.ndarray:
    """native 棱柱专属的 Persson-Peraire 传感器，`s_e = log10(S_e)`。

    `S_e = Σ_{top} M_mm chat_m^2 / Σ_all M_mm chat_m^2`，`M_mm` 是对角
    质量（基正交但不归一，见 `_build_native_prism_sensor_operators`）。

    Args:
        field_nodal: `(n_cells, n_sps)`，只使用**前 n_native 列**（其余是
            冻结在初值的零填充槽位，不是自由度）。
        order: 当前多项式阶数；`order==0` 时返回 `-inf`。
    """
    xp = _array_module(field_nodal)
    n_cells = field_nodal.shape[0]
    if order == 0:
        return xp.full(n_cells, -np.inf)

    V_inv, mass_diag, top_mask, n_native = (
        _build_native_prism_sensor_operators(order))
    V_inv, mass_diag, top_mask = _operators_on(
        xp, ("native_prism", order), (V_inv, mass_diag, top_mask))
    if field_nodal.shape[1] < n_native:
        raise ValueError(
            f"field 每单元只有 {field_nodal.shape[1]} 个解点，少于 native "
            f"棱柱 order={order} 所需的 {n_native} 个真实自由度")

    real = xp.ascontiguousarray(field_nodal[:, :n_native])
    modal = xp.einsum("ij,cj->ci", V_inv, real)
    energy = modal * modal * mass_diag[xp.newaxis, :]
    energy_all = xp.sum(energy, axis=1)
    energy_top = xp.sum(xp.where(top_mask[xp.newaxis, :], energy, 0.0), axis=1)

    S_e = energy_top / xp.maximum(energy_all, 1e-300)
    with np.errstate(divide="ignore"):      # 见 native 四面体版同一处说明
        return xp.log10(xp.maximum(S_e, 1e-300))


def compute_persson_peraire_sensor(
    field_nodal: np.ndarray, cell_type: str, order: int, ref_cube_sps: np.ndarray
) -> np.ndarray:
    """计算 Persson-Peraire 模态传感器 s_e = log10(S_e)。

    S_e = <u_trunc, u_trunc>_e / <u, u>_e，其中 u_trunc 是把 u 变换到
    模态系数空间、只保留 max(i,j,k)==order 的最高阶模态（其余模态清零）
    后再变换回节点值——即 u 与"截断掉最高阶模态的 u"之间的差恰好等于
    u_trunc 本身（线性算子的性质：完整重构 - 截断重构 = 只保留被截断
    那部分模态的重构），<.,.>_e 是用节点所在 Gauss-Legendre 求积点权重
    做的离散 L2 内积（张量积三维权重，逐元素与场值相乘再求和，不是
    矩阵乘法意义上的质量矩阵内积，但对配置在求积点上的节点表示，两者
    对多项式被积函数是精确等价的——求积点与解点重合正是配置法的定义）。

    Args:
        field_nodal: (n_cells, n_sps) 待探测的场（通常是密度）
        cell_type: "tet" 或 "prism"
        order: 当前多项式阶数
        ref_cube_sps: (n_sps, 3) 计算立方体参考坐标，与 fr/operators.py
            生成 D_3d_tet/D_3d_prism 用的完全一致

    Returns:
        s_e: (n_cells,) 传感器值（已取 log10），order==0 时返回 -inf
            填充的数组（representing "无穷光滑"，任何下游 kappa 判据
            都不会触发人工粘性，与 order==0 没有可截断的高阶模态这一
            事实一致）
    """
    xp = _array_module(field_nodal)
    n_cells = field_nodal.shape[0]
    if order == 0:
        return xp.full(n_cells, -np.inf)

    V, V_inv, quad_weights_3d, top_mask = _build_sensor_operators(
        cell_type, order, ref_cube_sps)
    V, V_inv, quad_weights_3d, top_mask = _operators_on(
        xp, (cell_type, order), (V, V_inv, quad_weights_3d, top_mask))

    modal = xp.einsum("ij,cj->ci", V_inv, field_nodal)
    modal_top = xp.where(top_mask[xp.newaxis, :], modal, 0.0)
    diff_nodal = xp.einsum("ij,cj->ci", V, modal_top)

    num = xp.einsum("cs,s,cs->c", diff_nodal, quad_weights_3d, diff_nodal)
    den = xp.einsum("cs,s,cs->c", field_nodal, quad_weights_3d, field_nodal)

    S_e = num / xp.maximum(den, 1e-300)
    with np.errstate(divide="ignore"):      # 见 native 版同一处说明
        s_e = xp.log10(xp.maximum(S_e, 1e-300))
    return s_e
