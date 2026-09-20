"""AutoFlowCFD V2.0 - Order Continuation 的跨阶数解插值算子（按基分派）。

## 修的是什么（2026-09-20 发现的真实生产缺陷）

`core/utils/order_continuation.py::interpolate_to_new_order` 一直用
**一维 Gauss 点的张量积 Lagrange 插值**（`_build_linear_interp_matrix_3d`）
把解从 P_old 延拓到 P_new。那个矩阵的前提是"解点是 `[-1,1]^3` 上的张量积
Gauss 点"——只有**坍缩棱柱基**满足。

判据：**线性场**在 P1 与 P2 空间里都精确可表示，所以正确的延拓必须把
P1 上的线性场插成 P2 上同一个线性场（机器精度）。实测 P1->P2：

    棱柱基     单元类型   线性场插值最大相对误差
    坍缩       棱柱       4.3e-16    <- 正确
    坍缩       四面体     **7.0e-01**
    原生       棱柱       **1.4e-01**
    原生       四面体     **7.0e-01**

也就是说：

* **四面体那一条自 2026-09-03 起就是错的** —— 那天坍缩四面体基被整体
  删除、native 成为四面体唯一实现，而这个插值算子没有跟着改。P0->P1
  不受影响（P0 是常数场，任何插值都给出同一个常数），所以走
  `P0 -> P1 -> P2` 的生产运行只在**最后那一跳**被污染，前面看不出来；
* 棱柱那一条在默认基改成 native（2026-09-20）之后同样会错。

## 正确的做法

延拓的语义是"把 P_old 的那个多项式，在 P_new 的解点上求值"。所以

    c      = V_old(nodes_old)^{-1} f_old        <- 模态系数
    f_new  = V_old(nodes_new) c                 <- 在新解点上求值
    =>  W  = V_old(nodes_new) @ V_old(nodes_old)^{-1}

三条基各自用**自己的**模态族与解点：

* 坍缩棱柱：解点就是张量积 Gauss 点，`W` 退化成一维 Lagrange 的张量积，
  与既有实现逐位相同（本模块直接复用它，不再写第二份）；
* 原生棱柱：PKD(三角形)⊗Legendre(挤出) 模态 + Warp&Blend⊗Gauss 解点；
* 原生四面体：受限 PKD/Dubiner 模态 + Warp&Blend 解点。

## 零填充槽位的处理

全局 SP 宽度恒为 `(p+1)^3`（坍缩棱柱的自由度数），原生基只用前
`real_sps_per_cell` 个槽位（见 `fr/native_padding.py`）：

* **源**的填充列必须取零 —— 那些槽位的值是"初始化时复制真实 SP #0"
  之后被冻结的，不是该处多项式的取值，让它们参与插值就是把馊值搬进
  新阶数；
* **目标**的填充行取与"真实 SP #0"完全相同的那一行 —— 这正好复现
  `native_padding.py` 的初始化约定（`U[pad] = U[0]`），使延拓后的场满足
  与新建场相同的不变量。
"""

from functools import lru_cache
from typing import Tuple

import numpy as np

__all__ = ["build_order_interp_matrices", "apply_order_interp"]


def _embed_padded(w_real: np.ndarray, n_new_global: int,
                  n_old_global: int) -> np.ndarray:
    """把真实自由度上的 `(n_real_new, n_real_old)` 矩阵嵌入全局宽度。

    填充列为零、填充行复制真实 SP #0 那一行（见模块文档"零填充槽位"）。
    """
    n_real_new, n_real_old = w_real.shape
    w = np.zeros((n_new_global, n_old_global), dtype=np.float64)
    w[:n_real_new, :n_real_old] = w_real
    if n_real_new < n_new_global:
        # 填充行与"真实 SP #0"那一行完全相同（全局宽度，含已置零的填充列）
        w[n_real_new:, :] = w[0, :][None, :]
    return w


def _native_prism_interp(old_order: int, new_order: int) -> np.ndarray:
    from .native_prism.basis import (
        build_native_prism_nodes, build_native_prism_vandermonde,
    )

    nodes_old = build_native_prism_nodes(old_order)
    nodes_new = build_native_prism_nodes(new_order)
    v_old_at_old, _, _, _ = build_native_prism_vandermonde(
        old_order, nodes_old)
    v_old_at_new, _, _, _ = build_native_prism_vandermonde(
        old_order, nodes_new)
    w_real = np.linalg.solve(v_old_at_old.T, v_old_at_new.T).T
    return _embed_padded(w_real, (new_order + 1) ** 3, (old_order + 1) ** 3)


def _native_tet_interp(old_order: int, new_order: int) -> np.ndarray:
    from .native_tet.basis import build_native_tet_operators, restricted_tet_modes
    from .native_tet.overintegration import _native_modal_vandermonde

    nodes_old, _ = build_native_tet_operators(old_order)
    nodes_new, _ = build_native_tet_operators(new_order)
    modes_old = restricted_tet_modes(old_order)
    v_old_at_old = _native_modal_vandermonde(nodes_old, modes_old)
    v_old_at_new = _native_modal_vandermonde(nodes_new, modes_old)
    w_real = np.linalg.solve(v_old_at_old.T, v_old_at_new.T).T
    return _embed_padded(w_real, (new_order + 1) ** 3, (old_order + 1) ** 3)


@lru_cache(maxsize=None)
def _build_cached(old_order: int, new_order: int, prism_native: bool
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """真正的构造 + 缓存。**缓存键必须含棱柱基**：矩阵依赖解点位置，
    而那由 `AFCFD_PRISM_BASIS` 决定。第一版只按 `(old, new)` 缓存，于是
    同一个进程里先跑坍缩档、再跑原生档时会拿到坍缩的矩阵 —— 被
    `tests/unit/test_order_interp_basis_aware.py` 的参数化当场抓到
    （原生棱柱 P1->P2 相对误差 1.44e-01）。
    """
    if prism_native:
        w_prism = _native_prism_interp(old_order, new_order)
    else:
        # 坍缩棱柱：复用既有的一维 Lagrange 张量积实现（唯一事实来源）
        from ..core.utils.order_continuation import (
            _build_linear_interp_matrix_3d,
        )
        from .quadrature_points import gauss_legendre

        old_1d, _ = gauss_legendre(old_order + 1)
        new_1d, _ = gauss_legendre(new_order + 1)
        w_prism = np.ascontiguousarray(
            _build_linear_interp_matrix_3d(old_1d, new_1d))

    w_tet = _native_tet_interp(old_order, new_order)
    for w in (w_prism, w_tet):
        w.setflags(write=False)
    return w_prism, w_tet


def build_order_interp_matrices(old_order: int, new_order: int
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """返回 `(W_prism, W_tet)`，形状均为 `((new+1)^3, (old+1)^3)`。

    `W @ f_old` 给出解在新阶数解点上的取值（全局填充布局，见模块文档）。
    两个矩阵是**只读**的缓存对象，调用方不要就地修改。
    """
    if old_order < 0 or new_order < 0:
        raise ValueError(
            f"阶数必须非负，收到 old_order={old_order}, "
            f"new_order={new_order}")
    from .native_prism.mode import prism_basis_is_native

    return _build_cached(int(old_order), int(new_order),
                         bool(prism_basis_is_native()))


def apply_order_interp(field: np.ndarray, n_prism_cells: int,
                       old_order: int, new_order: int) -> np.ndarray:
    """把 `(n_cells, n_sps[, n_vars])` 的场延拓到新阶数。

    单元顺序按项目约定"棱柱在前、四面体在后"（见 `HighOrderMesh` 模块
    文档），两段各用自己的矩阵。

    Args:
        field: `(n_cells, n_sps_old)` 或 `(n_cells, n_sps_old, n_vars)`
        n_prism_cells: 棱柱单元数（前缀长度）
        old_order / new_order: 源与目标阶数

    Returns:
        同维度、SP 轴换成 `(new_order+1)^3` 的新数组。
    """
    field = np.asarray(field)
    w_prism, w_tet = build_order_interp_matrices(old_order, new_order)
    n_old = (old_order + 1) ** 3
    if field.shape[1] != n_old:
        raise ValueError(
            f"场的 SP 轴长度 {field.shape[1]} 与 old_order={old_order} 的"
            f"全局宽度 {n_old} 不符 —— 延拓算子与场的阶数必须一致")

    n_cells = field.shape[0]
    n_prism = int(n_prism_cells)
    if not (0 <= n_prism <= n_cells):
        raise ValueError(
            f"n_prism_cells={n_prism} 超出 [0, n_cells={n_cells}]")

    spec = "ab,cb->ca" if field.ndim == 2 else "ab,cbv->cav"
    out_shape = (n_cells, (new_order + 1) ** 3) + field.shape[2:]
    out = np.empty(out_shape, dtype=np.float64)
    if n_prism > 0:
        out[:n_prism] = np.einsum(spec, w_prism, field[:n_prism])
    if n_prism < n_cells:
        out[n_prism:] = np.einsum(spec, w_tet, field[n_prism:])
    return out
