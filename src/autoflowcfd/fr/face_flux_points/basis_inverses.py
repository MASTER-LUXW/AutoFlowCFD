"""AutoFlowCFD V2.0 - 原生四面体 / 原生棱柱的节点 Vandermonde 逆矩阵与模态索引。

`build_fp_newton_parallel` / `build_ms_interp_parallel` 两个 numba kernel 在内部构造"面点 -> 体积节点"插值矩阵
时需要 `V_sps^{-1}` 与模态索引三元组，这里一次性备好，kernel 只做查表与矩阵乘。

2026-10-19 以前另有坍缩棱柱 / 坍缩四面体两条基的 `(order+1)^3` 阶 Vandermonde 逆（面编码 0~5 才用，生产网格的面
编码全在 [6,15)，从未用到），并按面编码"自动探测"是否构造原生两条；现在只有原生基，无条件构造（代价是两个小矩阵）。

## 一个必须一致的约定

`lu_factor(V.T)` 之后再 `.T` —— 也就是 `v_sps_inv = V^{-1}`，**不是** `V^{-T}`。两条基必须用同一个约定：搞反了
不会报错，只会让插值矩阵变成另一个矩阵（实测相对误差 5.44，而正确时是 1.0e-15）。
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class BasisInverses:
    """原生四面体（`*_native`）与原生棱柱（`*_np`）的 Vandermonde 逆与模态索引。"""

    v_sps_inv_native: np.ndarray
    native_mode_i: np.ndarray
    native_mode_j: np.ndarray
    native_mode_k: np.ndarray
    v_sps_inv_np: np.ndarray
    np_mode_i: np.ndarray
    np_mode_j: np.ndarray
    np_mode_k: np.ndarray


def _mode_columns(modes) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    i, j, k = (np.array([m[c] for m in modes], dtype=np.int32) for c in range(3))
    return i, j, k


def build_basis_inverses(order: int) -> BasisInverses:
    """原生四面体与原生棱柱的 `V_sps^{-1}` 与模态索引。"""
    from scipy.linalg import lu_factor, lu_solve

    from autoflowcfd.fr.native_prism.basis import (
        build_native_prism_nodes, build_native_prism_vandermonde, restricted_prism_modes,
    )
    from .geometry import _get_v_sps_lu_native

    lu_native, modes_native = _get_v_sps_lu_native(order)
    v_sps_inv_native = np.ascontiguousarray(lu_solve(lu_native, np.eye(len(modes_native))).T)

    modes_np = restricted_prism_modes(order)
    V_np, _, _, _ = build_native_prism_vandermonde(order, build_native_prism_nodes(order))
    if V_np.shape != (len(modes_np), len(modes_np)):
        raise ValueError(f"原生棱柱节点 Vandermonde 形状 {V_np.shape} 应为方阵 ({len(modes_np)}, {len(modes_np)})")
    v_sps_inv_np = np.ascontiguousarray(lu_solve(lu_factor(V_np.T), np.eye(len(modes_np))).T)

    return BasisInverses(v_sps_inv_native, *_mode_columns(modes_native), v_sps_inv_np, *_mode_columns(modes_np))
