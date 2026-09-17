# -*- coding: utf-8 -*-
"""微分矩阵的**精确常数零化**修正（2026-09-17）。

## 缺口

参考空间上的微分矩阵 `D` 按 `D = V_grad @ V^{-1}` 构造，解析上必然满足
`D @ 1 = 0`（常数的导数为零）——但 LU 求解带来的舍入让它只成立到
`eps * cond(V)`，而这个残余**直接决定体积项的自由流保持性**。

对直边（native）四面体这条链是纯粹的：Jacobian 逐单元为常数，均匀自由流
下体积项散度恰好是

    div = f2c @ [ sum_m adj(J)_m * F_m * (D_m @ 1) ] / det(J)

也就是说算出来的非零残差**全部**来自 `D_m @ 1` 的舍入残余，再被
`adj(J)/det(J)` 在小体积单元上放大。实测（`build_native_tet_operators`，
`max|D @ 1|`）：

    order/oo    2        3        4        5        6
    max|D@1|  9.8e-16  3.7e-15  9.3e-14  5.0e-13  1.7e-12

于是提高过积分阶数（去混叠精度的来源）会同时抬高自由流保持性的误差底。
真实网格（plate_coarse_volume，168340 单元）P2 实测，四面体段均匀自由流
残差 `max`：oo=3 时 4.22e-4，oo=4 时 3.06e-3（恶化约 7 倍）。这与坍缩基
当年"上限不能放宽"的那条依据是同一个机制，只是量级温和得多。

## 修正

把每行的行和从该行对角元上减掉：

    D'[i, j] = D[i, j] - delta_ij * sum_k D[i, k]

于是 `D' @ 1 = 0` **逐位精确**（对角元的一次减法不会再引入行和）。代价是
把 `D` 扰动了 `max|D@1|` 这么多（最坏 1.7e-12，相对 `max|D|~14` 是 1.2e-13），
也就是把"精确到 eps*cond(V)"换成"常数模态精确、其余模态多一个 1e-13 量级的
扰动"——对 degree>=1 的多项式精确性的影响远在舍入以下，而常数模态的精确性
换来的是自由流保持性直接落到 eps 量级。

这是曲线坐标有限差分里的标准做法（Pulliam & Steger 的度量抵消、
Kopriva《Implementing Spectral Methods for PDEs》Ch.8 对 GCL 的讨论）：
**离散**守恒律的自由流保持性靠强制离散恒等式成立来保证，而不是指望它在
浮点下自动成立。

## 适用范围（一条重要限制）

对**直边四面体**，`D' @ 1 = 0` 是自由流保持的**充分**条件（度量逐单元
常数，可以提到求和外面）。对**棱柱**（坍缩坐标 + 双线性侧面）度量逐点
变化，自由流保持需要完整的离散 GCL `sum_m D_m(adj(J)_m) = 0`，行和修正
只消掉其中"度量恒定部分"那一项——所以棱柱段的改善是部分的，不应期待落到
eps。真实网格实测见 `tests/unit/test_diff_matrix_constant_annihilation.py`。
"""

from typing import Optional

import numpy as np

__all__ = ["enforce_constant_annihilation", "constant_annihilation_error"]


def constant_annihilation_error(D: np.ndarray) -> float:
    """`max_m max_i |sum_j D[i, j, m]|` —— 修正前后的度量指标。"""
    return float(np.abs(D.sum(axis=1)).max())


def enforce_constant_annihilation(
    D: np.ndarray, *, copy: bool = False, atol: Optional[float] = None
) -> np.ndarray:
    """让微分矩阵逐位精确地零化常数。

    Args:
        D: 形状 `(n, n, 3)` 的参考空间微分矩阵（`D[:, :, m]` 是对第 m 个
            参考方向的导数），或形状 `(n, n)` 的单方向矩阵。
        copy: True 时在副本上修正并返回副本；默认原地修正并返回同一数组。
        atol: 修正量的上界护栏。行和本应只有舍入量级；若超过这个值，说明
            该矩阵根本不是一个"解析上零化常数"的微分算子（例如基/节点
            不匹配、模态索引写错），静默修正会把那种真实 bug 抹平成看起来
            正常的结果。默认按 `1e-8 * max(|D|, 1)` 自适应。

    Returns:
        修正后的 D（`copy=False` 时与入参同一对象）。
    """
    if D.ndim == 2:
        D3 = D[:, :, None]
    elif D.ndim == 3 and D.shape[2] in (1, 2, 3):
        D3 = D
    else:
        raise ValueError(f"D 形状 {D.shape} 既不是 (n,n) 也不是 (n,n,m<=3)")
    n = D3.shape[0]
    if D3.shape[1] != n:
        raise ValueError(f"D 前两轴必须同长，收到 {D3.shape}")

    if not np.isfinite(D3).all():
        # NaN/inf 会让下面的 `worst > limit` 比较恒为 False 从而静默穿过，
        # 把一个彻底损坏的矩阵当成"行和已经很小"。实际踩过一次：给
        # `build_collapsed_diff_matrices` 传 Gauss-Lobatto 点集（含坍缩基
        # 的退化端点）会让模态 Vandermonde 奇异、D 全是 NaN。
        n_bad = int((~np.isfinite(D3)).sum())
        raise ValueError(
            f"微分矩阵含 {n_bad} 个非有限元素（NaN/inf）——矩阵构造已经失败"
            f"（典型原因：节点集落在基函数的退化点上使 Vandermonde 奇异），"
            f"行和修正对它没有意义"
        )

    rowsum = D3.sum(axis=1)                      # (n, m)
    limit = atol if atol is not None else 1e-8 * max(float(np.abs(D3).max()), 1.0)
    worst = float(np.abs(rowsum).max())
    if worst > limit:
        raise ValueError(
            f"微分矩阵的行和达到 {worst:.3e}，远超舍入量级（阈值 {limit:.3e}）"
            f"——解析上 D @ 1 必须为零，这么大的行和说明矩阵构造本身有误"
            f"（基/节点/模态索引不匹配），不能用行和修正掩盖"
        )

    out = D.copy() if copy else D
    out3 = out[:, :, None] if out.ndim == 2 else out
    idx = np.arange(n)
    out3[idx, idx, :] -= rowsum
    return out
