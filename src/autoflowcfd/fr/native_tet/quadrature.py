"""AutoFlowCFD V2.0 - 原生四面体基的解点求积权重（参考四面体上的精确积分）。

与 `fr/native_prism/quadrature.py` 是同一个构造、同一个自洽检查，只是
参考单元换成四面体。两者合起来给出"单元均值"的唯一口径：本格式是在解点上
逐点除以 `det(J)` 的配点形式，对每个解点乘参考求积权重 `w_s` 再求和，
体积项积出界面通量、提升项积出公共通量，所以**守恒的离散量就是
`Σ_s w_s det(J)_s U_s`**。正性保持限制器向单元均值收缩时必须用这套权重，
否则限制器本身不守恒。

## 构造：`w = V^{-T} m`

要求"对空间内任意 `f` 都有 `Σ_i w_i f(x_i) = ∫_ref f`"。按模态展开
`f = Σ_j c_j φ_j`，节点值 `f_nodal = V c`，于是 `Σ_i w_i f_i = w^T V c`、
`∫ f = m^T c`，对任意 `c` 成立 ⟺ `w = V^{-T} m`，其中 `m_j = ∫_ref φ_j`。

`m` 用 Duffy 变换后的张量积 Gauss 求积算（被积函数是多项式，点数足够即
精确）。参考四面体 `{r,s,t >= -1, r+s+t <= -1}` 的坍缩坐标
`(a,b,c) ∈ [-1,1]^3`：

    r = (1+a)(1-b)(1-c)/4 - 1,  s = (1+b)(1-c)/2 - 1,  t = c
    dr ds dt = ((1-b)/2) ((1-c)/2)^2 da db dc

`simplex3d_value` 本来就在 `(a,b,c)` 上求值，直接用。

注意 P>=2 时部分权重为负或为零（例如二次四面体顶点基函数的积分为负）——
这是 Lagrange 基在单纯形上的固有性质，不是数值问题；向均值收缩对任意
符号的权重都严格保持 `Σ w J U`。
"""

from functools import lru_cache

import numpy as np

from ..quadrature_points import gauss_legendre
from .basis import build_native_tet_operators, restricted_tet_modes, rst_to_abc, simplex3d_value

__all__ = [
    "NATIVE_REF_TET_VOLUME",
    "build_native_tet_sp_weights",
]

#: 参考四面体体积 = 8/6。
NATIVE_REF_TET_VOLUME = 4.0 / 3.0


def _mode_integrals(order: int) -> np.ndarray:
    """`m_j = ∫_ref φ_j`，形状 `(n_modes,)`。"""
    # 被积次数最高 order（模态）+ Duffy 因子 (1-b)/2 的 1 次、((1-c)/2)^2 的
    # 2 次；一维 Gauss n 点精确到 2n-1 次，取 order+3 点足够。
    n_q = order + 3
    x, w = gauss_legendre(n_q)
    a, b, c = np.meshgrid(x, x, x, indexing="ij")
    wa, wb, wc = np.meshgrid(w, w, w, indexing="ij")
    a, b, c = a.ravel(), b.ravel(), c.ravel()
    weight = (wa * wb * wc).ravel() * ((1.0 - b) / 2.0) * ((1.0 - c) / 2.0) ** 2
    modes = restricted_tet_modes(order)
    return np.array([weight @ simplex3d_value(a, b, c, i, j, k) for (i, j, k) in modes])


@lru_cache(maxsize=None)
def build_native_tet_sp_weights(order: int) -> np.ndarray:
    """原生四面体解点上的求积权重，形状 `(n_sps_native,)`。

    满足 `Σ_i w_i f(x_i) = ∫_ref f` 对多项式空间内任意 `f` 精确；特别地
    `Σ_i w_i = NATIVE_REF_TET_VOLUME = 4/3`。返回只读数组（缓存对象）。
    """
    ref_rst, _D = build_native_tet_operators(order)
    a, b, c = rst_to_abc(ref_rst[:, 0], ref_rst[:, 1], ref_rst[:, 2])
    modes = restricted_tet_modes(order)
    V = np.column_stack([simplex3d_value(a, b, c, i, j, k) for (i, j, k) in modes])
    w = np.linalg.solve(V.T, _mode_integrals(order))
    total = float(w.sum())
    if not np.isclose(total, NATIVE_REF_TET_VOLUME, rtol=1e-12, atol=0.0):
        raise ValueError(
            f"P{order} 原生四面体求积权重之和 {total!r} 不等于参考四面体体积 "
            f"{NATIVE_REF_TET_VOLUME} —— 常数函数的积分是这套权重最基本的自洽"
            f"条件，不通过说明 Vandermonde 或模态积分有错，不静默使用")
    w.setflags(write=False)
    return w
