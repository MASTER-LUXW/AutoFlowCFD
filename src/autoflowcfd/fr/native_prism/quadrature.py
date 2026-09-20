"""AutoFlowCFD V2.0 - 原生棱柱基的解点求积权重（参考棱柱上的精确积分）。

## 为什么需要它

单元体积是 `∫_ref det(J) dV_ref`。坍缩棱柱基下解点就是 `[-1,1]^3` 上的
张量积 Gauss 点，所以 `HighOrderMesh.get_all_cell_volumes` 直接用张量积
Gauss 权重求和 —— 那套权重的和是 8（立方体体积）。

原生棱柱基的解点是**三角形 Warp&Blend ⊗ 挤出方向 Gauss** 的
`(p+1)^2(p+2)/2` 个点，它们既不是张量积点、也没有现成权重；而参考棱柱的
体积是 4（参考三角形面积 2 × 挤出方向长度 2），不是 8。直接沿用张量积
权重会把棱柱体积**算成 2 倍**（实测：合成网格上 1.0 vs 真值 0.5；det(J)
恒为 0.125 的仿射棱柱，8×0.125=1.0 而 4×0.125=0.5）。

体积被 CFL 的网格尺度估计、LES 滤波宽度、退化单元诊断等多处消费，所以
这不是诊断层面的小事。

## 构造：`w = V^{-T} m`

要求"对空间内任意 `f` 都有 `Σ_i w_i f(x_i) = ∫_ref f`"。把 `f` 按模态
展开 `f = Σ_j c_j φ_j`，节点值向量 `f_nodal = V c`，于是

    Σ_i w_i f_i = w^T V c ,    ∫ f = Σ_j c_j ∫φ_j = m^T c

对任意 `c` 成立 ⟺ `V^T w = m` ⟺ `w = V^{-T} m`。

`V` 是解点上的 Vandermonde（方阵，`build_native_prism_operators` 已经
验证过它可逆），`m_j = ∫_ref φ_j`。

`m` 本身用 **Duffy 变换后的张量积 Gauss 求积**算（被积函数是多项式，
取足够多的点就是精确的，不是近似）：参考三角形
`T = {(r,s): r >= -1, s >= -1, r+s <= 0}` 的 Duffy 映射

    r = (1+a)(1-b)/2 - 1,   s = b,   (a,b) in [-1,1]^2
    dr ds = ((1-b)/2) da db

挤出方向 `t` 直接用一维 Gauss。这里只用来算 `m`（每个阶数一次、可缓存），
不进入任何热路径。

## 为什么不是"照抄四面体那条"

原生四面体用的是 `det(J) * 4/3`（参考四面体体积），因为直边四面体的
`det(J)` **逐单元为常数**。棱柱不是：挤出方向可以非均匀、三角形侧面可以
不平行，`det(J)` 在单元内是真实变化的多项式（合成网格上实测坍缩档同一
单元内跨度 0.0722）。所以棱柱必须有真正的求积权重，不能用"常数 × 参考
体积"。
"""

from functools import lru_cache

import numpy as np

from ..quadrature_points import gauss_legendre
from .basis import build_native_prism_nodes, build_native_prism_vandermonde

__all__ = [
    "NATIVE_REF_PRISM_VOLUME",
    "build_native_prism_sp_weights",
]

#: 参考棱柱体积：三角形面积 2 × 挤出长度 2。
NATIVE_REF_PRISM_VOLUME = 4.0


def _mode_integrals(order: int) -> np.ndarray:
    """`m_j = ∫_ref φ_j`，形状 `(n_modes,)`。"""
    # 被积函数最高次数是 order（模态本身）加上 Duffy 因子 (1-b)/2 的 1 次，
    # 一维 Gauss 用 n 点精确到 2n-1 次，取 order+2 点足够（order+2 点精确
    # 到 2*order+3 >= order+1）。
    n_q = order + 2
    a_1d, w_a = gauss_legendre(n_q)
    b_1d, w_b = gauss_legendre(n_q)
    t_1d, w_t = gauss_legendre(n_q)

    a, b, t = np.meshgrid(a_1d, b_1d, t_1d, indexing="ij")
    wa, wb, wt = np.meshgrid(w_a, w_b, w_t, indexing="ij")
    a, b, t = a.ravel(), b.ravel(), t.ravel()
    weight = (wa * wb * wt).ravel() * ((1.0 - b) / 2.0)

    r = (1.0 + a) * (1.0 - b) / 2.0 - 1.0
    s = b
    V, _, _, _ = build_native_prism_vandermonde(
        order, np.stack([r, s, t], axis=1))
    return weight @ V


@lru_cache(maxsize=None)
def build_native_prism_sp_weights(order: int) -> np.ndarray:
    """原生棱柱解点上的求积权重，形状 `(n_sps_native,)`。

    满足 `Σ_i w_i f(x_i) = ∫_ref f` 对多项式空间内任意 `f` 精确；
    特别地 `Σ_i w_i = NATIVE_REF_PRISM_VOLUME = 4`。

    返回的数组是**只读**的（`lru_cache` 缓存的同一个对象），调用方不要
    就地修改。
    """
    nodes = build_native_prism_nodes(order)
    V, _, _, _ = build_native_prism_vandermonde(order, nodes)
    w = np.linalg.solve(V.T, _mode_integrals(order))
    total = float(w.sum())
    if not np.isclose(total, NATIVE_REF_PRISM_VOLUME, rtol=1e-12, atol=0.0):
        raise ValueError(
            f"P{order} 原生棱柱求积权重之和 {total!r} 不等于参考棱柱体积 "
            f"{NATIVE_REF_PRISM_VOLUME} —— 常数函数的积分是这套权重最基本"
            f"的自洽条件，不通过说明 Vandermonde 或模态积分有错，不静默使用")
    w.setflags(write=False)
    return w
