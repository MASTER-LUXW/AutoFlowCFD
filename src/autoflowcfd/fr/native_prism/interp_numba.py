"""AutoFlowCFD V2.0 - 原生棱柱基的 numba 侧插值矩阵（残差 kernel 消费）。

## 为什么这一份非常短

面通量点的**定位**完全复用坍缩路径（`_newton_locate_nb`）—— 依据是两条基
的几何映射**逐位恒等**（2000 个随机点上 `max|diff|` 恰好 0.0，见
`fr/native_prism/__init__.py` 模块文档）。所以原生化只剩"把 Vandermonde
换成原生的"这一件事。

而这件事又比预想的更简单：**原生棱柱模态可以直接在坍缩立方体坐标
`(a,b,c)` 上求值**，零坐标换算。理由是坐标往返恰好是恒等 ——

    cube_to_tri_rs:  r = (1+a)(1-b)/2 - 1,   s = b
    rs_to_ab:        a' = 2(1+r)/(1-s) - 1,  b' = s
    代入：1+r = (1+a)(1-b)/2、1-s = 1-b  =>  a' = (1+a) - 1 = a，b' = b

于是

    phi_ijk(点) = psi_ij(a, b) * P_k^{0,0}(c)

实测与走 `build_native_prism_vandermonde` 的结果吻合到 **9.4e-16 相对**
（3000 随机点、P2）。所以这里不需要 `cube_to_tri_rs`/`rs_to_ab` 任何一步，
也就不会引入它们在退化边附近的除法。

## 与四面体那一份的关系

`face_flux_points/native_geometry_nb.py::_native_interp_matrix_nb` 是四面体版
（输入是原生 `(r,s,t)`，内部还要 `_rst_to_abc_nb`）。两者输入空间不同
（四面体那边拿到的是解析定位器直接给出的原生坐标，这边拿到的是坍缩 Newton
给出的立方体坐标），所以不是重复实现，而是两种输入各一条最短路径。
"""

import numpy as np
from numba import njit

from ..collapsed_basis import jacobi_polynomial
from .triangle_basis import simplex2d_value

__all__ = ["native_prism_interp_matrix_nb"]


@njit(cache=True)
def native_prism_interp_matrix_nb(abc, mode_i, mode_j, mode_k,
                                  v_sps_inv_native, n_sps):
    """原生棱柱目标单元的"面点 -> 体积节点"插值矩阵，形状 `(n_pts, n_sps)`。

    Args:
        abc: `(n_pts, 3)` 目标点在**坍缩立方体**参考坐标 `(a,b,c)` 下的位置
            （坍缩 Newton 定位的直接输出，见模块文档为什么不需要换算）
        mode_i / mode_j / mode_k: `(n_native,)` 原生棱柱模态索引，排列与
            `v_sps_inv_native` 的行一致（`restricted_prism_modes` 的顺序）
        v_sps_inv_native: `(n_native, n_native)`，原生棱柱节点 Vandermonde
            的逆（转置形式，与四面体那条同一约定）
        n_sps: 全局统一宽度 `(order+1)^3`

    Returns:
        `(n_pts, n_sps)`，按"补位对齐"原则只写前 `n_native` 列，其余列保持
        零 —— 与 `_native_interp_matrix_nb`（四面体版）同一个约定，下游
        kernel 因此不需要知道目标单元是哪一类。
    """
    n_pts = abc.shape[0]
    n_native = mode_i.shape[0]
    a_v = abc[:, 0]
    b_v = abc[:, 1]
    c_v = abc[:, 2]
    V_t = np.empty((n_pts, n_native))
    for m in range(n_native):
        psi = simplex2d_value(a_v, b_v, mode_i[m], mode_j[m])
        leg = jacobi_polynomial(c_v, 0.0, 0.0, mode_k[m])
        for p in range(n_pts):
            V_t[p, m] = psi[p] * leg[p]
    interp = np.zeros((n_pts, n_sps))
    for p in range(n_pts):
        for s in range(n_native):
            val = 0.0
            for m in range(n_native):
                val += V_t[p, m] * v_sps_inv_native[m, s]
            interp[p, s] = val
    return interp
