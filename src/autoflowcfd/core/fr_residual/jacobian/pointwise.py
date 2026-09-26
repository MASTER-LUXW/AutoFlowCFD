"""AutoFlowCFD V2.0 - 解析 Jacobian 的逐点导数（numba）。

非线性只出现在逐点函数里（原始变量换算、物理通量、界面跳变量）；其余都是
线性算子（插值、微分、外插、提升）。这里给出逐点导数：

* `primitive_jacobian`：`dQ/dU`，`Q=(rho,u,v,w,p)`，解析式；
* `temperature_gradient_row`：`dT/dQ`，`T = p/(rho R)`，解析式；
* `euler_flux_jacobian` / `viscous_flux_jacobian`：对**残差实际调用的那个逐点
  通量函数**做前向差分。不写解析式是刻意的：通量函数若日后修改（正性保护、
  常数），差分自动跟着变，Jacobian 不会与残差脱节（通量对梯度是线性的，差分
  在舍入误差内精确；对 Q 的非线性部分误差 `O(sqrt(eps))`，远小于预处理子
  需要的精度）。

## 差分步长

逐输入 `h = sqrt(eps) * (|x| + 尺度)`：密度、压力取自身量级，速度分量取
`|V| + c`（零速度分量也有合理步长），梯度分量取该点梯度张量的模（全零时
通量对它线性，步长取任意正数都精确，用 `sqrt(eps)`）。
"""

import numpy as np
from numba import njit, prange

from autoflowcfd.core.fr_operators.flux_kernels import (
    euler_physical_flux_point, viscous_physical_flux_point,
)
from autoflowcfd.core.fr_residual.viscous_flux.constants import GAMMA, R_AIR

_SQRT_EPS = float(np.sqrt(np.finfo(np.float64).eps))

#: 粘性通量输入的排布：Q(5) | grad_vel 行主序 (a,b) -> 5+3a+b (9) | grad_T (3)。
N_VISC_INPUTS = 17


@njit(cache=True, inline='always')
def primitive_step(Q, v):
    """原始变量第 `v` 分量的差分步长（见模块文档）。"""
    if v == 0 or v == 4:
        return _SQRT_EPS * (abs(Q[v]) + 1e-300)
    rho = max(Q[0], 1e-300)
    c = np.sqrt(max(GAMMA * Q[4] / rho, 0.0))
    speed = np.sqrt(Q[1] * Q[1] + Q[2] * Q[2] + Q[3] * Q[3])
    return _SQRT_EPS * (abs(Q[v]) + speed + c + 1e-300)


@njit(cache=True, parallel=True)
def primitive_jacobian(U):
    """`dQ/dU`，U `(N, >=5)` 守恒变量 -> `(N, 5, 5)`。"""
    n = U.shape[0]
    out = np.zeros((n, 5, 5))
    gm1 = GAMMA - 1.0
    for k in prange(n):
        rho = U[k, 0]
        u = U[k, 1] / rho
        v = U[k, 2] / rho
        w = U[k, 3] / rho
        out[k, 0, 0] = 1.0
        out[k, 1, 0] = -u / rho
        out[k, 1, 1] = 1.0 / rho
        out[k, 2, 0] = -v / rho
        out[k, 2, 2] = 1.0 / rho
        out[k, 3, 0] = -w / rho
        out[k, 3, 3] = 1.0 / rho
        out[k, 4, 0] = gm1 * 0.5 * (u * u + v * v + w * w)
        out[k, 4, 1] = -gm1 * u
        out[k, 4, 2] = -gm1 * v
        out[k, 4, 3] = -gm1 * w
        out[k, 4, 4] = gm1
    return out


@njit(cache=True, parallel=True)
def temperature_gradient_row(Q):
    """`dT/dQ`，Q `(N, 5)` -> `(N, 5)`（只有 rho 与 p 两个非零分量）。"""
    n = Q.shape[0]
    out = np.zeros((n, 5))
    for k in prange(n):
        rho = Q[k, 0]
        out[k, 0] = -Q[k, 4] / (rho * rho * R_AIR)
        out[k, 4] = 1.0 / (rho * R_AIR)
    return out


@njit(cache=True, parallel=True)
def euler_flux_jacobian(Q):
    """`dF_i/dQ`，Q `(N, 5)` -> `(N, 3, 5, 5)`，`[k, i, out, in]`。"""
    n = Q.shape[0]
    out = np.empty((n, 3, 5, 5))
    for k in prange(n):
        f0 = euler_physical_flux_point(Q[k])
        for v in range(5):
            h = primitive_step(Q[k], v)
            q = Q[k].copy()
            q[v] += h
            f1 = euler_physical_flux_point(q)
            for i in range(3):
                for o in range(5):
                    out[k, i, o, v] = (f1[i, o] - f0[i, o]) / h
    return out


@njit(cache=True, inline='always')
def _gradient_scale(gv, gT):
    s_v = 0.0
    for a in range(3):
        for b in range(3):
            s_v += gv[a, b] * gv[a, b]
    s_t = 0.0
    for a in range(3):
        s_t += gT[a] * gT[a]
    return np.sqrt(s_v), np.sqrt(s_t)


@njit(cache=True, parallel=True)
def viscous_flux_jacobian(Q, gv, gT, mut, mu, Pr, Pr_t):
    """`dG_i/d(Q, grad_vel, grad_T)`，逐点 -> `(N, 3, 5, 17)`，输入排布见 `N_VISC_INPUTS`。

    `mut` 是冻结的涡粘（不作为输入求导，与平均流残差里 `mu_t` 整步冻结一致）。
    """
    n = Q.shape[0]
    out = np.empty((n, 3, 5, N_VISC_INPUTS))
    for k in prange(n):
        g0 = viscous_physical_flux_point(Q[k], gv[k], gT[k], mu, Pr, mut[k], Pr_t)
        sv, st = _gradient_scale(gv[k], gT[k])
        for j in range(N_VISC_INPUTS):
            q = Q[k].copy()
            g = gv[k].copy()
            t = gT[k].copy()
            if j < 5:
                h = primitive_step(Q[k], j)
                q[j] += h
            elif j < 14:
                a = (j - 5) // 3
                b = (j - 5) % 3
                h = _SQRT_EPS * (abs(g[a, b]) + sv + 1.0e-300) if sv > 0.0 else _SQRT_EPS
                g[a, b] += h
            else:
                h = _SQRT_EPS * (abs(t[j - 14]) + st + 1.0e-300) if st > 0.0 else _SQRT_EPS
                t[j - 14] += h
            g1 = viscous_physical_flux_point(q, g, t, mu, Pr, mut[k], Pr_t)
            for i in range(3):
                for o in range(5):
                    out[k, i, o, j] = (g1[i, o] - g0[i, o]) / h
    return out
