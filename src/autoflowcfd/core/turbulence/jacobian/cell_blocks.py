"""AutoFlowCFD V2.0 - k-omega 残差的单元内项（源项 + 对流/扩散体积项）对对角块的贡献。

记 `phi = (k, omega)`，`Gop[s,b,t] = sum_m inv[s,m,b] D[s,t,m]` 为单元内局部物理梯度算子
（与 `fr_operators/gradients.py` 同一定义）。`d(rho phi_v)/dt` 的单元内部分：

    源项        S_v(s)                          逐点，依赖 (k, omega, grad k, grad omega)
    对流体积项  -(1/det_s) sum_{r,m} W[s,r,m] rut[r,m] (c2f phi_v)_r
    扩散体积项  +(1/det_s) sum_{r,m} W[s,r,m] sum_i adj[r,m,i] (c2f Gamma_v)_r (c2f grad phi_v)_{r,i}

（`rut = adj (rho u)` 在通量点上，平均流冻结；`W = f2c D_f`、`c2f`、`adj` 与平均流体积项
同一套过积分算子，湍流不过积分时 `c2f = I`、`W = D`。）逐点量的导数
`dS/dq`、`dGamma/dq`（`q = (k, omega, grad k, grad omega)`）经梯度算子落到单元解点上：

    d X(s) / d phi_u(t) = dX/du|_s delta_st + sum_b dX/d(grad u)_b|_s Gop[s,b,t]

输出分两份：乘了 det 的体积项 `acc`、不乘 det 的源项 `src`（`assemble.py` 合并）。
"""

import numpy as np
from numba import njit, prange


@njit(cache=True, inline='always')
def _gradient_operator(inv_c, D, n):
    Gop = np.zeros((n, 3, n))
    for s in range(n):
        for t in range(n):
            for m in range(3):
                d = D[s, t, m]
                if d != 0.0:
                    for b in range(3):
                        Gop[s, b, t] += inv_c[s, m, b] * d
    return Gop


@njit(cache=True, inline='always')
def _pointwise_chain(dX, Gop, n):
    """`out[s, v, t, u] = dX_v(s)/d phi_u(t)`（`dX (n, 2, 8)` 为逐点导数）。"""
    out = np.zeros((n, 2, n, 2))
    for s in range(n):
        for v in range(2):
            for u in range(2):
                out[s, v, s, u] += dX[s, v, u]
                for b in range(3):
                    g = dX[s, v, 2 + 3 * u + b]
                    if g != 0.0:
                        for t in range(n):
                            out[s, v, t, u] += g * Gop[s, b, t]
    return out


@njit(cache=True, parallel=True)
def turbulence_cell_blocks(acc, src, slot0, phi, gam, dS, dG, inv_sp, D, c2f, W, adj, rut):
    """逐单元的单元内导数，累加到 `acc[slot0+c]`（乘 det）与 `src[slot0+c]`（源项）。

    形状（单元 c 为本块第 c 个）：`phi/gam (C, n, 2)`、`dS/dG (C, n, 2, 8)`、
    `inv_sp (C, n, 3, 3)`、`D (n, n, 3)`、`c2f (nf, n)`、`W (n, nf, 3)`、`adj (C, nf, 3, 3)`、
    `rut (C, nf, 3)`；输出块 `(n, 2, n, 2)`。
    """
    n_cells = phi.shape[0]
    n = D.shape[0]
    nf = c2f.shape[0]
    for c in prange(n_cells):
        Gop = _gradient_operator(inv_sp[c], D, n)
        # ---- 源项 ----
        blk = _pointwise_chain(dS[c], Gop, n)
        k0 = slot0 + c
        for s in range(n):
            for v in range(2):
                for t in range(n):
                    for u in range(2):
                        src[k0, s, v, t, u] += blk[s, v, t, u]
        # ---- 体积项（乘 det）----
        out = np.zeros((n, 2, n, 2))
        # A[s,r,i] = sum_m W[s,r,m] adj[r,m,i]；conv[s,r] = sum_m W[s,r,m] rut[r,m]
        A = np.zeros((n, nf, 3))
        conv = np.zeros((n, nf))
        for s in range(n):
            for r in range(nf):
                for m in range(3):
                    w = W[s, r, m]
                    if w != 0.0:
                        conv[s, r] += w * rut[c, r, m]
                        for i in range(3):
                            A[s, r, i] += w * adj[c, r, m, i]
        # 通量点上的 Gamma、grad phi 与梯度迹算子 GF = c2f Gop
        gam_f = np.zeros((nf, 2))
        grad_f = np.zeros((nf, 2, 3))
        GF = np.zeros((nf, 3, n))
        for r in range(nf):
            for sp in range(n):
                e = c2f[r, sp]
                if e != 0.0:
                    for v in range(2):
                        gam_f[r, v] += e * gam[c, sp, v]
                    for b in range(3):
                        for t in range(n):
                            GF[r, b, t] += e * Gop[sp, b, t]
            for v in range(2):
                for b in range(3):
                    acc_g = 0.0
                    for t in range(n):
                        acc_g += GF[r, b, t] * phi[c, t, v]
                    grad_f[r, v, b] = acc_g
        DGc = _pointwise_chain(dG[c], Gop, n)       # dGamma_v(s')/dphi_u(t)
        # 对流体积项取对流形式 div(rho u phi) - phi div(rho u)（transport/convection.py 模块
        # 文档）：后一项对 phi_s 的导数是对角的 div_vol(rho u)_s = sum_r conv[s,r] (c2f 1)_r
        for s in range(n):
            d1 = 0.0
            for r in range(nf):
                csum = 0.0
                for t in range(n):
                    csum += c2f[r, t]
                d1 += conv[s, r] * csum
            for v in range(2):
                out[s, v, s, v] += d1
        for s in range(n):
            for r in range(nf):
                cv = conv[s, r]
                for t in range(n):
                    e = c2f[r, t]
                    if e != 0.0 and cv != 0.0:
                        for v in range(2):
                            out[s, v, t, v] -= cv * e
                for v in range(2):
                    g = gam_f[r, v]
                    # 扩散：Gamma 固定、对 phi_v 的梯度求导
                    for t in range(n):
                        a = A[s, r, 0] * GF[r, 0, t] + A[s, r, 1] * GF[r, 1, t] + A[s, r, 2] * GF[r, 2, t]
                        out[s, v, t, v] += g * a
                    # 扩散：grad phi 固定、对 Gamma_v 求导（经 Gamma 依赖两个未知量）
                    ag = A[s, r, 0] * grad_f[r, v, 0] + A[s, r, 1] * grad_f[r, v, 1] + A[s, r, 2] * grad_f[r, v, 2]
                    if ag != 0.0:
                        for sp in range(n):
                            e = c2f[r, sp]
                            if e != 0.0:
                                for t in range(n):
                                    for u in range(2):
                                        out[s, v, t, u] += ag * e * DGc[sp, v, t, u]
        for s in range(n):
            for v in range(2):
                for t in range(n):
                    for u in range(2):
                        acc[k0, s, v, t, u] += out[s, v, t, u]
