"""AutoFlowCFD V2.0 - 块从原始变量空间换到 Newton 未知量空间（numba）。

装配在原始变量空间里累加 `K[s,a,t,b] = det_s * d(dU/dt)_{s,a} / dQ_{t,b}`。Newton 在
守恒变量上解 `R = Gamma (-dU/dt)`，所以每个块都要

    B[s,:,t,:] = Gamma_s (-K[s,:,t,:] / det_s) (dQ/dU)_t

对角块另加 `Gamma` 随状态变化的那一项 `d(Gamma R_raw)/dU_s |_{R_raw}`（只在 s = t）。
耦合块在变换之前先把同一 (行, 列) 单元对的多个槽位求和（变换是线性的、且只
依赖行列单元）。逐块并行、块内无共享写。
"""

import numpy as np
from numba import njit, prange


@njit(cache=True, inline='always')
def _transform_block(X, det_r, TQ_c, gamma_r, use_gamma, out):
    """`out[s,a,t,d] = sum_e G[s,a,e] * (-X[s,e,t,b]/det[s]) * TQ[t,b,d]`（G 缺省为单位阵）。"""
    n_r, _, n_c, _ = X.shape
    tmp = np.empty((5, 5))
    for s in range(n_r):
        inv_det = -1.0 / det_r[s]
        for t in range(n_c):
            for e in range(5):
                for d in range(5):
                    acc = 0.0
                    for b in range(5):
                        acc += X[s, e, t, b] * TQ_c[t, b, d]
                    tmp[e, d] = acc * inv_det
            if use_gamma:
                for a in range(5):
                    for d in range(5):
                        acc = 0.0
                        for e in range(5):
                            acc += gamma_r[s, a, e] * tmp[e, d]
                        out[s, a, t, d] = acc
            else:
                for a in range(5):
                    for d in range(5):
                        out[s, a, t, d] = tmp[a, d]


@njit(cache=True, parallel=True)
def finalize_diag_kernel(K, cells, n, det, TQ, gamma, dgamma, use_gamma):
    """对角块就地变换：`K[k]` 属于单元 `cells[k]`（float32，形状 (n,5,n,5)）。"""
    for k in prange(cells.shape[0]):
        c = cells[k]
        X = np.empty((n, 5, n, 5))
        for s in range(n):
            for a in range(5):
                for t in range(n):
                    for b in range(5):
                        X[s, a, t, b] = K[k, s, a, t, b]
        out = np.empty((n, 5, n, 5))
        _transform_block(X, det[c], TQ[c], gamma[c] if use_gamma else gamma[0], use_gamma, out)
        if use_gamma:
            for s in range(n):
                for a in range(5):
                    for d in range(5):
                        out[s, a, s, d] += dgamma[c, s, a, d]
        for s in range(n):
            for a in range(5):
                for t in range(n):
                    for b in range(5):
                        K[k, s, a, t, b] = out[s, a, t, b]


@njit(cache=True, parallel=True)
def finalize_pairs_kernel(data, base, n_r, n_c, first, n_slots, rows, cols, det, TQ, gamma, use_gamma, out):
    """同组耦合槽位去重求和并变换：唯一单元对 `u` 的槽位是 `[first[u], first[u+1])`。"""
    blk = 25 * n_r * n_c
    n_u = first.shape[0]
    for u in prange(n_u):
        k0 = first[u]
        k1 = first[u + 1] if u + 1 < n_u else n_slots
        X = np.zeros((n_r, 5, n_c, 5))
        for k in range(k0, k1):
            p = base + k * blk
            for s in range(n_r):
                for a in range(5):
                    for t in range(n_c):
                        for b in range(5):
                            X[s, a, t, b] += data[p]
                            p += 1
        r, c = rows[k0], cols[k0]
        res = np.empty((n_r, 5, n_c, 5))
        _transform_block(X, det[r], TQ[c], gamma[r] if use_gamma else gamma[0], use_gamma, res)
        for s in range(n_r):
            for a in range(5):
                for t in range(n_c):
                    for b in range(5):
                        out[u, s, a, t, b] = res[s, a, t, b]
