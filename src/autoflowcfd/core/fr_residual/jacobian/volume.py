"""AutoFlowCFD V2.0 - 体积项对单元对角块的贡献（原始变量空间，numba）。

体积项只依赖单元自身的解点，所以只进对角块。记 `Q_t` 为解点 `t` 的原始变量，
一段同类单元（棱柱或四面体）共用参考算子：

    c2f (nf, n)      解点 -> 细点插值（无粘与粘性都过积分）
    W   (n, nf, 3)   `K = f2c·D_f - Σ_面 lift·diag(w)·Tn`（体积散度的 L2 投影减去修正项的本侧
                     通量迹，`fr/face_flux_trace.py`），无粘与粘性共用
    adj (C, nf, 3, 3) 通量点上的 adj(J)[m,i] = det * inv_jac[m,i]

无粘 `dU/dt = -(1/det_s) sum_{r,m} W[s,r,m] sum_i adj[r,m,i] F_i(c2f Q)`、粘性同式
取正号且通量还依赖梯度 `grad = c2f Gop Q`（`Gop[s,b,t] = sum_m inv[s,m,b] D[s,t,m]`
是单元内局部物理梯度算子，与 `fr_operators/gradients.py::compute_physical_gradient`
同一个定义）。于是（乘 det_s 之后）

    dQdot_s/dQ_t = sum_r Yq[s,r] c2f[r,t]
                 + sum_r sum_b GF[r,b,t] ( Z_vel[s,r,:,b] 进速度列 + Z_T[s,r,b] dT/dQ_t )

`Yq`/`Z` 是逐点通量导数经 `W` 与 `adj` 收缩的结果，`GF = c2f Gop`。

逐单元的稠密收缩在一个 prange 核里做（按单元并行，单元内无共享写）；逐点通量
导数用 `pointwise.py` 的批量核按块求出。
"""

import numpy as np
from numba import njit, prange

from .pointwise import N_VISC_INPUTS, euler_flux_jacobian, viscous_flux_jacobian

#: 分块瞬态内存上限（字节），逐点导数数组按它决定每块单元数。
CHUNK_BYTES = 256 * 2 ** 20


class VolumeSegment:
    """一段同类单元的体积项参考算子（只含真实解点）。无粘与粘性共用同一组细点插值
    `c2f` 与体积算子 `W = K`（两者都过积分，修正项本侧通量迹已并入 K）。"""

    __slots__ = ("lo", "hi", "n", "c2f", "W", "D_sp")

    def __init__(self, lo, hi, n, c2f, W, D_sp):
        self.lo, self.hi, self.n = lo, hi, n
        self.c2f = np.ascontiguousarray(c2f)
        self.W = np.ascontiguousarray(W)
        self.D_sp = np.ascontiguousarray(D_sp)    # (n, n, 3) 解点微分算子 D[s,t,m]

    def chunk_cells(self) -> int:
        nf = self.c2f.shape[0]
        per_cell = 8 * (nf * 3 * 25 + nf * 3 * 5 * N_VISC_INPUTS + 2 * nf * 9)
        return max(64, int(CHUNK_BYTES // max(per_cell, 1)))


@njit(cache=True, parallel=True)
def _volume_kernel(K, slot0, A, adj_i, W_i, c2f_i, PJ, adj_v, W_v, c2f_v, inv_sp, D, dTdQ):
    """逐单元：`K[slot0 + c] += dQdot/dQ * det`（float32 累加目标）。"""
    n_cells = A.shape[0]
    n = W_i.shape[0]
    nf = W_i.shape[1]
    nfv = W_v.shape[1]
    for c in prange(n_cells):
        out = np.zeros((n, 5, n, 5))
        # ---- 无粘：Bq[r,m] = sum_i adj[r,m,i] A[r,i]（5x5）----
        Bq = np.zeros((nf, 3, 5, 5))
        for r in range(nf):
            for m in range(3):
                for i in range(3):
                    a_ = adj_i[c, r, m, i]
                    if a_ != 0.0:
                        for a in range(5):
                            for b in range(5):
                                Bq[r, m, a, b] += a_ * A[c, r, i, a, b]
        M = np.empty((5, 5))
        for s in range(n):
            for r in range(nf):
                for a in range(5):
                    for b in range(5):
                        M[a, b] = -(W_i[s, r, 0] * Bq[r, 0, a, b] + W_i[s, r, 1] * Bq[r, 1, a, b]
                                    + W_i[s, r, 2] * Bq[r, 2, a, b])
                for t in range(n):
                    e = c2f_i[r, t]
                    if e != 0.0:
                        for a in range(5):
                            for b in range(5):
                                out[s, a, t, b] += M[a, b] * e
        # ---- 粘性：梯度迹算子 GF[r,b,t] = sum_s c2f_v[r,s] sum_m inv[s,m,b] D[s,t,m] ----
        Gop = np.zeros((n, 3, n))
        for s in range(n):
            for t in range(n):
                for m in range(3):
                    d = D[s, t, m]
                    if d != 0.0:
                        for b in range(3):
                            Gop[s, b, t] += inv_sp[c, s, m, b] * d
        GF = np.zeros((nfv, 3, n))
        for r in range(nfv):
            for s in range(n):
                e = c2f_v[r, s]
                if e != 0.0:
                    for b in range(3):
                        for t in range(n):
                            GF[r, b, t] += e * Gop[s, b, t]
        Bv = np.zeros((nfv, 3, 5, N_VISC_INPUTS))
        for r in range(nfv):
            for m in range(3):
                for i in range(3):
                    a_ = adj_v[c, r, m, i]
                    if a_ != 0.0:
                        for a in range(5):
                            for j in range(N_VISC_INPUTS):
                                Bv[r, m, a, j] += a_ * PJ[c, r, i, a, j]
        Z = np.empty((5, N_VISC_INPUTS))
        H = np.empty((5, 4))
        for s in range(n):
            for r in range(nfv):
                for a in range(5):
                    for j in range(N_VISC_INPUTS):
                        Z[a, j] = (W_v[s, r, 0] * Bv[r, 0, a, j] + W_v[s, r, 1] * Bv[r, 1, a, j]
                                   + W_v[s, r, 2] * Bv[r, 2, a, j])
                for t in range(n):
                    e = c2f_v[r, t]
                    # H[a, k]：k=0..2 速度分量 u_k 的梯度链，k=3 温度梯度链
                    for a in range(5):
                        for k in range(4):
                            H[a, k] = (Z[a, 5 + 3 * k] * GF[r, 0, t] + Z[a, 6 + 3 * k] * GF[r, 1, t]
                                       + Z[a, 7 + 3 * k] * GF[r, 2, t])
                    for a in range(5):
                        for b in range(5):
                            out[s, a, t, b] += Z[a, b] * e + H[a, 3] * dTdQ[c, t, b]
                        out[s, a, t, 1] += H[a, 0]
                        out[s, a, t, 2] += H[a, 1]
                        out[s, a, t, 3] += H[a, 2]
        k0 = slot0 + c
        for s in range(n):
            for a in range(5):
                for t in range(n):
                    for b in range(5):
                        K[k0, s, a, t, b] += out[s, a, t, b]


def add_volume_blocks(K, slot0, seg, Q, dTdQ, inv_sp, adj_f, mu_t_f, mu, Pr, Pr_t, gv_f, gT_f):
    """把一块单元（与 `Q` 等同长）的体积项导数加到 `K[slot0:slot0+len]`（乘 det 之后的量）。
    `adj_f`/`mu_t_f`/`gv_f`/`gT_f` 是细点上的度量、涡粘与梯度。"""
    nb = Q.shape[0]
    nf = seg.c2f.shape[0]
    Qf = np.einsum("rt,ctv->crv", seg.c2f, Q, optimize=True)
    Qf_flat = np.ascontiguousarray(Qf.reshape(-1, 5))
    A = euler_flux_jacobian(Qf_flat).reshape(nb, nf, 3, 5, 5)
    PJ = viscous_flux_jacobian(
        Qf_flat,
        np.ascontiguousarray(gv_f.reshape(-1, 3, 3)),
        np.ascontiguousarray(gT_f.reshape(-1, 3)),
        np.ascontiguousarray(mu_t_f.reshape(-1)), mu, Pr, Pr_t,
    ).reshape(nb, nf, 3, 5, N_VISC_INPUTS)
    adj_f = np.ascontiguousarray(adj_f)
    _volume_kernel(K, int(slot0), A, adj_f, seg.W, seg.c2f, PJ, adj_f, seg.W, seg.c2f,
                   np.ascontiguousarray(inv_sp), seg.D_sp, np.ascontiguousarray(dTdQ))
