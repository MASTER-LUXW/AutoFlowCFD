"""AutoFlowCFD V2.0 - Zhang–Shu 正性保持限制器的数值核心（CPU numba / 数组模块无关两版）。

两版实现同一套数学，逐单元：

1. 用守恒权重 `W_cs = w_s * det(J)_cs` 求单元均值 Ū（权重的来历见
   `fr/native_tet/quadrature.py` 模块文档：本格式守恒的离散量就是
   `Σ_s W_cs U_cs`）。补零槽位权重为 0，天然不参与；
2. 点集 = 全部**真实解点** + 全部**面通量点**（`E @ U`，E 的行和为 1）。
   通量点必须在内：Riemann 求解器用的是它们，而 P1 的极值落在顶点、解点未必
   覆盖（原生棱柱 P1 的挤出方向解点是 Gauss 点、不含端面）；
3. 密度：`ρ_min < ε_ρ` 时 `θ1 = (ρ̄ - ε_ρ)/(ρ̄ - ρ_min)`，
   `ρ_s ← ρ̄ + θ1 (ρ_s - ρ̄)`（E 是线性的且行和为 1，通量点值同比收缩）；
4. 压力：对每个 `p_q < ε_p` 的点，在线段 `Ū + t (U_q - Ū)` 上解 `p = ε_p`。
   `p·ρ/(γ-1) = ρE - |m|²/2` 让条件变成 t 的二次方程
   `a t² + b t + c = 0`，`c = ρ̄ (p̄ - ε_p) >= 0`、`t=1` 处值 `< 0`；压力是
   守恒量的凹函数（ρ>0 时），可容许集是凸的，线段上恰有一个交点。
   `θ2 = min_q t_q`，全部 5 个守恒量 `U_s ← Ū + θ2 (U_s - Ū)`；
5. **只改 θ<1 的单元**。其余单元一个比特都不动 —— 正常运行里这个限制器
   从不触发，结果与"没有限制器"逐位相同。

`ε_ρ = 1e-10 ρ̄`、`ε_p = 1e-10 p̄`（**相对**下限）：

* 不引入任何带量纲的下限 —— 此前逐点硬钳的 `p_floor = 1 Pa` 按大气压标定，
  等熵涡验证算例因此不得不从文献的无量纲形式改写成国际单位制；
* 不能用 Zhang–Shu 原文的绝对 `1e-13`：那是针对无量纲变量的。有量纲时
  （能量 ~2.5e5 J/m³）`p = (γ-1)(E - |m|²/2ρ)` 的舍入噪声约 1e-10~1e-8，
  绝对 1e-13 低于浮点分辨率，实测限制后的点压力为 -6.6e-9（本模块第一版
  的单元测试当场抓到）。相对 1e-10 比舍入噪声高 5 个数量级以上，且量纲不变。

单元均值本身不可容许（ρ̄<=0、p̄<=0 或非有限）时**不修**：均值的正性由格式
在 CFL 条件下保证，它坏了就是真正的发散，调用方据此报错。

湍流分量（第 5、6 列）不在这里处理 —— k/omega 的正性由湍流模型自己的
`apply_positivity_limiter` 负责，这里只动前 5 个守恒量。
"""

import numpy as np
from numba import njit, prange

#: 相对下限系数：`ε = _EPS_REL * 单元均值`（理由见模块文档）。
_EPS_REL = 1e-10


@njit(cache=True, inline="always")
def _pressure(rho, mx, my, mz, E, gamma):
    return (gamma - 1.0) * (E - 0.5 * (mx * mx + my * my + mz * mz) / rho)


@njit(cache=True, inline="always")
def _crossing_t(rb, mxb, myb, mzb, Eb, dr, dmx, dmy, dmz, dE, eps_p, gamma):
    """线段 Ū + t·D 上压力恰为 eps_p 的 t ∈ [0,1]（见模块文档第 4 步）。"""
    g1 = gamma - 1.0
    a = g1 * (dr * dE - 0.5 * (dmx * dmx + dmy * dmy + dmz * dmz))
    b = g1 * (rb * dE + dr * Eb - (mxb * dmx + myb * dmy + mzb * dmz)) - eps_p * dr
    c = g1 * (rb * Eb - 0.5 * (mxb * mxb + myb * myb + mzb * mzb)) - eps_p * rb
    if c <= 0.0:
        return 0.0
    if abs(a) < 1e-300:
        if b >= 0.0:
            return 1.0
        t = -c / b
    else:
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            disc = 0.0
        sq = np.sqrt(disc)
        # 数值稳定的求根式，取 [0,1] 内较小的正根
        q = -0.5 * (b + (sq if b >= 0.0 else -sq))
        t1 = q / a
        t2 = c / q if q != 0.0 else 2.0
        t = 2.0
        if 0.0 <= t1 <= 1.0:
            t = t1
        if 0.0 <= t2 <= 1.0 and t2 < t:
            t = t2
        if t > 1.0:
            t = 1.0
    if t < 0.0:
        t = 0.0
    return t


@njit(cache=True, parallel=True)
def zhang_shu_numba(U, W, E_prism, E_tet, cell_is_prism, n_real_prism, n_real_tet,
                    gamma, theta_out, bad_mean_out):
    """就地限制 `U`（形状 `(n_cells, n_sps, n_vars)`）。

    Args:
        W: `(n_cells, n_sps)` 守恒权重 `w_s * det(J)_cs`（补零槽位为 0）。
        cell_is_prism: `(n_cells,)` 布尔，逐单元类型。**不假设"棱柱在前"**：
            单机网格是棱柱在前，但分布式本地编号里两类单元交错。
        E_prism / E_tet: `(m, n_sps)` 该类单元全部面的通量点外插行，按面
            堆叠（棱柱 5 面、四面体 4 面）。
        theta_out: `(n_cells,)` 输出，每单元实际用的 θ（未动的单元为 1）。
        bad_mean_out: `(n_cells,)` 输出，单元均值不可容许的标记。
    """
    n_cells = U.shape[0]
    for c in prange(n_cells):
        theta_out[c] = 1.0
        bad_mean_out[c] = False
        is_prism = cell_is_prism[c]
        nr = n_real_prism if is_prism else n_real_tet
        E = E_prism if is_prism else E_tet
        n_fp = E.shape[0]

        wsum = 0.0
        m0 = 0.0
        m1 = 0.0
        m2 = 0.0
        m3 = 0.0
        m4 = 0.0
        for s in range(nr):
            w = W[c, s]
            wsum += w
            m0 += w * U[c, s, 0]
            m1 += w * U[c, s, 1]
            m2 += w * U[c, s, 2]
            m3 += w * U[c, s, 3]
            m4 += w * U[c, s, 4]
        if not (wsum > 0.0):
            bad_mean_out[c] = True
            continue
        rb = m0 / wsum
        mxb = m1 / wsum
        myb = m2 / wsum
        mzb = m3 / wsum
        Eb = m4 / wsum
        if not (rb > 0.0) or not np.isfinite(rb) or not np.isfinite(Eb):
            bad_mean_out[c] = True
            continue
        pb = _pressure(rb, mxb, myb, mzb, Eb, gamma)
        if not (pb > 0.0) or not np.isfinite(pb):
            bad_mean_out[c] = True
            continue
        eps_r = _EPS_REL * rb
        eps_p = _EPS_REL * pb

        # ---- 密度
        rmin = U[c, 0, 0]
        for s in range(nr):
            if U[c, s, 0] < rmin:
                rmin = U[c, s, 0]
        for q in range(n_fp):
            v = 0.0
            for s in range(nr):
                v += E[q, s] * U[c, s, 0]
            if v < rmin:
                rmin = v
        th1 = 1.0
        if rmin < eps_r:
            th1 = (rb - eps_r) / (rb - rmin)
            if th1 < 0.0:
                th1 = 0.0
            for s in range(nr):
                U[c, s, 0] = rb + th1 * (U[c, s, 0] - rb)

        # ---- 压力（在密度已限制的状态上）
        th2 = 1.0
        for s in range(nr):
            r = U[c, s, 0]
            p = _pressure(r, U[c, s, 1], U[c, s, 2], U[c, s, 3], U[c, s, 4], gamma)
            if p < eps_p:
                t = _crossing_t(rb, mxb, myb, mzb, Eb,
                                r - rb, U[c, s, 1] - mxb, U[c, s, 2] - myb,
                                U[c, s, 3] - mzb, U[c, s, 4] - Eb, eps_p, gamma)
                if t < th2:
                    th2 = t
        for q in range(n_fp):
            r = 0.0
            x1 = 0.0
            x2 = 0.0
            x3 = 0.0
            x4 = 0.0
            for s in range(nr):
                e = E[q, s]
                r += e * U[c, s, 0]
                x1 += e * U[c, s, 1]
                x2 += e * U[c, s, 2]
                x3 += e * U[c, s, 3]
                x4 += e * U[c, s, 4]
            p = _pressure(r, x1, x2, x3, x4, gamma)
            if p < eps_p:
                t = _crossing_t(rb, mxb, myb, mzb, Eb,
                                r - rb, x1 - mxb, x2 - myb, x3 - mzb, x4 - Eb, eps_p, gamma)
                if t < th2:
                    th2 = t
        if th2 < 1.0:
            for s in range(nr):
                U[c, s, 0] = rb + th2 * (U[c, s, 0] - rb)
                U[c, s, 1] = mxb + th2 * (U[c, s, 1] - mxb)
                U[c, s, 2] = myb + th2 * (U[c, s, 2] - myb)
                U[c, s, 3] = mzb + th2 * (U[c, s, 3] - mzb)
                U[c, s, 4] = Eb + th2 * (U[c, s, 4] - Eb)
        theta_out[c] = th1 * th2 if th2 < 1.0 else th1


def zhang_shu_xp(xp, U, W, E_prism, E_tet, cell_is_prism, n_real_prism, n_real_tet, gamma):
    """数组模块无关的向量化实现（GPU 路径用；numpy 上与 numba 版逐位对照）。

    就地修改 `U`；返回 `(theta, bad_mean)`，两者都是 `(n_cells,)`。
    只改 θ<1 的单元（`xp.where` 对未选中位置原样保留，逐位不变）。
    两类单元按 `cell_is_prism` 掩码各自收集、各自写回（同 numba 版，不假设
    单元顺序）。
    """
    n_cells = U.shape[0]
    theta = xp.ones(n_cells, dtype=U.dtype)
    bad = xp.zeros(n_cells, dtype=bool)
    for sel, nr, E in ((cell_is_prism, n_real_prism, E_prism),
                       (~cell_is_prism, n_real_tet, E_tet)):
        idx = xp.nonzero(sel)[0]
        if idx.size == 0:
            continue
        Uc = U[idx, :nr, :5]
        Wc = W[idx, :nr]
        Ec = E[:, :nr]
        wsum = Wc.sum(axis=1)
        mean = (Wc[:, :, None] * Uc).sum(axis=1) / xp.where(wsum > 0, wsum, 1.0)[:, None]
        rb, mxb, myb, mzb, Eb = (mean[:, k] for k in range(5))
        pb = (gamma - 1.0) * (Eb - 0.5 * (mxb ** 2 + myb ** 2 + mzb ** 2) / xp.where(rb > 0, rb, 1.0))
        ok = (wsum > 0) & (rb > 0) & (pb > 0) & xp.isfinite(rb) & xp.isfinite(Eb) & xp.isfinite(pb)
        bad[idx] = ~ok
        eps_r = _EPS_REL * xp.where(ok, rb, 1.0)
        eps_p = _EPS_REL * xp.where(ok, pb, 1.0)

        # ---- 密度
        rho_fp = xp.einsum("qs,cs->cq", Ec, Uc[:, :, 0])
        rmin = xp.minimum(Uc[:, :, 0].min(axis=1), rho_fp.min(axis=1))
        need1 = ok & (rmin < eps_r)
        th1 = xp.where(need1, xp.clip((rb - eps_r) / xp.where(need1, rb - rmin, 1.0), 0.0, 1.0), 1.0)
        rho_new = xp.where(need1[:, None], rb[:, None] + th1[:, None] * (Uc[:, :, 0] - rb[:, None]),
                           Uc[:, :, 0])
        Uc[:, :, 0] = rho_new          # 花式索引取出的本来就是副本

        # ---- 压力：点集 = 解点 + 通量点
        U_fp = xp.einsum("qs,csv->cqv", Ec, Uc)
        P = xp.concatenate([Uc, U_fp], axis=1)                     # (c, n_pts, 5)
        r = P[..., 0]
        p = (gamma - 1.0) * (P[..., 4] - 0.5 * (P[..., 1] ** 2 + P[..., 2] ** 2 + P[..., 3] ** 2)
                             / xp.where(r > 0, r, 1.0))
        viol = ok[:, None] & (p < eps_p[:, None])
        D = P - mean[:, None, :]
        g1 = gamma - 1.0
        a = g1 * (D[..., 0] * D[..., 4] - 0.5 * (D[..., 1] ** 2 + D[..., 2] ** 2 + D[..., 3] ** 2))
        b = g1 * (rb[:, None] * D[..., 4] + D[..., 0] * Eb[:, None]
                  - (mxb[:, None] * D[..., 1] + myb[:, None] * D[..., 2] + mzb[:, None] * D[..., 3])) \
            - eps_p[:, None] * D[..., 0]
        cc = (g1 * (rb * Eb - 0.5 * (mxb ** 2 + myb ** 2 + mzb ** 2)) - eps_p * rb)[:, None] \
            * xp.ones_like(a)
        small_a = xp.abs(a) < 1e-300
        disc = xp.maximum(b * b - 4.0 * a * cc, 0.0)
        sq = xp.sqrt(disc)
        q = -0.5 * (b + xp.where(b >= 0, sq, -sq))
        t1 = q / xp.where(small_a, 1.0, a)
        t2 = xp.where(q != 0, cc / xp.where(q != 0, q, 1.0), 2.0)
        in1 = (t1 >= 0) & (t1 <= 1)
        in2 = (t2 >= 0) & (t2 <= 1)
        tq = xp.where(in1, t1, 2.0)
        tq = xp.where(in2 & (t2 < tq), t2, tq)
        tq = xp.minimum(tq, 1.0)
        t_lin = xp.where(b >= 0, 1.0, -cc / xp.where(b != 0, b, -1.0))
        tq = xp.where(small_a, t_lin, tq)
        tq = xp.where(cc <= 0, 0.0, tq)
        tq = xp.clip(tq, 0.0, 1.0)
        th2 = xp.where(viol, tq, 1.0).min(axis=1)
        need2 = th2 < 1.0
        U[idx, :nr, :5] = xp.where(need2[:, None, None],
                                   mean[:, None, :] + th2[:, None, None] * (Uc - mean[:, None, :]),
                                   Uc)
        theta[idx] = xp.where(need2, th1 * th2, th1)
    return theta, bad
