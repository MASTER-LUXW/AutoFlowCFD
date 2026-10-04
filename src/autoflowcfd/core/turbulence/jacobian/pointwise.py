"""AutoFlowCFD V2.0 - 湍流方程逐点量对 `(phi, grad phi)` 的导数（`phi` 为模型的 Newton 未知量）。

冻结平均流下，湍流残差里的非线性全部落在两个逐点函数上：源项 `S` 与有效扩散系数
`Gamma`（各 `nv` 个分量，`nv` 为模型的输运标量个数）。两者在每个解点上只依赖该点的
`(phi, grad phi)`（梯度按模型自己的裁剪规则处理之后使用，与残差同一顺序）。这里对**残差用的
那两个函数**逐输入做前向差分：逐点函数对全场同时平移一个输入分量，每个解点得到的就是它自己
的偏导（`4 nv` 个输入、共 `4 nv + 1` 次整场求值），不写解析式，模型函数日后修改导数自动跟随。

输入排布：`phi_0 .. phi_{nv-1} | grad phi_0 (3) | ... | grad phi_{nv-1} (3)`。

求值器（`evaluate(u, g) -> (S, Gamma)`，`u (n_cells, n_sps, nv)`、`g (n_cells, n_sps, nv, 3)`）由模型
给出：

* SST（`phi = (k, ln omega)`）：`TurbulencePointwise`。`S_k`、`S_omega` 来自
  `sst/source.py::compute_source_terms`（模型项取 realizability 有效值，求值用物理
  `omega = exp(w)` 与 `grad omega = omega grad w`），`S_w = S_omega / omega + Gamma_w |grad w|^2`
  （后一项是 `sst/log_omega.py` 的变换带出的），`Gamma = mu + sigma(F1) rho nu_t`
  （`transport/residual.py::turbulence_diffusivities`，`F1` 与 `nu_t` 由源项求值在同一组场上刷新）；
* SA-neg（`phi = nu_tilde`）：见 `turbulence/sa`。
"""

import numpy as np

from autoflowcfd.core.turbulence.sst.bounds import clip_gradient_magnitude
from autoflowcfd.core.turbulence.sst.log_omega import log_omega_gradient_source, omega_from_log
from autoflowcfd.core.turbulence.transport.residual import turbulence_diffusivities

_SQRT_EPS = float(np.sqrt(np.finfo(np.float64).eps))


def n_pointwise_inputs(nv: int) -> int:
    """`nv` 个未知量的逐点输入个数（未知量本身 + 梯度三个分量）。"""
    return 4 * int(nv)


class TurbulencePointwise:
    """SST 的逐点 `(S, Gamma)` 求值器（主机数组，各 `(n_cells, n_sps, 2)`），全部后端共用；后端只
    注入它自己的源项求值（与该后端残差同一份）
    `source(Q, grad_vel, d_wall, mu, grad_k, grad_omega) -> (S_k, S_omega)`，扩散系数读源项刷新的
    模型缓存（`turbulence_diffusivities`，numpy / cupy 同一份）。

    求值时临时把 `k/omega` 场换成试探值（`omega = exp(w)`），结束后恢复场与源项刷新的缓存。
    做成类而不是闭包（项目规范）。CPU 用 `cpu_turbulence_pointwise` 构造。
    """

    __slots__ = ("xp", "turb", "Q", "grad_vel", "d_wall", "mu", "source")

    def __init__(self, xp, turb, Q, grad_vel, d_wall, mu, source):
        self.xp, self.turb, self.Q, self.grad_vel, self.d_wall = xp, turb, Q, grad_vel, d_wall
        self.mu, self.source = float(mu), source

    def __call__(self, u, g):
        xp, turb = self.xp, self.turb
        snap = turb.field_snapshot()
        w = xp.ascontiguousarray(xp.asarray(u[..., 1]))
        omega = omega_from_log(w, turb.omega_max, xp)
        turb.k_field, turb.omega_field = xp.ascontiguousarray(xp.asarray(u[..., 0])), omega
        try:
            gk = clip_gradient_magnitude(xp.ascontiguousarray(xp.asarray(g[..., 0, :])), xp)
            gw = clip_gradient_magnitude(xp.ascontiguousarray(xp.asarray(g[..., 1, :])), xp)
            Sk, Sw = self.source(self.Q, self.grad_vel, self.d_wall, self.mu, gk, omega[..., None] * gw)
            Gk, Gw = turbulence_diffusivities(turb, self.Q[:, :, 0] * turb.nu_t, self.mu)
        finally:
            turb.field_restore(snap)
        S_w = Sw / omega + log_omega_gradient_source(Gw, gw, xp)
        return _host(xp.stack([Sk, S_w], axis=-1)), _host(xp.stack([Gk, Gw], axis=-1))


def cpu_turbulence_pointwise(turb, Q, grad_vel, d_wall, mu) -> TurbulencePointwise:
    """单机 / CPU 分布式的 SST 逐点求值器（`SSTModelFR` 的源项）。"""
    return TurbulencePointwise(np, turb, Q, grad_vel, d_wall, mu, turb.compute_source_terms)


def turbulence_pointwise_partials(evaluate, u, g, scales):
    """返回 `(S, Gamma, dS, dGamma)`（主机 numpy）：`S/Gamma (n_cells, n_sps, nv)`，
    `dS/dGamma (n_cells, n_sps, nv, 4 nv)`（最后一维是输入，排布见模块文档）。

    `evaluate(u, g) -> (S, Gamma)` 是模型给出的逐点求值，输入与输出可以在该后端的数组模块上；
    `u (n_cells, n_sps, nv)` 为未知量、`g (n_cells, n_sps, nv, 3)` 为其梯度；`scales` 为各未知量的
    尺度（`TransportedTurbulence.unknown_scales`）。
    """
    xp = _array_module(u)
    nv = int(u.shape[-1])
    n_in = n_pointwise_inputs(nv)
    S0, G0 = evaluate(u, g)
    shape = u.shape[:-1]
    dS = np.empty(shape + (nv, n_in))
    dG = np.empty(shape + (nv, n_in))
    # 梯度分量的步长尺度：该点梯度模长，零梯度点取全场平均模长（逐点函数对梯度
    # 光滑，步长只需量级合理）
    g_norm = []
    for v in range(nv):
        nrm = xp.linalg.norm(g[..., v, :], axis=-1)
        g_norm.append(xp.maximum(nrm, max(float(nrm.mean()), 1e-12)))
    for j in range(n_in):
        uu, gg = u, g
        if j < nv:
            h = _SQRT_EPS * (xp.abs(u[..., j]) + scales[j])
            uu = u.copy()
            uu[..., j] += h
        else:
            v, b = divmod(j - nv, 3)
            h = _SQRT_EPS * (xp.abs(g[..., v, b]) + g_norm[v])
            gg = g.copy()
            gg[..., v, b] += h
        S1, G1 = evaluate(uu, gg)
        dS[..., :, j] = _host((S1 - S0) / h[..., None])
        dG[..., :, j] = _host((G1 - G0) / h[..., None])
    return _host(S0), _host(G0), dS, dG


def _array_module(a):
    from autoflowcfd.core.utils.array_module import array_module
    return array_module(a)


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)
