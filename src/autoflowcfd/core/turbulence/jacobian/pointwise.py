"""AutoFlowCFD V2.0 - k-ln(omega) 方程逐点量对 `(k, w, grad k, grad w)` 的导数（`w = ln omega`）。

冻结平均流下，湍流残差里的非线性全部落在两个逐点函数上：

* 源项 `S = (S_k, S_w)`：`S_k` 与 `S_omega` 来自 `sst/source.py::compute_source_terms`
  （产生、耗散、交叉扩散、环境维持项，模型项取 realizability 有效值，求值用物理
  `omega = exp(w)` 与 `grad omega = omega grad w`），`S_w = S_omega / omega + Gamma_w |grad w|^2`
  （后一项是 `sst/log_omega.py` 的变换带出的，逐点、依赖 `grad w`）；
* 有效扩散系数 `Gamma = (Gamma_k, Gamma_w)`（`transport/residual.py::
  turbulence_diffusivities`，`mu + sigma(F1) rho nu_t`，`nu_t` 由源项求值在同一组
  `(k, omega)` 上刷新）。

两者在每个解点上只依赖该点的 `(k, w, grad k, grad w)`（梯度按
`sst/bounds.py::clip_gradient_magnitude` 裁剪之后使用，与残差同一顺序）。
这里对**残差用的那两个函数**逐输入做前向差分：逐点函数对全场同时平移一个
输入分量，每个解点得到的就是它自己的偏导（8 个输入、共 9 次整场求值），不写
解析式，模型函数日后修改导数自动跟随。

输入排布：`k | w | grad k (3) | grad w (3)`。
"""

import numpy as np

from autoflowcfd.core.turbulence.sst.bounds import clip_gradient_magnitude
from autoflowcfd.core.turbulence.sst.log_omega import log_omega_gradient_source, omega_from_log
from autoflowcfd.core.turbulence.transport.residual import turbulence_diffusivities

#: 逐点输入个数。
N_TURB_INPUTS = 8

_SQRT_EPS = float(np.sqrt(np.finfo(np.float64).eps))

#: 模型上被源项求值刷新的缓存属性：试探求值之后必须恢复，否则试探场会泄漏进平均流用的
#: `nu_t`（CPU 与 GPU 的 SST 模型同名；耦合 Newton 的快照还原与这里共用这一份）。
CACHED_MODEL_ATTRS = ("nu_t", "_last_beta_blend", "_omega_realizability_min")


class TurbulencePointwise:
    """在模型对象上求逐点 `(S, Gamma)`（主机数组，`(n_cells, n_sps, 2)` 两份），全部后端
    共用；后端只注入它自己的源项与扩散系数求值（与该后端残差同一份）：

    * `source(Q, grad_vel, d_wall, mu, grad_k, grad_omega) -> (S_k, S_omega)`；
    * `diffusivities(turb, k, omega, grad_k, grad_w, rho, rho_nu_t, nu, mu, S_mag, d_wall)
      -> (Gamma_k, Gamma_w)`。

    求值时临时把 `k/omega` 场换成试探值（`omega = exp(w)`），结束后恢复场与源项刷新
    的缓存。做成类而不是闭包（项目规范）。CPU 用 `cpu_turbulence_pointwise` 构造。
    """

    __slots__ = ("xp", "turb", "Q", "grad_vel", "d_wall", "mu", "S_mag", "source", "diffusivities")

    def __init__(self, xp, turb, Q, grad_vel, d_wall, mu, S_mag, source, diffusivities):
        self.xp, self.turb, self.Q, self.grad_vel, self.d_wall = xp, turb, Q, grad_vel, d_wall
        self.mu, self.S_mag, self.source, self.diffusivities = float(mu), S_mag, source, diffusivities

    def __call__(self, k, w, grad_k, grad_w):
        xp, turb = self.xp, self.turb
        saved = (turb.k_field, turb.omega_field)
        saved_cache = {a: getattr(turb, a) for a in CACHED_MODEL_ATTRS if hasattr(turb, a)}
        omega = omega_from_log(xp.asarray(w), turb.omega_max, xp)
        turb.k_field, turb.omega_field = xp.asarray(k), omega
        try:
            gk = clip_gradient_magnitude(xp.asarray(grad_k), xp)
            gw = clip_gradient_magnitude(xp.asarray(grad_w), xp)
            Sk, Sw = self.source(self.Q, self.grad_vel, self.d_wall, self.mu, gk, omega[..., None] * gw)
            rho = self.Q[:, :, 0]
            nu = self.mu / xp.maximum(rho, 1e-10)
            Gk, Gw = self.diffusivities(turb, turb.k_field, omega, gk, gw, rho, rho * turb.nu_t, nu, self.mu,
                                        self.S_mag, self.d_wall)
        finally:
            turb.k_field, turb.omega_field = saved
            for a, v in saved_cache.items():
                setattr(turb, a, v)
        S_w = Sw / omega + log_omega_gradient_source(Gw, gw, xp)
        return _host(xp.stack([Sk, S_w], axis=-1)), _host(xp.stack([Gk, Gw], axis=-1))


def cpu_turbulence_pointwise(turb, Q, grad_vel, d_wall, mu) -> TurbulencePointwise:
    """单机 / CPU 分布式的逐点求值器（`SSTModelFR` 的源项与 `turbulence_diffusivities`）。"""
    return TurbulencePointwise(np, turb, Q, grad_vel, d_wall, mu, turb.compute_strain_rate_magnitude(grad_vel),
                               turb.compute_source_terms, turbulence_diffusivities)


def turbulence_pointwise_partials(evaluate, k, omega, grad_k, grad_omega, scales):
    """返回 `(S, Gamma, dS, dGamma)`（主机 numpy）：`S/Gamma (n_cells, n_sps, 2)`，
    `dS/dGamma (n_cells, n_sps, 2, 8)`（最后一维是输入，排布见模块文档）。

    `evaluate(k, w, grad_k, grad_w) -> (S, Gamma)` 是后端给出的逐点求值
    （`TurbulencePointwise`），输入与输出可以在该后端的数组模块上；参数名
    `omega/grad_omega` 在这里就是第二个未知量 `w = ln omega` 及其梯度；`scales`
    是 `(k_scale, 1)`（`sst/bounds.py::turbulence_scales`）。
    """
    xp = _array_module(k)
    S0, G0 = evaluate(k, omega, grad_k, grad_omega)
    shape = k.shape
    dS = np.empty(shape + (2, N_TURB_INPUTS))
    dG = np.empty(shape + (2, N_TURB_INPUTS))
    k_scale, w_scale = scales
    # 梯度分量的步长尺度：该点梯度模长，零梯度点取全场平均模长（逐点函数对梯度
    # 光滑，步长只需量级合理）
    gk_norm = xp.linalg.norm(grad_k, axis=-1)
    gw_norm = xp.linalg.norm(grad_omega, axis=-1)
    gk_norm = xp.maximum(gk_norm, max(float(gk_norm.mean()), 1e-12))
    gw_norm = xp.maximum(gw_norm, max(float(gw_norm.mean()), 1e-12))
    for j in range(N_TURB_INPUTS):
        kk, ww, gk, gw = k, omega, grad_k, grad_omega
        if j == 0:
            h = _SQRT_EPS * (xp.abs(k) + k_scale)
            kk = k + h
        elif j == 1:
            h = _SQRT_EPS * (xp.abs(omega) + w_scale)
            ww = omega + h
        elif j < 5:
            h = _SQRT_EPS * (xp.abs(grad_k[..., j - 2]) + gk_norm)
            gk = grad_k.copy()
            gk[..., j - 2] += h
        else:
            h = _SQRT_EPS * (xp.abs(grad_omega[..., j - 5]) + gw_norm)
            gw = grad_omega.copy()
            gw[..., j - 5] += h
        S1, G1 = evaluate(kk, ww, gk, gw)
        dS[..., :, j] = _host((S1 - S0) / h[..., None])
        dG[..., :, j] = _host((G1 - G0) / h[..., None])
    return _host(S0), _host(G0), dS, dG


def _array_module(a):
    from autoflowcfd.core.utils.array_module import array_module
    return array_module(a)


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)
