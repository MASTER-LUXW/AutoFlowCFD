"""AutoFlowCFD V2.0 - k-omega 方程逐点量对 `(k, omega, grad k, grad omega)` 的导数。

冻结平均流下，湍流残差里的非线性全部落在两个逐点函数上：

* 源项 `S = (S_k, S_omega)`（`sst/source.py::compute_source_terms`：产生、耗散、
  交叉扩散、环境维持项，模型项取 realizability 有效值）；
* 有效扩散系数 `Gamma = (Gamma_k, Gamma_omega)`（`transport/residual.py::
  turbulence_diffusivities`，`mu + sigma(F1) rho nu_t`，`nu_t` 由源项求值在同一组
  `(k, omega)` 上刷新）。

两者在每个解点上只依赖该点的 `(k, omega, grad k, grad omega)`（梯度按
`sst/bounds.py::clip_gradient_magnitude` 裁剪之后使用，与残差同一顺序）。
这里对**残差用的那两个函数**逐输入做前向差分：逐点函数对全场同时平移一个
输入分量，每个解点得到的就是它自己的偏导（8 个输入、共 9 次整场求值），不写
解析式，模型函数日后修改导数自动跟随。

输入排布：`k | omega | grad k (3) | grad omega (3)`。
"""

import numpy as np

from autoflowcfd.core.turbulence.sst.bounds import clip_gradient_magnitude
from autoflowcfd.core.turbulence.transport.residual import turbulence_diffusivities

#: 逐点输入个数。
N_TURB_INPUTS = 8

_SQRT_EPS = float(np.sqrt(np.finfo(np.float64).eps))

#: 模型上被源项求值刷新的缓存属性（与 `fr_solver/turbulence/implicit.py` 同一组）。
_CACHED_ATTRS = ("nu_t", "_last_beta_blend", "_omega_realizability_min")


class CpuTurbulencePointwise:
    """单机 / CPU 分布式：在模型对象上求逐点 `(S, Gamma)`，`(n_cells, n_sps, 2)` 两份。

    求值时临时把 `k/omega` 场换成试探值，结束后恢复场与源项刷新的缓存。做成类
    而不是闭包（项目规范）。
    """

    __slots__ = ("turb", "Q", "grad_vel", "d_wall", "mu", "S_mag")

    def __init__(self, turb, Q, grad_vel, d_wall, mu):
        self.turb, self.Q, self.grad_vel, self.d_wall, self.mu = turb, Q, grad_vel, d_wall, float(mu)
        self.S_mag = turb.compute_strain_rate_magnitude(grad_vel)

    def __call__(self, k, omega, grad_k, grad_omega):
        turb = self.turb
        saved = (turb.k_field, turb.omega_field)
        saved_cache = {a: getattr(turb, a) for a in _CACHED_ATTRS if hasattr(turb, a)}
        turb.k_field, turb.omega_field = k, omega
        try:
            gk = clip_gradient_magnitude(grad_k, np)
            gw = clip_gradient_magnitude(grad_omega, np)
            Sk, Sw = turb.compute_source_terms(self.Q, self.grad_vel, self.d_wall, self.mu,
                                               grad_k=gk, grad_omega=gw)
            rho = self.Q[:, :, 0]
            nu = self.mu / np.maximum(rho, 1e-10)
            Gk, Gw = turbulence_diffusivities(turb, k, omega, gk, gw, rho, rho * turb.nu_t, nu, self.mu,
                                              self.S_mag, self.d_wall)
        finally:
            turb.k_field, turb.omega_field = saved
            for a, v in saved_cache.items():
                setattr(turb, a, v)
        return np.stack([Sk, Sw], axis=-1), np.stack([Gk, Gw], axis=-1)


def turbulence_pointwise_partials(evaluate, k, omega, grad_k, grad_omega, scales):
    """返回 `(S, Gamma, dS, dGamma)`（主机 numpy）：`S/Gamma (n_cells, n_sps, 2)`，
    `dS/dGamma (n_cells, n_sps, 2, 8)`（最后一维是输入，排布见模块文档）。

    `evaluate(k, omega, grad_k, grad_omega) -> (S, Gamma)` 是后端给出的逐点求值
    （`CpuTurbulencePointwise` 等），输入与输出可以在该后端的数组模块上；`scales`
    是 `(k_scale, omega_scale)`（`sst/bounds.py::turbulence_scales`）。
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
