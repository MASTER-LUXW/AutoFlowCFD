"""Numba CPU kernel：SST k-omega 湍流的逐单元计算（涡粘性 + 源项）。

对应 fvm_viscous_residual.py 的 ViscousRANSResidual._eddy_viscosity /
_f1_blend / _sst_sources。这几个都是纯逐单元的公式（不涉及跨单元的面
scatter），本来向量化 numpy 实现效率就不差，这里翻译成 @njit(parallel=True)
主要是为了和无粘/粘性通量部分风格一致、进一步减少 Python 层开销，公式与
原 numpy 实现逐项对应。
"""

import numpy as np

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

SST_A1 = 0.31
SST_BETA_STAR = 0.09
SST_SIGMA_W2 = 0.856


@njit(parallel=True, cache=True)
def _eddy_viscosity_kernel(
    rho: np.ndarray, k: np.ndarray, omega: np.ndarray,
    grad_vel: np.ndarray,  # (n_cells, 3, 3)
    wall_distance: np.ndarray, mu_lam: float,
) -> np.ndarray:
    """SST eddy viscosity mu_t = rho*a1*k / max(a1*omega, |S|*F2)."""
    n = rho.shape[0]
    mu_t = np.zeros(n, dtype=np.float64)
    mu_t_cap = 1e5 * mu_lam

    for i in prange(n):
        # Symmetric strain rate magnitude |S| = sqrt(2 Sij Sij),
        # S = 0.5*(grad_vel + grad_vel^T).
        s2 = 0.0
        for a in range(3):
            for b in range(3):
                sab = 0.5 * (grad_vel[i, a, b] + grad_vel[i, b, a])
                s2 += sab * sab
        Smag = np.sqrt(2.0 * s2 + 1e-30)

        nu = mu_lam / rho[i]
        d = wall_distance[i]
        omega_safe = max(omega[i], 1e-8)
        k_pos = max(k[i], 0.0)

        arg2 = max(
            2.0 * np.sqrt(k_pos) / (SST_BETA_STAR * omega_safe * d),
            500.0 * nu / (d * d * omega_safe),
        )
        F2 = np.tanh(arg2 * arg2)
        denom = max(SST_A1 * omega_safe, Smag * F2)
        mt = rho[i] * SST_A1 * k_pos / max(denom, 1e-12)
        if mt < 0.0:
            mt = 0.0
        if mt > mu_t_cap:
            mt = mu_t_cap
        mu_t[i] = mt

    return mu_t


@njit(parallel=True, cache=True)
def _sst_sources_kernel(
    rho: np.ndarray, k: np.ndarray, omega: np.ndarray, mu_t: np.ndarray,
    grad_vel: np.ndarray,      # (n_cells, 3, 3)
    grad_k: np.ndarray, grad_omega: np.ndarray,  # (n_cells, 3)
    wall_distance: np.ndarray, mu_lam: float,
    sigma_w1: float, sigma_w2: float,
    beta1: float, beta2: float,
    gamma1: float, gamma2: float,
) -> np.ndarray:
    """SST production/dissipation/cross-diffusion source terms.

    Returns (n_cells, 2): column 0 is (Pk - Dk) for the rho*k equation,
    column 1 is (Pw - Dw + cross) for the rho*omega equation - the caller
    subtracts these from the residual (residual[:,5] -= col0,
    residual[:,6] -= col1), matching ViscousRANSResidual._sst_sources.
    """
    n = rho.shape[0]
    out = np.zeros((n, 2), dtype=np.float64)

    for i in prange(n):
        s2 = 0.0
        for a in range(3):
            for b in range(3):
                sab = 0.5 * (grad_vel[i, a, b] + grad_vel[i, b, a])
                s2 += sab * sab
        Smag = np.sqrt(2.0 * s2 + 1e-30)

        d = wall_distance[i]
        nu = mu_lam / rho[i]
        omega_safe = max(omega[i], 1e-8)
        k_pos = max(k[i], 0.0)

        gkdotgw = grad_k[i, 0] * grad_omega[i, 0] + grad_k[i, 1] * grad_omega[i, 1] + grad_k[i, 2] * grad_omega[i, 2]
        CDkw = max(2.0 * rho[i] * SST_SIGMA_W2 / omega_safe * gkdotgw, 1e-10)

        arg1a = max(
            np.sqrt(k_pos) / (SST_BETA_STAR * omega_safe * d),
            500.0 * nu / (d * d * omega_safe),
        )
        arg1b = 4.0 * rho[i] * SST_SIGMA_W2 * k[i] / (CDkw * d * d)
        arg1 = min(arg1a, arg1b)
        F1 = np.tanh(arg1 ** 4)

        beta = F1 * beta1 + (1.0 - F1) * beta2
        gamma = F1 * gamma1 + (1.0 - F1) * gamma2
        sigma_w = F1 * sigma_w1 + (1.0 - F1) * sigma_w2

        Pk = mu_t[i] * Smag * Smag
        Pk_lim = 10.0 * SST_BETA_STAR * rho[i] * k[i] * omega[i]
        if Pk > Pk_lim:
            Pk = Pk_lim
        Dk = SST_BETA_STAR * rho[i] * k[i] * omega[i]

        Pw = gamma * rho[i] * Smag * Smag
        Dw = beta * rho[i] * omega[i] * omega[i]

        cross = 2.0 * (1.0 - F1) * rho[i] * sigma_w / omega_safe * gkdotgw
        max_cross = 10.0 * max(abs(Pw), abs(Dw))
        if cross > max_cross:
            cross = max_cross
        if cross < -max_cross:
            cross = -max_cross

        out[i, 0] = Pk - Dk
        out[i, 1] = Pw - Dw + cross

    return out
