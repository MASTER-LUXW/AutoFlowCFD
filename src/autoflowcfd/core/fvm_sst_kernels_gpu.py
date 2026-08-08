"""CUDA GPU kernel：SST k-omega 湍流的逐单元计算（涡粘性 + 源项），未经真实 GPU 硬件验证。

⚠️ 本文件是 fvm_sst_kernels.py（已用真实数值结果验证，mu_t 位级一致、
源项误差 ~1e-11/1e-14）的逐行 CUDA 翻译。开发环境没有可用 GPU，这里的
kernel **从未在真实 GPU 硬件上编译、运行、验证过**。纯逐单元计算，两个
kernel 都不涉及跨线程的读写，天然无 race condition 风险，但仍然只做了
逐行公式审查，没有真实运行结果比对。

在任何生产环境依赖这条路径之前，必须先在有 GPU 的机器上用和
fvm_sst_kernels.py 同样的方法做数值对比验证。
"""

import numpy as np

try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False

SST_A1 = 0.31
SST_BETA_STAR = 0.09
SST_SIGMA_W2 = 0.856


if CUDA_AVAILABLE:
    import math

    @cuda.jit(cache=True)
    def _eddy_viscosity_kernel_gpu(rho, k, omega, grad_vel, wall_distance, mu_lam, mu_t):
        i = cuda.grid(1)
        if i >= rho.shape[0]:
            return

        s2 = 0.0
        for a in range(3):
            for b in range(3):
                sab = 0.5 * (grad_vel[i, a, b] + grad_vel[i, b, a])
                s2 += sab * sab
        Smag = math.sqrt(2.0 * s2 + 1e-30)

        nu = mu_lam / rho[i]
        d = wall_distance[i]
        omega_safe = max(omega[i], 1e-8)
        k_pos = max(k[i], 0.0)

        arg2 = max(
            2.0 * math.sqrt(k_pos) / (SST_BETA_STAR * omega_safe * d),
            500.0 * nu / (d * d * omega_safe),
        )
        F2 = math.tanh(arg2 * arg2)
        denom = max(SST_A1 * omega_safe, Smag * F2)
        mt = rho[i] * SST_A1 * k_pos / max(denom, 1e-12)
        if mt < 0.0:
            mt = 0.0
        mu_t_cap = 1e5 * mu_lam
        if mt > mu_t_cap:
            mt = mu_t_cap
        mu_t[i] = mt

    @cuda.jit(cache=True)
    def _sst_sources_kernel_gpu(
        rho, k, omega, mu_t, grad_vel, grad_k, grad_omega,
        wall_distance, mu_lam, sigma_w1, sigma_w2, beta1, beta2, gamma1, gamma2,
        out,
    ):
        i = cuda.grid(1)
        if i >= rho.shape[0]:
            return

        s2 = 0.0
        for a in range(3):
            for b in range(3):
                sab = 0.5 * (grad_vel[i, a, b] + grad_vel[i, b, a])
                s2 += sab * sab
        Smag = math.sqrt(2.0 * s2 + 1e-30)

        d = wall_distance[i]
        nu = mu_lam / rho[i]
        omega_safe = max(omega[i], 1e-8)
        k_pos = max(k[i], 0.0)

        gkdotgw = (grad_k[i, 0] * grad_omega[i, 0] + grad_k[i, 1] * grad_omega[i, 1]
                   + grad_k[i, 2] * grad_omega[i, 2])
        CDkw = max(2.0 * rho[i] * SST_SIGMA_W2 / omega_safe * gkdotgw, 1e-10)

        arg1a = max(
            math.sqrt(k_pos) / (SST_BETA_STAR * omega_safe * d),
            500.0 * nu / (d * d * omega_safe),
        )
        arg1b = 4.0 * rho[i] * SST_SIGMA_W2 * k[i] / (CDkw * d * d)
        arg1 = min(arg1a, arg1b)
        F1 = math.tanh(arg1 ** 4)

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


def eddy_viscosity_gpu(rho: np.ndarray, k: np.ndarray, omega: np.ndarray,
                        grad_vel: np.ndarray, wall_distance: np.ndarray,
                        mu_lam: float) -> np.ndarray:
    """⚠️ 未经真实 GPU 硬件验证，见本文件模块级文档字符串。"""
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA GPU not available in this environment")
    n = rho.shape[0]
    d_mu_t = cuda.device_array(n, dtype=np.float64)
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block
    _eddy_viscosity_kernel_gpu[blocks, threads_per_block](
        cuda.to_device(np.ascontiguousarray(rho, dtype=np.float64)),
        cuda.to_device(np.ascontiguousarray(k, dtype=np.float64)),
        cuda.to_device(np.ascontiguousarray(omega, dtype=np.float64)),
        cuda.to_device(np.ascontiguousarray(grad_vel, dtype=np.float64)),
        cuda.to_device(np.ascontiguousarray(wall_distance, dtype=np.float64)),
        mu_lam, d_mu_t,
    )
    return d_mu_t.copy_to_host()


def sst_sources_gpu(
    rho: np.ndarray, k: np.ndarray, omega: np.ndarray, mu_t: np.ndarray,
    grad_vel: np.ndarray, grad_k: np.ndarray, grad_omega: np.ndarray,
    wall_distance: np.ndarray, mu_lam: float,
    sigma_w1: float, sigma_w2: float, beta1: float, beta2: float,
    gamma1: float, gamma2: float,
) -> np.ndarray:
    """⚠️ 未经真实 GPU 硬件验证，见本文件模块级文档字符串。"""
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA GPU not available in this environment")
    n = rho.shape[0]
    d_out = cuda.device_array((n, 2), dtype=np.float64)
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block
    _sst_sources_kernel_gpu[blocks, threads_per_block](
        cuda.to_device(np.ascontiguousarray(rho, dtype=np.float64)),
        cuda.to_device(np.ascontiguousarray(k, dtype=np.float64)),
        cuda.to_device(np.ascontiguousarray(omega, dtype=np.float64)),
        cuda.to_device(np.ascontiguousarray(mu_t, dtype=np.float64)),
        cuda.to_device(np.ascontiguousarray(grad_vel, dtype=np.float64)),
        cuda.to_device(np.ascontiguousarray(grad_k, dtype=np.float64)),
        cuda.to_device(np.ascontiguousarray(grad_omega, dtype=np.float64)),
        cuda.to_device(np.ascontiguousarray(wall_distance, dtype=np.float64)),
        mu_lam, sigma_w1, sigma_w2, beta1, beta2, gamma1, gamma2,
        d_out,
    )
    return d_out.copy_to_host()
