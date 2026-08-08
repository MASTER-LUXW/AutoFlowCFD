"""CUDA GPU kernel：AUSM+up 无粘通量（未经真实 GPU 硬件验证）。

⚠️ 重要说明：本文件是 fvm_inviscid_kernels.py 里已经用真实数值结果验证过
（与 numpy 参考实现在随机测试数据上误差 ~1e-7 绝对/机器精度级相对误差）
的 Numba CPU kernel 的逐行 CUDA 翻译。写这个文件时的开发环境没有可用的
GPU（numba.cuda.is_available() 为 False，未安装 cupy），所以这里的
kernel **从未在真实 GPU 硬件上编译、运行、验证过**——只能靠对照 CPU 版本
公式做逐行审查来把关，没有 fvm_inviscid_kernels.py 那样的真实运行结果
比对。

在任何生产环境依赖这条路径之前，必须先在有 GPU 的机器上跑一次和
fvm_inviscid_kernels.py 同样的数值对比验证（把 CPU 参考结果和这里的 GPU
结果对比，确认误差在浮点精度范围内），并跑一次真实算例确认收敛行为一致。
"""

import math

import numpy as np

try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False

GAMMA = 1.4
_AUSM_KP = 0.25
_AUSM_KU = 0.75
_AUSM_SIGMA = 1.0
_AUSM_BETA = 1.0 / 8.0
_MAX_VELOCITY = 1e4
_MAX_ENERGY = 1e12


if CUDA_AVAILABLE:
    @cuda.jit(device=True, inline=True)
    def _m1_plus(M):
        return 0.5 * (M + abs(M))

    @cuda.jit(device=True, inline=True)
    def _m1_minus(M):
        return 0.5 * (M - abs(M))

    @cuda.jit(device=True, inline=True)
    def _m2_plus(M):
        return 0.25 * (M + 1.0) ** 2

    @cuda.jit(device=True, inline=True)
    def _m2_minus(M):
        return -0.25 * (M - 1.0) ** 2

    @cuda.jit(cache=True)
    def _ausm_up_flux_kernel_gpu(primL, primR, normal, mach_ref, flux):
        """一个线程处理一个面 - 逐面独立，无需原子操作（结果直接写各自的
        flux[i,:] 行，不同线程互不干扰）。公式和
        fvm_inviscid_kernels._ausm_up_flux_kernel 逐行对应。"""
        i = cuda.grid(1)
        if i >= primL.shape[0]:
            return

        rhoL = primL[i, 0]; uL = primL[i, 1]; vL = primL[i, 2]; wL = primL[i, 3]
        pL = primL[i, 4]; kL = primL[i, 5]; wkL = primL[i, 6]
        rhoR = primR[i, 0]; uR = primR[i, 1]; vR = primR[i, 2]; wR = primR[i, 3]
        pR = primR[i, 4]; kR = primR[i, 5]; wkR = primR[i, 6]
        nx = normal[i, 0]; ny = normal[i, 1]; nz = normal[i, 2]

        vel_mag_L = math.sqrt(uL * uL + vL * vL + wL * wL)
        vel_mag_R = math.sqrt(uR * uR + vR * vR + wR * wR)
        clip_L = min(1.0, _MAX_VELOCITY / max(vel_mag_L, 1e-12))
        clip_R = min(1.0, _MAX_VELOCITY / max(vel_mag_R, 1e-12))
        uL *= clip_L; vL *= clip_L; wL *= clip_L
        uR *= clip_R; vR *= clip_R; wR *= clip_R

        rhoL = max(rhoL, 1e-9)
        rhoR = max(rhoR, 1e-9)
        pL = max(pL, 1.0)
        pR = max(pR, 1.0)

        unL = uL * nx + vL * ny + wL * nz
        unR = uR * nx + vR * ny + wR * nz

        EL = pL / (GAMMA - 1.0) + 0.5 * rhoL * (uL * uL + vL * vL + wL * wL)
        ER = pR / (GAMMA - 1.0) + 0.5 * rhoR * (uR * uR + vR * vR + wR * wR)
        EL = min(EL, _MAX_ENERGY)
        ER = min(ER, _MAX_ENERGY)
        HL = (EL + pL) / rhoL
        HR = (ER + pR) / rhoR

        a_crit_L = math.sqrt(max(2.0 * (GAMMA - 1.0) / (GAMMA + 1.0) * HL, 1e-12))
        a_crit_R = math.sqrt(max(2.0 * (GAMMA - 1.0) / (GAMMA + 1.0) * HR, 1e-12))
        a_hat_L = a_crit_L * a_crit_L / max(a_crit_L, unL)
        a_hat_R = a_crit_R * a_crit_R / max(a_crit_R, -unR)
        a_half = max(min(a_hat_L, a_hat_R), 1e-6)

        ML = unL / a_half
        MR = unR / a_half

        rho_half = 0.5 * (rhoL + rhoR)
        Mbar2 = (unL * unL + unR * unR) / (2.0 * a_half * a_half)
        M0_2 = max(Mbar2, mach_ref * mach_ref)
        if M0_2 > 1.0:
            M0_2 = 1.0
        if M0_2 < 0.0:
            M0_2 = 0.0
        sqrt_M0_2 = math.sqrt(M0_2)
        f_a = max(sqrt_M0_2 * (2.0 - sqrt_M0_2), 1e-6)
        alpha = 3.0 / 16.0 * (-4.0 + 5.0 * f_a * f_a)

        subL = abs(ML) < 1.0
        subR = abs(MR) < 1.0

        if subL:
            M4_plus = _m2_plus(ML) * (1.0 - 16.0 * _AUSM_BETA * _m2_minus(ML))
        else:
            M4_plus = _m1_plus(ML)
        if subR:
            M4_minus = _m2_minus(MR) * (1.0 + 16.0 * _AUSM_BETA * _m2_plus(MR))
        else:
            M4_minus = _m1_minus(MR)

        if subL:
            P5_plus = _m2_plus(ML) * ((2.0 - ML) - 16.0 * alpha * ML * _m2_minus(ML))
        else:
            ML_safe = ML if ML != 0.0 else 1.0
            P5_plus = _m1_plus(ML) / ML_safe
        if subR:
            P5_minus = _m2_minus(MR) * ((-2.0 - MR) + 16.0 * alpha * MR * _m2_plus(MR))
        else:
            MR_safe = MR if MR != 0.0 else 1.0
            P5_minus = _m1_minus(MR) / MR_safe

        Mp = (-_AUSM_KP / f_a) * max(1.0 - _AUSM_SIGMA * Mbar2, 0.0) \
            * (pR - pL) / (rho_half * a_half * a_half)
        M_half = M4_plus + M4_minus + Mp

        pu = -_AUSM_KU * P5_plus * P5_minus * (rhoL + rhoR) * f_a * a_half * (unR - unL)
        p_half = P5_plus * pL + P5_minus * pR + pu

        mdot = a_half * M_half * (rhoL if M_half > 0 else rhoR)

        pos = mdot >= 0
        u_up = uL if pos else uR
        v_up = vL if pos else vR
        w_up = wL if pos else wR
        H_up = HL if pos else HR
        k_up = kL if pos else kR
        wk_up = wkL if pos else wkR

        flux[i, 0] = mdot
        flux[i, 1] = mdot * u_up + p_half * nx
        flux[i, 2] = mdot * v_up + p_half * ny
        flux[i, 3] = mdot * w_up + p_half * nz
        flux[i, 4] = mdot * H_up
        flux[i, 5] = mdot * k_up
        flux[i, 6] = mdot * wk_up


def ausm_up_flux_gpu(primL: np.ndarray, primR: np.ndarray, normal: np.ndarray,
                      mach_ref: float) -> np.ndarray:
    """Host-side launcher: copy to device, launch, copy back.

    ⚠️ 未经真实 GPU 硬件验证，见本文件模块级文档字符串。
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA GPU not available in this environment")
    n = primL.shape[0]
    d_primL = cuda.to_device(np.ascontiguousarray(primL, dtype=np.float64))
    d_primR = cuda.to_device(np.ascontiguousarray(primR, dtype=np.float64))
    d_normal = cuda.to_device(np.ascontiguousarray(normal, dtype=np.float64))
    d_flux = cuda.device_array((n, 7), dtype=np.float64)

    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block
    _ausm_up_flux_kernel_gpu[blocks, threads_per_block](d_primL, d_primR, d_normal, mach_ref, d_flux)
    return d_flux.copy_to_host()
