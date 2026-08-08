"""Numba CPU kernel：AUSM+up 无粘通量（逐面循环版）。

fvm_viscous_residual.py 的 `_ausm_up` 是这套 RANS-SST 求解器实际使用的
无粘通量格式（HLLC 保留作参考，未接入主流程）。这里把同一套 Liou 2006
AUSM+up 公式从向量化 numpy 逐项翻译成显式的逐面循环 + @njit(parallel=True)
（每个面独立计算，天然无数据竞争，可以安全并行）。公式、每一步的数值
保护（速度裁剪、分母下限等）都和原 numpy 实现逐行对应，不是重新推导。
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

GAMMA = 1.4
_AUSM_KP = 0.25
_AUSM_KU = 0.75
_AUSM_SIGMA = 1.0
_AUSM_BETA = 1.0 / 8.0
_MAX_VELOCITY = 1e4
_MAX_ENERGY = 1e12


@njit(inline='always', cache=True)
def _m1_plus(M):
    return 0.5 * (M + abs(M))


@njit(inline='always', cache=True)
def _m1_minus(M):
    return 0.5 * (M - abs(M))


@njit(inline='always', cache=True)
def _m2_plus(M):
    return 0.25 * (M + 1.0) ** 2


@njit(inline='always', cache=True)
def _m2_minus(M):
    return -0.25 * (M - 1.0) ** 2


@njit(parallel=True, cache=True)
def _ausm_up_flux_kernel(
    primL: np.ndarray,   # (n_faces, 7): rho, u, v, w, p, k, omega
    primR: np.ndarray,
    normal: np.ndarray,  # (n_faces, 3)
    mach_ref: float,
) -> np.ndarray:
    """AUSM+up flux, one row per face - direct per-face translation of
    ViscousRANSResidual._ausm_up (see that method's own docstring for the
    physical derivation/references). Returns (n_faces, 7)."""
    n = primL.shape[0]
    flux = np.zeros((n, 7), dtype=np.float64)

    for i in prange(n):
        rhoL = primL[i, 0]; uL = primL[i, 1]; vL = primL[i, 2]; wL = primL[i, 3]
        pL = primL[i, 4]; kL = primL[i, 5]; wkL = primL[i, 6]
        rhoR = primR[i, 0]; uR = primR[i, 1]; vR = primR[i, 2]; wR = primR[i, 3]
        pR = primR[i, 4]; kR = primR[i, 5]; wkR = primR[i, 6]
        nx = normal[i, 0]; ny = normal[i, 1]; nz = normal[i, 2]

        # === NUMERICAL STABILITY: clip velocity magnitude. ===
        vel_mag_L = np.sqrt(uL * uL + vL * vL + wL * wL)
        vel_mag_R = np.sqrt(uR * uR + vR * vR + wR * wR)
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

        # Interface (critical) speed of sound, Liou 2006 eq. 30-33.
        a_crit_L = np.sqrt(max(2.0 * (GAMMA - 1.0) / (GAMMA + 1.0) * HL, 1e-12))
        a_crit_R = np.sqrt(max(2.0 * (GAMMA - 1.0) / (GAMMA + 1.0) * HR, 1e-12))
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
        sqrt_M0_2 = np.sqrt(M0_2)
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

    return flux
