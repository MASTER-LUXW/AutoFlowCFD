"""AutoFlowCFD V2.0 - AUSM+up 批量通量(GPU, 与 kernels.py 逐字对应)"""


from autoflowcfd.core.gpu import get_cupy
from autoflowcfd.core.thermo import GAMMA


def _ausm_up_flux_batch_gpu(Q_L, Q_R, normal, mach_ref):
    """GPU 批量 AUSM+up 通量计算（CuPy 向量化版本，通量内部全部用物理声速）。与 kernels.py::compute_ausm_up_flux 逐字对应，理由/推导
    见该函数文档，这里不重复；两处必须同步修改（该文件模块文档要求
    "逐字对应"）。

    `normal` 是逐 FP 各自独立的方向 `(N, n_fp, 3)`（与 CPU numba kernel 逐 FP 调用
    `compute_ausm_up_flux(qL, qR, normal_at_this_fp, ...)` 对应），分量取 `normal[..., d]`（不保留末尾长度 1 的维度），
    与 `uL` 等 `(N, n_fp)` 逐 FP 量逐元素对齐，不依赖隐式广播。

    Args:
        Q_L, Q_R: (N, n_fp, 5) 左右状态
        normal: (N, n_fp, 3) 单位法向量——逐 FP 各自独立
        mach_ref: 参考（自由来流）马赫数，见 kernels.py::
            compute_ausm_up_flux 文档

    Returns:
        flux: (N, n_fp, 5) 数值通量
    """
    cp = get_cupy()
    gamma = GAMMA

    # 两侧状态不钳制（与 CPU `fr_operators/kernels.py::compute_ausm_up_flux` 同一约定，I-18）
    rhoL = Q_L[..., 0]
    uL, vL, wL = Q_L[..., 1], Q_L[..., 2], Q_L[..., 3]
    pL = Q_L[..., 4]

    rhoR = Q_R[..., 0]
    uR, vR, wR = Q_R[..., 1], Q_R[..., 2], Q_R[..., 3]
    pR = Q_R[..., 4]

    nx = normal[..., 0]
    ny = normal[..., 1]
    nz = normal[..., 2]

    unL = uL * nx + vL * ny + wL * nz
    unR = uR * nx + vR * ny + wR * nz

    aL = cp.sqrt(gamma * pL / rhoL)
    aR = cp.sqrt(gamma * pR / rhoR)

    a_half = 0.5 * (aL + aR)
    rho_half = 0.5 * (rhoL + rhoR)
    Mbar2 = (unL**2 + unR**2) / (2.0 * a_half**2)

    M0_sq = cp.minimum(1.0, cp.maximum(Mbar2, mach_ref**2))
    sqrt_M0_sq = cp.sqrt(M0_sq)
    fa = sqrt_M0_sq * (2.0 - sqrt_M0_sq)
    fa = cp.maximum(fa, 1e-6)

    # M4±/P5± 耗散系数——真实 bug 修复（V2.0 专家组盲审第四次评审，
    # 2026-08-28，#12），与 kernels.py::compute_ausm_up_flux 逐字对应，
    # 完整推导/文献交叉核实见该文件同名注释，这里不重复。
    beta_mass = 1.0 / 8.0
    # P5± 的 α 项不乘 1/4（Liou 2006 式 (24)；2026-09-25 修正，见
    # core/fr_operators/kernels.py::compute_ausm_up_flux 的 P5 注释）。
    alpha_pressure = 3.0 / 16.0 * (-4.0 + 5.0 * fa * fa)

    # 两侧马赫数用界面公共声速归一、质量通量取迎风密度（标准 AUSM+up，与 CPU `kernels.compute_ausm_up_flux` 同式）
    M_L = unL / cp.maximum(a_half, 1e-10)
    M_R = unR / cp.maximum(a_half, 1e-10)

    # M+ / M-
    abs_ML = cp.abs(M_L)
    abs_MR = cp.abs(M_R)
    Mp_L = cp.where(
        abs_ML >= 1.0,
        0.5 * (M_L + abs_ML),
        0.25 * (M_L + 1.0)**2 + beta_mass * (M_L**2 - 1.0)**2,
    )
    Mm_R = cp.where(
        abs_MR >= 1.0,
        0.5 * (M_R - abs_MR),
        -0.25 * (M_R - 1.0)**2 - beta_mass * (M_R**2 - 1.0)**2,
    )
    M_half = Mp_L + Mm_R

    # Mp 压力扩散
    Kp = 0.25
    sigma_p = 1.0
    one_minus_sigma = cp.maximum(1.0 - sigma_p * Mbar2, 0.0)
    Mp = -(Kp / fa) * one_minus_sigma * (pR - pL) / (rho_half * a_half**2)
    M_tot = M_half + Mp
    mass_flux = a_half * cp.where(M_tot >= 0.0, M_tot * rhoL, M_tot * rhoR)

    # P+ / P-（用压力分裂专属的马赫数 M_*_pr，见上方 s_pres 分派）
    abs_ML_pr = cp.abs(M_L)
    abs_MR_pr = cp.abs(M_R)
    Pp_L = cp.where(
        abs_ML_pr >= 1.0,
        0.5 * (1.0 + cp.sign(M_L)),
        0.25 * (M_L + 1.0)**2 * (2.0 - M_L)
        + alpha_pressure * M_L * (M_L**2 - 1.0)**2,
    )
    Pm_R = cp.where(
        abs_MR_pr >= 1.0,
        0.5 * (1.0 - cp.sign(M_R)),
        0.25 * (M_R - 1.0)**2 * (2.0 + M_R)
        - alpha_pressure * M_R * (M_R**2 - 1.0)**2,
    )

    # pu 速度扩散
    Ku = 0.75
    p_half = (Pp_L * pL + Pm_R * pR
              - Ku * Pp_L * Pm_R * (rhoL + rhoR) * fa * a_half * (unR - unL))

    # 上风通量
    upwind_L = (mass_flux >= 0.0)
    u_up = cp.where(upwind_L, uL, uR)
    v_up = cp.where(upwind_L, vL, vR)
    w_up = cp.where(upwind_L, wL, wR)

    hL = gamma / (gamma - 1.0) * pL / rhoL + 0.5 * (uL**2 + vL**2 + wL**2)
    hR = gamma / (gamma - 1.0) * pR / rhoR + 0.5 * (uR**2 + vR**2 + wR**2)
    h_up = cp.where(upwind_L, hL, hR)

    flux = cp.stack([
        mass_flux,
        mass_flux * u_up + p_half * nx,
        mass_flux * v_up + p_half * ny,
        mass_flux * w_up + p_half * nz,
        mass_flux * h_up,
    ], axis=-1)

    return flux
