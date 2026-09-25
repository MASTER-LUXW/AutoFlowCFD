"""AutoFlowCFD V2.0 - AUSM+up 批量通量(GPU, 与 kernels.py 逐字对应)

从 `src/autoflowcfd/core/gpu/residual/gpu_inviscid.py`(原 710 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""


from autoflowcfd.core.fr_operators.kernels import (
    PRECOND_LEGACY,
    PRECOND_PHYSICAL,
    PRECOND_PRESSURE_PHYSICAL,
)

from autoflowcfd.core.gpu import get_cupy


def _ausm_up_flux_batch_gpu(Q_L, Q_R, normal, mach_ref, precond_mode):
    """GPU 批量 AUSM+up 通量计算（CuPy 向量化版本，含 Weiss-Smith 低马赫
    数预处理）。与 kernels.py::compute_ausm_up_flux 逐字对应，理由/推导
    见该函数文档，这里不重复；两处必须同步修改（该文件模块文档要求
    "逐字对应"）。

    真实 bug 修复（问题清单 #5，2026-09-02，用真实含多源/混合拆分面的
    合成网格 + numpy-as-cupy 替身 + 完整 `GPUFRSolver.step()` 首次真正
    走到这里才发现——`test_gpu_solver_order_continuation.py`"发现但
    本次未修复"一节记录过这个缺口，本次专项排查修复）：`normal` 形参
    此前文档标注/实现都当作 `(N, 3)`（逐面一个法向，`nx=normal[...,
    0:1]` 故意保留末尾长度 1 的维度，为的是广播到 `uL` 等 `(N,n_fp)`
    形状——对应 CPU kernel 里"同一面内全部 FP 共用同一个法向"这个
    从未真正成立的假设）——但两个真实调用点（本文件
    `_compute_interface_correction_gpu` 的 owner/neighbor 两个分支）
    传入的 `direction_o`/`direction_n` 来自 `_ausm_direction_with_
    fallback`，形状恒为 `(N, n_fp, 3)`（逐 FP 各自独立的方向，因为
    对齐安全阀是逐 FP 用该 FP 自己的 `adjrow`/`true_normal_ref` 判定
    的，不是全面共享同一个值——与 CPU numba kernel 逐 FP 独立调用
    `compute_ausm_up_flux(qL, qR, normal_at_this_fp, ...)` 完全对应）。
    `nx=normal[...,0:1]` 对 `(N,n_fp,3)` 输入会产出 `(N,n_fp,1)`，
    与 `uL`（`(N,n_fp)`）相乘时 NumPy/CuPy 广播规则按从右往左对齐，
    `uL` 的 `n_fp` 维度会被错误地对上 `nx` 的长度-1 维度、`uL` 隐式
    补出的前导维度又被要求匹配 `nx` 的 `n_fp` 维——只有 `N==n_fp`（本
    次合成测试网格的巧合，两者都恰好是 4）时这个错误广播才会"成功"
    但产出一个多出一维、内容完全错误的结果，`N!=n_fp` 的一般网格上
    会在这里直接 `ValueError`（这正是该测试文件此前报告的"P1/P2 直接
    构造时同样复现"的崩溃点，只是那次调查停在了更下游的
    `distribute_face_correction_to_sps`（已随坍缩路径删除））。修复：`nx=normal[...,0]`
    （不保留末尾维度）——这样在真实的 `(N,n_fp,3)` 输入下 `nx` 形状
    恰好是 `(N,n_fp)`，与 `uL` 等逐 FP 量逐元素精确匹配，不再依赖任何
    隐式广播。

    Args:
        Q_L, Q_R: (N, n_fp, 5) 左右状态
        normal: (N, n_fp, 3) 单位法向量——逐 FP 各自独立（不是逐面共享
            同一个值，见上方"真实 bug 修复"说明）
        precond_mode: AUSM+up 预处理声速在通量内部的作用域（
            kernels.py 的 PRECOND_PHYSICAL/PRECOND_PRESSURE_PHYSICAL/
            PRECOND_LEGACY，语义与 CPU 端逐字对应，推导见该文件模块级
            常量上方的长注释）。GPU 侧不存在 numba 磁盘缓存冻结全局量的
            问题，但仍然按实参传入，以保证 CPU/GPU 交叉一致性测试能对
            同一档逐项比对。
        mach_ref: 参考（自由来流）马赫数，见 kernels.py::
            compute_ausm_up_flux 文档

    Returns:
        flux: (N, n_fp, 5) 数值通量
    """
    cp = get_cupy()
    gamma = 1.4

    rhoL = cp.maximum(Q_L[..., 0], 1e-6)
    uL, vL, wL = Q_L[..., 1], Q_L[..., 2], Q_L[..., 3]
    pL = cp.maximum(Q_L[..., 4], 10.0)

    rhoR = cp.maximum(Q_R[..., 0], 1e-6)
    uR, vR, wR = Q_R[..., 1], Q_R[..., 2], Q_R[..., 3]
    pR = cp.maximum(Q_R[..., 4], 10.0)

    nx = normal[..., 0]
    ny = normal[..., 1]
    nz = normal[..., 2]

    unL = uL * nx + vL * ny + wL * nz
    unR = uR * nx + vR * ny + wR * nz

    aL = cp.sqrt(cp.maximum(gamma * pL / rhoL, 1e-10))
    aR = cp.sqrt(cp.maximum(gamma * pR / rhoR, 1e-10))

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

    # Weiss-Smith 预处理声速（与 kernels.py::compute_ausm_up_flux 的
    # _WEISS_SMITH_K=1.1 同一个安全裕度常数、同一套 beta2 公式）。
    beta2 = cp.minimum(1.0, cp.maximum(cp.maximum(Mbar2, 1.1 * mach_ref**2), 1e-10))
    sqrt_beta2 = cp.sqrt(beta2)

    # 预处理声速的作用域按 precond_mode 分派，与 kernels.py::
    # compute_ausm_up_flux 的 s_mass/s_pres 逐字对应（PRECOND_PHYSICAL
    # 是默认值，此时两个因子都是 1.0、本函数精确退化为标准 AUSM+up）。
    if precond_mode == PRECOND_PHYSICAL:
        s_mass = 1.0
        s_pres = 1.0
    elif precond_mode == PRECOND_PRESSURE_PHYSICAL:
        s_mass = sqrt_beta2
        s_pres = 1.0
    elif precond_mode == PRECOND_LEGACY:
        s_mass = sqrt_beta2
        s_pres = sqrt_beta2
    else:
        raise ValueError(f'未知 precond_mode: {precond_mode!r}')

    aL_m = s_mass * aL
    aR_m = s_mass * aR
    a_half_m = s_mass * a_half
    a_half_pr = s_pres * a_half

    M_L = unL / cp.maximum(aL_m, 1e-10)
    M_R = unR / cp.maximum(aR_m, 1e-10)
    M_L_pr = unL / cp.maximum(s_pres * aL, 1e-10)
    M_R_pr = unR / cp.maximum(s_pres * aR, 1e-10)

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
    Mp = -(Kp / fa) * one_minus_sigma * (pR - pL) / (rho_half * a_half_m**2)
    mass_flux = 0.5 * (rhoL * aL_m + rhoR * aR_m) * (M_half + Mp)

    # P+ / P-（用压力分裂专属的马赫数 M_*_pr，见上方 s_pres 分派）
    abs_ML_pr = cp.abs(M_L_pr)
    abs_MR_pr = cp.abs(M_R_pr)
    Pp_L = cp.where(
        abs_ML_pr >= 1.0,
        0.5 * (1.0 + cp.sign(M_L_pr)),
        0.25 * (M_L_pr + 1.0)**2 * (2.0 - M_L_pr)
        + alpha_pressure * M_L_pr * (M_L_pr**2 - 1.0)**2,
    )
    Pm_R = cp.where(
        abs_MR_pr >= 1.0,
        0.5 * (1.0 - cp.sign(M_R_pr)),
        0.25 * (M_R_pr - 1.0)**2 * (2.0 + M_R_pr)
        - alpha_pressure * M_R_pr * (M_R_pr**2 - 1.0)**2,
    )

    # pu 速度扩散
    Ku = 0.75
    p_half = (Pp_L * pL + Pm_R * pR
              - Ku * Pp_L * Pm_R * (rhoL + rhoR) * fa * a_half_pr * (unR - unL))

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
