"""
AutoFlowCFD - FR 求解器 AUSM+up 黎曼求解器内核 (Numba 加速版)。

本模块曾经还包含 compute_fr_residual_kernel（把计算空间微分算子 D_3d
直接当物理空间导数用，没有度量项变换，对本代码库任何非笛卡尔映射单元
都是错误导数）、compute_viscous_ldg_term/apply_correction_term/
compute_interface_flux_jump/compute_ldg_penalty_flux/
apply_correction_term_full 五个函数——V2.0 二次评审确认这五个全仓库
零调用点（真正参与残差组装的实现在 core/fr_residual_inviscid.py 与
core/fr_viscous_flux.py），已删除而不是继续留作"看起来完整、实则孤立"
的死代码。真正参与求解主循环的只有 compute_ausm_up_flux（无粘界面
黎曼求解器），保留在本文件。
"""

import numpy as np
from numba import njit


@njit(cache=True, inline='always')
def compute_ausm_up_flux(qL: np.ndarray, qR: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """
    计算 AUSM+up 数值通量（工业级稳定性增强版）。

    增强功能:
    1. 压力/密度正性保护 (Pressure/Density Positivity Preservation)
    2. 低马赫数修正 Mp/pu 项 (Liou 2006, AUSM+up 压力/速度扩散项)

    Args:
        qL: 左侧状态 (rho, u, v, w, p)，形状 (5,)
        qR: 右侧状态 (rho, u, v, w, p)，形状 (5,)
        normal: 单位法向量，形状 (3,)

    Returns:
        flux: 守恒变量通量，形状 (5,)
    """
    gamma = 1.4
    alpha = 0.1875  # AUSM+up 参数
    beta = 0.5      # 压力分裂参数

    # === 1. 正性保护与状态限制 ===
    rhoL = max(qL[0], 1e-6)
    rhoR = max(qR[0], 1e-6)
    pL = max(qL[4], 10.0)   # 最小压力 10 Pa
    pR = max(qR[4], 10.0)

    uL, vL, wL = qL[1], qL[2], qL[3]
    uR, vR, wR = qR[1], qR[2], qR[3]

    # 计算法向速度
    unL = uL * normal[0] + vL * normal[1] + wL * normal[2]
    unR = uR * normal[0] + vR * normal[1] + wR * normal[2]

    # 声速
    aL = np.sqrt(max(gamma * pL / rhoL, 1e-10))
    aR = np.sqrt(max(gamma * pR / rhoR, 1e-10))

    # 马赫数
    M_L = unL / max(aL, 1e-10)
    M_R = unR / max(aR, 1e-10)

    # === 2. 界面声速与低马赫数标度函数 (Liou 2006, AUSM+up) ===
    # a_half 用简单算术平均（工程上常见的近似，非 Liou 原文的临界声速构造，
    # 但对当前亚声速外流场景足够，且不影响下面 Mp/pu 项的反对称性证明）。
    a_half = 0.5 * (aL + aR)
    rho_half = 0.5 * (rhoL + rhoR)

    # Mbar^2 = (unL^2+unR^2)/(2*a_half^2) 在 (L,R,n)->(R,L,-n) 变换下不变
    # （法向翻转使 unL/unR 同时变号但平方不变），fa 因此也不变——这是下面
    # Mp/pu 项能保持通量反对称性 F(A,B,n)=-F(B,A,-n) 的前提。
    Mbar2 = (unL**2 + unR**2) / (2.0 * a_half**2)
    Ma_ref = 0.1  # 截断参考马赫数（未接入自由来流马赫数时的局部近似）
    M0_sq = min(1.0, max(Mbar2, Ma_ref**2))
    fa = np.sqrt(M0_sq) * (2.0 - np.sqrt(M0_sq))
    fa = max(fa, 1e-6)

    # === 3. AUSM+ 质量通量分裂 (van Leer 多项式分裂函数) ===
    # 标准形式（Liou 1996, AUSM+）: M+(M)+M-(M) ≡ M（相容性要求：qL=qR时
    # mass_flux 必须精确退化为 rho*u_n）。此前版本 M_minus 的亚声速分支
    # 缺少整体负号（写成 +0.25*(M-1)^2 而不是 -0.25*(M-1)^2），导致
    # M_plus(M)+M_minus(M) = 0.5*(M^2+1) 而不是 M —— 通量在 qL=qR
    # 时不等于精确物理通量，已用数值一致性测试验证发现并在此修复
    # （见 tests/unit/test_fr_residual_inviscid.py::test_ausm_up_consistency）。
    def M_plus(M):
        """M+ 函数"""
        if abs(M) >= 1:
            return 0.5 * (M + abs(M))
        else:
            return 0.25 * (M + 1)**2 + alpha * (M**2 - 1)**2

    def M_minus(M):
        """M- 函数"""
        if abs(M) >= 1:
            return 0.5 * (M - abs(M))
        else:
            return -0.25 * (M - 1)**2 - alpha * (M**2 - 1)**2

    # === 4. 压力扩散项 Mp (Liou 2006, AUSM+up 式17) ===
    # 取代此前版本的"熵修正"：旧实现在 mass_flux 上叠加 0.5*(rhoL+rhoR)*a_half
    # *|M_L-M_R|*0.1，而 |M_L-M_R| 在 (L,R,n)->(R,L,-n) 变换下不翻号（是偶量），
    # 直接破坏了 mass_flux 必须满足的反对称性 F(A,B,n)=-F(B,A,-n)（已用受控
    # 数值算例验证：|M|<0.1 时两次调用之和最大相对不平衡达 1.08%，且触发窗口
    # 恰好覆盖本项目 30 m/s / M≈0.087 的目标工况，即全域每个内部面都不守恒）。
    # Mp 项是 Liou 原始 AUSM+up 方案自带的低马赫数稳定化机制，(pR-pL) 在同一
    # 变换下翻号、其余因子（Mbar2/rho_half/a_half/fa）不变，故 Mp 本身翻号，
    # 叠加到已验证满足反对称性的 M_half 上不会破坏该性质。
    Kp = 0.25
    sigma_p = 1.0
    M_half = M_plus(M_L) + M_minus(M_R)
    Mp = -(Kp / fa) * max(1.0 - sigma_p * Mbar2, 0.0) * (pR - pL) / (rho_half * a_half**2)
    mass_flux = 0.5 * (rhoL * aL + rhoR * aR) * (M_half + Mp)

    # === 5. AUSM+up 压力通量分裂 ===
    def P_plus(M):
        """P+ 函数"""
        if abs(M) >= 1:
            return 0.5 * (1 + np.sign(M))
        else:
            return 0.25 * ((M + 1)**2 * (2 - M) + beta * M * (M**2 - 1)**2)

    def P_minus(M):
        """P- 函数"""
        if abs(M) >= 1:
            return 0.5 * (1 - np.sign(M))
        else:
            return 0.25 * ((M - 1)**2 * (2 + M) - beta * M * (M**2 - 1)**2)

    # 速度扩散项 pu (Liou 2006, AUSM+up 式18)：与 Mp 项配套的压力项低马赫
    # 稳定化。(unR-unL) 在 (L,R,n)->(R,L,-n) 变换下不变（法向翻转与 L/R 互换
    # 相互抵消），P_plus(M_L)*P_minus(M_R) 乘积也不变，故 p_half 整体保持对称
    # ——这正是需要的性质：p_half 只通过外层的 normal 分量翻号来满足动量/能量
    # 通量的反对称性，pu 项不破坏这一点。
    Ku = 0.75
    p_half = P_plus(M_L) * pL + P_minus(M_R) * pR \
        - Ku * P_plus(M_L) * P_minus(M_R) * (rhoL + rhoR) * fa * a_half * (unR - unL)

    # === 6. 构造最终通量 ===
    # 动量/能量的对流部分必须按 mass_flux 的符号做简单迎风选择（AUSM 族
    # 方法的标准做法），而不是用压力分裂函数 P+/P- 做加权混合——P+/P-
    # 是为压力项设计的相容分裂（P+(M)+P-(M)≡1），把它们套用到速度/焓的
    # 迎风选择上没有理论依据，此前版本正是这样做的（已在此修复）：
    # 当 qL=qR 时会得到与真实通量不一致的动量/能量分量。
    upwind_L = mass_flux >= 0.0
    flux = np.zeros(5)
    flux[0] = mass_flux
    flux[1] = mass_flux * (uL if upwind_L else uR) + p_half * normal[0]
    flux[2] = mass_flux * (vL if upwind_L else vR) + p_half * normal[1]
    flux[3] = mass_flux * (wL if upwind_L else wR) + p_half * normal[2]

    # 能量通量：用比总焓 h = H = e + p/rho + 0.5|u|^2 做迎风选择
    hL = gamma / (gamma - 1) * pL / rhoL + 0.5 * (uL**2 + vL**2 + wL**2)
    hR = gamma / (gamma - 1) * pR / rhoR + 0.5 * (uR**2 + vR**2 + wR**2)

    flux[4] = mass_flux * (hL if upwind_L else hR)

    return flux
