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

# Weiss-Smith 预处理安全裕度倍数，与 core/utils/preconditioning.py::
# preconditioned_acoustic_eigs 的同名默认值保持一致（同一套 beta2 下限
# 构造，理由见该文件模块文档）。
_WEISS_SMITH_K = 1.1


@njit(cache=True, inline='always')
def compute_ausm_up_flux(qL: np.ndarray, qR: np.ndarray, normal: np.ndarray, mach_ref: float) -> np.ndarray:
    """
    计算 AUSM+up 数值通量（工业级稳定性增强版，含 Weiss-Smith 低马赫数预处理）。

    增强功能:
    1. 压力/密度正性保护 (Pressure/Density Positivity Preservation)
    2. 低马赫数修正 Mp/pu 项 (Liou 2006, AUSM+up 压力/速度扩散项)
    3. Weiss-Smith 特征值预处理（2026-08-23 新增，见下方"预处理声速"一节）

    Args:
        qL: 左侧状态 (rho, u, v, w, p)，形状 (5,)
        qR: 右侧状态 (rho, u, v, w, p)，形状 (5,)
        normal: 单位法向量，形状 (3,)
        mach_ref: 参考（自由来流）马赫数，必须显式传入（无默认值——上一次
            "只改 CFL 不改通量"的预处理尝试（2026-08-14，见 cfl.py 模块
            文档"已撤销"一节）就是因为参考马赫数在两处不同步才导致失稳，
            这次要求调用方在每次调用都显式给出同一个值，任何遗漏都会在
            调用点直接报 TypeError 而不是静默用错的默认值）。

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

    # === 2. 界面声速、Weiss-Smith 预处理声速、低马赫数标度函数 ===
    # a_half 用简单算术平均（工程上常见的近似，非 Liou 原文的临界声速构造，
    # 但对当前亚声速外流场景足够，且不影响下面 Mp/pu 项的反对称性证明）。
    a_half = 0.5 * (aL + aR)
    rho_half = 0.5 * (rhoL + rhoR)

    # Mbar^2 = (unL^2+unR^2)/(2*a_half^2)（用*物理*声速算，不能用下面的
    # 预处理声速，否则 beta2 的定义会自我循环）在 (L,R,n)->(R,L,-n) 变换
    # 下不变（法向翻转使 unL/unR 同时变号但平方不变）——这是 fa 和下面
    # beta2 都能保持通量反对称性 F(A,B,n)=-F(B,A,-n) 的共同前提。
    Mbar2 = (unL**2 + unR**2) / (2.0 * a_half**2)

    # fa：Liou (2006) AUSM+up 自带的低马赫数标度函数，只缩放 Mp/pu 修正项
    # 的幅度。此前这里的 Ma_ref 硬编码成 0.1，不接真实自由来流马赫数，
    # 现在直接用调用方传入的 mach_ref（与下面 beta2 用的是同一个真实值，
    # 不再是两个互相不知道对方存在的"低马赫数修正"）。
    M0_sq = min(1.0, max(Mbar2, mach_ref**2))
    fa = np.sqrt(M0_sq) * (2.0 - np.sqrt(M0_sq))
    fa = max(fa, 1e-6)

    # Weiss-Smith 特征值预处理（core/utils/preconditioning.py::
    # preconditioned_acoustic_eigs 的公式，逐字对应，理由/推导见该文件
    # 模块文档；这里内联而不是直接调用该函数，是因为那个函数按"单个
    # 特征速度 un/a"设计（HLLC 的 SL/SR 场景），AUSM+up 要求 beta2 是
    # 界面共享的单一值（用上面已经证明在 L/R 互换下不变的 Mbar2 算），
    # 不是分别给 L、R 各算一个——直接调用会破坏这个共享不变量、进而破坏
    # 反对称性证明）。beta2 只在局部马赫数趋于 1（跨/超声速）时精确等于
    # 1（此时预处理声速退化为物理声速，通量退化为未预处理形式）；本项目
    # 典型亚声速外流工况（M~0.09）下 beta2 会被压到 _WEISS_SMITH_K*
    # mach_ref^2 这个下限附近，预处理声速比物理声速小一个数量级，这是
    # Weiss-Smith 预处理设计上就要做的事（减少低马赫数下的过量声学耗散），
    # 不是数值不稳定的迹象。qL=qR 时 Mp=0（(pR-pL)=0）、pu 项的 (unR-unL)=0
    # ——mass_flux 和 p_half 的相容性 F(U,U)=F(U) 与 beta2 取值无关（已用
    # M+(M)+M-(M)≡M、P+(M)+P-(M)≡1 两个恒等式在任意 beta2 下验证过）。
    beta2 = min(1.0, max(max(Mbar2, _WEISS_SMITH_K * mach_ref**2), 1e-10))
    sqrt_beta2 = np.sqrt(beta2)
    aL_p = sqrt_beta2 * aL
    aR_p = sqrt_beta2 * aR
    a_half_p = sqrt_beta2 * a_half

    # 马赫数：用预处理声速算（这是预处理真正改变通量数值的地方——
    # 预处理声速变小 ⇒ 局部马赫数被放大 ⇒ M±/P± 分裂函数对这个面的
    # 响应更接近"高马赫"区域的行为，从而降低人为声学耗散）。
    M_L = unL / max(aL_p, 1e-10)
    M_R = unR / max(aR_p, 1e-10)

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
    # Mp 的声速项、mass_flux 前面的声阻抗项都改用预处理声速 aL_p/aR_p/
    # a_half_p（原来是 aL/aR/a_half）——这是 Weiss-Smith 预处理在
    # mass_flux 上生效的地方；qL=qR 时 (pR-pL)=0 ⇒ Mp=0，
    # M_plus(M)+M_minus(M)≡M 恒等式与 M 具体等于多少（物理还是预处理
    # Mach）无关，mass_flux = rho*aL_p*(M_half+0) = rho*aL_p*(un/aL_p)
    # = rho*un——相容性不因预处理声速的引入而破坏，见函数文档"Weiss-
    # Smith 特征值预处理"一节。
    Kp = 0.25
    sigma_p = 1.0
    M_half = M_plus(M_L) + M_minus(M_R)
    Mp = -(Kp / fa) * max(1.0 - sigma_p * Mbar2, 0.0) * (pR - pL) / (rho_half * a_half_p**2)
    mass_flux = 0.5 * (rhoL * aL_p + rhoR * aR_p) * (M_half + Mp)

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
        - Ku * P_plus(M_L) * P_minus(M_R) * (rhoL + rhoR) * fa * a_half_p * (unR - unL)

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
