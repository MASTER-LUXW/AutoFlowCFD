"""
AutoFlowCFD - AUSM+up 无粘界面黎曼求解器（Numba 版，`compute_ausm_up_flux`）。
"""

import numpy as np
from numba import njit

from autoflowcfd.core.thermo import GAMMA


@njit(cache=True, inline='always')
def compute_ausm_up_flux(qL: np.ndarray, qR: np.ndarray, normal: np.ndarray,
                         mach_ref: float) -> np.ndarray:
    """
    计算 AUSM+up 数值通量（Liou 2006 原文 / SU2 `CUpwAUSMPLUSUP_Flow`，通量内部全部用物理声速）。

    增强功能:
    1. 压力/密度正性保护 (Pressure/Density Positivity Preservation)
    2. 低马赫数修正 Mp/pu 项 (Liou 2006, AUSM+up 压力/速度扩散项)

    低马赫数修正完全由 AUSM+up 自带的 fa/Kp/Ku/alpha 机制提供；Weiss-Smith 预处理只作用在伪时间系统
    （`utils/preconditioning.py` 的 Gamma 矩阵）与 CFL 步长估计上，**不进入通量**。2026-10-18 以前另有两档
    把预处理声速放进 M4±/P5± 的马赫数归一（`AFCFD_AUSM_PRECOND_MODE=legacy/pressure_physical`）：亚声速外流下
    预处理声速 ≈ 来流速度，法向速度接近来流量级的面上压力分裂进入 |M|>=1 的饱和分支，静止固壁镜像态上
    p_half ≈ 2p（实测 7 倍过预测）、dp_half/du_n = 0 使不可穿透条件失效；两档随环境变量一并删除。

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
    gamma = GAMMA

    # === 1. 两侧状态 ===
    # 正性由正性限制器在解点、全部面通量点与过积分细点上保证（`time_integration/positivity/`
    # ），幽灵态由这些合法状态按边界条件构造。2026-10-19（I-18）以前这里另把 rho 钳到 1e-6、p
    # 钳到 10 Pa——带量纲的绝对下限：有量纲空气算例从不触发（p~1e5），无量纲 /
    # 低压算例（p~1/GAMMA）则被静默改写通量；非正状态到达这里说明上游失效，应以 NaN
    # 暴露给发散检测，而不是在通量里掩盖。
    rhoL = qL[0]
    rhoR = qR[0]
    pL = qL[4]
    pR = qR[4]

    uL, vL, wL = qL[1], qL[2], qL[3]
    uR, vR, wR = qR[1], qR[2], qR[3]

    # 计算法向速度
    unL = uL * normal[0] + vL * normal[1] + wL * normal[2]
    unR = uR * normal[0] + vR * normal[1] + wR * normal[2]

    # 声速
    aL = np.sqrt(gamma * pL / rhoL)
    aR = np.sqrt(gamma * pR / rhoR)

    # === 2. 界面声速、低马赫数标度函数 ===
    # a_half 用简单算术平均（工程上常见的近似，非 Liou 原文的临界声速构造，
    # 但对当前亚声速外流场景足够，且不影响下面 Mp/pu 项的反对称性证明）。
    a_half = 0.5 * (aL + aR)
    rho_half = 0.5 * (rhoL + rhoR)

    # Mbar^2 = (unL^2+unR^2)/(2*a_half^2) 在 (L,R,n)->(R,L,-n) 变换下不变（法向翻转使 unL/unR 同时变号
    # 但平方不变）——这是 fa 保持通量反对称性 F(A,B,n)=-F(B,A,-n) 的前提。
    Mbar2 = (unL**2 + unR**2) / (2.0 * a_half**2)

    # fa：Liou (2006) AUSM+up 自带的低马赫数标度函数，只缩放 Mp/pu 修正项
    # 的幅度。此前这里的 Ma_ref 硬编码成 0.1，不接真实自由来流马赫数，
    # 现在直接用调用方传入的 mach_ref（与 CFL 步长、伪时间预处理同一个值）。
    M0_sq = min(1.0, max(Mbar2, mach_ref**2))
    fa = np.sqrt(M0_sq) * (2.0 - np.sqrt(M0_sq))
    fa = max(fa, 1e-6)

    # M4±（质量分裂，下方 M_plus/M_minus）与 P5±（压力分裂，下方
    # P_plus/P_minus）各自的耗散系数——真实 bug 修复（V2.0 专家组盲审
    # 第四次评审，2026-08-28，#12）：此前把两者用反了（0.1875 用在质量
    # 分裂里、0.5 用在压力分裂里），已用 Liou (2006) AUSM+up 原始文献的
    # 独立生产实现（SU2 `CUpwAUSMPLUSUP_Flow::ComputeMassAndPressureFluxes`，
    # su2code/SU2 GitHub 仓库 ausm_slau.cpp）逐项核对确认：
    #   - 质量分裂系数（beta_mass）是固定常数 1/8=0.125（SU2 源码
    #     `beta = 1.0/8.0`，与该文档"质量分裂通常记作 β"的记号一致）；
    #   - 压力分裂系数（alpha_pressure）不是固定常数，而是随 fa 变化：
    #     3/16*(-4+5*fa²)（SU2 源码 `alpha = 3.0/16.0*(-4.0+5.0*fa*fa)`）
    #     ——fa=1（跨/超声速）时退化为标准值 3/16=0.1875，fa→0（低马赫
    #     极限）时趋于 -3/4，恰好落在此前代码注释记录的"α 常见有效区间
    #     [-3/4,3/16]"两端，证实此前把 beta=0.5（超出这个区间）错配给
    #     压力分裂就是这处 bug 的直接后果，不只是"偏离推荐默认值"。
    # 与 M+(M)+M-(M)≡M、P+(M)+P-(M)≡1 两个相容性恒等式（对任意系数值
    # 代数成立，不依赖具体系数）互不冲突，只改变通量的耗散幅度/低马赫
    # 数区域的数值行为。
    beta_mass = 1.0 / 8.0
    alpha_pressure = 3.0 / 16.0 * (-4.0 + 5.0 * fa * fa)

    # 两侧马赫数都用**界面公共声速**归一（Liou 2006 式 (13)；SU2 `mL = ProjVelL/aF`）。2026-10-17 以前
    # 各用本侧声速 aL/aR：接触间断两侧声速不同（等压、密度跳变），M4+(uL/aL)+M4-(uR/aR) 不再等于 u/a，
    # 运动接触间断上的质量通量不等于迎风值。
    M_L = unL / max(a_half, 1e-10)
    M_R = unR / max(a_half, 1e-10)

    # === 3. AUSM+ 质量通量分裂 (van Leer 多项式分裂函数) ===
    # 标准形式（Liou 1996, AUSM+）: M+(M)+M-(M) ≡ M（相容性要求：qL=qR时
    # mass_flux 必须精确退化为 rho*u_n）。此前版本 M_minus 的亚声速分支
    # 缺少整体负号（写成 +0.25*(M-1)^2 而不是 -0.25*(M-1)^2），导致
    # M_plus(M)+M_minus(M) = 0.5*(M^2+1) 而不是 M —— 通量在 qL=qR
    # 时不等于精确物理通量，已用数值一致性测试验证发现并在此修复
    # （见 tests/unit/test_ausm_up_flux.py::TestAusmUpFluxIdentities::test_consistency）。

    def M_plus(M):
        """M+ 函数"""
        if abs(M) >= 1:
            return 0.5 * (M + abs(M))
        else:
            return 0.25 * (M + 1)**2 + beta_mass * (M**2 - 1)**2

    def M_minus(M):
        """M- 函数"""
        if abs(M) >= 1:
            return 0.5 * (M - abs(M))
        else:
            return -0.25 * (M - 1)**2 - beta_mass * (M**2 - 1)**2

    # === 4. 压力扩散项 Mp (Liou 2006, AUSM+up 式17) ===
    # (pR-pL) 在 (L,R,n)->(R,L,-n) 下翻号、其余因子（Mbar2/rho_half/a_half/fa）不变，Mp 随之翻号，
    # 质量通量的反对称性 F(A,B,n) = -F(B,A,-n) 保持。相容性：qL=qR 时 Mp=0，M_plus(M)+M_minus(M) = M，
    # mass_flux = a_half * (un/a_half) * rho = rho*un。
    Kp = 0.25
    sigma_p = 1.0
    M_half = M_plus(M_L) + M_minus(M_R)
    Mp = -(Kp / fa) * max(1.0 - sigma_p * Mbar2, 0.0) * (pR - pL) / (rho_half * a_half**2)
    # 质量通量取**迎风密度**（Liou 2006 式 (12)；SU2 `mdot = aF*(max(mF,0)*rhoL + min(mF,0)*rhoR)`）。
    # 2026-10-17 以前是 0.5*(rhoL*aL + rhoR*aR)*M：密度按中心平均进入质量通量，接触间断与熵波
    # （等压的密度/温度扰动）上没有迎风耗散。
    M_tot = M_half + Mp
    mass_flux = a_half * (M_tot * rhoL if M_tot >= 0.0 else M_tot * rhoR)

    # === 5. AUSM+up 压力通量分裂 ===
    # P5±（Liou 2006, AUSM+up 式 (24)）：
    #     P5±(M) = M2±(M)·[(±2 − M) ∓ 16·α·M·M2∓(M)],  M2±(M) = ±(M±1)²/4
    #   展开：P5+ = (M+1)²(2−M)/4 + α·M·(M²−1)²（α 项**没有** 1/4；SU2
    #   `pLP = 0.25*(mL+1)^2*(2-mL) + alpha*mL*(mL^2-1)^2` 同）。
    #
    # **2026-09-25 修正的真实缺陷**：此前四份实现（本函数、GPU CuPy 版、GPU
    # CUDA P0 源串、GPU P0 无粘残差）都写成 `0.25*((M+1)²(2−M) + α·M·(M²−1)²)`，
    # α 项被多乘了 1/4。P5 分裂在 M=0 处的斜率本应是 0.75+α，低马赫下
    # α → −3/4 使斜率 = O(fa²)、压力扰动 O(M²)；缩小 4 倍后斜率 ≈ 0.57，压力
    # 分裂退化成声阻抗响应 p*−p ≈ ρ·a·u_n（O(M)，Guillard–Viozat 型低马赫失效）。
    # 后果：驻点区出现 Cp ~ 2/M 量级的虚假超压（plate_demo，M=0.098：P0 稳态
    # 驻点平台 Cp≈21.5；P1 长程运行驻点线总压 Cp_t≈2 并持续向上游推进，
    # Cd≈4.2 而实验值≈1.2）。2026-08-28 #12 对照 SU2 核对 α/β 时漏看了这个系数。
    def P_plus(M):
        """P+ 函数"""
        if abs(M) >= 1:
            return 0.5 * (1 + np.sign(M))
        else:
            return 0.25 * (M + 1)**2 * (2 - M) + alpha_pressure * M * (M**2 - 1)**2

    def P_minus(M):
        """P- 函数"""
        if abs(M) >= 1:
            return 0.5 * (1 - np.sign(M))
        else:
            return 0.25 * (M - 1)**2 * (2 + M) - alpha_pressure * M * (M**2 - 1)**2

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
