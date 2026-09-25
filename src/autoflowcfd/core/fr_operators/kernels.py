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

# ===== 预处理声速在通量内部的作用域（precond_mode）=====
#
# 三档语义（数值上的差别只在"喂给 M4±/P5± 分裂函数的马赫数用哪个声速
# 归一"，以及与之配套的 Mp/pu 耗散项的声速因子）：
#
#   PRECOND_PHYSICAL (0)  —— 标准 AUSM+up（Liou 2006 原文 / SU2
#       `CUpwAUSMPLUSUP_Flow`）：通量内部**全部用物理声速**。低马赫数
#       所需的修正完全由 AUSM+up 自带的 fa/Kp/Ku/alpha_pressure 机制
#       提供；Weiss-Smith 预处理只作用在伪时间系统（`step.py` 的 Gamma
#       矩阵）与 CFL 步长估计（`cfl.py`）上，不进入通量。**这是默认值。**
#
#   PRECOND_PRESSURE_PHYSICAL (1) —— 只把压力分裂 P5± 及其配套 pu 项
#       改回物理马赫数，质量分裂 M4± 与 Mp 项仍用预处理声速。用于隔离
#       "P5± 饱和"这一单一因素的诊断/对照。
#
#   PRECOND_LEGACY (2) —— 2026-08-23 引入的历史行为：M4± 与 P5± 都用
#       预处理马赫数。**已确认不可用**，保留仅为回归对照，见下。
#
# 为什么 LEGACY 不能再做默认（2026-09-16 定位，不是口味问题）：
#
#   预处理声速 a_p = sqrt(beta2)*a，而亚声速外流下 beta2 被压到下限
#   `_WEISS_SMITH_K * mach_ref^2` 附近，于是 a_p ≈ u_inf。这使得任何
#   法向速度接近来流量级的面上"预处理马赫数"≈1。而 P5± 在 |M|>=1 时
#   **恒等于 1 / 0，导数精确为零**——压力分裂进入饱和分支：
#
#     * 静止绝热固壁的镜像幽灵态满足 M_R = -M_L，故
#       P5+(M_L)+P5-(M_R) = 2*P5+(M_L)。M_L≈0.95 时该和 ≈ 1.994，
#       即 p_half ≈ p_L + p_R = 2p（实测 p_half/p = 2.0018，虚假超压
#       约 149 q_inf）。物理马赫数下同一状态给出 p_half - p = 11310 Pa，
#       与镜像黎曼问题的精确解 rho*a*u_n = 13890 Pa 同量级——**物理
#       马赫数给的是真解，预处理版本是 7 倍过预测**。
#     * 更本质的后果：饱和分支上 dp_half/du_n = 0，壁面压力对法向速度
#       的**恢复梯度消失**，不可穿透条件不再被强制执行。这正是
#       [[stagnation_face_overpressure_localized]] 记录的"迎风面贴壁
#       薄层压力超过来流滞止压力、且随迭代零曲率线性增长"的成因，也
#       解释了为什么压低 CFL（0.03/0.01/0.009/0.008/0.007 全试过）只能
#       推迟显形而不能避免。
#
#   P5± 是 p_L 与 p_R 的**迎风权重**（相容性要求 P5+(M)+P5-(M) ≡ 1，
#   且必须是同一个 M），不是耗散系数；把它推到超声速极限等于把界面
#   压力从"加权平均"改成"两侧之和"，直接破坏了"p_L=p_R 的滞止面上
#   p_half 必须等于 p"这条要求。
#
#   这与项目自身历史完全同类：2026-08-14 那次"把预处理声速用在 CFL
#   估计里"导致 dt 高估约 10 倍而失稳（已撤销，见 `cfl.py` 模块文档）。
#   预处理矩阵作用在 dU/dtau 上是对的，把预处理声速塞进通量的马赫数
#   归一里是错的。
PRECOND_PHYSICAL = 0
PRECOND_PRESSURE_PHYSICAL = 1
PRECOND_LEGACY = 2

_PRECOND_MODE_NAMES = {
    'physical': PRECOND_PHYSICAL,
    'pressure_physical': PRECOND_PRESSURE_PHYSICAL,
    'legacy': PRECOND_LEGACY,
}
_PRECOND_MODE_LABELS = {v: k for k, v in _PRECOND_MODE_NAMES.items()}

DEFAULT_PRECOND_MODE = PRECOND_PHYSICAL


def resolve_ausm_precond_mode(value=None) -> int:
    """把 `AFCFD_AUSM_PRECOND_MODE` 解析成 `precond_mode` 整数。

    必须是**纯 Python**函数、且结果作为**运行期实参**传进 njit 内核：
    numba 的 `cache=True` 会把 njit 函数里读到的模块级全局量当成编译期
    常量冻结进磁盘缓存，用全局变量做开关会在换值后静默沿用旧编译产物
    （本项目 2026-09-16 已经真实踩过一次：改了 `kernels.py` 只清了本
    目录的 `__pycache__`，因为 `inline='always'` 把旧函数体烘进了调用方
    缓存，A/B 跑出逐位相同的结果）。

    Args:
        value: 显式取值（'physical'/'pressure_physical'/'legacy' 或对应
            整数）。为 None 时读环境变量，未设置则用 `DEFAULT_PRECOND_MODE`。

    Returns:
        precond_mode 整数。

    Raises:
        ValueError: 取值不在三档之内（不静默回退——静默回退会让一次
            拼写错误伪装成"默认行为"，掩盖 A/B 结果）。
    """
    import os

    if value is None:
        value = os.environ.get('AFCFD_AUSM_PRECOND_MODE', '').strip()
        if not value:
            return DEFAULT_PRECOND_MODE
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        iv = int(value)
        if iv in _PRECOND_MODE_LABELS:
            return iv
        raise ValueError(
            f'AUSM+up precond_mode 整数取值非法: {iv}；'
            f'合法值 {sorted(_PRECOND_MODE_LABELS)}'
        )
    key = str(value).strip().lower()
    if key in _PRECOND_MODE_NAMES:
        return _PRECOND_MODE_NAMES[key]
    raise ValueError(
        f'AFCFD_AUSM_PRECOND_MODE 取值非法: {value!r}；'
        f'合法值 {sorted(_PRECOND_MODE_NAMES)}'
    )


def ausm_precond_mode_label(mode: int) -> str:
    """precond_mode 整数 -> 人类可读名字（用于启动日志）。"""
    try:
        return _PRECOND_MODE_LABELS[int(mode)]
    except KeyError:
        raise ValueError(f'未知 precond_mode: {mode!r}')


@njit(cache=True, inline='always')
def compute_ausm_up_flux(qL: np.ndarray, qR: np.ndarray, normal: np.ndarray,
                         mach_ref: float, precond_mode: int) -> np.ndarray:
    """
    计算 AUSM+up 数值通量（工业级稳定性增强版，含 Weiss-Smith 低马赫数预处理）。

    增强功能:
    1. 压力/密度正性保护 (Pressure/Density Positivity Preservation)
    2. 低马赫数修正 Mp/pu 项 (Liou 2006, AUSM+up 压力/速度扩散项)
    3. Weiss-Smith 特征值预处理（2026-08-23 新增；2026-09-16 起其在通量
       内部的作用域由 `precond_mode` 控制，默认已改为"不进入通量"，
       原因见模块级 PRECOND_* 常量上方的长注释）

    Args:
        qL: 左侧状态 (rho, u, v, w, p)，形状 (5,)
        qR: 右侧状态 (rho, u, v, w, p)，形状 (5,)
        normal: 单位法向量，形状 (3,)
        mach_ref: 参考（自由来流）马赫数，必须显式传入（无默认值——上一次
            "只改 CFL 不改通量"的预处理尝试（2026-08-14，见 cfl.py 模块
            文档"已撤销"一节）就是因为参考马赫数在两处不同步才导致失稳，
            这次要求调用方在每次调用都显式给出同一个值，任何遗漏都会在
            调用点直接报 TypeError 而不是静默用错的默认值）。
        precond_mode: 预处理声速在通量**内部**的作用域，取值见模块级
            PRECOND_PHYSICAL / PRECOND_PRESSURE_PHYSICAL / PRECOND_LEGACY
            三个常量及其上方的长注释。同样**无默认值**，且必须由调用方用
            `resolve_ausm_precond_mode()` 在纯 Python 层解析后作为实参传入
            ——不能做成 njit 里读的模块级全局量，那会被 numba 的磁盘缓存
            冻结成编译期常量（理由与 2026-09-16 的真实事故见该解析器文档）。

    Returns:
        flux: 守恒变量通量，形状 (5,)
    """
    gamma = 1.4

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

    # 预处理声速的作用域按 precond_mode 分派：三档语义与"为什么 LEGACY
    # 不能继续做默认"的完整推导见模块级 PRECOND_* 常量上方的长注释。
    # s_mass 作用在质量分裂 M4± 及其配套的 Mp 压力扩散项上，s_pres 作用
    # 在压力分裂 P5± 及其配套的 pu 速度扩散项上；两者都等于 1.0 时本函数
    # 精确退化为标准 AUSM+up（Liou 2006 / SU2 实现）。
    #
    # 三档都保持 beta2 在 (L,R,n)->(R,L,-n) 变换下的不变性（Mbar2 是偶量、
    # mach_ref 是标量），所以通量反对称性 F(A,B,n) = -F(B,A,-n) 与 mode
    # 取值无关；相容性 F(U,U) = F(U) 同样与 mode 无关——qL=qR 时
    # (pR-pL)=0 使 Mp=0、(unR-unL)=0 使 pu 项为 0，剩下的
    # M+(M)+M-(M) = M 与 P+(M)+P-(M) = 1 两个恒等式对任意声速归一都代数
    # 成立（mass_flux = rho*a_m*(un/a_m) = rho*un，归一用的声速自行约掉）。
    # 这两条性质已由单元测试对三档逐一覆盖。
    if precond_mode == PRECOND_PHYSICAL:
        s_mass = 1.0
        s_pres = 1.0
    elif precond_mode == PRECOND_PRESSURE_PHYSICAL:
        s_mass = sqrt_beta2
        s_pres = 1.0
    else:
        s_mass = sqrt_beta2
        s_pres = sqrt_beta2

    aL_m = s_mass * aL
    aR_m = s_mass * aR
    a_half_m = s_mass * a_half
    a_half_pr = s_pres * a_half

    # 质量分裂 M4± 用的马赫数
    M_L = unL / max(aL_m, 1e-10)
    M_R = unR / max(aR_m, 1e-10)
    # 压力分裂 P5± 用的马赫数
    M_L_pr = unL / max(s_pres * aL, 1e-10)
    M_R_pr = unR / max(s_pres * aR, 1e-10)


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
            return 0.25 * (M + 1)**2 + beta_mass * (M**2 - 1)**2

    def M_minus(M):
        """M- 函数"""
        if abs(M) >= 1:
            return 0.5 * (M - abs(M))
        else:
            return -0.25 * (M - 1)**2 - beta_mass * (M**2 - 1)**2

    # === 4. 压力扩散项 Mp (Liou 2006, AUSM+up 式17) ===
    # 取代此前版本的"熵修正"：旧实现在 mass_flux 上叠加 0.5*(rhoL+rhoR)*a_half
    # *|M_L-M_R|*0.1，而 |M_L-M_R| 在 (L,R,n)->(R,L,-n) 变换下不翻号（是偶量），
    # 直接破坏了 mass_flux 必须满足的反对称性 F(A,B,n)=-F(B,A,-n)（已用受控
    # 数值算例验证：|M|<0.1 时两次调用之和最大相对不平衡达 1.08%，且触发窗口
    # 恰好覆盖本项目 30 m/s / M≈0.087 的目标工况，即全域每个内部面都不守恒）。
    # Mp 项是 Liou 原始 AUSM+up 方案自带的低马赫数稳定化机制，(pR-pL) 在同一
    # 变换下翻号、其余因子（Mbar2/rho_half/a_half/fa）不变，故 Mp 本身翻号，
    # 叠加到已验证满足反对称性的 M_half 上不会破坏该性质。
    # Mp 的声速项与 mass_flux 前面的声阻抗项都用上面按 precond_mode 定好
    # 的 aL_m/aR_m/a_half_m（PRECOND_PHYSICAL 档等于物理声速 aL/aR/a_half，
    # 另两档等于预处理声速）——这是 Weiss-Smith 预处理在 mass_flux 上生效
    # 的地方。相容性不因归一声速的选择而破坏：qL=qR 时 (pR-pL)=0 使 Mp=0，
    # 而 M_plus(M)+M_minus(M) = M 这个恒等式与 M 用哪个声速归一无关，
    # mass_flux = rho*aL_m*(M_half+0) = rho*aL_m*(un/aL_m) = rho*un。
    Kp = 0.25
    sigma_p = 1.0
    M_half = M_plus(M_L) + M_minus(M_R)
    Mp = -(Kp / fa) * max(1.0 - sigma_p * Mbar2, 0.0) * (pR - pL) / (rho_half * a_half_m**2)
    mass_flux = 0.5 * (rhoL * aL_m + rhoR * aR_m) * (M_half + Mp)

    # === 5. AUSM+up 压力通量分裂 ===
    # P5±（Liou 2006, AUSM+up 式 (24)）：
    #     P5±(M) = M2±(M)·[(±2 − M) ∓ 16·α·M·M2∓(M)],  M2±(M) = ±(M±1)²/4
    #   展开：P5+ = (M+1)²(2−M)/4 + α·M·(M²−1)²（α 项**没有** 1/4；SU2
    #   `pLP = 0.25*(mL+1)^2*(2-mL) + alpha*mL*(mL^2-1)^2` 同）。
    #
    # **2026-09-25 修正的真实缺陷**：此前四份实现（本函数、GPU CuPy 版、GPU
    # CUDA P0 源串、backend/fr_gpu_p0.py）都写成 `0.25*((M+1)²(2−M) + α·M·(M²−1)²)`，
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
    p_half = P_plus(M_L_pr) * pL + P_minus(M_R_pr) * pR \
        - Ku * P_plus(M_L_pr) * P_minus(M_R_pr) * (rhoL + rhoR) * fa * a_half_pr * (unR - unL)

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
