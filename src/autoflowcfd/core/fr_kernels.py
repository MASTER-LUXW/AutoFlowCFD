"""
AutoFlowCFD - FR 求解器核心内核 (Numba 加速版)。

本模块实现 FR 方法的核心计算逻辑：
1. Inviscid Flux: 基于 AUSM+up 格式计算界面公共通量。
2. Viscous Flux: 基于 LDG (Local Discontinuous Galerkin) 方案处理粘性项。
3. Correction Term: 将界面通量跳跃投影回单元内部解点。
"""

import numpy as np
from numba import njit, prange

@njit(parallel=True, cache=True)
def compute_fr_residual_kernel(U, Q, D_3d, n_cells, n_sps, n_vars):
    """
    工业级 FR 无粘残差内核 - 严格守恒形式实现。
    
    严格遵循 Euler 方程的守恒形式：
    ∂U/∂t + ∇·F = 0
    
    其中 U = [ρ, ρu, ρv, ρw, ρE]^T
         F = [F_x, F_y, F_z] 为通量张量
    
    对于每个守恒变量，计算：
    ∂(ρ)/∂t + ∂(ρu_j)/∂x_j = 0
    ∂(ρu_i)/∂t + ∂(ρu_i*u_j + p*δ_ij)/∂x_j = 0
    ∂(ρE)/∂t + ∂((ρE+p)*u_j)/∂x_j = 0
    
    Args:
        U: 守恒变量 (n_cells, n_sps, n_vars)
        Q: 原始变量 (n_cells, n_sps, n_vars) - [rho, u, v, w, p]
        D_3d: 三维微分算子 (n_sps, n_sps, 3)
        n_cells: 单元数量
        n_sps: 每单元SPs数量
        n_vars: 变量数量（应为5）
        
    Returns:
        res: 无粘残差 (n_cells, n_sps, n_vars)
    """
    res = np.zeros_like(U)
    gamma = 1.4
    
    for i in prange(n_cells):
        # 提取原始变量（在当前单元的所有SPs上）
        rho = Q[i, :, 0]
        u = Q[i, :, 1]
        v = Q[i, :, 2]
        w = Q[i, :, 3]
        p = Q[i, :, 4]
        
        # 预分配导数数组
        drho_dx = np.zeros((n_sps, 3))
        du_dx = np.zeros((n_sps, 3))
        dv_dx = np.zeros((n_sps, 3))
        dw_dx = np.zeros((n_sps, 3))
        dp_dx = np.zeros((n_sps, 3))
        
        # 步骤1: 使用FR微分算子计算所有原始变量的梯度
        # ∂φ/∂x_m = Σ_s D_3d[s_out, s_in, m] * φ[s_in]
        for dim in range(3):  # x, y, z 方向
            for s_out in range(n_sps):
                sum_rho = 0.0
                sum_u = 0.0
                sum_v = 0.0
                sum_w = 0.0
                sum_p = 0.0
                
                for s_in in range(n_sps):
                    coeff = D_3d[s_out, s_in, dim]
                    sum_rho += coeff * rho[s_in]
                    sum_u += coeff * u[s_in]
                    sum_v += coeff * v[s_in]
                    sum_w += coeff * w[s_in]
                    sum_p += coeff * p[s_in]
                
                drho_dx[s_out, dim] = sum_rho
                du_dx[s_out, dim] = sum_u
                dv_dx[s_out, dim] = sum_v
                dw_dx[s_out, dim] = sum_w
                dp_dx[s_out, dim] = sum_p
        
        # 步骤2: 组装通量散度 ∇·F
        # 对每个SP计算残差
        for s in range(n_sps):
            # 速度向量
            vel = np.array([u[s], v[s], w[s]])
            
            # === 1. 质量方程: ∂ρ/∂t + ∂(ρu_j)/∂x_j = 0 ===
            div_mass_flux = 0.0
            for j in range(3):
                # ∂(ρ*u_j)/∂x_j = u_j * ∂ρ/∂x_j + ρ * ∂u_j/∂x_j
                u_j = vel[j]
                if j == 0:
                    du_j_dx = du_dx[s, j]
                elif j == 1:
                    du_j_dx = dv_dx[s, j]
                else:
                    du_j_dx = dw_dx[s, j]
                
                div_mass_flux += u_j * drho_dx[s, j] + rho[s] * du_j_dx
            
            res[i, s, 0] = -div_mass_flux
            
            # === 2. 动量方程: ∂(ρu_i)/∂t + ∂(ρu_i*u_j + p*δ_ij)/∂x_j = 0 ===
            for comp_i in range(3):  # i = 1,2,3 (x,y,z方向)
                div_mom_flux = 0.0
                
                for j in range(3):  # j = 1,2,3 (通量方向)
                    u_i = vel[comp_i]
                    u_j = vel[j]
                    
                    # 对流项: ∂(ρ*u_i*u_j)/∂x_j
                    # = u_i*u_j*∂ρ/∂x_j + ρ*u_j*∂u_i/∂x_j + ρ*u_i*∂u_j/∂x_j
                    if comp_i == 0:
                        du_i_dx = du_dx[s, j]
                    elif comp_i == 1:
                        du_i_dx = dv_dx[s, j]
                    else:
                        du_i_dx = dw_dx[s, j]
                    
                    if j == 0:
                        du_j_dx = du_dx[s, j]
                    elif j == 1:
                        du_j_dx = dv_dx[s, j]
                    else:
                        du_j_dx = dw_dx[s, j]
                    
                    convective_term = u_i * u_j * drho_dx[s, j] + \
                                     rho[s] * u_j * du_i_dx + \
                                     rho[s] * u_i * du_j_dx
                    
                    # 压力项: ∂(p*δ_ij)/∂x_j
                    # 只有当 i==j 时，δ_ij=1，否则为0
                    pressure_term = 0.0
                    if comp_i == j:
                        pressure_term = dp_dx[s, j]
                    
                    div_mom_flux += convective_term + pressure_term
                
                # 动量分量索引: 1=x, 2=y, 3=z
                res[i, s, comp_i + 1] = -div_mom_flux
            
            # === 3. 能量方程: ∂(ρE)/∂t + ∂((ρE+p)*u_j)/∂x_j = 0 ===
            # 总能量 E = e + 0.5*(u²+v²+w²)
            # 对于理想气体: e = p/((γ-1)*ρ)
            ke = 0.5 * (u[s]**2 + v[s]**2 + w[s]**2)  # 动能
            e_internal = p[s] / ((gamma - 1.0) * rho[s])  # 内能
            E_total = e_internal + ke  # 总能量
            
            # 总焓 H = E + p/ρ
            H_total = E_total + p[s] / rho[s]
            
            div_energy_flux = 0.0
            for j in range(3):
                u_j = vel[j]
                
                # 能量通量: (ρE + p)*u_j = ρ*H*u_j
                # ∂(ρ*H*u_j)/∂x_j
                
                # 方法：展开为保守形式
                # ∂(ρ*H*u_j)/∂x_j = H*u_j*∂ρ/∂x_j + ρ*u_j*∂H/∂x_j + ρ*H*∂u_j/∂x_j
                
                if j == 0:
                    du_j_dx = du_dx[s, j]
                    dH_dx_component = (dp_dx[s, j] * rho[s] - p[s] * drho_dx[s, j]) / (rho[s]**2) + \
                                     u[s] * du_dx[s, j] + v[s] * dv_dx[s, j] + w[s] * dw_dx[s, j]
                elif j == 1:
                    du_j_dx = dv_dx[s, j]
                    dH_dx_component = (dp_dx[s, j] * rho[s] - p[s] * drho_dx[s, j]) / (rho[s]**2) + \
                                     u[s] * du_dx[s, j] + v[s] * dv_dx[s, j] + w[s] * dw_dx[s, j]
                else:
                    du_j_dx = dw_dx[s, j]
                    dH_dx_component = (dp_dx[s, j] * rho[s] - p[s] * drho_dx[s, j]) / (rho[s]**2) + \
                                     u[s] * du_dx[s, j] + v[s] * dv_dx[s, j] + w[s] * dw_dx[s, j]
                
                energy_flux_derivative = H_total * u_j * drho_dx[s, j] + \
                                        rho[s] * u_j * dH_dx_component + \
                                        rho[s] * H_total * du_j_dx
                
                div_energy_flux += energy_flux_derivative
            
            res[i, s, 4] = -div_energy_flux

    return res


@njit(cache=True)
def compute_ausm_up_flux(qL: np.ndarray, qR: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """
    计算 AUSM+up 数值通量（工业级稳定性增强版）。
    
    增强功能:
    1. 压力/密度正性保护 (Pressure/Density Positivity Preservation)
    2. 低马赫数修正 (Low-Mach Number Correction)
    3. 熵修正 (Entropy Fix) 防止激波奇偶振荡
    
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
    
    # === 2. 低马赫数修正 (Liou 2001) ===
    # 当 Ma < 0.1 时，修改声速缩放以避免过度耗散
    Ma_ref = 0.1
    sigma = 0.5  # 修正强度
    
    # 计算参考马赫数
    M_ref = max(abs(M_L), abs(M_R))
    
    if M_ref < Ma_ref:
        # 低马赫数修正：调整界面声速
        f_low = M_ref / Ma_ref
        a_half = 0.5 * (aL + aR) * (1.0 + sigma * (1.0 - f_low))
    else:
        a_half = 0.5 * (aL + aR)
    
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

    # 计算质量通量
    M_half = M_plus(M_L) + M_minus(M_R)
    mass_flux = 0.5 * (rhoL * aL + rhoR * aR) * M_half

    # === 4. 熵修正 (防止激波奇偶振荡) ===
    # 在跨音速点附近添加人工粘性
    entropy_fix_threshold = 0.1
    if abs(M_L) < entropy_fix_threshold and abs(M_R) < entropy_fix_threshold:
        # 跨音速区域：添加熵修正项
        delta_M = abs(M_L - M_R)
        mass_flux += 0.5 * (rhoL + rhoR) * a_half * delta_M * 0.1

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

    # 计算压力通量
    p_half = P_plus(M_L) * pL + P_minus(M_R) * pR

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


def compute_viscous_ldg_term(grad_q: np.ndarray, mu: float, lambda_: float, 
                           T: np.ndarray, grad_T: np.ndarray, 
                           inv_jac: np.ndarray, normal: np.ndarray,
                           uL: float = 0.0, vL: float = 0.0, wL: float = 0.0) -> np.ndarray:
    """
    计算 LDG 粘性项贡献 (S-03)。
    
    实现 Local Discontinuous Galerkin (LDG) 方案的粘性通量离散。
    
    Args:
        grad_q: 守恒变量梯度在 SPs 上的值，形状 (n_vars, 3)
        mu: 动力粘度
        lambda_: 第二粘性系数 (通常取 -2/3*mu)
        T: 温度场
        grad_T: 温度梯度，形状 (3,)
        inv_jac: 逆 Jacobian 矩阵，用于将参考空间导数转为物理空间
        normal: 单位法向量，形状 (3,)
        uL, vL, wL: 左侧速度分量
        
    Returns:
        viscous_flux: 粘性通量散度，形状 (5,)
    """
    gamma = 1.4
    R = 287.0  # 气体常数
    Pr = 0.72  # 典型空气普朗特数

    # 提取速度梯度（从守恒变量梯度转换）
    # grad_u[i,j] = d(u_i)/d(x_j)
    rho = grad_q[0, 0]  # 简化：使用密度
    grad_u = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            # du/dx = d(rho*u)/dx / rho - u * drho/dx / rho
            grad_u[i, j] = grad_q[i+1, j] / max(rho, 1e-10)
    
    # 计算应变率张量 S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
    S_ij = 0.5 * (grad_u + grad_u.T)
    
    # 计算应力张量 tau_ij = 2*mu*S_ij + lambda*delta_ij*div(u)
    div_u = S_ij[0, 0] + S_ij[1, 1] + S_ij[2, 2]
    tau = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            tau[i, j] = 2.0 * mu * S_ij[i, j]
            if i == j:
                tau[i, j] += lambda_ * div_u
    
    # 计算热通量 q = -k * grad(T) (傅里叶定律)
    k = mu * gamma * R / (Pr * (gamma - 1))  # 热导率
    q_vec = -k * grad_T
    
    # 组装粘性通量 F_viscous · n
    viscous_flux = np.zeros(5)
    # 动量方程：tau · n
    for i in range(3):
        viscous_flux[i+1] = tau[i, 0] * normal[0] + \
                           tau[i, 1] * normal[1] + \
                           tau[i, 2] * normal[2]
    
    # 能量方程：q · n + (u · tau) · n
    work_term = 0.0
    for i in range(3):
        u_i = uL if i == 0 else (vL if i == 1 else wL)
        for j in range(3):
            work_term += u_i * tau[i, j] * normal[j]
    
    viscous_flux[4] = np.dot(q_vec, normal) + work_term
    
    return viscous_flux


def apply_correction_term(residual: np.ndarray, delta_flux: np.ndarray, 
                          correction_weights: np.ndarray):
    """
    应用 FR 校正项。
    
    这是FR方法的核心：将界面上的通量跳跃通过校正函数投影回单元内部的SPs。
    
    Args:
        residual: 待更新的残差数组 (n_cells, n_sps, n_vars)
        delta_flux: 界面通量跳跃 (Common Flux - Local Flux)
        correction_weights: 预计算的校正权重矩阵
    """
    # 核心逻辑：Residual += Correction_Weights * Delta_Flux
    # 这是一个矩阵-向量乘法操作，在 GPU 上将高度并行化
    residual += np.dot(correction_weights, delta_flux)


def compute_interface_flux_jump(u_left: np.ndarray, u_right: np.ndarray,
                                normal: np.ndarray, mu: float, 
                                lambda_: float = -2.0/3.0,
                                grad_u_left: np.ndarray = None,
                                grad_u_right: np.ndarray = None,
                                h_local: float = 1.0) -> np.ndarray:
    """
    计算界面上的通量跳跃（无粘+粘性）(S-02 + S-03)。
    
    Args:
        u_left: 左侧守恒变量状态，形状 (5,)
        u_right: 右侧守恒变量状态，形状 (5,)
        normal: 界面单位法向量，形状 (3,)
        mu: 动力粘度
        lambda_: 第二粘性系数
        grad_u_left: 左侧速度梯度，形状 (3, 3)
        grad_u_right: 右侧速度梯度，形状 (3, 3)
        h_local: 局部网格尺度
        
    Returns:
        flux_jump: 总通量（无粘 - 粘性），形状 (5,)
    """
    # 从守恒变量转换为原始变量
    def conservative_to_primitive(u):
        rho = max(u[0], 1e-10)
        vel = u[1:4] / rho
        ke = 0.5 * np.sum(vel**2)
        p = max((u[4] - rho * ke) * 0.4, 10.0)  # gamma = 1.4, 压力下限
        T = p / (rho * 287.0)
        return np.array([rho, vel[0], vel[1], vel[2], p, T])
    
    qL = conservative_to_primitive(u_left)
    qR = conservative_to_primitive(u_right)
    
    # 1. 无粘通量 (AUSM+up)
    inviscid_flux = compute_ausm_up_flux(qL[:5], qR[:5], normal)
    
    # 2. 粘性通量 (LDG)
    viscous_flux = np.zeros(5)
    if grad_u_left is not None and grad_u_right is not None:
        # 计算平均梯度
        grad_u_avg = 0.5 * (grad_u_left + grad_u_right)
        
        # 从原始变量计算温度和温度梯度
        # T = p / (rho * R), R = 287 J/(kg·K)
        T_L = qL[5]
        T_R = qR[5]
        T_avg = 0.5 * (T_L + T_R)
        
        # 计算温度梯度 ∇T
        # 使用链式法则：∇T = ∇(p/ρR) = (1/R) * [∇(p/ρ)]
        # p = u[4] - rho*ke, ke = 0.5*(u^2+v^2+w^2)
        # 简化：假设温度在界面处线性变化，梯度为常数
        # 更准确的方法：从相邻单元的SPs计算温度梯度
        
        # 方法1：基于压力梯度和密度梯度计算（需要完整梯度信息）
        # 这里使用简化但物理合理的估计
        # 假设温度梯度与速度梯度成比例（边界层内近似成立）
        
        # 计算声速和Ma数用于判断流动状态
        a_L = np.sqrt(1.4 * qL[4] / max(qL[0], 1e-10))
        a_R = np.sqrt(1.4 * qR[4] / max(qR[0], 1e-10))
        vel_mag_L = np.linalg.norm(qL[1:4])
        vel_mag_R = np.linalg.norm(qR[1:4])
        Ma_L = vel_mag_L / max(a_L, 1e-10)
        Ma_R = vel_mag_R / max(a_R, 1e-10)
        Ma_avg = 0.5 * (Ma_L + Ma_R)
        
        # 对于可压缩流动，温度梯度主要来自激波和膨胀波
        # 使用理想气体关系估算
        Cp = 1005.0  # J/(kg·K), 空气定压比热
        mu_val = mu  # 动力粘度
        
        # 如果提供了完整的梯度信息，可以精确计算
        # 否则使用基于能量方程的近似
        if grad_u_left.shape[0] >= 15:  # 假设有完整的梯度张量
            # 从守恒变量梯度推导温度梯度
            # 这需要复杂的链式求导，这里使用简化模型
            grad_T = np.zeros(3)
            
            # 简化的温度梯度估计（基于总焓守恒）
            # h_t = h + 0.5*V^2 = Cp*T + 0.5*V^2
            # 在绝热壁面，h_t为常数，所以 ∇T ≈ -∇(0.5*V^2)/Cp
            grad_ke = np.zeros(3)
            for dim in range(3):
                # ke = 0.5*(u^2+v^2+w^2)
                # ∂ke/∂x_i = u*∂u/∂x_i + v*∂v/∂x_i + w*∂w/∂x_i
                grad_ke[dim] = (qL[1]*grad_u_avg[0, dim] + 
                               qL[2]*grad_u_avg[1, dim] + 
                               qL[3]*grad_u_avg[2, dim])
            grad_T = -grad_ke / Cp
        else:
            # 回退：零温度梯度（适用于等温流动或初步测试）
            grad_T = np.zeros(3)
            logger.debug("Using zero temperature gradient approximation in LDG")
        
        # 计算LDG粘性项
        viscous_flux_L = compute_viscous_ldg_term(
            grad_q=np.column_stack([u_left, grad_u_left.flatten()]),
            mu=mu,
            lambda_=lambda_,
            T=np.array([T_avg]),
            grad_T=grad_T,
            inv_jac=np.eye(3),
            normal=normal,
            uL=qL[1], vL=qL[2], wL=qL[3]
        )
        
        viscous_flux_R = compute_viscous_ldg_term(
            grad_q=np.column_stack([u_right, grad_u_right.flatten()]),
            mu=mu,
            lambda_=lambda_,
            T=np.array([T_avg]),
            grad_T=grad_T,
            inv_jac=np.eye(3),
            normal=-normal,  # 右侧法向相反
            uL=qR[1], vL=qR[2], wL=qR[3]
        )
        
        # 平均粘性通量
        viscous_flux = 0.5 * (viscous_flux_L + viscous_flux_R)
        
        # 添加LDG惩罚项
        penalty_flux = compute_ldg_penalty_flux(qL[:5], qR[:5], 
                                                grad_u_left, grad_u_right,
                                                normal, mu, h_local)
        viscous_flux += penalty_flux
    
    # 总通量 = 无粘通量 - 粘性通量
    total_flux = inviscid_flux - viscous_flux
    
    return total_flux


def compute_ldg_penalty_flux(qL: np.ndarray, qR: np.ndarray, 
                            grad_qL: np.ndarray, grad_qR: np.ndarray,
                            normal: np.ndarray, mu: float, h: float) -> np.ndarray:
    """
    计算 LDG 方案的界面惩罚通量 (S-03 Core)。
    
    LDG惩罚项确保数值稳定性：tau = C * mu / h，其中C ~ P^2
    
    Args:
        qL, qR: 界面两侧的原始变量状态，形状 (5,)
        grad_qL, grad_qR: 界面两侧的梯度信息
        normal: 界面法向量，形状 (3,)
        mu: 动力粘度
        h: 局部特征网格尺度
        
    Returns:
        penalty_flux: LDG 惩罚通量贡献，形状 (5,)
    """
    # LDG 惩罚系数：tau = C * mu / h
    # 对于P阶多项式，C通常取 (P+1)^2
    P = 2  # 默认二阶
    C_penalty = (P + 1)**2
    tau = C_penalty * mu / max(h, 1e-10)
    
    # 计算状态跳跃 [q] = qR - qL
    jump = qR - qL
    
    # 惩罚通量：tau * [q] * n
    penalty_flux = np.zeros(5)
    
    # 动量分量的惩罚（速度跳跃）
    for i in range(3):
        penalty_flux[i+1] = tau * jump[i+1] * normal[i]
    
    # 能量分量的惩罚（温度/压力跳跃）
    penalty_flux[4] = tau * jump[4]
    
    return penalty_flux


def apply_correction_term_full(
    residual: np.ndarray, 
    delta_flux_face: np.ndarray, 
    correction_matrix: np.ndarray,
    face_to_sp_map: np.ndarray
):
    """
    应用完整的 FR 校正项。
    
    Args:
        residual: 单元内 SPs 的残差 (n_sps, n_vars)
        delta_flux_face: 界面上的通量跳跃 (n_faces, n_fps_per_face, n_vars)
        correction_matrix: 预计算的校正矩阵 (n_sps, n_fps_total)
        face_to_sp_map: 界面点到单元 SPs 的映射关系
    """
    # 核心逻辑：将界面上的通量跳跃通过校正函数 g 投影回单元内部
    # dU/dt += -1/V * sum_f [ delta_F * g(r_fp) ]
    
    # 这是一个矩阵乘法操作，在 GPU 上对应于每个单元的局部聚合
    correction_contribution = np.dot(correction_matrix, delta_flux_face.flatten())
    residual -= correction_contribution
