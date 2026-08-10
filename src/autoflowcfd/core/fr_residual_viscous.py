"""
AutoFlowCFD V2.0 - FR 粘性残差计算 (S-03)

本模块实现基于 LDG (Local Discontinuous Galerkin) 方案的粘性项离散。

核心功能:
1. 速度梯度重构
2. LDG 惩罚项计算
3. 应力张量和热通量计算
4. 粘性通量散度组装
"""

import numpy as np
from typing import Tuple, Optional
from loguru import logger


def compute_viscous_residual(
    state_U: np.ndarray,
    state_Q: np.ndarray,
    ops,
    mesh,
    mu: float = 1.8e-5,
    Pr: float = 0.72,
    gamma: float = 1.4
) -> np.ndarray:
    """
    计算粘性残差（LDG方案）- 完整实现。
    
    实现完整的Navier-Stokes方程粘性项：
    - 动量方程：∂/∂x_j(τ_ij)
    - 能量方程：∂/∂x_j(u_i*τ_ij + q_j)
    
    Args:
        state_U: 守恒变量，形状 (n_cells, n_sps, n_vars)
        state_Q: 原始变量，形状 (n_cells, n_sps, n_vars)
        ops: FR算子集合
        mesh: 高阶网格对象
        mu: 动力粘度
        Pr: 普朗特数
        gamma: 比热比
        
    Returns:
        viscous_res: 粘性残差，形状 (n_cells, n_sps, n_vars)
    """
    n_cells, n_sps, n_vars = state_U.shape
    
    # 初始化粘性残差
    viscous_res = np.zeros_like(state_U)
    
    # 步骤1: 计算速度梯度和温度梯度
    grad_U = compute_gradients(state_U, ops)
    
    # 预计算所有单元的温度和梯度
    T_all = np.zeros((n_cells, n_sps))
    grad_T_all = np.zeros((n_cells, n_sps, 3))
    
    for i in range(n_cells):
        Q_cell = state_Q[i]
        rho = Q_cell[:, 0]
        u = Q_cell[:, 1] / np.maximum(rho, 1e-10)
        v = Q_cell[:, 2] / np.maximum(rho, 1e-10)
        w = Q_cell[:, 3] / np.maximum(rho, 1e-10)
        p = (gamma - 1.0) * (Q_cell[:, 4] - 0.5 * rho * (u**2 + v**2 + w**2))
        
        # 温度: T = p/(ρR)
        T_all[i] = p / np.maximum(rho * 287.0, 1e-10)
        
        # 温度梯度: ∂T/∂x_m = Σ D_3d[s_out, s_in, m] * T[s_in]
        if hasattr(ops, 'D_3d') and ops.D_3d is not None:
            for d in range(3):
                grad_T_all[i, :, d] = ops.D_3d[:, :, d] @ T_all[i]
    
    # 步骤2: 对每个单元计算粘性通量散度
    for i in range(n_cells):
        Q_cell = state_Q[i]
        grad_Q_cell = grad_U[i]
        
        # 提取流场变量
        rho = Q_cell[:, 0]
        u_vel = Q_cell[:, 1] / np.maximum(rho, 1e-10)
        v_vel = Q_cell[:, 2] / np.maximum(rho, 1e-10)
        w_vel = Q_cell[:, 3] / np.maximum(rho, 1e-10)
        p = (gamma - 1.0) * (Q_cell[:, 4] - 0.5 * rho * (u_vel**2 + v_vel**2 + w_vel**2))
        T = T_all[i]
        
        # 速度梯度
        grad_u = grad_Q_cell[:, 1:4, :]  # (n_sps, 3, 3)
        
        # 应变率张量 S_ij = 0.5*(∂u_i/∂x_j + ∂u_j/∂x_i)
        S_ij = 0.5 * (grad_u + np.transpose(grad_u, (0, 2, 1)))
        
        # 第二粘性系数（Stokes假设）
        lambda_ = -2.0/3.0 * mu
        
        # 速度散度
        div_u = S_ij[:, 0, 0] + S_ij[:, 1, 1] + S_ij[:, 2, 2]
        
        # 应力张量 τ_ij = 2μS_ij + λδ_ij(∇·u)
        tau = np.zeros((n_sps, 3, 3))
        for ii in range(3):
            for jj in range(3):
                tau[:, ii, jj] = 2.0 * mu * S_ij[:, ii, jj]
                if ii == jj:
                    tau[:, ii, jj] += lambda_ * div_u
        
        # 热通量 q_j = -k*∂T/∂x_j（傅里叶定律）
        k = mu * gamma * 287.0 / (Pr * (gamma - 1.0))  # 热导率
        q_vec = -k * grad_T_all[i]  # (n_sps, 3)
        
        # === 组装粘性残差 ===
        
        # 动量方程残差：∂τ_ij/∂x_j
        for comp_i in range(3):  # i = x, y, z
            div_tau_i = np.zeros(n_sps)
            for j in range(3):  # j = x, y, z方向
                if hasattr(ops, 'D_3d') and ops.D_3d is not None:
                    # ∂τ_ij/∂x_j
                    div_tau_i += ops.D_3d[:, :, j] @ tau[:, comp_i, j]
            viscous_res[i, :, comp_i + 1] = div_tau_i
        
        # 能量方程残差：∂/∂x_j(u_i*τ_ij + q_j)
        div_energy_viscous = np.zeros(n_sps)
        
        for j in range(3):  # j = x, y, z方向
            # 功项：u_i * τ_ij
            work_term = np.zeros(n_sps)
            for comp_i in range(3):
                vel_i = u_vel if comp_i == 0 else (v_vel if comp_i == 1 else w_vel)
                work_term += vel_i * tau[:, comp_i, j]
            
            # 总能量粘性通量：u_i*τ_ij + q_j
            total_energy_flux = work_term + q_vec[:, j]
            
            # 散度：∂/∂x_j(total_energy_flux)
            if hasattr(ops, 'D_3d') and ops.D_3d is not None:
                div_energy_viscous += ops.D_3d[:, :, j] @ total_energy_flux
        
        viscous_res[i, :, 4] = div_energy_viscous
    
    # 步骤3: 添加 LDG 惩罚项（确保数值稳定性）
    penalty_res = compute_ldg_penalty_term(state_U, state_Q, ops, mesh, mu)
    viscous_res -= penalty_res
    
    return viscous_res


def compute_gradients(U: np.ndarray, ops) -> np.ndarray:
    """
    计算守恒变量的梯度。
    
    Args:
        U: 守恒变量，形状 (n_cells, n_sps, n_vars)
        ops: FR算子
        
    Returns:
        grad_U: 梯度，形状 (n_cells, n_sps, n_vars, 3)
    """
    n_cells, n_sps, n_vars = U.shape
    grad_U = np.zeros((n_cells, n_sps, n_vars, 3))
    
    if not hasattr(ops, 'D_3d') or ops.D_3d is None:
        logger.warning("D_3d operator not available, returning zero gradients")
        return grad_U
    
    D_3d = ops.D_3d
    
    # 关键修复：确保D_3d的第一维与U的第二维(n_sps)匹配
    # 在Order Continuation中，不同阶数会有不同的n_sps
    expected_n_sps = D_3d.shape[0]
    
    for i in range(n_cells):
        # 检查当前单元的SPs数量是否与算子匹配
        actual_n_sps = U.shape[1]
        
        if actual_n_sps != expected_n_sps:
            # Order Continuation场景：状态变量的SPs数量与预计算算子不匹配
            # 需要插值或重新生成算子
            logger.warning(
                f"Cell {i}: SPs mismatch - U has {actual_n_sps} SPs but "
                f"D_3d expects {expected_n_sps}. This indicates Order Continuation "
                f"is active. Skipping gradient computation for this cell."
            )
            continue
        
        for v in range(n_vars):
            for d in range(3):
                # 确保向量维度匹配
                u_component = U[i, :, v]  # 形状: (n_sps,)
                
                # 安全检查：验证矩阵乘法维度
                if D_3d.shape[1] == u_component.shape[0]:
                    grad_U[i, :, v, d] = D_3d[:, :, d] @ u_component
                else:
                    logger.error(
                        f"Dimension mismatch in gradient computation: "
                        f"D_3d shape {D_3d.shape}, u_component shape {u_component.shape}"
                    )
                    raise ValueError(
                        f"Cannot compute gradient: D_3d[:, :, {d}] has shape "
                        f"{D_3d[:, :, d].shape} but U[{i}, :, {v}] has shape {u_component.shape}. "
                        f"This typically happens during Order Continuation when operators "
                        f"haven't been regenerated for the new polynomial order."
                    )
    
    return grad_U


def compute_scalar_gradient(scalar_field: np.ndarray, ops) -> np.ndarray:
    """
    计算标量场的梯度（用于湍流模型）。
    
    Args:
        scalar_field: 标量场，形状 (n_cells, n_sps, 1) 或 (n_cells, n_sps)
        ops: FR算子
        
    Returns:
        grad_scalar: 梯度，形状 (n_cells, n_sps, 3)
    """
    # 处理输入形状
    if scalar_field.ndim == 3:
        # 形状为 (n_cells, n_sps, 1)，展平最后一维
        scalar_field = scalar_field.squeeze(axis=-1)
    
    n_cells, n_sps = scalar_field.shape
    grad_scalar = np.zeros((n_cells, n_sps, 3))
    
    if not hasattr(ops, 'D_3d') or ops.D_3d is None:
        logger.warning("D_3d operator not available, returning zero gradients")
        return grad_scalar
    
    D_3d = ops.D_3d
    
    # 安全检查：确保维度匹配
    expected_n_sps = D_3d.shape[0]
    
    for i in range(n_cells):
        actual_n_sps = scalar_field.shape[1]
        
        if actual_n_sps != expected_n_sps:
            logger.warning(
                f"Cell {i}: Scalar field has {actual_n_sps} SPs but "
                f"D_3d expects {expected_n_sps}. Skipping."
            )
            continue
        
        for d in range(3):
            if D_3d.shape[1] == actual_n_sps:
                grad_scalar[i, :, d] = D_3d[:, :, d] @ scalar_field[i]
            else:
                raise ValueError(
                    f"Dimension mismatch in scalar gradient: "
                    f"D_3d[:, :, {d}] shape {D_3d[:, :, d].shape} vs "
                    f"scalar_field[{i}] shape {scalar_field[i].shape}"
                )
    
    return grad_scalar


def compute_ldg_penalty_term(
    U: np.ndarray,
    Q: np.ndarray,
    ops,
    mesh,
    mu: float,
    C_penalty: float = None
) -> np.ndarray:
    """
    计算 LDG 惩罚项。
    
    LDG (Local Discontinuous Galerkin) 方案通过在单元界面引入数值通量
    来处理粘性项的不连续性。惩罚项的形式为：
    
    τ = η * (P+1)^2 * μ / h
    
    其中 η 是用户定义的常数（通常取4-10），P是多项式阶数，h是局部网格尺度。
    
    惩罚项对残差的贡献为：
    R_penalty = -τ * [U] / h
    
    其中 [U] 是界面处的状态跳跃。
    
    Args:
        U: 守恒变量，形状 (n_cells, n_sps, n_vars)
        Q: 原始变量，形状 (n_cells, n_sps, n_vars)
        ops: FR算子
        mesh: 高阶网格
        mu: 动力粘度
        C_penalty: 惩罚系数（默认使用 4*(P+1)^2）
        
    Returns:
        penalty: 惩罚项残差，形状 (n_cells, n_sps, n_vars)
    """
    n_cells, n_sps, n_vars = U.shape
    
    # 关键修复：根据实际状态变量的SPs数量调整h_local和tau的维度
    # 在Order Continuation期间，U的SPs数量可能与网格预定义的n_sps_per_cell不同
    actual_n_sps = n_sps
    
    # 确定惩罚系数
    if C_penalty is None:
        P = getattr(mesh, 'order', 2)
        # LDG文献推荐的惩罚系数：η*(P+1)^2，η通常取4-10
        eta = 4.0
        C_penalty = eta * (P + 1)**2
    
    # 估计局部网格尺度 h
    h_local_full = estimate_local_grid_scale(mesh)
    
    # 如果h_local的SPs维度与U不匹配，进行调整
    if h_local_full.shape[1] != actual_n_sps:
        logger.warning(
            f"h_local shape {h_local_full.shape} mismatch with U shape {U.shape}. "
            f"Adjusting h_local to match current state."
        )
        # 取每个单元的平均h，然后广播到实际SPs数量
        h_mean = np.mean(h_local_full, axis=1, keepdims=True)
        h_local = np.tile(h_mean, (1, actual_n_sps))
    else:
        h_local = h_local_full
    
    # 计算惩罚系数 tau = C * mu / h^2
    # 注意：LDG惩罚项通常除以h^2而不是h
    # tau 形状: (n_cells, n_sps)
    tau = C_penalty * mu / (h_local**2 + 1e-10)
    
    # 初始化惩罚项
    penalty = np.zeros_like(U)
    
    # 如果有网格连通性信息，计算真实的界面跳跃
    if hasattr(mesh, 'face_connectivity') and mesh.face_connectivity is not None:
        # 完整的LDG实现：遍历所有内部面和边界面
        penalty = _compute_full_ldg_penalty(U, Q, mesh, tau, ops)
    else:
        # 简化但物理合理的近似：基于单元内梯度估计界面跳跃
        # [U] ≈ h * ∇U · n
        for i in range(n_cells):
            # 对每个SP，估计与"虚拟邻居"的跳跃
            # 使用局部梯度作为跳跃的代理
            for var_idx in range(n_vars):
                # 提取当前变量的场
                u_var = U[i, :, var_idx]  # (n_sps,)
                
                # 使用FR微分算子计算梯度
                if hasattr(ops, 'D_3d') and ops.D_3d is not None:
                    # 计算梯度的模 |∇u|
                    grad_u = np.zeros((n_sps, 3))
                    for dim in range(3):
                        grad_u[:, dim] = ops.D_3d[:, :, dim] @ u_var
                    
                    grad_mag = np.sqrt(np.sum(grad_u**2, axis=1))
                    
                    # 跳跃估计：[u] ≈ h * |∇u|
                    jump_estimate = h_local[i] * grad_mag
                    
                    # 惩罚项：-tau * [u]
                    penalty[i, :, var_idx] = -tau[i] * jump_estimate
    
    return penalty


def _compute_full_ldg_penalty(U: np.ndarray, Q: np.ndarray, mesh, 
                              tau: np.ndarray, ops) -> np.ndarray:
    """
    完整的LDG惩罚项计算，考虑真实的网格连通性。
    
    Args:
        U: 守恒变量
        Q: 原始变量
        mesh: 包含face_connectivity的网格对象
        tau: 惩罚系数
        ops: FR算子
        
    Returns:
        penalty: 惩罚项残差
    """
    n_cells, n_sps, n_vars = U.shape
    penalty = np.zeros_like(U)
    
    # 获取面连通性
    faces = mesh.face_connectivity  # 列表 of (cell_L, cell_R, face_normal, face_area)
    
    for face_data in faces:
        cell_L, cell_R, normal, area = face_data
        
        if cell_L >= 0 and cell_L < n_cells:
            # 左侧单元的状态（在靠近界面的SPs上）
            U_L = U[cell_L]  # (n_sps, n_vars)
            
            if cell_R >= 0 and cell_R < n_cells:
                # 内部面：右侧单元存在
                U_R = U[cell_R]
                
                # 计算界面跳跃 [U] = U_R - U_L
                jump = U_R - U_L
                
                # 左侧单元的惩罚贡献
                penalty[cell_L] -= tau[cell_L][:, np.newaxis] * jump * area
                
                # 右侧单元的惩罚贡献（符号相反）
                penalty[cell_R] += tau[cell_R][:, np.newaxis] * jump * area
            else:
                # 边界面：需要边界条件
                # 对于粘性边界，通常使用镜像或Dirichlet条件
                U_bc = _get_boundary_state_viscous(Q[cell_L], normal, mesh.boundary_type.get(cell_L, 'WALL'))
                jump = U_bc - U_L
                penalty[cell_L] -= tau[cell_L][:, np.newaxis] * jump * area
    
    return penalty


def _get_boundary_state_viscous(Q_cell: np.ndarray, normal: np.ndarray, 
                                bc_type: str) -> np.ndarray:
    """
    获取粘性流动的边界状态。
    
    Args:
        Q_cell: 单元原始变量 (n_sps, n_vars)
        normal: 边界法向量
        bc_type: 边界类型
        
    Returns:
        Q_bc: 边界状态 (n_sps, n_vars)
    """
    n_sps = Q_cell.shape[0]
    Q_bc = Q_cell.copy()
    
    if bc_type.upper() == 'WALL':
        # 无滑移壁面：u=0, T=T_wall（假设绝热则dT/dn=0）
        Q_bc[:, 1:4] = 0.0  # 速度置零
        # 保持压力和密度不变（绝热近似）
    
    elif bc_type.upper() == 'SYMMETRY':
        # 对称边界：反射法向速度
        vel = Q_bc[:, 1:4]
        vel_normal = np.sum(vel * normal[np.newaxis, :], axis=1, keepdims=True)
        vel_reflected = vel - 2.0 * vel_normal * normal[np.newaxis, :]
        Q_bc[:, 1:4] = vel_reflected
    
    # 其他边界类型可根据需要扩展
    
    return Q_bc


def estimate_local_grid_scale(mesh) -> np.ndarray:
    """
    估计局部网格尺度。
    
    Args:
        mesh: 高阶网格对象
        
    Returns:
        h: 局部网格尺度，形状 (n_cells, n_sps)
    """
    n_cells = getattr(mesh, 'n_cells', 1)
    n_sps = getattr(mesh, 'n_sps_per_cell', 8)
    
    # 如果有Jacobian行列式，使用它来估计体积
    if hasattr(mesh, 'jacobians') and mesh.jacobians is not None:
        det_jacs = mesh.jacobians.get('det_jacs', None)
        if det_jacs is not None:
            # det_jacs 可能是一维 (n_cells*n_sps,) 或二维 (n_cells, n_sps)
            if det_jacs.ndim == 1:
                # 重塑为 (n_cells, n_sps)
                det_jacs = det_jacs.reshape(n_cells, n_sps)
            
            # h ~ V^(1/3)，其中V是单元体积
            # 对每个单元的所有SPs取平均Jacobian行列式
            volumes = np.mean(det_jacs, axis=1) * 8.0  # 参考单元体积为8
            h = np.power(np.abs(volumes), 1.0/3.0)
            # 扩展到所有SPs
            h_expanded = np.tile(h[:, np.newaxis], (1, n_sps))
            return h_expanded
    
    # 回退：使用均匀估计
    return np.ones((n_cells, n_sps)) * 0.01
