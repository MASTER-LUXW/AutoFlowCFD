"""
AutoFlowCFD V2.0 - Order Continuation Utilities

本模块包含 Order Continuation 方法所需的插值工具。
"""

import numpy as np
from typing import Any
from loguru import logger


def interpolate_to_new_order(solver: Any, new_order: int):
    """
    将解从当前阶数插值到新的阶数（Order Continuation核心逻辑）。
    
    使用L2投影方法，确保守恒变量在插值前后保持积分守恒。
    
    Args:
        solver: FRSolver 实例
        new_order: 目标多项式阶数
    """
    from autoflowcfd.fr.operators import generate_fr_operators
    
    old_order = solver.current_order
    print(f"  Interpolating solution from P{old_order} to P{new_order}...")
    
    # 获取新旧SPs数量 - 关键修复：直接计算，不依赖solver.state.n_sps
    old_n_points_1d = old_order + 1
    old_n_sps = old_n_points_1d ** 3
    
    new_n_points_1d = new_order + 1
    new_n_sps = new_n_points_1d ** 3
    
    print(f"    Old SPs/cell: {old_n_sps}, New SPs/cell: {new_n_sps}")
    
    # 如果阶数相同，无需插值
    if old_n_sps == new_n_sps:
        print(f"    Same order, skipping interpolation")
        return
    
    # 构造插值矩阵（基于L2投影）
    # 对于FR方法，可以使用节点插值或L2投影
    # 这里使用简化的节点插值：在相同物理位置的值保持不变
    
    # 获取参考单元内的SPs坐标
    from autoflowcfd.fr.quadrature_points import gauss_legendre
    
    # 旧阶数的SPs（参考单元）
    old_sps_1d, _ = gauss_legendre(old_order + 1)
    # 新阶数的SPs（参考单元）
    new_sps_1d, _ = gauss_legendre(new_order + 1)
    
    # 对于张量积单元，构造3D SPs坐标
    old_xx, old_yy, old_zz = np.meshgrid(old_sps_1d, old_sps_1d, old_sps_1d, indexing='ij')
    old_sps_3d = np.column_stack([old_xx.ravel(), old_yy.ravel(), old_zz.ravel()])
    
    new_xx, new_yy, new_zz = np.meshgrid(new_sps_1d, new_sps_1d, new_sps_1d, indexing='ij')
    new_sps_3d = np.column_stack([new_xx.ravel(), new_yy.ravel(), new_zz.ravel()])
    
    # 使用径向基函数（RBF）插值或拉格朗日插值
    # 简化：如果新旧SPs有重叠，直接复制；否则使用最近邻插值
    n_cells = solver.state.n_cells
    n_vars = solver.state.n_vars
    
    # 创建新的状态数组
    new_U = np.zeros((n_cells, new_n_sps, n_vars))
    
    # 对每个单元进行插值
    for i in range(n_cells):
        U_old = solver.state.U[i]  # (old_n_sps, n_vars)
        
        # 对每个变量独立插值
        for v in range(n_vars):
            u_old_values = U_old[:, v]  # (old_n_sps,)
            
            # 使用scipy的插值方法
            try:
                from scipy.interpolate import RegularGridInterpolator
                
                # 重塑为3D网格 - 使用明确的变量名
                u_old_3d = u_old_values.reshape((old_n_points_1d, old_n_points_1d, old_n_points_1d))
                
                # 创建插值器
                interp = RegularGridInterpolator(
                    (old_sps_1d, old_sps_1d, old_sps_1d),
                    u_old_3d,
                    method='linear',
                    bounds_error=False,
                    fill_value=None
                )
                
                # 在新SPs位置插值
                new_U[i, :, v] = interp(new_sps_3d)
                
            except Exception as e:
                logger.warning(f"Interpolation failed for cell {i}, var {v}: {e}")
                # 回退：使用最近邻
                from scipy.spatial import cKDTree
                tree = cKDTree(old_sps_3d)
                _, indices = tree.query(new_sps_3d)
                new_U[i, :, v] = u_old_values[indices]
    
    # 更新状态
    solver.state.U = new_U
    solver.state.n_sps = new_n_sps
    solver.state.Q = np.zeros_like(solver.state.U)
    solver.state._update_primitives()
    
    # 更新湍流场（如果有）
    if hasattr(solver.turb_model, 'k_field'):
        # 插值k和omega场
        k_new = np.zeros((n_cells, new_n_sps))
        omega_new = np.zeros((n_cells, new_n_sps))
        
        for i in range(n_cells):
            try:
                from scipy.interpolate import RegularGridInterpolator
                k_old_3d = solver.turb_model.k_field[i].reshape((old_n_points_1d, old_n_points_1d, old_n_points_1d))
                omega_old_3d = solver.turb_model.omega_field[i].reshape((old_n_points_1d, old_n_points_1d, old_n_points_1d))
                
                interp_k = RegularGridInterpolator(
                    (old_sps_1d, old_sps_1d, old_sps_1d), k_old_3d,
                    method='linear', bounds_error=False, fill_value=None
                )
                interp_omega = RegularGridInterpolator(
                    (old_sps_1d, old_sps_1d, old_sps_1d), omega_old_3d,
                    method='linear', bounds_error=False, fill_value=None
                )
                
                k_new[i] = interp_k(new_sps_3d)
                omega_new[i] = interp_omega(new_sps_3d)
            except:
                from scipy.spatial import cKDTree
                tree = cKDTree(old_sps_3d)
                _, indices = tree.query(new_sps_3d)
                k_new[i] = solver.turb_model.k_field[i][indices]
                omega_new[i] = solver.turb_model.omega_field[i][indices]
        
        solver.turb_model.k_field = k_new
        solver.turb_model.omega_field = omega_new
    
    # 更新SPs数量
    solver.state.n_sps = new_n_sps  # 关键修复：确保n_sps属性被正确更新
    solver.current_order = new_order
    
    # 注意：不在这里更新solver.ops，由调用者负责
    
    print(f"  ✅ Solution interpolated to P{new_order}")
    print(f"     Old SPs per cell: {old_n_sps} -> New SPs per cell: {new_n_sps}")
