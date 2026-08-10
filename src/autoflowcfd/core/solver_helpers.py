"""
AutoFlowCFD V2.0 - Solver Helper Functions

本模块包含 FRSolver 的辅助函数，用于处理壁面边界信息、WMLES应用等。
目的是减少 fr_solver.py 的代码复杂度。
"""

import numpy as np
from typing import Dict, Any, Optional
from loguru import logger


def compute_scalar_gradient_simple(scalar_field: np.ndarray, ops: Any) -> np.ndarray:
    """
    计算标量场的梯度（简化版本）。
    
    Args:
        scalar_field: 标量场，形状 (n_cells, n_sps)
        ops: FR 算子对象，包含 D_3d 微分矩阵
        
    Returns:
        gradient: 梯度张量，形状 (n_cells, n_sps, 3)
    """
    n_cells, n_sps = scalar_field.shape
    
    # 使用FR微分算子
    if hasattr(ops, 'D_3d') and ops.D_3d is not None:
        # D_3d 形状: (n_sps, n_sps, 3)
        gradient = np.zeros((n_cells, n_sps, 3))
        for dim in range(3):
            # 对每个单元和每个SP，计算梯度分量
            for i in range(n_cells):
                gradient[i, :, dim] = np.dot(ops.D_3d[:, :, dim], scalar_field[i])
        return gradient
    else:
        # 回退：使用有限差分近似
        logger.warning("FR operators not available, using finite difference approximation")
        gradient = np.zeros((n_cells, n_sps, 3))
        # 简化的中心差分（假设均匀网格）
        dx = 0.01  # 假设网格尺度
        for i in range(1, n_cells-1):
            gradient[i, :, 0] = (scalar_field[i+1] - scalar_field[i-1]) / (2*dx)
        return gradient


def extract_wall_boundary_info(solver: Any) -> Optional[Dict[str, Any]]:
    """
    从求解器关联的网格或边界管理器中提取壁面边界信息。
    
    Args:
        solver: FRSolver 实例
        
    Returns:
        wall_sp_info: 字典，包含 'cell_idx', 'sp_idx', 'normals', 'boundary_names'
                      如果未找到壁面，返回 None
    """
    # 方法1: 尝试从boundary_manager获取
    if hasattr(solver, 'boundary_manager') and solver.boundary_manager is not None:
        try:
            boundary_map = solver.boundary_manager.boundary_map
            
            # 查找所有WALL类型的边界
            wall_cell_indices = []
            wall_boundary_names = []
            
            for bname in boundary_map.boundary_names:
                btype = boundary_map.get_boundary_type(bname)
                if btype == 'WALL' or 'wall' in bname.lower():
                    cell_indices = boundary_map.get_cell_indices(bname)
                    wall_cell_indices.extend(cell_indices)
                    wall_boundary_names.append(bname)
            
            if len(wall_cell_indices) > 0:
                wall_cell_indices = np.array(wall_cell_indices)
                
                # 对于每个壁面单元，获取其所有SPs
                n_wall_cells = len(wall_cell_indices)
                n_sps_per_cell = solver.mesh.n_sps_per_cell
                
                # 构造cell_idx和sp_idx数组
                cell_idx_list = []
                sp_idx_list = []
                
                for c_idx in wall_cell_indices:
                    for s_idx in range(n_sps_per_cell):
                        cell_idx_list.append(c_idx)
                        sp_idx_list.append(s_idx)
                
                cell_idx = np.array(cell_idx_list)
                sp_idx = np.array(sp_idx_list)
                
                wall_sp_info = {
                    'cell_idx': cell_idx,
                    'sp_idx': sp_idx,
                    'normals': None,  # 法向量需要从网格几何计算
                    'boundary_names': wall_boundary_names
                }
                
                logger.info(f"Extracted {len(cell_idx)} wall SPs from {len(wall_cell_indices)} wall cells (from boundary_manager)")
                return wall_sp_info
        except Exception as e:
            logger.warning(f"Failed to extract wall info from boundary_manager: {e}")
    
    # 方法2: 从网格的boundary_faces中提取（如果有）
    if hasattr(solver.mesh, 'boundary_faces') and solver.mesh.boundary_faces is not None:
        try:
            boundary_faces = solver.mesh.boundary_faces
            
            # 查找WALL类型的边界face
            wall_face_indices = []
            for face_idx, face_data in enumerate(boundary_faces):
                if hasattr(face_data, 'bc_type') and face_data.bc_type == 'WALL':
                    wall_face_indices.append(face_idx)
                elif isinstance(face_data, dict) and face_data.get('bc_type') == 'WALL':
                    wall_face_indices.append(face_idx)
            
            if len(wall_face_indices) > 0:
                logger.info(f"Found {len(wall_face_indices)} wall faces")
                
                # 实现从face到cell/SP的映射
                # 假设mesh提供了face_to_cell映射: (n_faces, 2) [owner, neighbor]
                if hasattr(solver.mesh, 'face_to_cell'):
                    face_to_cell = solver.mesh.face_to_cell
                    
                    wall_cells = set()
                    for face_idx in wall_face_indices:
                        owner = face_to_cell[face_idx, 0]
                        if owner >= 0:
                            wall_cells.add(owner)
                    
                    if wall_cells:
                        wall_cell_array = np.array(list(wall_cells))
                        n_wall_cells = len(wall_cell_array)
                        n_sps = solver.state.n_sps_per_cell
                        
                        # 为每个壁面单元的所有SPs标记
                        cell_indices = np.repeat(wall_cell_array, n_sps)
                        sp_indices = np.tile(np.arange(n_sps), n_wall_cells)
                        
                        # 计算法向量（简化：使用y方向）
                        normals = np.zeros((len(cell_indices), 3))
                        normals[:, 1] = 1.0  # 假设壁面法向沿y轴
                        
                        logger.info(f"Successfully mapped {n_wall_cells} wall cells to {len(cell_indices)} SPs")
                        
                        return {
                            'cell_idx': cell_indices,
                            'sp_idx': sp_indices,
                            'normals': normals,
                            'boundary_names': ['WALL']
                        }
        except Exception as e:
            logger.debug(f"Failed to extract from mesh.boundary_faces: {e}")
    
    # 方法3: 基于几何特征自动识别（简化版）
    logger.info("Attempting automatic wall detection based on geometry...")
    return auto_detect_wall_boundaries(solver)


def auto_detect_wall_boundaries(solver: Any) -> Optional[Dict[str, Any]]:
    """
    基于几何特征自动检测壁面边界。
    
    Args:
        solver: FRSolver 实例
        
    Returns:
        wall_sp_info: 字典，包含 'cell_idx', 'sp_idx', 'normals'
    """
    # 获取网格节点坐标
    if not hasattr(solver.mesh, 'nodes') or solver.mesh.nodes is None:
        logger.warning("Mesh nodes not available for auto-detection")
        return {'cell_idx': np.array([]), 'sp_idx': np.array([]), 'normals': None}
    
    nodes = solver.mesh.nodes  # (n_nodes, 3)
    
    # 简化的壁面检测：查找y接近0的节点
    # 实际应用中应该更复杂，考虑多个方向的壁面
    wall_tolerance = 1e-6  # 容差
    
    # 检查各个方向的壁面
    # Y方向壁面（y=0或y=max）
    y_min = nodes[:, 1].min()
    y_max = nodes[:, 1].max()
    
    detected_wall_nodes = []
    if abs(y_min) < wall_tolerance:
        logger.info(f"Detected potential wall at y={y_min:.6f}")
        # 找到y接近y_min的节点索引
        wall_node_indices = np.where(np.abs(nodes[:, 1] - y_min) < wall_tolerance)[0]
        detected_wall_nodes.extend(wall_node_indices)
        
    if abs(y_max) < wall_tolerance or abs(y_max - nodes[:, 1].max()) < wall_tolerance:
         # 注意：这里逻辑可能需要调整，取决于网格坐标系
         pass

    if len(detected_wall_nodes) == 0:
        logger.warning("No automatic wall boundaries detected.")
        return {'cell_idx': np.array([]), 'sp_idx': np.array([]), 'normals': None}
    
    # 找到包含这些节点的单元
    # 这是一个昂贵的操作，简化处理：假设前几个单元是壁面单元（仅用于测试）
    # 实际生产环境需要高效的 node-to-cell 映射
    logger.warning("Automatic wall detection found nodes but mapping to cells is not fully implemented.")
    return {'cell_idx': np.array([]), 'sp_idx': np.array([]), 'normals': None}


def apply_wmles_wall_stress(solver: Any):
    """
    应用WMLES壁面剪应力到动量方程残差。
    
    Args:
        solver: FRSolver 实例
    """
    # 检查是否有壁面边界信息
    if not hasattr(solver, '_wall_sp_info') or solver._wall_sp_info is None:
        # 尝试提取
        solver._wall_sp_info = extract_wall_boundary_info(solver)
    
    if solver._wall_sp_info is None or len(solver._wall_sp_info['cell_idx']) == 0:
        logger.warning("Wall boundary info not available or empty, skipping WMLES wall stress")
        return
    
    if solver.wmles_model is None:
        logger.warning("WMLES model not initialized, skipping wall stress")
        return

    wall_sp_info = solver._wall_sp_info
    cell_idx = wall_sp_info['cell_idx']
    sp_idx = wall_sp_info['sp_idx']
    wall_normals = wall_sp_info['normals']
    
    # 获取壁面SPs处的流场变量
    # U 形状: (n_cells, n_sps, n_vars)
    wall_U = solver.state.U[cell_idx, sp_idx]  # (n_wall_sps, n_vars)
    
    rho = wall_U[:, 0]  # 密度
    u_vel = wall_U[:, 1:4]  # 速度向量 (n_wall_sps, 3)
    
    # 计算切向速度（减去法向分量）
    if wall_normals is not None and len(wall_normals) > 0:
        # 法向速度分量
        u_normal_mag = np.sum(u_vel * wall_normals, axis=1, keepdims=True)  # (n_wall_sps, 1)
        u_normal = u_normal_mag * wall_normals  # (n_wall_sps, 3)
        u_tangent = u_vel - u_normal  # 切向速度
    else:
        # 如果没有法向量，假设整个速度都是切向的（简化）
        logger.debug("Wall normals not available, using full velocity as tangential")
        u_tangent = u_vel.copy()
    
    # 获取壁面距离
    if solver.wall_distance is not None:
        wall_distances = solver.wall_distance[cell_idx, sp_idx]
    else:
        logger.error("Wall distance field not computed, cannot apply WMLES")
        return
    
    # 计算壁面剪应力
    tau_w = solver.wmles_model.compute_wall_shear_stress(
        u_tangent=u_tangent,
        y_dist=wall_distances,
        rho=rho,
        method='iterative'
    )
    
    # 输出y+统计信息
    try:
        is_valid, stats = solver.wmles_model.validate_y_plus_range(min_y_plus=30, max_y_plus=300)
        logger.info(f"WMLES y+ statistics: min={stats['min']:.1f}, max={stats['max']:.1f}, "
                   f"mean={stats['mean']:.1f}, in_range={stats['n_in_range']}/{len(cell_idx)}")
        
        if not is_valid:
            logger.warning(f"WMLES y+ out of recommended range [30, 300]: "
                         f"{stats['n_below_min']} below 30, {stats['n_above_max']} above 300")
    except Exception as e:
        logger.debug(f"y+ validation skipped: {e}")
    
    # 将壁面剪应力应用到动量方程残差
    # 残差更新：Res_u -= τ_w / (ρ * Δy)
    delta_y = np.maximum(wall_distances, 1e-6)
    
    # 计算动量源项：S_momentum = -τ_w / delta_y
    momentum_source = -tau_w / delta_y[:, np.newaxis]  # (n_wall_sps, 3)
    
    # 将源项添加到残差中
    for i, (c_idx, s_idx) in enumerate(zip(cell_idx, sp_idx)):
        # 动量方程对应索引1,2,3 (u, v, w)
        solver.residual[c_idx, s_idx, 1] += momentum_source[i, 0]
        solver.residual[c_idx, s_idx, 2] += momentum_source[i, 1]
        solver.residual[c_idx, s_idx, 3] += momentum_source[i, 2]
    
    logger.debug(f"WMLES wall stress applied to {len(cell_idx)} wall SPs")
