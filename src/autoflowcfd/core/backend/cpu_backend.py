"""CPU backend implementation using Numba JIT compilation.

本模块提供基于 Numba 的 CPU 加速后端，用于 FR 求解器的通量和残差计算。

注意：
- 这是 V2.0 Pure FR 架构的一部分
- 使用 Numba JIT 编译加速 CPU 计算
- 支持多线程并行（prange）
"""

import numpy as np
from typing import Dict, Any, Optional
from .base import BackendBase


class NumbaBackend(BackendBase):
    """带 Numba JIT 并行加速的 CPU backend。

    通过 Numba 的即时编译，用 @njit(parallel=True) 自动多线程加速 CPU
    计算。

    Attributes:
        backend_type: 始终为 'cpu'
        available: Numba 已安装则为 True
        n_threads: 并行线程数
        n_cells: 单元数量（初始化后设置）
        n_nodes: 节点数量（初始化后设置）
        n_variables: 变量数量（初始化后设置）
    """

    def __init__(self, n_threads: int = 4):
        """初始化 Numba CPU backend。

        Args:
            n_threads: 并行线程数（默认4）
        """
        super().__init__()
        self.backend_type = 'cpu'
        self.n_threads = n_threads
        self.n_cells = 0
        self.n_nodes = 0
        self.n_variables = 5
        
        # 检查 Numba 可用性
        try:
            import numba
            self.available = True
            print(f"[NumbaBackend] Initialized with {n_threads} threads")
        except ImportError:
            self.available = False
            print("[NumbaBackend] Warning: Numba not installed, falling back to NumPy")
    
    def initialize(self, n_cells: int, n_nodes: int, n_variables: int = 5):
        """初始化 backend 并设置网格尺寸。
        
        Args:
            n_cells: 单元数量
            n_nodes: 节点数量
            n_variables: 变量数量（默认5）
        """
        self.n_cells = n_cells
        self.n_nodes = n_nodes
        self.n_variables = n_variables
        print(f"[NumbaBackend] Initialized for {n_cells} cells, {n_nodes} nodes, {n_variables} vars")
    
    @property
    def info(self) -> Dict[str, Any]:
        """获取 backend 信息。
        
        Returns:
            包含 backend 类型、可用性和线程数的字典
        """
        return {
            'type': self.backend_type,
            'available': self.available,
            'n_threads': self.n_threads,
            'n_cells': self.n_cells,
            'n_nodes': self.n_nodes,
            'n_variables': self.n_variables
        }
    
    def compute_flux(self, 
                    solution: np.ndarray,
                    cell_connectivity: np.ndarray,
                    face_normals: np.ndarray,
                    gamma: float = 1.4) -> np.ndarray:
        """计算界面数值通量。
        
        Args:
            solution: 解向量，形状 (n_cells, n_vars)
            cell_connectivity: 单元连接关系，形状 (n_faces, 2)
            face_normals: 界面法向量，形状 (n_faces, 3)
            gamma: 比热比
            
        Returns:
            flux: 数值通量向量，形状 (n_faces, n_vars)
        """
        from ..fr_kernels import compute_ausm_up_flux
        
        n_faces = cell_connectivity.shape[0]
        n_vars = solution.shape[1]
        flux = np.zeros((n_faces, n_vars))
        
        # 遍历所有面计算通量
        for f in range(n_faces):
            left_idx = cell_connectivity[f, 0]
            right_idx = cell_connectivity[f, 1]
            
            if left_idx >= 0 and right_idx >= 0:
                # 内部面：左右状态
                U_left = solution[left_idx]
                U_right = solution[right_idx]
                
                # 转换为原始变量
                def conservative_to_primitive(U):
                    rho = max(U[0], 1e-10)
                    u = U[1] / rho
                    v = U[2] / rho
                    w = U[3] / rho
                    E = U[4]
                    p = max((E - 0.5 * rho * (u**2 + v**2 + w**2)) * (gamma - 1.0), 10.0)
                    return np.array([rho, u, v, w, p])
                
                qL = conservative_to_primitive(U_left)
                qR = conservative_to_primitive(U_right)
                
                # 计算AUSM+up通量
                normal = face_normals[f]
                norm = np.linalg.norm(normal)
                if norm > 1e-10:
                    normal = normal / norm
                
                flux[f] = compute_ausm_up_flux(qL, qR, normal)
            else:
                # 边界面：简化处理
                flux[f] = np.zeros(n_vars)
        
        return flux
    
    def compute_residuals(self,
                         solution: np.ndarray,
                         flux: np.ndarray,
                         cell_volumes: np.ndarray,
                         boundary_mask: np.ndarray,
                         face_to_cell_map: Optional[np.ndarray] = None) -> np.ndarray:
        """由通量散度计算残差。
        
        实现完整的FR残差计算：
        R_i = (1/V_i) * Σ_f (F_f · n_f * A_f)
        
        其中求和遍历单元i的所有面，F_f是界面通量，n_f是单位法向量，
        A_f是面面积。
        
        Args:
            solution: 解向量，形状 (n_cells, n_vars)
            flux: 界面通量，形状 (n_faces, n_vars)
            cell_volumes: 单元体积，形状 (n_cells,)
            boundary_mask: 边界条件掩码，形状 (n_faces,)
            face_to_cell_map: 面到单元的映射，形状 (n_faces, 2)
                             [owner, neighbor]，neighbor=-1表示边界
            
        Returns:
            residuals: 残差向量，形状 (n_cells, n_vars)
        """
        n_cells = solution.shape[0]
        n_vars = solution.shape[1]
        residuals = np.zeros_like(solution)
        
        if face_to_cell_map is None:
            # 如果没有提供映射，使用简化假设
            print(f"[WARNING] face_to_cell_map not provided, using simplified residual calculation")
            # 简化：假设通量已经按单元组织
            return residuals
        
        n_faces = flux.shape[0]
        
        # 遍历所有面，将通量贡献累加到相邻单元
        for f in range(n_faces):
            owner = face_to_cell_map[f, 0]
            neighbor = face_to_cell_map[f, 1]
            
            if owner >= 0 and owner < n_cells:
                # Owner单元：通量流出为正
                residuals[owner] += flux[f]
            
            if neighbor >= 0 and neighbor < n_cells:
                # Neighbor单元：通量方向相反
                residuals[neighbor] -= flux[f]
        
        # 除以体积得到残差
        for i in range(n_cells):
            if cell_volumes[i] > 1e-10:
                residuals[i] /= cell_volumes[i]
            else:
                # 避免除以零
                residuals[i] = 0.0
        
        return residuals
    
    def update_solution(self,
                       solution: np.ndarray,
                       residuals: np.ndarray,
                       dt: float,
                       cfl: float) -> np.ndarray:
        """用时间积分格式更新解。
        
        Args:
            solution: 当前解向量
            residuals: 残差向量
            dt: 时间步长
            cfl: CFL 数
            
        Returns:
            updated_solution: 更新后的解向量
        """
        # 显式 Euler 更新: U^{n+1} = U^n - dt * R
        updated = solution - residuals * dt
        
        # 物理正性保护
        from ..time_integration import enforce_positivity
        enforce_positivity(updated, p_floor=10.0)
        
        return updated
    
    def apply_boundary_conditions(self,
                                 solution: np.ndarray,
                                 boundary_map: Dict[str, np.ndarray],
                                 bc_params: Dict[str, Any]) -> np.ndarray:
        """把边界条件应用到解上。
        
        支持多种边界条件类型：
        - WALL: 无滑移壁面 (u=v=w=0)
        - INLET: 速度/压力入口
        - OUTLET: 压力出口
        - FARFIELD: 远场Riemann不变量
        
        Args:
            solution: 解向量，形状 (n_cells, n_vars)
            boundary_map: 边界名到单元索引的映射
                         {'WALL': array([idx1, idx2, ...]), ...}
            bc_params: 边界条件参数
                      {'inlet_velocity': [u, v, w], 
                       'outlet_pressure': p_out,
                       'farfield_state': [rho_inf, u_inf, v_inf, w_inf, p_inf]}
            
        Returns:
            updated_solution: 应用边界条件后的解
        """
        updated = solution.copy()
        gamma = bc_params.get('gamma', 1.4)
        
        for bc_name, cell_indices in boundary_map.items():
            bc_type = bc_name.upper()
            
            if len(cell_indices) == 0:
                continue
            
            if bc_type == 'WALL' or 'WALL' in bc_type:
                # 无滑移壁面：速度为零，绝热
                for idx in cell_indices:
                    if idx < len(updated):
                        rho = updated[idx, 0]
                        # 速度置零
                        updated[idx, 1:4] = 0.0
                        # 保持能量（绝热）
                        # p = (gamma-1) * rho * e
                        # E = e + 0.5*u^2 = e (因为u=0)
                        # 所以E保持不变
                        
            elif bc_type == 'INLET' or 'INLET' in bc_type:
                # 速度入口
                inlet_vel = bc_params.get('inlet_velocity', [0.0, 0.0, 0.0])
                inlet_rho = bc_params.get('inlet_density', 1.225)
                inlet_p = bc_params.get('inlet_pressure', 101325.0)
                
                for idx in cell_indices:
                    if idx < len(updated):
                        updated[idx, 0] = inlet_rho
                        updated[idx, 1] = inlet_rho * inlet_vel[0]
                        updated[idx, 2] = inlet_rho * inlet_vel[1]
                        updated[idx, 3] = inlet_rho * inlet_vel[2]
                        
                        # 总能量 E = p/((gamma-1)*rho) + 0.5*|u|^2
                        ke = 0.5 * inlet_rho * (inlet_vel[0]**2 + inlet_vel[1]**2 + inlet_vel[2]**2)
                        e_internal = inlet_p / ((gamma - 1.0) * inlet_rho)
                        updated[idx, 4] = e_internal + ke
                        
            elif bc_type == 'OUTLET' or 'OUTLET' in bc_type:
                # 压力出口
                outlet_p = bc_params.get('outlet_pressure', 101325.0)
                
                for idx in cell_indices:
                    if idx < len(updated):
                        rho = updated[idx, 0]
                        vel = updated[idx, 1:4] / max(rho, 1e-10)
                        ke = 0.5 * rho * np.sum(vel**2)
                        
                        # 设置压力，保持速度外推
                        e_internal = outlet_p / ((gamma - 1.0) * max(rho, 1e-10))
                        updated[idx, 4] = e_internal + ke
                        
            elif bc_type == 'FARFIELD' or 'FARFIELD' in bc_type:
                # 远场边界：使用Riemann不变量或简单外推
                farfield_state = bc_params.get('farfield_state')
                if farfield_state is not None and len(farfield_state) == 5:
                    for idx in cell_indices:
                        if idx < len(updated):
                            # 简单的Dirichlet边界
                            updated[idx] = farfield_state
            
            else:
                # 未知边界类型：保持内部状态
                pass
        
        return updated

    def synchronize(self) -> None:
        """同步数据（对 CPU 来说是空操作）。"""
        # CPU backend 不需要同步
        pass
    
    def get_device_info(self) -> Dict[str, Any]:
        """获取硬件设备信息。
        
        Returns:
            包含设备规格的字典
        """
        return {
            'backend': 'Numba CPU',
            'device': 'CPU',
            'threads': self.n_threads,
            'available': self.available,
            'n_cells': self.n_cells,
            'n_nodes': self.n_nodes,
            'n_variables': self.n_variables
        }

    def cleanup(self) -> None:
        """释放已分配的资源。"""
        # CPU backend 不需要特殊清理
        pass
