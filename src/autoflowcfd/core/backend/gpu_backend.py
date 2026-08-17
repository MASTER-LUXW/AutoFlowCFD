"""基于 CUDA 加速的 GPU 后端实现。

本模块提供基于 CUDA 的 GPU 加速后端，用于 FR 求解器的通量和残差计算。

注意：
- 这是 V2.0 Pure FR 架构的一部分
- 使用 Numba CUDA 进行 GPU 加速
- 支持多 GPU 并行（未来扩展）
"""

import numpy as np
from typing import Dict, Any, Optional
from .base import BackendBase


class CUDABackend(BackendBase):
    """带 CUDA 加速的 GPU backend。

    通过 NVIDIA GPU 并行计算加速 FR 求解器的核心运算。

    Attributes:
        backend_type: 始终为 'gpu'
        available: CUDA 可用则为 True
        device_id: GPU 设备 ID
        n_cells: 单元数量（初始化后设置）
        n_nodes: 节点数量（初始化后设置）
        n_variables: 变量数量（初始化后设置）
    """

    def __init__(self, device_id: int = 0):
        """初始化 CUDA GPU backend。

        Args:
            device_id: GPU 设备 ID（默认0）
        """
        super().__init__()
        self.backend_type = 'gpu'
        self.device_id = device_id
        self.n_cells = 0
        self.n_nodes = 0
        self.n_variables = 5
        
        # CUDA相关资源
        self.stream = None
        self.device_arrays = {}
        
        # 检查 CUDA 可用性
        try:
            from numba import cuda
            if cuda.is_available():
                self.available = True
                self.cuda = cuda
                print(f"[CUDABackend] Initialized on GPU device {device_id}")
                
                # 显示GPU信息
                gpu = cuda.get_current_device()
                print(f"[CUDABackend] GPU: {gpu.name}")
                print(f"[CUDABackend] Compute Capability: {gpu.compute_capability}")
            else:
                self.available = False
                print("[CUDABackend] Warning: CUDA not available")
        except ImportError:
            self.available = False
            print("[CUDABackend] Warning: Numba/CUDA not installed")
    
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
        
        # 创建CUDA stream用于异步操作
        if self.available:
            self.stream = self.cuda.stream()
            print(f"[CUDABackend] Initialized for {n_cells} cells, {n_nodes} nodes, {n_variables} vars")
            print(f"[CUDABackend] Using CUDA stream for async operations")
    
    def compute_flux(self, 
                    solution: np.ndarray,
                    cell_connectivity: np.ndarray,
                    face_normals: np.ndarray,
                    gamma: float = 1.4) -> np.ndarray:
        """计算界面数值通量（GPU 版本）。
        
        Args:
            solution: 解向量 (n_cells, n_vars)
            cell_connectivity: 单元连接关系 (n_faces, 2)
            face_normals: 界面法向量 (n_faces, 3)
            gamma: 比热比
            
        Returns:
            flux: 数值通量向量 (n_faces, n_vars)
        """
        if not self.available:
            print("[CUDABackend] Warning: GPU not available, falling back to CPU")
            return self._compute_flux_cpu(solution, cell_connectivity, face_normals, gamma)
        
        n_faces = cell_connectivity.shape[0]
        n_vars = solution.shape[1]
        
        print(f"[CUDABackend] Computing flux on GPU for {n_faces} faces")
        
        # 传输数据到GPU
        d_solution = self.cuda.to_device(solution, stream=self.stream)
        d_connectivity = self.cuda.to_device(cell_connectivity, stream=self.stream)
        d_normals = self.cuda.to_device(face_normals, stream=self.stream)
        d_flux = self.cuda.device_array((n_faces, n_vars), dtype=np.float64, stream=self.stream)
        
        # 配置线程块
        threads_per_block = 256
        blocks_per_grid = (n_faces + threads_per_block - 1) // threads_per_block
        
        # 启动GPU内核
        self._cuda_flux_kernel[blocks_per_grid, threads_per_block, self.stream](
            d_solution, d_connectivity, d_normals, d_flux, n_faces, n_vars, gamma
        )
        
        # 同步并取回结果
        self.stream.synchronize()
        flux = d_flux.copy_to_host()
        
        return flux
    
    @staticmethod
    def _cuda_flux_kernel(d_solution, d_connectivity, d_normals, d_flux, 
                          n_faces, n_vars, gamma):
        """CUDA通量计算内核。
        
        每个线程处理一个界面。
        """
        from numba import cuda
        
        # 计算全局线程ID
        face_idx = cuda.grid(1)
        
        if face_idx >= n_faces:
            return
        
        # 获取左右单元索引
        left_cell = d_connectivity[face_idx, 0]
        right_cell = d_connectivity[face_idx, 1]
        
        # 获取法向量
        nx = d_normals[face_idx, 0]
        ny = d_normals[face_idx, 1]
        nz = d_normals[face_idx, 2]
        
        # AUSM+up通量计算（简化版）
        for v in range(n_vars):
            # 简单平均作为示例
            U_L = d_solution[left_cell, v]
            U_R = d_solution[right_cell, v]
            
            # 中心差分通量
            d_flux[face_idx, v] = 0.5 * (U_L + U_R)
    
    def _compute_flux_cpu(self, solution, cell_connectivity, face_normals, gamma):
        """CPU备用通量计算。"""
        n_faces = cell_connectivity.shape[0]
        n_vars = solution.shape[1]
        flux = np.zeros((n_faces, n_vars))
        
        for f in range(n_faces):
            left = cell_connectivity[f, 0]
            right = cell_connectivity[f, 1]
            flux[f] = 0.5 * (solution[left] + solution[right])
        
        return flux
    
    def compute_residuals(self,
                         solution: np.ndarray,
                         flux: np.ndarray,
                         cell_volumes: np.ndarray,
                         boundary_mask: np.ndarray) -> np.ndarray:
        """由通量散度计算残差（GPU 版本）。
        
        Args:
            solution: 解向量 (n_cells, n_vars)
            flux: 界面通量 (n_faces, n_vars)
            cell_volumes: 单元体积 (n_cells,)
            boundary_mask: 边界掩码 (n_cells,)
            
        Returns:
            residuals: 残差向量 (n_cells, n_vars)
        """
        if not self.available:
            print("[CUDABackend] Warning: GPU not available, falling back to CPU")
            return self._compute_residuals_cpu(solution, flux, cell_volumes, boundary_mask)
        
        n_cells = solution.shape[0]
        n_vars = solution.shape[1]
        
        print(f"[CUDABackend] Computing residuals on GPU for {n_cells} cells")
        
        # 传输数据到GPU
        d_solution = self.cuda.to_device(solution, stream=self.stream)
        d_flux = self.cuda.to_device(flux, stream=self.stream)
        d_volumes = self.cuda.to_device(cell_volumes, stream=self.stream)
        d_boundary = self.cuda.to_device(boundary_mask, stream=self.stream)
        d_residuals = self.cuda.device_array((n_cells, n_vars), dtype=np.float64, stream=self.stream)
        
        # 配置线程块
        threads_per_block = 256
        blocks_per_grid = (n_cells + threads_per_block - 1) // threads_per_block
        
        # 启动GPU内核
        self._cuda_residual_kernel[blocks_per_grid, threads_per_block, self.stream](
            d_solution, d_flux, d_volumes, d_boundary, d_residuals, n_cells, n_vars
        )
        
        # 同步并取回结果
        self.stream.synchronize()
        residuals = d_residuals.copy_to_host()
        
        return residuals
    
    @staticmethod
    def _cuda_residual_kernel(d_solution, d_flux, d_volumes, d_boundary, 
                              d_residuals, n_cells, n_vars):
        """CUDA残差计算内核。
        
        每个线程处理一个单元。
        """
        from numba import cuda
        
        cell_idx = cuda.grid(1)
        
        if cell_idx >= n_cells:
            return
        
        vol = d_volumes[cell_idx]
        is_boundary = d_boundary[cell_idx]
        
        # 简化的残差计算（实际应聚合相邻界面的通量）
        for v in range(n_vars):
            # 这里需要知道单元的界面连接关系
            # 简化：假设flux已经按单元聚合
            d_residuals[cell_idx, v] = 0.0
    
    def _compute_residuals_cpu(self, solution, flux, cell_volumes, boundary_mask):
        """CPU备用残差计算。"""
        n_cells = solution.shape[0]
        n_vars = solution.shape[1]
        residuals = np.zeros((n_cells, n_vars))
        
        # 简化：均匀分配通量
        avg_flux = np.mean(flux, axis=0)
        for c in range(n_cells):
            residuals[c] = -avg_flux / cell_volumes[c]
        
        return residuals
    
    def update_solution(self,
                       solution: np.ndarray,
                       residuals: np.ndarray,
                       dt: float,
                       cfl: float) -> np.ndarray:
        """用时间积分格式更新解（GPU 版本）。

        Args:
            solution: 当前解向量
            residuals: 残差向量
            dt: 时间步长
            cfl: CFL 数

        Returns:
            updated_solution: 更新后的解向量
        """
        if not self.available:
            print("[CUDABackend] Warning: GPU not available, falling back to CPU")
            return self._update_solution_cpu(solution, residuals, dt)

        n_cells, n_vars = solution.shape
        print(f"[CUDABackend] Updating solution on GPU for {n_cells} cells")

        d_solution = self.cuda.to_device(solution, stream=self.stream)
        d_residuals = self.cuda.to_device(residuals, stream=self.stream)
        d_updated = self.cuda.device_array((n_cells, n_vars), dtype=np.float64, stream=self.stream)

        threads_per_block = 256
        blocks_per_grid = (n_cells + threads_per_block - 1) // threads_per_block
        self._cuda_update_kernel[blocks_per_grid, threads_per_block, self.stream](
            d_solution, d_residuals, d_updated, dt, n_cells, n_vars
        )

        self.stream.synchronize()
        updated = d_updated.copy_to_host()

        # 物理正性保护（与 NumbaBackend.update_solution 同一套约束，见该
        # 方法文档：显式 Euler 更新后必须夹持密度/压力下限）。
        from ..time_integration import enforce_positivity
        enforce_positivity(updated, p_floor=10.0)

        return updated

    @staticmethod
    def _cuda_update_kernel(d_solution, d_residuals, d_updated, dt, n_cells, n_vars):
        """CUDA 显式 Euler 更新内核：U^{n+1} = U^n - dt * R，每个线程处理一个单元。"""
        from numba import cuda

        cell_idx = cuda.grid(1)
        if cell_idx >= n_cells:
            return

        for v in range(n_vars):
            d_updated[cell_idx, v] = d_solution[cell_idx, v] - dt * d_residuals[cell_idx, v]

    def _update_solution_cpu(self, solution, residuals, dt):
        """CPU 备用解更新（显式 Euler + 正性保护，与 NumbaBackend.update_solution 一致）。"""
        updated = solution - residuals * dt
        from ..time_integration import enforce_positivity
        enforce_positivity(updated, p_floor=10.0)
        return updated

    def apply_boundary_conditions(self,
                                 solution: np.ndarray,
                                 boundary_map: Dict[str, np.ndarray],
                                 bc_params: Dict[str, Any]) -> np.ndarray:
        """把边界条件应用到解上（CPU 实现：边界索引列表的散列写入不是有意义
        的 GPU 并行工作负载——每个边界组的单元数通常远小于总单元数，核启动
        开销会超过收益——与本文件 `compute_flux`/`compute_residuals` GPU
        不可用时的 CPU 回退是同一种权衡，公式与 NumbaBackend.
        apply_boundary_conditions 完全一致，理由/边界类型说明见该方法文档）。

        Args:
            solution: 解向量，形状 (n_cells, n_vars)
            boundary_map: 边界名到单元索引的映射
            bc_params: 边界条件参数

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
                for idx in cell_indices:
                    if idx < len(updated):
                        updated[idx, 1:4] = 0.0

            elif bc_type == 'INLET' or 'INLET' in bc_type:
                inlet_vel = bc_params.get('inlet_velocity', [0.0, 0.0, 0.0])
                inlet_rho = bc_params.get('inlet_density', 1.225)
                inlet_p = bc_params.get('inlet_pressure', 101325.0)

                for idx in cell_indices:
                    if idx < len(updated):
                        updated[idx, 0] = inlet_rho
                        updated[idx, 1] = inlet_rho * inlet_vel[0]
                        updated[idx, 2] = inlet_rho * inlet_vel[1]
                        updated[idx, 3] = inlet_rho * inlet_vel[2]
                        ke = 0.5 * inlet_rho * (inlet_vel[0]**2 + inlet_vel[1]**2 + inlet_vel[2]**2)
                        e_internal = inlet_p / ((gamma - 1.0) * inlet_rho)
                        updated[idx, 4] = e_internal + ke

            elif bc_type == 'OUTLET' or 'OUTLET' in bc_type:
                outlet_p = bc_params.get('outlet_pressure', 101325.0)

                for idx in cell_indices:
                    if idx < len(updated):
                        rho = updated[idx, 0]
                        vel = updated[idx, 1:4] / max(rho, 1e-10)
                        ke = 0.5 * rho * np.sum(vel**2)
                        e_internal = outlet_p / ((gamma - 1.0) * max(rho, 1e-10))
                        updated[idx, 4] = e_internal + ke

            elif bc_type == 'FARFIELD' or 'FARFIELD' in bc_type:
                farfield_state = bc_params.get('farfield_state')
                if farfield_state is not None and len(farfield_state) == 5:
                    for idx in cell_indices:
                        if idx < len(updated):
                            updated[idx] = farfield_state

            else:
                pass

        return updated

    def synchronize(self):
        """同步GPU操作。"""
        if self.available and self.stream:
            self.stream.synchronize()

    def get_device_info(self) -> Dict[str, Any]:
        """获取硬件设备信息。

        Returns:
            包含设备规格的字典
        """
        info = {
            'backend': 'CUDA GPU',
            'device': 'GPU',
            'available': self.available,
            'device_id': self.device_id,
            'n_cells': self.n_cells,
            'n_nodes': self.n_nodes,
            'n_variables': self.n_variables,
        }
        if self.available:
            gpu = self.cuda.get_current_device()
            info['name'] = gpu.name
            info['compute_capability'] = gpu.compute_capability
        return info

    def cleanup(self):
        """清理GPU资源。"""
        if self.available:
            self.device_arrays.clear()
            if self.stream:
                self.stream.synchronize()
            print("[CUDABackend] Resources cleaned up")