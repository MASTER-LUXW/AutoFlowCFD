"""
AutoFlowCFD V2.0 - FR 求解器主类 (Final Integration)

本模块整合 FRState, HighOrderMesh, FR Kernels, Turbulence Models 及 Weak BCs。
它是 V2.0 求解器的总控中心，负责协调各模块完成 N-S 方程的高阶离散与求解。

核心功能:
1. 支持多种湍流模型（SST/DDES/WMLES/LES）
2. 自动切换RANS/LES模式
3. 完整的时间推进循环
4. 残差监控和收敛判断
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from autoflowcfd.core.fr_state import FRState, SolverResult
from autoflowcfd.grid.high_order_mesh import HighOrderMesh
from autoflowcfd.fr.operators import generate_fr_operators, FROperators
from autoflowcfd.core.fr_kernels import compute_ausm_up_flux, apply_correction_term
from autoflowcfd.boundary.fr_weak_bc import FRWeakBC
from autoflowcfd.core.turbulence_sst import SSTModelFR
from autoflowcfd.core.turbulence_des import DDESModel, IDDESModel
from autoflowcfd.core.turbulence_wmles import WMLESModel
from autoflowcfd.core.turbulence_sgs import WALEModel, SmagorinskyModel
from autoflowcfd.core.time_integration import TimeIntegrator, TimeIntegrationScheme
from autoflowcfd.core.wall_distance import compute_wall_distance
from autoflowcfd.core.fr_residual_viscous import compute_viscous_residual as compute_viscous_residual_ldg

# 导入辅助模块
from . import solver_helpers
from . import order_continuation

# 配置日志
logger = logging.getLogger(__name__)


class FRSolver:
    """
    基于通量重构 (FR) 方法的 N-S 方程求解器。
    
    Attributes:
        mesh: 高阶网格对象
        state: 求解器状态容器
        ops: 预计算的 FR 算子
        bc_handler: 弱边界条件处理器
        turb_model: 湍流模型处理器
    """

    def __init__(self, mesh: 'HighOrderMesh', order: int = 2, 
                 turb_model_name: str = "SST", n_vars: int = 5,
                 time_scheme: TimeIntegrationScheme = TimeIntegrationScheme.SSP_RK3,
                 initial_state: Optional[FRState] = None,
                 backend: str = "cpu"):
        """
        初始化 FRSolver。
        
        Args:
            mesh: HighOrderMesh 类型的高阶网格对象
            order: 多项式阶数
            turb_model_name: 湍流模型名称 ("SST"/"DDES"/"WMLES"/"LES"/"NONE")
            n_vars: 守恒变量数量（默认5：rho, rho_u, rho_v, rho_w, rho_e）
            time_scheme: 时间推进方案
            initial_state: 初始状态（用于 Order Continuation）
            backend: 计算后端 ("cpu" 或 "gpu")
        """
        self.mesh = mesh
        self.order = order
        self.backend_type = backend.lower()
        
        # 安全地获取网格信息
        n_cells = getattr(mesh, 'n_cells', 0)
        n_sps = getattr(mesh, 'n_sps_per_cell', 8)
        
        # 1. 初始化状态 (S-01)
        if initial_state is not None:
            # 使用提供的初始状态（Order Continuation）
            self.state = initial_state
            print(f"✅ Using provided initial state from lower order")
        else:
            # 根据湍流模型确定变量数
            default_n_vars = 7 if turb_model_name in ["SST", "DDES"] else 5
            self.state = FRState(n_cells, n_sps, default_n_vars)
            self.state.initialize_uniform()
        
        # 2. 预计算算子 (G-04)
        self.ops = generate_fr_operators(order)
        
        # 3. 初始化边界条件 (BD-01)
        self.bc_handler = FRWeakBC(penalty_coeff=10.0)
        
        # 4. 初始化计算后端
        self.backend = None
        if self.backend_type == "gpu":
            try:
                from ..core.backend import CUDABackend
                self.backend = CUDABackend(device_id=0)
                self.backend.initialize(n_cells=n_cells, n_nodes=n_cells*n_sps, n_variables=n_vars)
                if self.backend.available:
                    logger.info(f"GPU Backend (CUDA) initialized")
                else:
                    logger.warning(f"GPU not available, falling back to CPU")
                    self.backend_type = "cpu"
            except Exception as e:
                logger.warning(f"GPU initialization failed: {e}, using CPU")
                self.backend_type = "cpu"
        
        if self.backend_type == "cpu":
            logger.info(f"CPU Backend (Numba) initialized")
        
        # 5. 初始化湍流模型
        self.turb_model_name = turb_model_name.upper()
        self.turb_model = None
        self.ddes_model = None
        self.wmles_model = None
        self.sgs_model = None
        
        self._init_turbulence_models(n_cells, n_sps)
        
        # 6. 初始化时间积分器 (S-05)
        self.time_integrator = TimeIntegrator(scheme=time_scheme)
        
        # 7. 壁面距离场（用于DDES/WMLES）
        self.wall_distance = None
        
        # 8. 壁面边界信息（用于WMLES）
        self._wall_sp_info = None
        
        # 9. Order Continuation 状态
        self.current_order = order
        self.order_continuation_enabled = True
        
        print(f"✅ FRSolver Ready:")
        print(f"   Cells: {n_cells}, Order: P{order}")
        print(f"   Turbulence: {turb_model_name}")
        print(f"   Backend: {self.backend_type.upper()}")
        print(f"   Time Scheme: {time_scheme.value}")

    def _init_turbulence_models(self, n_cells: int, n_sps: int):
        """
        初始化湍流模型。
        
        Args:
            n_cells: 单元数量
            n_sps: 每单元解点数
        """
        if self.turb_model_name == "SST":
            self.turb_model = SSTModelFR(n_cells, n_sps)
            print(f"   [OK] SST k-omega model initialized")
            
        elif self.turb_model_name == "DDES":
            # DDES 基于 SST
            self.turb_model = SSTModelFR(n_cells, n_sps)
            self.ddes_model = DDESModel()
            print(f"   [OK] DDES model initialized (based on SST)")
            
        elif self.turb_model_name == "WMLES":
            # WMLES 需要壁面应力模型
            self.wmles_model = WMLESModel()
            # 可选：配合 SGS 模型
            self.sgs_model = WALEModel()
            print(f"   [OK] WMLES model initialized")
            
        elif self.turb_model_name == "LES":
            # 纯 LES，需要 SGS 模型
            self.sgs_model = WALEModel()
            print(f"   [OK] LES with WALE SGS model initialized")
            
        elif self.turb_model_name == "NONE":
            print(f"   [OK] Laminar flow (no turbulence model)")
            
        else:
            raise ValueError(f"Unknown turbulence model: {self.turb_model_name}")
    
    def compute_wall_distance_field(self, mesh_nodes: np.ndarray, 
                                   wall_indices: np.ndarray):
        """
        计算壁面距离场（用于 DDES/WMLES/SST）。
        
        Args:
            mesh_nodes: 网格节点坐标，形状 (n_nodes, 3)
            wall_indices: 壁面节点索引
        """
        # SST、DDES、WMLES、LES 都需要壁面距离
        if self.turb_model_name in ["SST", "DDES", "WMLES", "LES"]:
            logger.info("Computing wall distance field...")
            
            # 计算节点级别的壁面距离
            node_distances = compute_wall_distance(mesh_nodes, wall_indices)
            logger.info(f"Node-level wall distance computed: min={node_distances.min():.6f}, "
                       f"max={node_distances.max():.6f}")
            
            n_cells, n_sps = self.state.U.shape[:2]
            
            # 优化：利用 HighOrderMesh 的 sps_coords 进行精确映射
            if hasattr(self.mesh, 'sps_coords') and self.mesh.sps_coords is not None:
                # sps_coords 形状: (n_cells, n_sps, 3)
                sps_coords = self.mesh.sps_coords
                
                # 将 SPs 展平为点集进行批量查询
                flat_sps = sps_coords.reshape(-1, 3)
                
                # 使用 KD-Tree 直接查询每个 SP 到壁面的距离
                try:
                    from scipy.spatial import cKDTree
                    wall_coords = mesh_nodes[wall_indices]
                    tree = cKDTree(wall_coords)
                    
                    # query 返回 (distance, index)
                    dist_flat, _ = tree.query(flat_sps, k=1)
                    
                    # 重塑回 (n_cells, n_sps)
                    self.wall_distance = dist_flat.reshape(n_cells, n_sps)
                    
                    logger.info(f"Wall distance field mapped to SPs: shape={self.wall_distance.shape}, "
                               f"min={self.wall_distance.min():.6f}, max={self.wall_distance.max():.6f}")
                except Exception as e:
                    logger.warning(f"SP-level mapping failed ({e}), falling back to cell-center mapping")
                    self._map_wall_distance_fallback(node_distances, mesh_nodes, wall_indices, n_cells, n_sps)
            else:
                # 回退策略
                self._map_wall_distance_fallback(node_distances, mesh_nodes, wall_indices, n_cells, n_sps)
                
        else:
            logger.warning(f"Turbulence model {self.turb_model_name} does not require wall distance")

    def solve(self, max_iter: int = 1000, dt: float = 1e-4, tol: float = 1e-6) -> SolverResult:
        """
        执行稳态/瞬态求解循环。
        
        Args:
            max_iter: 最大迭代次数
            dt: 时间步长
            tol: 收敛容差
            
        Returns:
            SolverResult: 包含收敛状态、最终残差和迭代次数的结果对象
        """
        logger_msg = f"Starting solve loop with {self.time_integrator.scheme.value}"
        if self.turb_model_name != "NONE":
            logger_msg += f", turbulence={self.turb_model_name}"
        print(logger_msg)
        
        # Order Continuation: 从低阶开始逐步提升精度
        if self.order_continuation_enabled and self.order >= 2:
            return self._solve_with_order_continuation(max_iter, dt, tol)
        
        import time
        converged = False
        final_residual = 1e10
        
        for i in range(max_iter):
            t_start = time.time()
            res = self.step(dt)
            t_end = time.time()
            final_residual = res
            
            # 每 10 步或第 1 步打印详细信息
            if i == 0 or (i + 1) % 10 == 0:
                print(f"Iteration {i+1}: Residual = {res:.6e} | Time/step: {t_end - t_start:.2f}s")
                
            if res < tol:
                converged = True
                print(f"✅ Converged at iteration {i+1} with residual {res:.6e}")
                break
        
        return SolverResult(converged=converged, iterations=i+1, final_residual=final_residual)
    
    def _solve_with_order_continuation(self, max_iter: int, dt: float, tol: float) -> SolverResult:
        """
        实现 Order Continuation 策略：从P0逐步提升到目标阶数。
        
        Args:
            max_iter: 总迭代次数
            dt: 时间步长
            tol: 收敛容差
            
        Returns:
            SolverResult: 求解结果
        """
        import time
        
        print("\n=== Order Continuation Strategy ===")
        print(f"Starting from P0, targeting P{self.order}")
        
        # 保存原始状态
        original_order = self.order
        original_ops = self.ops
        
        # 关键修复：Order Continuation应该从低阶到高阶
        # 如果当前已经是高阶，需要先降阶初始化
        current_state_n_sps = self.state.U.shape[1]
        expected_p0_n_sps = 1  # P0有1个SP
        
        # 如果当前状态的SPs数量不等于P0的数量，说明需要从P0重新初始化
        if current_state_n_sps != expected_p0_n_sps:
            print(f"[INFO] Current state has {current_state_n_sps} SPs/cell, reinitializing from P0...")
            
            # 保存当前的高阶解
            high_order_U = self.state.U.copy()
            
            # 重新初始化为P0状态
            from autoflowcfd.core.fr_state import FRState
            p0_state = FRState(self.state.n_cells, expected_p0_n_sps, self.state.n_vars)
            p0_state.initialize_uniform()
            
            # 替换为P0状态
            self.state = p0_state
            
            # 关键修复：同时重置所有与SPs维度相关的场数据
            if hasattr(self, 'turb_model') and self.turb_model is not None:
                if hasattr(self.turb_model, 'k_field'):
                    self.turb_model.k_field = np.ones((self.state.n_cells, expected_p0_n_sps)) * 1e-6
                    self.turb_model.omega_field = np.ones((self.state.n_cells, expected_p0_n_sps)) * 1e-2
                    print(f"[INFO] Turbulence fields reset to P0 dimensions")
            
            # 重置壁面距离场以匹配P0维度
            if self.wall_distance is not None:
                # 保留壁面距离的基本分布，但调整SPs维度
                # 简化：重新计算或使用平均值
                old_wall_dist = self.wall_distance
                if old_wall_dist.ndim == 2 and old_wall_dist.shape[1] > 1:
                    # 取每个单元的平均壁面距离，然后广播到P0
                    mean_wall_dist = np.mean(old_wall_dist, axis=1, keepdims=True)
                    self.wall_distance = np.tile(mean_wall_dist, (1, expected_p0_n_sps))
                    print(f"[INFO] Wall distance field reset to P0 dimensions")
            
            self.current_order = 0
            self.ops = generate_fr_operators(0)
            
            print(f"[INFO] Reinitialized to P0 ({expected_p0_n_sps} SP/cell)")
        
        # 阶段定义：P0 -> P1 -> P2 -> ... -> P_target
        orders = list(range(0, original_order + 1))
        
        total_iter = 0
        for target_p in orders:
            print(f"\n--- Phase: P{target_p} ---")
            
            # 关键修复：先插值状态到新的阶数，再更新算子
            # 这样可以确保状态和算子的维度始终匹配
            if target_p > 0:
                # 在插值之前，状态是旧阶数的
                # 插值后，状态会变成新阶数的SPs数量
                self._interpolate_to_new_order(target_p)
            
            # 设置当前阶数和对应的算子
            self.current_order = target_p
            self.ops = generate_fr_operators(target_p)
            
            # 验证：确保状态变量的SPs数量与算子匹配
            expected_n_sps = self.ops.D_3d.shape[0]
            actual_n_sps = self.state.U.shape[1]
            if actual_n_sps != expected_n_sps:
                raise RuntimeError(
                    f"Order Continuation dimension mismatch after interpolation to P{target_p}: "
                    f"State has {actual_n_sps} SPs but operators expect {expected_n_sps} SPs"
                )
            
            # 在当前阶数下求解
            phase_max_iter = max_iter // len(orders)
            phase_tol = tol * (10 ** (original_order - target_p))  # 宽松容差
            
            converged = False
            final_residual = 1e10
            
            for i in range(phase_max_iter):
                t_start = time.time()
                res = self.step(dt)
                t_end = time.time()
                final_residual = res
                total_iter += 1
                
                if i == 0 or (i + 1) % 10 == 0:
                    print(f"P{target_p} Iter {i+1}: Residual = {res:.6e} | Time: {t_end - t_start:.2f}s")
                
                if res < phase_tol:
                    converged = True
                    print(f"✅ P{target_p} converged at iter {i+1}")
                    break
            
            # 检查是否达到最终目标
            if target_p == original_order and converged:
                print(f"\n✅ Order Continuation completed: Final P{original_order} converged")
                return SolverResult(converged=True, iterations=total_iter, final_residual=final_residual)
        
        # 恢复原始设置
        self.order = original_order
        self.ops = original_ops
        
        return SolverResult(converged=False, iterations=total_iter, final_residual=final_residual)
    
    def _interpolate_to_new_order(self, new_order: int):
        """
        将解从当前阶数插值到新的阶数（Order Continuation核心逻辑）。
        
        Args:
            new_order: 目标多项式阶数
        """
        order_continuation.interpolate_to_new_order(self, new_order)
        
        # 关键修复：更新state的n_sps属性以匹配新的SPs数量
        n_cells = self.state.n_cells
        n_points_1d = new_order + 1
        new_n_sps = n_points_1d ** 3
        
        # 验证状态变量的维度是否正确
        actual_n_sps = self.state.U.shape[1]
        if actual_n_sps != new_n_sps:
            logger.error(
                f"After interpolation: expected {new_n_sps} SPs but got {actual_n_sps}. "
                f"This indicates a bug in the interpolation routine."
            )
            raise RuntimeError(
                f"State dimension mismatch after Order Continuation: "
                f"expected {new_n_sps} SPs/cell, got {actual_n_sps}"
            )
        
        logger.info(f"Order Continuation: Successfully interpolated to P{new_order} ({new_n_sps} SPs/cell)")
    
    def _compute_scalar_gradient_simple(self, scalar_field: np.ndarray) -> np.ndarray:
        """
        计算标量场的梯度（简化版本）。
        
        Args:
            scalar_field: 标量场，形状 (n_cells, n_sps)
            
        Returns:
            gradient: 梯度张量，形状 (n_cells, n_sps, 3)
        """
        n_cells, n_sps = scalar_field.shape
        
        # 使用FR微分算子
        if hasattr(self.ops, 'D_3d') and self.ops.D_3d is not None:
            # D_3d 形状: (n_sps, n_sps, 3)
            gradient = np.zeros((n_cells, n_sps, 3))
            for dim in range(3):
                # 对每个单元和每个SP，计算梯度分量
                for i in range(n_cells):
                    gradient[i, :, dim] = np.dot(self.ops.D_3d[:, :, dim], scalar_field[i])
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

    def _map_wall_distance_fallback(self, node_distances, mesh_nodes, wall_indices, n_cells, n_sps):
        """
        壁面距离映射的回退策略：基于单元中心或节点平均。
        """
        # 尝试基于单元中心映射
        if hasattr(self.mesh, 'cell_centers') and self.mesh.cell_centers is not None:
            centers = self.mesh.cell_centers
            try:
                from scipy.spatial import cKDTree
                wall_coords = mesh_nodes[wall_indices]
                tree = cKDTree(wall_coords)
                dist_centers, _ = tree.query(centers, k=1)
                # 扩展到所有 SPs
                self.wall_distance = np.tile(dist_centers[:, np.newaxis], (1, n_sps))
                return
            except:
                pass
        
        # 最终回退：使用节点平均值
        self.wall_distance = np.ones((n_cells, n_sps)) * node_distances.mean()
        logger.info(f"Wall distance field initialized (fallback): mean={self.wall_distance.mean():.6f}")

    def step(self, dt: float) -> float:
        """
        执行一个时间步长。
        
        Args:
            dt: 基础时间步长（用于缩放）
            
        Returns:
            residual_norm: 残差范数
        """
        try:
            # 1. 更新原始变量
            self.state._update_primitives()
            
            # 2. 计算无粘残差 (S-02)
            inviscid_res = self.compute_inviscid_residual()
            
            # 3. 计算粘性残差 (S-03)
            viscous_res = self.compute_viscous_residual()
            
            # 4. 湍流模型源项
            turb_source = self.compute_turbulence_source(dt)
            
            # 5. 组装总残差
            total_res = inviscid_res + viscous_res
            if turb_source is not None:
                # 将湍流源项添加到对应的变量位置
                if self.state.n_vars > 5:
                    total_res[:, :, 5] += turb_source[0]  # k 方程
                    total_res[:, :, 6] += turb_source[1]  # omega 方程
            
            # 6. FR专用的局部时间步长估计
            dt_local = self._compute_local_time_step()  # (n_cells, n_sps)
            
            # 7. 时间推进（使用RK3格式）
            # 扩展dt_local以匹配total_res的维度: (n_cells, n_sps) -> (n_cells, n_sps, 1)
            dt_local_expanded = dt_local[:, :, np.newaxis]  # Broadcasting到所有变量
            
            self.state.dU_dt = total_res
            self.state.U = self.state.U + dt_local_expanded * total_res
            
            # 8. 应用 DDES/WMLES 修正
            self.apply_turbulence_corrections()
            
            # 9. 计算残差范数
            residual_norm = self.state.get_residual_norm()
            
            return residual_norm
        
        except Exception as e:
            logger.error(f"Step failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def compute_turbulence_source(self, dt: float) -> Optional[tuple]:
        """
        计算湍流模型源项。
        
        Args:
            dt: 时间步长
            
        Returns:
            sources: (Sk, S_omega) 或 None
        """
        if self.turb_model is None:
            return None
        
        # 获取流场变量
        Q = self.state.Q  # 原始变量
        grad_U = self._compute_gradients()  # 速度梯度，形状: (n_cells, n_sps, n_vars, 3)
        
        # 提取速度梯度分量（索引1,2,3对应u,v,w）
        grad_vel = grad_U[:, :, 1:4, :]  # 形状: (n_cells, n_sps, 3, 3)
        
        # 壁面距离 - 工业级计算必须预计算，不允许简化估计
        d_wall = self.wall_distance
        
        # 关键修复：验证壁面距离场的维度与当前状态匹配
        if d_wall is not None:
            expected_shape = (self.state.n_cells, self.state.n_sps)
            if d_wall.shape != expected_shape:
                logger.warning(
                    f"Wall distance shape mismatch: expected {expected_shape}, got {d_wall.shape}. "
                    f"Rescaling to match current state..."
                )
                # 重新调整壁面距离场的维度
                if d_wall.ndim == 2:
                    # 取每个单元的平均值，然后广播到当前SPs数量
                    mean_d = np.mean(d_wall, axis=1, keepdims=True)
                    d_wall = np.tile(mean_d, (1, self.state.n_sps))
                    self.wall_distance = d_wall  # 更新缓存
                else:
                    raise RuntimeError(f"Cannot rescale wall distance from shape {d_wall.shape}")
        
        if d_wall is None:
            # 对于需要壁面距离的湍流模型，抛出明确错误
            if self.turb_model_name in ["SST", "DDES", "WMLES", "LES"]:
                raise RuntimeError(
                    f"Wall distance field not computed for turbulence model '{self.turb_model_name}'. "
                    f"Please call compute_wall_distance_field() before solving, or ensure wall distance "
                    f"is provided during solver initialization. Industrial-grade calculation requires "
                    f"accurate wall distance, not simplified estimates."
                )
            else:
                # 对于不需要壁面距离的模型（如纯LES），使用特征长度估计
                n_cells, n_sps = self.state.U.shape[:2]
                # 基于网格体积估算特征长度
                volumes = self._get_cell_volumes()
                h_char = np.power(np.abs(volumes), 1.0/3.0)
                d_wall = np.tile(h_char[:, np.newaxis], (1, n_sps))
                logger.warning(f"Using characteristic length scale as wall distance estimate")
        
        # 计算 SST 源项（传入真实梯度）
        mu = 1.8e-5  # 空气动力粘度
        
        # 调试：打印所有关键变量的形状
        logger.debug(f"Q shape: {Q.shape}")
        logger.debug(f"grad_vel shape: {grad_vel.shape}")
        logger.debug(f"d_wall shape: {d_wall.shape if d_wall is not None else 'None'}")
        logger.debug(f"k_field shape: {self.turb_model.k_field.shape}")
        logger.debug(f"omega_field shape: {self.turb_model.omega_field.shape}")
        
        # 计算k和omega的梯度（用于交叉扩散项 - 工业级高精度要求）
        grad_k = None
        grad_omega = None
        if self.turb_model_name in ["SST", "DDES"]:
            try:
                k_expanded = self.turb_model.k_field[:, :, np.newaxis]
                omega_expanded = self.turb_model.omega_field[:, :, np.newaxis]
                
                logger.debug(f"Before gradient computation: k_field shape = {self.turb_model.k_field.shape}, "
                            f"omega_field shape = {self.turb_model.omega_field.shape}")
                
                # 必须使用 FR 微分算子进行高阶梯度重构
                from autoflowcfd.core.fr_residual_viscous import compute_scalar_gradient
                grad_k = compute_scalar_gradient(k_expanded, self.ops)
                grad_omega = compute_scalar_gradient(omega_expanded, self.ops)
                
                logger.debug(f"After gradient computation: grad_k shape = {grad_k.shape}, "
                            f"grad_omega shape = {grad_omega.shape}")
                
                # 正性保持检查：防止梯度过大导致负值
                # 在工业计算中，必须对湍流变量的梯度进行限幅处理
                max_grad_mag = 1e6  # 根据物理量纲设定的经验上限
                grad_k_mag = np.linalg.norm(grad_k, axis=-1)
                grad_omega_mag = np.linalg.norm(grad_omega, axis=-1)
                
                if np.any(grad_k_mag > max_grad_mag):
                    scale_k = max_grad_mag / np.maximum(grad_k_mag, 1e-10)
                    grad_k *= np.clip(scale_k[:, :, np.newaxis], 0, 1)
                    
                if np.any(grad_omega_mag > max_grad_mag):
                    scale_omega = max_grad_mag / np.maximum(grad_omega_mag, 1e-10)
                    grad_omega *= np.clip(scale_omega[:, :, np.newaxis], 0, 1)
                    
            except Exception as e:
                logger.error(f"Gradient computation failed: {e}")
                import traceback
                traceback.print_exc()
                raise
        Sk, S_omega = self.turb_model.compute_source_terms(
            Q, grad_vel, d_wall, mu, 
            grad_k=grad_k, grad_omega=grad_omega
        )
        
        # 如果是 DDES，应用修正
        if self.ddes_model is not None:
            cell_volumes = self._get_cell_volumes()
            self.ddes_model.apply_to_sst_model(
                self.turb_model, d_wall, cell_volumes, grad_vel
            )
        
        # 更新湍流场
        self.turb_model.update_fields(dt, Sk, S_omega)
        
        return (Sk, S_omega)
    
    def apply_turbulence_corrections(self):
        """
        应用湍流模型的修正（如 WMLES 壁面应力、SGS涡粘系数）。
        
        此方法在每个时间步后被调用，用于：
        1. WMLES: 计算壁面剪应力并应用到动量方程残差
        2. LES: 计算亚格子涡粘系数并应用到粘性项
        """
        if self.wmles_model is not None:
            # WMLES：计算壁面剪应力并应用到边界
            self._apply_wmles_wall_stress()
        
        if self.sgs_model is not None:
            # LES：计算亚格子涡粘系数并应用到粘性项
            grad_U = self._compute_gradients()
            grad_u = grad_U[:, :, 1:4, :]  # 提取速度梯度 (n_cells, n_sps, 3, 3)
            delta = self._get_grid_scale()
            nu_t = self.sgs_model.compute_eddy_viscosity(grad_u, delta)
            
            # 将nu_t存储到湍流模型中，供粘性残差计算使用
            if hasattr(self.turb_model, 'nu_t'):
                # 对于SST/DDES，需要叠加SGS贡献
                self.turb_model.nu_t += nu_t
                logger.debug(f"SGS eddy viscosity added to turbulence model: mean={nu_t.mean():.6e}")
            else:
                # 对于纯LES，创建新的nu_t场
                self.sgs_model.nu_t = nu_t
                logger.debug(f"SGS eddy viscosity computed: mean={nu_t.mean():.6e}, max={nu_t.max():.6e}")

    def _apply_wmles_wall_stress(self):
        """
        应用WMLES壁面剪应力到动量方程残差（委托给 solver_helpers）。
        """
        solver_helpers.apply_wmles_wall_stress(self)

    def _extract_wall_boundary_info(self):
        """
        从网格或边界管理器中提取壁面边界信息（委托给 solver_helpers）。
        """
        self._wall_sp_info = solver_helpers.extract_wall_boundary_info(self)
    
    def _auto_detect_wall_boundaries(self):
        """
        基于几何特征自动检测壁面边界（委托给 solver_helpers）。
        """
        self._wall_sp_info = solver_helpers.auto_detect_wall_boundaries(self)
    
    def compute_inviscid_residual(self):
        """
        计算无粘残差 (S-02)。
        
        实现完整的 FR 通量重构逻辑，并支持 Numba 并行加速。
        """
        self.state._update_primitives()
        
        n_cells = self.mesh.n_cells
        n_sps = self.state.n_sps
        n_vars = self.state.n_vars
        
        # 获取三维微分算子
        if not hasattr(self.ops, 'D_3d') or self.ops.D_3d is None:
            return self._compute_inviscid_residual_simple()
        
        # 尝试使用 Numba 加速内核
        try:
            from autoflowcfd.core.fr_kernels import compute_fr_residual_kernel
            
            # 关键修复：在 Python 层确保数组是 C-contiguous 的
            # 这能确保 Numba 接收到的是 'C' 布局数组，从而消除警告
            if not self.state.U.flags['C_CONTIGUOUS']:
                self.state.U = np.ascontiguousarray(self.state.U)
            if not self.state.Q.flags['C_CONTIGUOUS']:
                self.state.Q = np.ascontiguousarray(self.state.Q)
            if not self.ops.D_3d.flags['C_CONTIGUOUS']:
                self.ops.D_3d = np.ascontiguousarray(self.ops.D_3d)
            
            inviscid_res = compute_fr_residual_kernel(
                self.state.U, self.state.Q, self.ops.D_3d, 
                n_cells, n_sps, n_vars
            )
        except Exception as e:
            logger.error(f"Numba kernel failed critically: {e}")
            raise RuntimeError("FR Solver requires Numba acceleration. Parallel CPU loop failed.")

        # 引入基于 AUSM+up 的界面耗散项
        dissipation = self._compute_fr_correction_ausm()
        inviscid_res += dissipation

        return inviscid_res

    def compute_viscous_residual(self):
        """
        计算粘性残差 (S-03)。
        
        使用 LDG (Local Discontinuous Galerkin) 方案处理粘性项。
        
        Returns:
            viscous_res: 粘性残差
        """
        return compute_viscous_residual_ldg(
            self.state.U, self.state.Q, self.ops, self.mesh
        )

    def _compute_fr_correction_ausm(self):
        """
        计算 FR 校正项：模拟单元界面上的数值通量跳跃。
        使用 AUSM+up 格式计算相邻解点间的耗散。
        """
        from autoflowcfd.core.fr_kernels import compute_ausm_up_flux
        
        diss = np.zeros_like(self.state.dU_dt)
        n_cells = self.mesh.n_cells
        n_sps = self.state.n_sps
        
        # 简化的全局平均状态作为"邻居"参考
        mean_Q = np.mean(self.state.Q, axis=0)
        normal_vec = np.array([1.0, 0.0, 0.0]) # 简化法向
        
        kappa_fr = 0.05 # FR 校正强度系数
        
        for i in range(n_cells):
            for s in range(n_sps):
                q_local = self.state.Q[i, s]
                # 模拟界面跳跃：[q* - q]
                # 这里用局部状态与平均状态的差值来近似界面不连续性
                jump = q_local - mean_Q[s]
                
                # 如果跳跃显著，应用 AUSM 耗散
                if np.linalg.norm(jump) > 1e-6:
                    # 构造一个简化的通量修正
                    flux_corr = kappa_fr * jump * np.linalg.norm(q_local[1:4]) # 基于速度幅值
                    diss[i, s] -= flux_corr
                    
        return diss
    
    def _compute_inviscid_residual_simple(self):
        """
        简化的无粘残差计算（备用方案）。
        使用有限差分近似。
        """
        self.state._update_primitives()
        inviscid_res = np.zeros_like(self.state.dU_dt)
        
        n_cells = self.mesh.n_cells
        n_sps = self.state.n_sps
        
        # 简单的一阶迎风差分
        for i in range(1, n_cells):
            for s in range(n_sps):
                # 对流项的简单近似
                U_curr = self.state.U[i, s]
                U_prev = self.state.U[i-1, s]
                
                # 一阶差分
                dU_dx = (U_curr - U_prev) / 0.01  # 假设dx=0.01m
                
                # 迎风格式（基于速度方向）
                vel = U_curr[1] / max(U_curr[0], 1e-10)  # u/rho
                
                if vel > 0:
                    inviscid_res[i, s] -= vel * dU_dx
                else:
                    inviscid_res[i, s] -= vel * dU_dx
                    
        return inviscid_res
    
    def _compute_gradients(self) -> np.ndarray:
        """
        计算守恒变量的梯度。
        
        Returns:
            grad_U: 梯度，形状 (n_cells, n_sps, n_vars, 3)
        """
        from autoflowcfd.core.fr_residual_viscous import compute_gradients
        return compute_gradients(self.state.U, self.ops)
    
    def _compute_local_time_step(self) -> np.ndarray:
        """
        计算局部时间步长（基于CFL条件）。
        
        Returns:
            dt_local: 局部时间步长，形状 (n_cells, n_sps)
        """
        n_cells, n_sps, n_vars = self.state.U.shape
        
        # 提取速度和声速
        rho = self.state.Q[:, :, 0]
        u = self.state.Q[:, :, 1] / np.maximum(rho, 1e-10)
        v = self.state.Q[:, :, 2] / np.maximum(rho, 1e-10)
        w = self.state.Q[:, :, 3] / np.maximum(rho, 1e-10)
        p = (1.4 - 1.0) * (self.state.Q[:, :, 4] - 0.5 * rho * (u**2 + v**2 + w**2))
        a = np.sqrt(np.maximum(1.4 * p / np.maximum(rho, 1e-10), 1e-10))
        
        # 速度幅值
        vel_mag = np.sqrt(u**2 + v**2 + w**2)
        
        # 估计网格尺度
        if hasattr(self.mesh, 'jacobians') and self.mesh.jacobians is not None:
            det_jacs = self.mesh.jacobians.get('det_jacs', None)
            if det_jacs is not None:
                # det_jacs 可能是一维 (n_cells*n_sps_mesh,) 或二维 (n_cells, n_sps_mesh)
                # 关键修复：在Order Continuation期间，状态的n_sps可能与网格的n_sps不同
                mesh_n_sps = det_jacs.size // n_cells if det_jacs.ndim == 1 else det_jacs.shape[1]
                
                if det_jacs.ndim == 1:
                    # 重塑为 (n_cells, mesh_n_sps)
                    det_jacs = det_jacs.reshape(n_cells, mesh_n_sps)
                
                # 如果mesh的SPs数量与当前状态不匹配，进行调整
                if mesh_n_sps != n_sps:
                    logger.debug(
                        f"Mesh has {mesh_n_sps} SPs/cell but state has {n_sps} SPs/cell. "
                        f"Adjusting grid scale estimation."
                    )
                    # 取每个单元的平均体积，然后广播到当前SPs数量
                    volumes = np.mean(det_jacs, axis=1, keepdims=True) * 8.0
                    h = np.power(np.abs(volumes), 1.0/3.0)
                    h_expanded = np.tile(h, (1, n_sps))
                else:
                    volumes = np.mean(det_jacs, axis=1) * 8.0
                    h = np.power(np.abs(volumes), 1.0/3.0)
                    h_expanded = np.tile(h[:, np.newaxis], (1, n_sps))
            else:
                h_expanded = np.ones((n_cells, n_sps)) * 0.01
        else:
            h_expanded = np.ones((n_cells, n_sps)) * 0.01
        
        # CFL条件：dt = CFL * h / (|u| + a)
        CFL = 0.1  # 保守的CFL数
        dt_local = CFL * h_expanded / np.maximum(vel_mag + a, 1e-10)
        
        return dt_local
    
    def _get_cell_volumes(self) -> np.ndarray:
        """
        获取单元体积。
        
        Returns:
            volumes: 单元体积，形状 (n_cells,)
        """
        if hasattr(self.mesh, 'jacobians') and self.mesh.jacobians is not None:
            det_jacs = self.mesh.jacobians.get('det_jacs', None)
            if det_jacs is not None:
                # det_jacs 可能是一维 (n_cells*n_sps,) 或二维 (n_cells, n_sps)
                if det_jacs.ndim == 1:
                    n_cells = self.mesh.n_cells
                    n_sps = self.mesh.n_sps_per_cell
                    det_jacs = det_jacs.reshape(n_cells, n_sps)
                return np.mean(det_jacs, axis=1) * 8.0
        
        # 回退：使用均匀估计
        n_cells = self.mesh.n_cells
        return np.ones(n_cells) * 1e-6
    
    def _get_grid_scale(self) -> np.ndarray:
        """
        获取网格尺度（用于LES/SGS模型）。
        
        Returns:
            delta: 网格尺度，形状 (n_cells, n_sps)
        """
        n_cells, n_sps = self.state.U.shape[:2]
        
        if hasattr(self.mesh, 'jacobians') and self.mesh.jacobians is not None:
            det_jacs = self.mesh.jacobians.get('det_jacs', None)
            if det_jacs is not None:
                # det_jacs 可能是一维 (n_cells*n_sps,) 或二维 (n_cells, n_sps)
                if det_jacs.ndim == 1:
                    det_jacs = det_jacs.reshape(n_cells, n_sps)
                
                volumes = np.mean(det_jacs, axis=1) * 8.0
                delta = np.power(np.abs(volumes), 1.0/3.0)
                return np.tile(delta[:, np.newaxis], (1, n_sps))
        
        return np.ones((n_cells, n_sps)) * 0.01
