"""
AutoFlowCFD V2.0 - WMLES 壁面应力模型 (T-05)

本模块实现 Wall-Modeled LES 的核心逻辑，通过平衡律假设提供壁面剪应力。

核心功能:
1. 基于平衡律的壁面应力模型
2. 迭代求解摩擦速度 u_tau
3. 支持滑移边界条件
4. 适配 y+ > 50 的工业级粗网格 LES
"""

import numpy as np
from typing import Tuple, Optional


class WMLESModel:
    """
    WMLES 壁面应力处理器。
    
    基于 Werner-Wengle 或平衡律模型，适用于 y+ > 50 的工业级粗网格 LES 仿真。
    
    核心思想：
    - 不在近壁面解析边界层剖面
    - 通过第一层网格点的速度信息反推壁面剪应力 τ_w
    - 将 τ_w 作为边界条件施加到动量方程
    
    Attributes:
        kappa: Von Karman 常数（默认0.41）
        B: Log-law 常数（默认5.2）
        nu: 运动粘度
        tau_w: 壁面剪应力场，形状 (n_wall_faces, 3)
        u_tau: 摩擦速度场，形状 (n_wall_faces,)
    """

    def __init__(self, kappa: float = 0.41, B: float = 5.2, nu: float = 1.5e-5):
        """
        初始化 WMLES 模型。
        
        Args:
            kappa: Von Karman 常数
            B: Log-law 积分常数
            nu: 运动粘度 (m²/s)
        """
        self.kappa = kappa
        self.B = B
        self.nu = nu
        
        # 输出变量
        self.tau_w = None  # 壁面剪应力
        self.u_tau = None  # 摩擦速度
        self.y_plus = None  # 无量纲壁面距离
        
    def compute_log_law_velocity(self, y_plus: np.ndarray) -> np.ndarray:
        """
        计算对数律速度剖面 u+.
        
        u+ = (1/kappa) * ln(y+) + B
        
        Args:
            y_plus: 无量纲壁面距离
            
        Returns:
            u_plus: 无量纲速度
        """
        y_plus = np.maximum(y_plus, 1.0)  # 确保在对数律区域
        u_plus = (1.0 / self.kappa) * np.log(y_plus) + self.B
        
        return u_plus
    
    def compute_spalding_law(self, y_plus: np.ndarray) -> np.ndarray:
        """
        计算 Spalding 律（全y+范围适用）。
        
        u+ = y+ + (1/(κ*B)) * [exp(κ*u+) - 1 - κ*u+ - (κ*u+)²/2 - (κ*u+)³/6]
        
        简化形式：使用隐式迭代求解
        
        Args:
            y_plus: 无量纲壁面距离
            
        Returns:
            u_plus: 无量纲速度
        """
        # 初始猜测：线性律
        u_plus = y_plus.copy()
        
        # Newton-Raphson 迭代求解
        for _ in range(10):
            # Spalding 律残差
            f_val = u_plus - y_plus - (1.0 / (self.kappa * self.B)) * (
                np.exp(self.kappa * u_plus) - 1 - self.kappa * u_plus - 
                0.5 * (self.kappa * u_plus)**2 - (1.0/6.0) * (self.kappa * u_plus)**3
            )
            
            # 导数
            df_du = 1.0 - (1.0 / self.B) * np.exp(self.kappa * u_plus)
            
            # 更新
            delta_u = f_val / np.maximum(np.abs(df_du), 1e-10)
            u_plus -= np.clip(delta_u, -1.0, 1.0)  # 限制步长避免发散
            
            # 检查收敛
            if np.max(np.abs(delta_u)) < 1e-6:
                break
        
        return u_plus
    
    def solve_friction_velocity_iterative(self, u_tangent: np.ndarray, 
                                         y_dist: np.ndarray,
                                         max_iter: int = 20,
                                         tol: float = 1e-6) -> np.ndarray:
        """
        迭代求解摩擦速度 u_tau。
        
        基于关系式：u_mag = u_tau * u+(y+)
        其中 y+ = y * u_tau / nu
        
        Args:
            u_tangent: 第一层 SPs 处的切向速度，形状 (n_points, 3)
            y_dist: 第一层 SPs 到壁面的距离，形状 (n_points,)
            max_iter: 最大迭代次数
            tol: 收敛容差
            
        Returns:
            u_tau: 摩擦速度，形状 (n_points,)
        """
        u_mag = np.linalg.norm(u_tangent, axis=-1)
        n_points = len(u_mag)
        
        # 防止除以零
        y_dist = np.maximum(y_dist, 1e-6)
        u_mag = np.maximum(u_mag, 1e-10)
        
        # 初始猜测：基于对数律
        # u_mag ≈ u_tau * [(1/kappa)*ln(y*u_tau/nu) + B]
        # 简化初始值
        u_tau = u_mag / 20.0  # 粗略估计
        u_tau = np.maximum(u_tau, 1e-6)
        
        # Newton-Raphson 迭代
        for iteration in range(max_iter):
            # 计算当前 y+
            y_plus = y_dist * u_tau / self.nu
            y_plus = np.maximum(y_plus, 1.0)  # 确保在对数律区
            
            # 计算 u+ (使用对数律)
            u_plus = self.compute_log_law_velocity(y_plus)
            
            # 残差：u_mag - u_tau * u+ = 0
            residual = u_mag - u_tau * u_plus
            
            # 导数：d(residual)/d(u_tau)
            # du+/du_tau = (1/kappa) * (1/y+) * (y_dist/nu) = 1/u_tau
            du_plus_du_tau = 1.0 / u_tau
            d_residual_d_u_tau = -(u_plus + u_tau * du_plus_du_tau)
            
            # Newton 更新
            delta_u_tau = residual / np.maximum(np.abs(d_residual_d_u_tau), 1e-10)
            
            # 限制步长避免发散
            delta_u_tau = np.clip(delta_u_tau, -0.5 * u_tau, 0.5 * u_tau)
            
            u_tau_new = u_tau + delta_u_tau
            u_tau_new = np.maximum(u_tau_new, 1e-6)  # 保持正值
            
            # 检查收敛
            max_change = np.max(np.abs(delta_u_tau) / u_tau)
            u_tau = u_tau_new
            
            if max_change < tol:
                break
        
        # 存储结果
        self.u_tau = u_tau
        self.y_plus = y_dist * u_tau / self.nu
        
        return u_tau
    
    def compute_wall_shear_stress(self, u_tangent: np.ndarray, 
                                 y_dist: np.ndarray,
                                 rho: np.ndarray,
                                 method: str = 'iterative') -> np.ndarray:
        """
        计算壁面剪应力 τ_w。
        
        Args:
            u_tangent: 第一层 SPs 处的切向速度，形状 (n_points, 3)
            y_dist: 第一层 SPs 到壁面的距离，形状 (n_points,)
            rho: 密度，形状 (n_points,)
            method: 计算方法
                - 'iterative': 迭代求解（推荐）
                - 'direct': 直接估算（快速但不准确）
                
        Returns:
            tau_w: 壁面剪应力向量，形状 (n_points, 3)
        """
        u_mag = np.linalg.norm(u_tangent, axis=-1)
        
        if method == 'iterative':
            # 迭代求解摩擦速度
            u_tau = self.solve_friction_velocity_iterative(u_tangent, y_dist)
        else:
            # 直接估算（简化版）
            y_dist_safe = np.maximum(y_dist, 1e-6)
            u_tau = u_mag / (1.0 / self.kappa * np.log(y_dist_safe * u_mag / self.nu + 1e-10) + self.B)
            u_tau = np.maximum(u_tau, 1e-6)
            self.u_tau = u_tau
            self.y_plus = y_dist * u_tau / self.nu
        
        # 计算壁面剪应力大小：τ_w = ρ * u_tau²
        tau_w_mag = rho * u_tau**2
        
        # 方向：与切向速度同向
        # 单位化切向速度
        u_tangent_unit = u_tangent / (u_mag[:, np.newaxis] + 1e-10)
        
        # 壁面剪应力向量
        tau_w = tau_w_mag[:, np.newaxis] * u_tangent_unit
        
        # 存储结果
        self.tau_w = tau_w
        
        return tau_w
    
    def apply_slip_boundary_condition(self, u_interior: np.ndarray, 
                                     normal: np.ndarray,
                                     tau_w: np.ndarray,
                                     rho: np.ndarray,
                                     dt: float) -> np.ndarray:
        """
        应用滑移边界条件（考虑壁面剪应力）。
        
        在 WMLES 中，壁面不强制无滑移，而是通过剪应力耦合。
        
        Args:
            u_interior: 内部点的速度，形状 (n_points, 3)
            normal: 壁面法向量，形状 (n_points, 3)
            tau_w: 壁面剪应力，形状 (n_points, 3)
            rho: 密度，形状 (n_points,)
            dt: 时间步长
            
        Returns:
            u_bc: 边界速度，形状 (n_points, 3)
        """
        # 分解为法向和切向分量
        u_normal = np.sum(u_interior * normal, axis=-1, keepdims=True)
        u_tangent = u_interior - u_normal * normal
        
        # 法向速度为零（不可穿透）
        u_bc_normal = np.zeros_like(u_normal)
        
        # 切向速度：考虑壁面剪应力的影响
        # τ_w = μ * (∂u/∂y) ≈ μ * (u_bc_tangent - u_tangent) / y
        # 简化：u_bc_tangent = u_tangent - τ_w * dt / (ρ * y)
        
        # 这里需要根据具体数值格式调整
        u_bc_tangent = u_tangent.copy()  # 暂时保持内部值
        
        # 组合
        u_bc = u_bc_normal * normal + u_bc_tangent
        
        return u_bc
    
    def get_y_plus_distribution(self) -> np.ndarray:
        """
        获取 y+ 分布统计。
        
        Returns:
            y_plus: 无量纲壁面距离
        """
        if self.y_plus is None:
            raise RuntimeError("Wall shear stress not computed yet")
        
        return self.y_plus.copy()
    
    def validate_y_plus_range(self, min_y_plus: float = 30.0, 
                             max_y_plus: float = 300.0) -> Tuple[bool, dict]:
        """
        验证 y+ 是否在 WMLES 适用范围内。
        
        Args:
            min_y_plus: 最小允许 y+
            max_y_plus: 最大允许 y+
            
        Returns:
            is_valid: 是否全部在范围内
            stats: 统计信息字典
        """
        if self.y_plus is None:
            raise RuntimeError("Wall shear stress not computed yet")
        
        y_plus = self.y_plus
        
        stats = {
            'min': float(np.min(y_plus)),
            'max': float(np.max(y_plus)),
            'mean': float(np.mean(y_plus)),
            'std': float(np.std(y_plus)),
            'n_below_min': int(np.sum(y_plus < min_y_plus)),
            'n_above_max': int(np.sum(y_plus > max_y_plus)),
            'n_in_range': int(np.sum((y_plus >= min_y_plus) & (y_plus <= max_y_plus)))
        }
        
        is_valid = (stats['n_below_min'] == 0) and (stats['n_above_max'] == 0)
        
        return is_valid, stats


if __name__ == "__main__":
    # 测试代码
    np.random.seed(42)
    
    # 创建测试数据
    n_points = 100
    u_tangent = np.random.rand(n_points, 3) * 10.0  # 0-10 m/s
    y_dist = np.random.rand(n_points) * 0.01 + 0.001  # 1-11 mm
    rho = np.ones(n_points) * 1.225  # 空气密度
    
    # 创建 WMLES 模型
    wmles = WMLESModel(nu=1.5e-5)
    
    # 计算壁面剪应力
    tau_w = wmles.compute_wall_shear_stress(u_tangent, y_dist, rho, method='iterative')
    
    print(f"Wall shear stress computed:")
    print(f"  tau_w magnitude: min={np.linalg.norm(tau_w, axis=1).min():.4f}, "
          f"max={np.linalg.norm(tau_w, axis=1).max():.4f} Pa")
    print(f"  u_tau: min={wmles.u_tau.min():.4f}, max={wmles.u_tau.max():.4f} m/s")
    print(f"  y+: min={wmles.y_plus.min():.1f}, max={wmles.y_plus.max():.1f}")
    
    # 验证 y+ 范围
    is_valid, stats = wmles.validate_y_plus_range()
    print(f"\ny+ Distribution:")
    print(f"  Range: [{stats['min']:.1f}, {stats['max']:.1f}]")
    print(f"  Mean: {stats['mean']:.1f} ± {stats['std']:.1f}")
    print(f"  In range [30, 300]: {stats['n_in_range']}/{n_points}")
    print(f"  Valid: {is_valid}")
