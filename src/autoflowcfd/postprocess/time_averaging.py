"""
AutoFlowCFD V2.0 - 时间平均统计模块

本模块实现在线时间平均统计功能，用于 LES/DDES 瞬态计算的后处理。

核心功能:
1. 滑动平均：在线计算平均值
2. 脉动量统计：RMS 值、雷诺应力
3. 收敛判断：基于平均值变化率
"""

import numpy as np
from typing import Dict, Optional, Tuple
from loguru import logger


class TimeAveraging:
    """
    时间平均统计器。
    
    用于在线累积统计量，支持：
    - 平均值：mean(phi)
    - 均方根：phi_rms = sqrt(mean(phi'^2))
    - 雷诺应力：mean(u'v'), mean(u'w'), mean(v'w')
    - 收敛判断：当平均值相对变化 < tol 时认为统计收敛
    
    Attributes:
        n_samples: 已累积的样本数
        sum_phi: 累积和（用于计算平均值）
        sum_phi_sq: 累积平方和（用于计算 RMS）
        sum_uv, sum_uw, sum_vw: 交叉项累积和（用于雷诺应力）
        prev_mean: 上一时刻的平均值（用于收敛判断）
    """
    
    def __init__(self, n_points: int, n_vars: int = 5):
        """
        初始化时间平均统计器。
        
        Args:
            n_points: 统计点数（n_cells * n_sps 或边界点数）
            n_vars: 变量数（默认5：rho, u, v, w, p）
        """
        self.n_points = n_points
        self.n_vars = n_vars
        self.n_samples = 0
        
        # 累积和
        self.sum_phi = np.zeros((n_points, n_vars))
        self.sum_phi_sq = np.zeros((n_points, n_vars))
        
        # 速度交叉项（用于雷诺应力）
        self.sum_uv = np.zeros(n_points)
        self.sum_uw = np.zeros(n_points)
        self.sum_vw = np.zeros(n_points)
        
        # 上一时刻的平均值（用于收敛判断）
        self.prev_mean = None
        self.converged = False
        
        logger.info(f"TimeAveraging initialized for {n_points} points, {n_vars} variables")
    
    def add_sample(self, phi: np.ndarray):
        """
        添加一个时间步的样本。
        
        Args:
            phi: 当前时间步的场变量，形状 (n_points, n_vars)
        """
        if phi.shape != (self.n_points, self.n_vars):
            raise ValueError(f"Expected shape ({self.n_points}, {self.n_vars}), got {phi.shape}")
        
        self.sum_phi += phi
        self.sum_phi_sq += phi**2
        
        # 提取速度分量（索引1,2,3对应u,v,w）
        u = phi[:, 1]
        v = phi[:, 2]
        w = phi[:, 3]
        
        self.sum_uv += u * v
        self.sum_uw += u * w
        self.sum_vw += v * w
        
        self.n_samples += 1
        
        # 每100个样本检查一次收敛性
        if self.n_samples % 100 == 0 and self.n_samples > 100:
            self._check_convergence()
    
    def get_mean(self) -> np.ndarray:
        """
        获取时间平均值。
        
        Returns:
            mean_phi: 时间平均值，形状 (n_points, n_vars)
        """
        if self.n_samples == 0:
            raise RuntimeError("No samples added yet")
        
        return self.sum_phi / self.n_samples
    
    def get_rms(self) -> np.ndarray:
        """
        获取均方根值（RMS）。
        
        RMS(phi) = sqrt(mean(phi'^2)) = sqrt(mean(phi^2) - mean(phi)^2)
        
        Returns:
            rms_phi: RMS 值，形状 (n_points, n_vars)
        """
        if self.n_samples < 2:
            raise RuntimeError("Need at least 2 samples for RMS calculation")
        
        mean_phi = self.get_mean()
        mean_phi_sq = self.sum_phi_sq / self.n_samples
        
        # 方差 = mean(phi^2) - mean(phi)^2
        variance = mean_phi_sq - mean_phi**2
        
        # 防止负值（数值误差）
        variance = np.maximum(variance, 0.0)
        
        return np.sqrt(variance)
    
    def get_reynolds_stresses(self) -> Dict[str, np.ndarray]:
        """
        获取雷诺应力张量分量。
        
        Returns:
            reynolds_stresses: 字典，包含 'uv', 'uw', 'vw' 分量
        """
        if self.n_samples < 2:
            raise RuntimeError("Need at least 2 samples for Reynolds stress calculation")
        
        mean_phi = self.get_mean()
        mean_u = mean_phi[:, 1]
        mean_v = mean_phi[:, 2]
        mean_w = mean_phi[:, 3]
        
        # 雷诺应力 = mean(u'v') = mean(uv) - mean(u)*mean(v)
        mean_uv = self.sum_uv / self.n_samples
        mean_uw = self.sum_uw / self.n_samples
        mean_vw = self.sum_vw / self.n_samples
        
        reynolds_stresses = {
            'uv': mean_uv - mean_u * mean_v,
            'uw': mean_uw - mean_u * mean_w,
            'vw': mean_vw - mean_v * mean_w,
        }
        
        return reynolds_stresses
    
    def _check_convergence(self, tol: float = 0.01):
        """
        检查统计收敛性。
        
        当平均值的相对变化 < tol 时认为收敛。
        
        Args:
            tol: 收敛容差（默认1%）
        """
        current_mean = self.get_mean()
        
        if self.prev_mean is not None:
            # 计算相对变化
            rel_change = np.abs(current_mean - self.prev_mean) / (np.abs(self.prev_mean) + 1e-10)
            max_rel_change = np.max(rel_change)
            
            if max_rel_change < tol:
                self.converged = True
                logger.info(f"Time averaging converged after {self.n_samples} samples "
                           f"(max relative change: {max_rel_change:.6f})")
        
        self.prev_mean = current_mean.copy()
    
    def reset(self):
        """重置统计器。"""
        self.n_samples = 0
        self.sum_phi.fill(0.0)
        self.sum_phi_sq.fill(0.0)
        self.sum_uv.fill(0.0)
        self.sum_uw.fill(0.0)
        self.sum_vw.fill(0.0)
        self.prev_mean = None
        self.converged = False
        logger.info("TimeAveraging reset")
    
    def get_statistics_summary(self) -> Dict:
        """
        获取统计摘要。
        
        Returns:
            summary: 字典，包含样本数、收敛状态等
        """
        return {
            'n_samples': self.n_samples,
            'converged': self.converged,
            'mean_range': {
                'min': float(self.get_mean().min()) if self.n_samples > 0 else None,
                'max': float(self.get_mean().max()) if self.n_samples > 0 else None,
            } if self.n_samples > 0 else None,
        }


if __name__ == "__main__":
    # 测试代码
    np.random.seed(42)
    
    n_points = 100
    n_vars = 5
    
    # 创建统计器
    stats = TimeAveraging(n_points, n_vars)
    
    # 模拟1000个时间步的随机数据
    print("Adding 1000 samples...")
    for i in range(1000):
        # 生成带有趋势的随机数据（模拟收敛过程）
        trend = np.exp(-i / 200.0)  # 指数衰减趋势
        phi = np.random.rand(n_points, n_vars) * 0.1 + trend
        
        stats.add_sample(phi)
    
    # 获取统计结果
    mean_val = stats.get_mean()
    rms_val = stats.get_rms()
    reynolds = stats.get_reynolds_stresses()
    
    print(f"\n统计摘要:")
    print(f"  样本数: {stats.n_samples}")
    print(f"  收敛状态: {stats.converged}")
    print(f"  平均值范围: [{mean_val.min():.6f}, {mean_val.max():.6f}]")
    print(f"  RMS 范围: [{rms_val.min():.6f}, {rms_val.max():.6f}]")
    print(f"  雷诺应力 uv 范围: [{reynolds['uv'].min():.6f}, {reynolds['uv'].max():.6f}]")
    
    summary = stats.get_statistics_summary()
    print(f"\n完整摘要: {summary}")
