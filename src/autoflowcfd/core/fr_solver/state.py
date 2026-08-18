"""
AutoFlowCFD V2.0 - FR 求解器状态数据结构 (S-01)

本模块定义 FRState 类，用于管理存储在 Solution Points (SPs) 上的
守恒变量 (Conservative Variables) 及其时间导数。
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class SolverResult:
    """求解结果数据类。"""
    converged: bool
    iterations: int
    final_residual: float


class FRState:
    """
    FR 求解器状态容器。
    
    Attributes:
        U: 守恒变量数组，形状为 (n_cells, n_sps_per_cell, n_vars)。
           n_vars = 5 (rho, rho_u, rho_v, rho_w, rho_e)
        dU_dt: 残差/时间导数数组，形状同 U。
        Q: 原始变量数组 (rho, u, v, w, p)，用于通量计算。
    """

    def __init__(self, n_cells: int, n_sps_per_cell: int, n_vars: int = 7):
        """
        初始化 FRState。
        
        Args:
            n_cells: 单元数量
            n_sps_per_cell: 每个单元的解点数量
            n_vars: 守恒变量数量 (5个流体 + 2个湍流)
        """
        self.n_cells = n_cells
        self.n_sps = n_sps_per_cell
        self.n_vars = n_vars
        
        # 采用 SoA (Structure of Arrays) 思想的连续内存布局优化
        # 形状: (n_cells, n_sps_per_cell, n_vars)
        self.U = np.zeros((n_cells, n_sps_per_cell, n_vars), dtype=np.float64)
        self.dU_dt = np.zeros_like(self.U)
        self.Q = np.zeros_like(self.U)  # 原始变量用于通量计算

    def initialize_uniform(self, rho=1.0, u=0.0, v=0.0, w=0.0, p=1.0, k=1e-6, omega=1e-2):
        """
        用均匀流场初始化状态。
        
        Args:
            rho: 密度
            u, v, w: 速度分量
            p: 压力
            k: 湍动能
            omega: 比耗散率
        """
        gamma = 1.4
        e = p / ((gamma - 1.0) * rho) + 0.5 * (u**2 + v**2 + w**2)
        
        self.U[:, :, 0] = rho
        self.U[:, :, 1] = rho * u
        self.U[:, :, 2] = rho * v
        self.U[:, :, 3] = rho * w
        self.U[:, :, 4] = rho * e
        
        if self.n_vars > 5:
            self.U[:, :, 5] = rho * k      # rho_k
            self.U[:, :, 6] = rho * omega  # rho_omega
        
        self._update_primitives()

    def _update_primitives(self):
        """从守恒变量 U 更新原始变量 Q (含湍流量)。"""
        rho = self.U[:, :, 0]
        rho = np.maximum(rho, 1e-10)
        
        u = self.U[:, :, 1] / rho
        v = self.U[:, :, 2] / rho
        w = self.U[:, :, 3] / rho
        
        energy = self.U[:, :, 4]
        ke = 0.5 * (u**2 + v**2 + w**2)
        p = np.maximum((energy - rho * ke) * 0.4, 1.0)  # gamma = 1.4
        
        self.Q[:, :, 0] = rho
        self.Q[:, :, 1] = u
        self.Q[:, :, 2] = v
        self.Q[:, :, 3] = w
        self.Q[:, :, 4] = p
        
        if self.n_vars > 5:
            self.Q[:, :, 5] = np.maximum(self.U[:, :, 5] / rho, 1e-12)  # k
            self.Q[:, :, 6] = np.maximum(self.U[:, :, 6] / rho, 1e-12)  # omega

    def get_residual_norm(self) -> float:
        """计算残差的 RMS 范数（按单元数归一化），用于收敛性判断。
        
        使用 RMS (Root Mean Square) 而非原始 L2 范数，使残差量级与网格尺寸无关，
        便于不同网格间的收敛行为对比。RMS = L2 / sqrt(N)，其中 N 是总自由度数。
        """
        n_total = self.dU_dt.size  # n_cells * n_sps * n_vars
        if n_total == 0:
            return 0.0
        return np.linalg.norm(self.dU_dt) / np.sqrt(n_total)