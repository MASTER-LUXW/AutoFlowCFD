"""
AutoFlowCFD V2.0 - DDES/IDDES 混合湍流模型 (T-04)

本模块实现 Delayed Detached Eddy Simulation (DDES) 逻辑，
通过屏蔽函数在边界层内保持 RANS，在分离区切换为 LES。

核心功能:
1. DDES 屏蔽函数 F_d 计算
2. 有效长度尺度 l_eff 计算（RANS/LES 切换）
3. IDDES 改进型延迟分离涡模拟
4. 与 SST k-ω 模型的无缝集成
"""

import numpy as np
from typing import Optional
from loguru import logger


class DDESModel:
    """
    DDES 混合模型处理器。
    
    基于 Spalart-Allmaras 或 SST k-ω 模型的 DDES 实现：
    - 在边界层内：F_d ≈ 0，使用 RANS 长度尺度 l_RANS = d_w
    - 在分离区：F_d ≈ 1，使用 LES 长度尺度 l_LES = C_DES * Δ
    
    Attributes:
        c_des: DES 常数（通常取 0.65）
        c_w1: DDES 延迟参数（通常取 8.0）
        psi: 屏蔽函数值场
        l_eff: 有效长度尺度场
    """

    def __init__(self, c_des: float = 0.65, c_w1: float = 8.0):
        """
        初始化 DDES 模型。
        
        Args:
            c_des: DES 常数
            c_w1: DDES 延迟参数（控制屏蔽函数的敏感度）
        """
        self.c_des = c_des
        self.c_w1 = c_w1
        self.psi = None  # 屏蔽函数场
        self.l_eff = None  # 有效长度尺度场
        
    def compute_grid_scale(self, cell_volumes: np.ndarray, 
                          method: str = 'cube_root') -> np.ndarray:
        """
        计算网格尺度 Δ。
        
        Args:
            cell_volumes: 单元体积数组
            method: 计算方法
                - 'cube_root': Δ = V^(1/3)（标准）
                - 'max_edge': Δ = max(Δx, Δy, Δz)
                - 'wurz': Δ = (Δx*Δy*Δz)^(1/3)
                
        Returns:
            delta: 网格尺度
        """
        if method == 'cube_root':
            return cell_volumes ** (1.0 / 3.0)
        elif method == 'max_edge':
            # 需要额外的网格边长信息，此处简化
            return cell_volumes ** (1.0 / 3.0)
        else:
            raise ValueError(f"Unknown method: {method}")

    def compute_strain_rate_magnitude(self, grad_u: np.ndarray) -> np.ndarray:
        """
        计算应变率张量的模 |S|。
        
        Args:
            grad_u: 速度梯度张量，形状 (n_cells, n_sps, 3, 3)
            
        Returns:
            S_mag: 应变率模，形状 (n_cells, n_sps)
        """
        # S_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)
        S_ij = 0.5 * (grad_u + np.transpose(grad_u, (0, 1, 3, 2)))
        
        # |S| = sqrt(2 * S_ij * S_ij)
        S_mag = np.sqrt(2.0 * np.einsum('nijm,nijm->ni', S_ij, S_ij))
        
        return S_mag

    def compute_shielding_function(self, d_w: np.ndarray, nu_t: np.ndarray, 
                                   omega: np.ndarray, kappa: float = 0.41,
                                   grad_u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算 DDES 屏蔽函数 F_d。
        
        F_d = 1 - tanh[(C_d1 * r_d)^3]
        
        其中 r_d = ν_t / (κ² * d_w² * |S|)
        
        Args:
            d_w: 壁面距离
            nu_t: 湍流涡粘系数
            omega: 比耗散率
            kappa: Von Karman 常数
            grad_u: 速度梯度张量（可选，用于精确计算|S|）
            
        Returns:
            f_d: 屏蔽函数，范围 [0, 1]
                 - F_d ≈ 0: 边界层内（RANS 模式）
                 - F_d ≈ 1: 分离区（LES 模式）
        """
        # 防止除以零
        d_w = np.maximum(d_w, 1e-6)
        omega = np.maximum(omega, 1e-6)
        nu_t = np.maximum(nu_t, 1e-10)
        
        # 计算应变率模 |S|
        if grad_u is not None:
            S_mag = self.compute_strain_rate_magnitude(grad_u)
        else:
            # 简化：用 omega 近似 |S|
            S_mag = omega.copy()
        
        S_mag = np.maximum(S_mag, 1e-6)
        
        # 计算 r_d
        r_d = nu_t / (kappa**2 * d_w**2 * S_mag)
        
        # 限制 r_d 的范围
        r_d = np.minimum(r_d, 10.0)
        
        # 计算屏蔽函数
        f_d = 1.0 - np.tanh((self.c_w1 * r_d)**3)
        
        # 存储供后续使用
        self.psi = f_d
        
        return f_d

    def compute_effective_length_scale(self, d_w: np.ndarray, 
                                      delta: np.ndarray,
                                      f_d: np.ndarray,
                                      c_des: Optional[float] = None) -> np.ndarray:
        """
        计算 DDES 有效长度尺度。
        
        Args:
            d_w: 壁面距离，形状 (n_cells, n_sps)
            delta: 网格尺度，形状 (n_cells,) 或 (n_cells, n_sps)
            f_d: 屏蔽函数，形状 (n_cells, n_sps)
            c_des: DES 常数（None 使用默认值）
            
        Returns:
            l_eff: 有效长度尺度，形状 (n_cells, n_sps)
        """
        if c_des is None:
            c_des = self.c_des
        
        # 确保 delta 与 d_w 维度一致
        if delta.ndim == 1:
            n_sps = d_w.shape[1]
            delta = np.tile(delta[:, np.newaxis], (1, n_sps))
        
        l_rans = d_w
        l_les = c_des * delta
        
        # DDES 公式
        l_eff = l_rans - f_d * np.maximum(0.0, l_rans - l_les)
        
        # 确保非负
        l_eff = np.maximum(l_eff, 1e-10)
        
        self.l_eff = l_eff
        
        return l_eff

    def apply_to_sst_model(self, sst_model, d_w: np.ndarray, 
                          cell_volumes: np.ndarray,
                          grad_u: Optional[np.ndarray] = None):
        """
        将 DDES 模型应用到 SST k-ω 模型。
        
        修改 SST 模型的耗散项，使用 l_eff 替代 d_w。
        
        Args:
            sst_model: SSTModelFR 实例
            d_w: 壁面距离，形状 (n_cells, n_sps)
            cell_volumes: 单元体积，形状 (n_cells,)
            grad_u: 速度梯度张量，形状 (n_cells, n_sps, 3, 3)
        """
        # 计算网格尺度
        delta = self.compute_grid_scale(cell_volumes)
        
        # 获取湍流变量
        k = sst_model.k_field
        omega = sst_model.omega_field
        nu_t = sst_model.get_turbulent_viscosity()
        
        # 计算屏蔽函数
        f_d = self.compute_shielding_function(d_w, nu_t, omega, grad_u=grad_u)
        
        # 计算有效长度尺度，直接替换 SST k 方程耗散项里的长度尺度
        # （turbulence_sst.py::compute_source_terms 里 D_k=rho*k^1.5/l_eff），
        # 而不是用启发式系数缩放 beta_star——那样做既不是标准 DES 公式，
        # 也会因为原地修改 sst_model.beta_star 这个"常数"而在多步迭代下
        # 产生复合误差（每次都把上一步已经改过的值当 original 再改一次）。
        # beta_star 本身保持不变，供纯 RANS 场合复用。
        l_eff = self.compute_effective_length_scale(d_w, delta, f_d)
        sst_model.des_length_scale = l_eff

        logger.debug(
            f"DDES applied: f_d range [{f_d.min():.4f}, {f_d.max():.4f}], "
            f"l_eff range [{l_eff.min():.4e}, {l_eff.max():.4e}]"
        )


class IDDESModel(DDESModel):
    """
    IDDES (Improved DDES) 模型。
    
    在 DDES 基础上增加以下改进：
    1. 更敏感的屏蔽函数
    2. 考虑亚格子尺度的混合
    3. 更好的对数层匹配
    """
    
    def __init__(self, c_des: float = 0.65, c_w1: float = 8.0, 
                 alpha_w: float = 0.2):
        """
        初始化 IDDES 模型。
        
        Args:
            c_des: DES 常数
            c_w1: DDES 延迟参数
            alpha_w: IDDES 权重参数
        """
        super().__init__(c_des, c_w1)
        self.alpha_w = alpha_w
        
    def compute_shielding_function(self, d_w: np.ndarray, nu_t: np.ndarray,
                                   omega: np.ndarray, kappa: float = 0.41,
                                   grad_u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        IDDES 改进的屏蔽函数。
        
        引入额外的敏感性参数，使 RANS-LES 过渡更平滑。
        """
        # 基础 DDES 屏蔽函数
        f_d_base = super().compute_shielding_function(d_w, nu_t, omega, kappa, grad_u)
        
        # IDDES 修正：增加近壁区域的敏感性
        # 使用 y+ 相关的修正因子
        y_plus = d_w * np.sqrt(omega / nu_t)
        correction = 1.0 - np.exp(-y_plus / 10.0)
        
        f_d_iddes = f_d_base * correction
        
        self.psi = f_d_iddes
        
        return f_d_iddes


if __name__ == "__main__":
    # 测试代码
    from turbulence_sst import SSTModelFR
    
    # 创建测试数据
    n_cells = 100
    n_sps = 8
    
    d_w = np.random.rand(n_cells, n_sps) * 0.01
    cell_volumes = np.ones(n_cells) * 1e-6
    nu_t = np.random.rand(n_cells, n_sps) * 1e-4
    omega = np.random.rand(n_cells, n_sps) * 100
    
    # 创建 SST 模型
    sst = SSTModelFR(n_cells, n_sps)
    sst.k_field = np.random.rand(n_cells, n_sps) * 1e-4
    
    # 应用 DDES
    ddes = DDESModel()
    ddes.apply_to_sst_model(sst, d_w, cell_volumes)
    
    print("DDES model test completed.")
