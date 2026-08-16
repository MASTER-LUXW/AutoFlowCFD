"""
AutoFlowCFD V2.0 - SST k-omega 湍流模型 FR 离散 (T-01, T-02)

核心功能:
1. SST k-ω 模型源项计算（产生项/耗散项/交叉扩散项）
2. 正性保持限制器 (Positivity-preserving Limiter)
3. blending functions F1, F2（Menter 1994 标准公式，V2.0 二次评审修复：
   此前 F1/F2 都误用 Von Karman 常数 kappa 顶替 beta_star，且各丢失了
   标准公式里的一项，已改正并用近壁/远场极限数值验证）

已知局限（诚实记录，不是本次会话修复范围）：k/omega 目前只有**逐点源项
ODE**（产生-耗散-交叉扩散，含 DES 长度尺度替换），没有独立的**对流+
扩散输运**——即 k/omega 不会被平均流速度场对流、也不会跨单元/跨面
扩散，只在原地随源项弛豫。规范 T-01 要求"离散 k 和 omega 输运方程"，
严格意义上并未完全满足。这是比 F1/F2 公式错误更大的独立工作量（需要
仿照 core/fr_residual_inviscid.py 给 k/omega 各自实现一套基于真实面
连接关系的标量对流数值通量，以及仿照 core/fr_viscous_flux.py 实现
BR1 标量扩散耦合），本次会话优先修复了标准公式错误、量纲/双重更新
bug、非物理限制器这几项影响更直接的正确性问题，完整输运方程留待
后续会话。
"""

import numpy as np
from typing import Tuple, Optional


class SSTModelFR:
    """
    FR 框架下的 SST k-omega 模型处理器。
    
    实现 Menter 的 SST (Shear Stress Transport) 模型，包括:
    - k 输运方程: ∂(ρk)/∂t + ∇·(ρUk) = P_k - D_k + ∇·[(μ+σ_k μ_t)∇k]
    - ω 输运方程: ∂(ρω)/∂t + ∇·(ρUω) = P_ω - D_ω + ∇·[(μ+σ_ω μ_t)∇ω] + CD_ω
    
    Attributes:
        k_field: 湍动能场，存储在 SPs 上，形状 (n_cells, n_sps)
        omega_field: 比耗散率场，存储在 SPs 上，形状 (n_cells, n_sps)
        nu_t: 湍流涡粘系数场，形状 (n_cells, n_sps)
    """

    def __init__(self, n_cells: int, n_sps: int):
        """
        初始化 SST 模型。
        
        Args:
            n_cells: 单元数量
            n_sps: 每单元解点数量
        """
        self.n_cells = n_cells
        self.n_sps = n_sps
        
        # 初始化湍流场（使用小正值避免除零）
        self.k_field = np.ones((n_cells, n_sps)) * 1e-6
        self.omega_field = np.ones((n_cells, n_sps)) * 1.0
        self.nu_t = np.zeros((n_cells, n_sps))
        
        # SST 模型常数
        self.sigma_k1 = 0.85
        self.sigma_k2 = 1.0
        self.sigma_w1 = 0.5
        self.sigma_w2 = 0.856
        self.beta1 = 0.075
        self.beta2 = 0.0828
        self.a1 = 0.31
        self.kappa = 0.41  # Von Karman 常数
        self.beta_star = 0.09

        # Blending function 相关常数
        self.CD_epsilon = 1e-10

        # DES/DDES 长度尺度替换 (T-04)：非 None 时，k 方程耗散项改用
        # D_k = rho*k^1.5/l_eff 替代标准 RANS 的 D_k=rho*beta_star*k*omega，
        # 由 turbulence_des.py::DDESModel.apply_to_sst_model 设置。
        # 之前的版本用 "beta_star *= (1+0.5*f_d)" 这个启发式系数冒充 DES
        # 修正，且该写法会在每次调用时把已经修改过的 self.beta_star 当成
        # "original" 再乘一次，多步迭代下 beta_star 会无界增长——是原地
        # 修改一个应保持不变的模型常数导致的复合 bug，不只是公式选择有误。
        self.des_length_scale: Optional[np.ndarray] = None

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

    def compute_blending_function_F1(self, k: np.ndarray, omega: np.ndarray,
                                     d: np.ndarray, nu: np.ndarray,
                                     S_mag: np.ndarray, rho: np.ndarray,
                                     CD_kw: np.ndarray) -> np.ndarray:
        """
        计算 SST 模型的 blending function F1（Menter 1994 标准公式）。

        arg1 = min[ max( sqrt(k)/(beta*·omega·d), 500·nu/(d²·omega) ),
                    4·rho·sigma_w2·k/(CD_kw·d²) ]
        F1 = tanh(arg1^4)

        此前实现有两处系统性错误（已用真实网格数值审计发现，见
        ProjectFiles/V2.0/6_整体专家组二次评审.md T-01 发现16）：
        1. `arg1 = d/(kappa*sqrt(k)/omega)` 是标准式
           `sqrt(k)/(beta_star*omega*d)` 的倒数，且把 beta_star(=0.09)
           误写成了 Von Karman 常数 kappa(=0.41)；
        2. 完全丢失了标准公式里的第三项（基于交叉扩散 CD_kw 的上界），
           导致 F1 在近壁区可能被错误抬高/压低，SST 的"近壁走 k-omega、
           远场走 k-epsilon"混合机制失真。

        Args:
            k, omega: 湍动能/比耗散率
            d: 到壁面的距离
            nu: 运动粘度
            S_mag: 应变率模（未直接用于 F1，保留参数以兼容既有调用签名）
            rho: 密度
            CD_kw: 交叉扩散项 max(2*rho*sigma_w2/omega*grad_k·grad_omega, 1e-10)

        Returns:
            F1: blending function，范围 [0, 1]
        """
        omega = np.maximum(omega, 1e-10)
        k = np.maximum(k, 1e-10)
        d = np.maximum(d, 1e-10)

        sqrt_k = np.sqrt(k)
        term1 = sqrt_k / (self.beta_star * omega * d)
        term2 = 500.0 * nu / (d**2 * omega)
        term3 = 4.0 * rho * self.sigma_w2 * k / (np.maximum(CD_kw, 1e-10) * d**2)

        arg1 = np.minimum(np.maximum(term1, term2), term3)
        F1 = np.tanh(arg1**4)

        return F1

    def compute_blending_function_F2(self, k: np.ndarray, omega: np.ndarray,
                                     d: np.ndarray, S_mag: np.ndarray,
                                     nu: np.ndarray) -> np.ndarray:
        """
        计算 SST 模型的 blending function F2（Menter 1994 标准公式）。

        arg2 = max( 2·sqrt(k)/(beta*·omega·d), 500·nu/(d²·omega) )
        F2 = tanh(arg2^2)

        此前实现同 F1：用 kappa 顶替 beta_star，且完全丢失了
        500·nu/(d²·omega) 这一项（粘性子层内该项主导，缺失会让 F2 在
        粘性子层内错误地过早趋于 0，SST 的剪应力限制器
        max(a1·omega, F2·S) 在边界层内失效，退化为标准 k-omega）。

        Args:
            k, omega: 湍动能/比耗散率
            d: 到壁面的距离
            S_mag: 应变率模（未直接用于 F2，保留参数以兼容既有调用签名）
            nu: 运动粘度

        Returns:
            F2: blending function，范围 [0, 1]
        """
        omega = np.maximum(omega, 1e-10)
        k = np.maximum(k, 1e-10)
        d = np.maximum(d, 1e-10)

        sqrt_k = np.sqrt(k)
        term1 = 2.0 * sqrt_k / (self.beta_star * omega * d)
        term2 = 500.0 * nu / (d**2 * omega)
        arg2 = np.maximum(term1, term2)

        F2 = np.tanh(arg2**2)

        return F2

    def compute_eddy_viscosity(self, k: np.ndarray, omega: np.ndarray, 
                              rho: np.ndarray, S_mag: np.ndarray, 
                              F2: np.ndarray) -> np.ndarray:
        """
        计算湍流涡粘系数 ν_t。
        
        ν_t = a1 * k / max(a1*ω, F2*S)
        
        Args:
            k: 湍动能
            omega: 比耗散率
            rho: 密度
            S_mag: 应变率模
            F2: blending function
            
        Returns:
            nu_t: 湍流涡粘系数
        """
        k = np.maximum(k, 1e-10)
        omega = np.maximum(omega, 1e-10)
        S_mag = np.maximum(S_mag, 1e-10)
        
        # Boussinesq 假设下的涡粘系数
        nu_t = self.a1 * k / np.maximum(self.a1 * omega, F2 * S_mag)
        
        # 限制最大值以避免数值不稳定
        nu_t = np.minimum(nu_t, 1e6)
        
        return nu_t

    def compute_source_terms(self, Q: np.ndarray, grad_U: np.ndarray,
                            d_wall: np.ndarray, mu: float,
                            grad_k: np.ndarray,
                            grad_omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 SST 模型的完整源项。

        Args:
            Q: 原始变量场 (rho, u, v, w, p)，形状 (n_cells, n_sps, 5)
            grad_U: 速度梯度张量，形状 (n_cells, n_sps, 3, 3)
            d_wall: 壁面距离场，形状 (n_cells, n_sps)
            mu: 动力粘度
            grad_k: k 的梯度，形状 (n_cells, n_sps, 3)——F1 的第三项与
                CD_omega 交叉扩散项都需要，工业级计算要求真实梯度，不再
                提供"简化估计"回退（此前的回退用 `|k|*10`/`|omega|*10`
                冒充梯度量级，物理上没有意义，已删除——调用方
                core/fr_solver_turbulence.py 现在总是提供真实梯度）。
            grad_omega: omega 的梯度，形状 (n_cells, n_sps, 3)

        Returns:
            Sk: 湍动能方程源项（rho*k 量纲），形状 (n_cells, n_sps)
            S_omega: 比耗散率方程源项（rho*omega 量纲），形状 (n_cells, n_sps)
        """
        # 提取流场变量
        rho = Q[:, :, 0]

        # 运动粘度
        nu = mu / np.maximum(rho, 1e-10)

        # 计算应变率模
        S_mag = self.compute_strain_rate_magnitude(grad_U)

        # 交叉扩散项 CD_kw（F1 与 S_omega 的 CD_omega 项共用同一个量，
        # 标准做法是先算这个再算两处，避免重复计算且保证一致）
        grad_dot_product = np.sum(grad_k * grad_omega, axis=2)  # (n_cells,n_sps)
        omega_safe = np.maximum(self.omega_field, 1e-10)
        CD_kw = np.maximum(
            2.0 * rho * self.sigma_w2 / omega_safe * grad_dot_product, 1e-10
        )

        # 计算 blending functions
        F1 = self.compute_blending_function_F1(
            self.k_field, self.omega_field, d_wall, nu, S_mag, rho, CD_kw
        )
        F2 = self.compute_blending_function_F2(
            self.k_field, self.omega_field, d_wall, S_mag, nu
        )

        # Blending 常数
        sigma_k = F1 * self.sigma_k1 + (1.0 - F1) * self.sigma_k2
        sigma_w = F1 * self.sigma_w1 + (1.0 - F1) * self.sigma_w2
        beta = F1 * self.beta1 + (1.0 - F1) * self.beta2

        # 计算涡粘系数
        self.nu_t = self.compute_eddy_viscosity(
            self.k_field, self.omega_field, rho, S_mag, F2
        )

        # === k 方程源项 ===
        # 产生项: P_k = μ_t * S^2
        P_k = self.nu_t * rho * S_mag**2

        # P_k 上限（标准 SST 要求，此前缺失）：P_k = min(P_k, 10*beta_star*rho*k*omega)，
        # 防止驻点/强剪切层附近产生项无界增长导致 k 失控。
        P_k = np.minimum(P_k, 10.0 * self.beta_star * rho * self.k_field * omega_safe)

        # 耗散项：标准 RANS 为 D_k = ρ*β**k*ω；DES/DDES 激活时（T-04）
        # 替换为 D_k = ρ*k^1.5/l_eff，用 DDES 的混合长度尺度直接替代
        # SST 隐含的 RANS 耗散长度尺度，而不是用启发式系数缩放 β*。
        if self.des_length_scale is not None:
            D_k = rho * self.k_field**1.5 / np.maximum(self.des_length_scale, 1e-10)
        else:
            D_k = rho * self.beta_star * self.k_field * self.omega_field

        # k 方程总源项
        Sk = P_k - D_k

        # === ω 方程源项 ===
        # 产生项: P_ω = ρ * γ * S^2
        gamma1 = self.beta1 / self.beta_star - self.sigma_w1 * self.kappa**2 / np.sqrt(self.beta_star)
        gamma2 = self.beta2 / self.beta_star - self.sigma_w2 * self.kappa**2 / np.sqrt(self.beta_star)
        gamma = F1 * gamma1 + (1.0 - F1) * gamma2

        P_omega = rho * gamma * S_mag**2

        # 耗散项: D_ω = ρ * β * ω^2
        D_omega = rho * beta * self.omega_field**2

        # 交叉扩散项: CD_ω = 2 * ρ * (1-F1) * σ_w2 / ω * ∇k · ∇ω（与上面
        # 算 CD_kw 用的是同一个 grad_dot_product，(1-F1) 权重是标准 SST
        # 公式要求的——CD_kw 用于 F1 判据时不带这个权重，两者不是同一个量，
        # 不能合并）。
        CD_omega = 2.0 * rho * (1.0 - F1) * self.sigma_w2 / omega_safe * grad_dot_product

        # ω 方程总源项
        S_omega = P_omega - D_omega + CD_omega

        return Sk, S_omega

    def apply_positivity_limiter(self, min_k: float = 1e-12, min_omega: float = 1e-12):
        """
        正性保持限制器 (T-02)：强制 k 和 omega 非负，并在重构过程中嵌入硬约束。
        
        Args:
            min_k: k 的最小允许值
            min_omega: omega 的最小允许值
        """
        # 硬截断确保物理合理性（T-02 规范要求的正性约束：k,omega >= 0）
        self.k_field = np.maximum(self.k_field, min_k)
        self.omega_field = np.maximum(self.omega_field, min_omega)

        # 此前这里还有一段"超过全场均值100倍就拍回100倍均值"的全局裁剪，
        # 已删除：这不是正性限制器要求的东西（T-02 只要求 k,omega>=0），
        # 全场均值在驻点/强剪切层附近偏低是正常物理现象，用它做裁剪阈值
        # 会把这些区域本应合法的高 k 值直接削平，是非物理的伪平滑，且
        # 阈值 100 没有任何依据（见 V2.0 二次评审 T-02 发现）。

    def update_fields(self, dt: float, Sk: np.ndarray, S_omega: np.ndarray,
                     diff_k: np.ndarray = None, diff_omega: np.ndarray = None):
        """
        执行一个时间步长的湍流场更新。
        
        Args:
            dt: 时间步长
            Sk: 湍动能源项
            S_omega: 比耗散率源项
            diff_k: k 的扩散项（可选）
            diff_omega: omega 的扩散项（可选）
        """
        # 如果提供了扩散项，则包含在更新中
        if diff_k is not None and diff_omega is not None:
            self.k_field += dt * (Sk + diff_k)
            self.omega_field += dt * (S_omega + diff_omega)
        else:
            # 仅源项更新
            self.k_field += dt * Sk
            self.omega_field += dt * S_omega
        
        # 应用正性限制器
        self.apply_positivity_limiter()
        
    def get_turbulent_viscosity(self) -> np.ndarray:
        """
        获取当前的湍流涡粘系数。
        
        Returns:
            nu_t: 涡粘系数场
        """
        return self.nu_t.copy()
