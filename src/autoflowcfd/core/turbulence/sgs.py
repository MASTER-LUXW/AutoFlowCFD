"""亚格子（SGS）应力模型：WALE 与 Smagorinsky-Lilly。

生产路径只用 WALE：`fr_solver/turbulence.py::init_turbulence_models` 的
WMLES 与 LES 两个分支都是 `solver.sgs_model = WALEModel()`。
`SmagorinskyModel` 从未被实例化，但它是 `core.turbulence.__init__` 导出的
公开 API、且 `compute_eddy_viscosity` 是标准且完整的 Smagorinsky-Lilly
公式（不是简化），因此保留。

## 2026-09-14 死代码清理（用户明确要求"本项目从不接受简化"后的普查）

删除了三个方法和一个类，全部经全仓库 grep 确认**零引用**（生产代码、
测试、CLI 都没有），而且每一个都带着"简化"标注或有实质缺陷——保留它们
等于在仓库里留下带简化标记的、无人使用的代码：

- `WALEModel.compute_subgrid_stress`：注释自述"简化：忽略各向同性部分"。
  Boussinesq 形式 `tau_ij = 2*rho*nu_t*S_ij` 缺了 `-(2/3)*rho*k_sgs*delta_ij`
  这一项。不可压 LES 里把它并进压力是标准做法，但**可压缩** LES 通常要
  保留（需要一个 k_sgs 闭合，例如 Yoshizawa）。这个方法从未被调用过
  （粘性残差走的是 `nu_t` 叠加进 mu_eff 那条路，不经过显式的 tau_ij），
  GPU 版移植时就已经 grep 确认过它是死代码而刻意不移植。
- `WALEModel.compute_velocity_gradients`：同样是 GPU 移植时已确认的死
  代码——真实路径用的是 `fr_operators/gradients.py::
  compute_physical_gradient`（真正的 FR 物理梯度），不是这里这份。
- `WALEModel.apply_van_driest_damping`：零引用。近壁阻尼在本项目里由
  WMLES 壁面模型承担，不走这条。
- `DynamicSmagorinskyModel`（原 `sgs_dynamic.py`，整文件删除）：既未在
  `__init__` 导出、也无任何引用，**而且数学上是坏的**。它的
  `L_ij = tau_fine - tau_coarse_filtered` 用的是两个涡粘模型应力之差，
  代入自己定义的 `M_ij` 后恒有 `L_ij == 2*M_ij`，于是 Lilly 最小二乘解
  `C_s^2 = <L:M>/<M:M> == 2` 恒成立，被 `min(..., 0.04)` 钳住后**恒定
  返回 c_s = 0.2**，与流场完全无关。一个"动态"模型却恒返回常数，比不
  存在更危险（会让人以为它在自适应）。
  真正的 Germano/Lilly 动态系数要求对速度**乘积**做测试滤波：
  `L_ij = filter(u_i u_j) - filter(u_i) filter(u_j)`，
  `M_ij = 2[(alpha*Delta)^2 |S_hat| S_hat_ij - Delta^2 filter(|S| S_ij)]`。
  原函数的签名只收梯度（`grad_u_coarse`/`grad_u_fine`），结构上就表达
  不了 `L_ij`。要正确实现必须改成接收速度场 + 一个测试滤波算子（本项目
  的模态滤波器可以充当），并且即便代数写对了，仍是一个没有 LES 验证
  数据支撑的新模型——因此本轮选择删除而不是"补完"，把这段推导记录在
  这里，需要时按上面两个公式重新实现。
"""

import numpy as np
from typing import Optional, Tuple


class WALEModel:
    """
    WALE (Wall-Adapting Local Eddy-viscosity) 亚格子模型。
    
    WALE 模型的优势：
    - 在近壁区域自动衰减，无需阻尼函数
    - 对旋转和剪切流动有更好的适应性
    - 基于速度梯度张量的二阶不变量
    
    Attributes:
        c_wale: WALE 模型常数（通常取 0.325-0.5）
        nu_t: 亚格子涡粘系数场
    """

    def __init__(self, c_wale: float = 0.325):
        """
        初始化 WALE 模型。
        
        Args:
            c_wale: WALE 模型常数
        """
        self.c_wale = c_wale
        self.nu_t = None  # 亚格子涡粘系数
        
    def compute_strain_and_rotation_tensors(self, grad_u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算应变率张量 S_ij 和旋转率张量 Ω_ij。
        
        Args:
            grad_u: 速度梯度张量，形状 (n_cells, n_sps, 3, 3)
            
        Returns:
            S_ij: 应变率张量，形状同 grad_u
            Omega_ij: 旋转率张量，形状同 grad_u
        """
        # S_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)
        S_ij = 0.5 * (grad_u + np.transpose(grad_u, (0, 1, 3, 2)))
        
        # Ω_ij = 0.5 * (∂u_i/∂x_j - ∂u_j/∂x_i)
        Omega_ij = 0.5 * (grad_u - np.transpose(grad_u, (0, 1, 3, 2)))
        
        return S_ij, Omega_ij
    
    def compute_second_invariants(self, S_ij: np.ndarray, 
                                 Omega_ij: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算应变率和旋转率张量的二阶不变量。
        
        Args:
            S_ij: 应变率张量
            Omega_ij: 旋转率张量
            
        Returns:
            S_sq: S_ij*S_ij，形状 (n_cells, n_sps)
            Omega_sq: Ω_ij*Ω_ij，形状 (n_cells, n_sps)
        """
        # S^2 = S_ij * S_ij (Einstein summation)
        S_sq = np.einsum('nijm,nijm->ni', S_ij, S_ij)
        Omega_sq = np.einsum('nijm,nijm->ni', Omega_ij, Omega_ij)
        
        return S_sq, Omega_sq
    
    def compute_wale_invariant(self, S_ij: np.ndarray, Omega_ij: np.ndarray) -> np.ndarray:
        """
        计算 WALE 模型的核心不变量。

        L_ij = S_ik * S_kj + Ω_ik * Ω_kj
        S_ij^d = L_ij - (1/3)*δ_ij*trace(L)  （迹的各向同性部分必须减去，
            Nicoud & Ducros 1999 原始定义）
        L_sq = S_ij^d * S_ij^d

        此前实现漏掉了减迹这一步，直接用未去迹的 L_ij 平方求和。用轴对称
        纯应变反例验证过：diag(a,-a/2,-a/2) 下正确值 L_sq=0.375*a^4，漏迹
        版本算出 1.125*a^4（3倍误差）。纯剪切流特例下两者恰好相等（trace
        本身为零），容易在简单验证用例下"看起来正确"，但一般三维应变场景
        （驻点/加速区）会系统性高估涡粘，破坏 WALE"近壁自动衰减为零"这一
        核心性质（该性质依赖去迹后 S_ij^d 在纯剪切下才自动满足，一般三维
        应变必须显式去迹）。

        Args:
            S_ij: 应变率张量，形状 (n_cells, n_sps, 3, 3)
            Omega_ij: 旋转率张量，形状 (n_cells, n_sps, 3, 3)

        Returns:
            L_sq: WALE 不变量（已去迹），形状 (n_cells, n_sps)
        """
        # L_ij = S_ik*S_kj + Ω_ik*Ω_kj，对最后一维（k）求和，向量化
        L_ij = (np.einsum('nsik,nskj->nsij', S_ij, S_ij)
                + np.einsum('nsik,nskj->nsij', Omega_ij, Omega_ij))

        # 减去迹的各向同性部分：S_ij^d = L_ij - (1/3)*δ_ij*trace(L)
        trace_L = np.einsum('nsii->ns', L_ij)
        eye3 = np.eye(3)
        L_ij_traceless = L_ij - (trace_L / 3.0)[:, :, np.newaxis, np.newaxis] * eye3

        L_sq = np.einsum('nsij,nsij->ns', L_ij_traceless, L_ij_traceless)

        return L_sq
    
    def compute_eddy_viscosity(self, grad_u: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """
        计算 WALE 亚格子涡粘系数 ν_t。
        
        核心公式：
        ν_t = (C_wale * Δ)^2 * (L_ij^2)^(3/2) / ((S_ij^2)^(5/2) + (L_ij^2)^(5/4))
        
        其中：
        - S_ij 是应变率张量
        - L_ij = S_ik*S_kj + Ω_ik*Ω_kj
        - Δ 是网格尺度
        
        Args:
            grad_u: 速度梯度张量，形状 (n_cells, n_sps, 3, 3)
            delta: 网格尺度，形状 (n_cells, n_sps)
            
        Returns:
            nu_t: 亚格子涡粘系数，形状 (n_cells, n_sps)
        """
        # 计算应变率和旋转率张量
        S_ij, Omega_ij = self.compute_strain_and_rotation_tensors(grad_u)
        
        # 计算二阶不变量
        S_sq, Omega_sq = self.compute_second_invariants(S_ij, Omega_ij)
        
        # 计算 WALE 不变量 L^2
        L_sq = self.compute_wale_invariant(S_ij, Omega_ij)
        
        # 防止除以零
        S_sq = np.maximum(S_sq, 1e-10)
        L_sq = np.maximum(L_sq, 1e-10)
        delta = np.maximum(delta, 1e-10)
        
        # WALE 核心公式
        numerator = L_sq**(3.0/2.0)
        denominator = S_sq**(5.0/2.0) + L_sq**(5.0/4.0)
        
        nu_t = (self.c_wale * delta)**2 * numerator / np.maximum(denominator, 1e-10)
        
        # 限制最大值以避免数值不稳定
        nu_t = np.minimum(nu_t, 1e-3)
        
        # 存储结果
        self.nu_t = nu_t
        
        return nu_t
    
class SmagorinskyModel:
    """
    Smagorinsky-Lilly 亚格子模型。
    
    经典的 SGS 模型，形式简单但在近壁区域需要阻尼函数。
    
    Attributes:
        c_s: Smagorinsky 常数（通常取 0.1-0.2）
        nu_t: 亚格子涡粘系数
    """
    
    def __init__(self, c_s: float = 0.1):
        """
        初始化 Smagorinsky 模型。
        
        Args:
            c_s: Smagorinsky 常数
        """
        self.c_s = c_s
        self.nu_t = None
    
    def compute_eddy_viscosity(self, grad_u: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """
        计算 Smagorinsky 涡粘系数。
        
        ν_t = (C_s * Δ)^2 * |S|
        
        其中 |S| = sqrt(2 * S_ij * S_ij)
        
        Args:
            grad_u: 速度梯度张量
            delta: 网格尺度
            
        Returns:
            nu_t: 亚格子涡粘系数
        """
        # 计算应变率张量
        S_ij = 0.5 * (grad_u + np.transpose(grad_u, (0, 1, 3, 2)))
        
        # 计算 |S|
        S_sq = np.einsum('nijm,nijm->ni', S_ij, S_ij)
        S_mag = np.sqrt(2.0 * S_sq)
        
        # Smagorinsky 公式
        nu_t = (self.c_s * delta)**2 * S_mag
        
        # 限制
        nu_t = np.minimum(nu_t, 1e-3)
        
        self.nu_t = nu_t
        
        return nu_t
