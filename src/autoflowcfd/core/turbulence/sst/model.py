"""AutoFlowCFD V2.0 - SSTModelFR 主类：构造与对外接口

从 `src/autoflowcfd/core/turbulence/sst.py` 拆出（2026-09-24）。方法按职责分到同目录的 mixin 里，
这里只留构造与对外接口。
"""

import numpy as np
from typing import Optional
from .blending import _SSTBlendingMixin
from .source import _SSTSourceMixin
from .update import _SSTUpdateMixin


class SSTModelFR(_SSTBlendingMixin, _SSTSourceMixin, _SSTUpdateMixin):
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

    def __init__(self, n_cells: int, n_sps: int,
                 k_inf: float = 1e-6, omega_inf: float = 1.0):
        """
        初始化 SST 模型。

        Args:
            n_cells: 单元数量
            n_sps: 每单元解点数量
            k_inf: 来流湍动能初值（默认 1e-6，工业标准由 Tu/VR 推导，
                见 turbulence.py::_set_freestream_turbulence）
            omega_inf: 来流比耗散率初值（默认 1.0，工业标准由 Tu/VR 推导）
        """
        self.n_cells = n_cells
        self.n_sps = n_sps

        # 初始化湍流场（工业标准：从 Tu/VR 推导物理自洽的 k/omega，
        # 而非拍脑袋的 k=1e-6, omega=1.0）。
        # 参考：Fluent 用户手册 Section 7.3.2，
        # k = 1.5*(U*Tu)^2, omega = k/(VR*nu)。
        self.k_field = np.ones((n_cells, n_sps)) * k_inf
        self.omega_field = np.ones((n_cells, n_sps)) * omega_inf
        self.nu_t = np.zeros((n_cells, n_sps))

        # 来流 omega/k（保留为持久属性）。omega_inf 供 compute_source_terms
        # 里 omega realizability 下限在 S_mag 恒零时使用（见该处真实 bug
        # 修复文档完整推导）；k_inf 供 apply_positivity_limiter 里 k 的
        # 同类下限使用（2026-09-11 真实 bug 修复，见该处文档）。
        self.omega_inf = omega_inf
        self.k_inf = k_inf

        # k/omega 物理上界（防止输运方程数值爆炸）。
        # 默认值保守（1e6），应在求解器初始化时根据来流条件设置：
        #   k_max = 0.5 * vel_inf^2（湍动能不超过平均流动能）
        #   omega_max = 1e6（远大于任何工程壁面 omega 值）
        self.k_max: float = 1e6
        self.omega_max: float = 1e6

        # 湍流产项渐变因子 [0, 1]（工业 RANS 标准做法）。
        # 初始为 0（抑制产生项），逐步增加到 1（全量产生）。
        # 防止初始流场未发展时 P_k >> D_k 导致 k/omega 指数爆炸。
        # 由求解器根据全局迭代步数控制，见 turbulence.py::_update_production_ramp。
        self.production_factor: float = 1.0

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

    def get_turbulent_viscosity(self) -> np.ndarray:
        """
        获取当前的湍流涡粘系数。

        Returns:
            nu_t: 涡粘系数场
        """
        return self.nu_t.copy()
