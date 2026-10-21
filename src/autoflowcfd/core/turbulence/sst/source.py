"""AutoFlowCFD V2.0 - k/omega 源项（产生、耗散、交叉扩散、环境维持项、realizability 下限）

**只含方法，没有状态**：全部属性由 `SSTModelFR` 的 `__init__` 建立，这里通过 `self` 访问。
数组模块无关（取自 `k_field` 所在的模块），CPU 与 GPU 模型共用这一份。

DES/DDES/IDDES：k 方程破坏项的长度尺度在**同一次**求值里、用本次的 F1 与 nu_t 算出
（`des` 参数，`core/turbulence/des/`），源项是当前状态的纯函数。此前长度尺度在源项求值之后
另算、供下一次求值使用（"慢一拍"），隐式步的残差因此依赖上一次求值的历史，换阶数与续算后
的第一次求值还要退回 RANS 破坏项，分布式路径为它维护了跨步持久状态与 halo 交换。
"""

from typing import Any, Tuple

import numpy as np
from autoflowcfd.core.turbulence.nonfinite import replace_nonfinite

from autoflowcfd.core.utils.array_module import array_module

from .ambient import ambient_sustaining_terms
from .bounds import model_evaluation_fields, omega_realizability_floor


class _SSTSourceMixin:
    """k/omega 源项（产生、耗散、交叉扩散、环境维持项、realizability 下限）"""

    # 宿主类在 __init__ / 其它 mixin 中提供的属性（仅类型声明）
    beta1: Any
    beta2: Any
    beta_star: Any
    compute_blending_function_F1: Any
    compute_blending_function_F2: Any
    compute_eddy_viscosity: Any
    compute_strain_rate_magnitude: Any
    compute_vorticity_magnitude: Any
    k_field: Any
    kappa: Any
    omega_field: Any
    production_factor: Any
    sigma_w1: Any
    sigma_w2: Any

    def compute_source_terms(self, Q: np.ndarray, grad_U: np.ndarray,
                             d_wall: np.ndarray, mu: float,
                             grad_k: np.ndarray,
                             grad_omega: np.ndarray, des=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 SST 模型的完整源项。

        Args:
            Q: 原始变量场 (rho, u, v, w, p)，形状 (n_cells, n_sps, 5)
            grad_U: 速度梯度张量，形状 (n_cells, n_sps, 3, 3)
            d_wall: 壁面距离场，形状 (n_cells, n_sps)
            mu: 动力粘度
            grad_k: k 的梯度，形状 (n_cells, n_sps, 3)——F1 的第三项与
                CD_omega 交叉扩散项都需要，必须是真实梯度（调用方 `core/fr_solver/turbulence/source.py`
                由 `compute_scalar_gradient` 给出）。
            grad_omega: omega 的梯度，形状 (n_cells, n_sps, 3)
            des: DES 长度尺度求值器（`des.ddes.DESLengthScale`，网格尺度与本模型同一索引空间），
                None 时为纯 RANS 破坏项

        Returns:
            Sk: 湍动能方程源项（rho*k 量纲），形状 (n_cells, n_sps)
            S_omega: 比耗散率方程源项（rho*omega 量纲），形状 (n_cells, n_sps)
        """
        xp = array_module(self.k_field)
        rho = Q[:, :, 0]
        nu = mu / xp.maximum(rho, 1e-10)

        # 被输运的原值（可以越过下限甚至为负，见 `bounds.py` 模块文档）；
        # 钳到 ±1e40 只防 k*omega 在 float64 上溢出
        k_raw = xp.clip(self.k_field, -1e40, 1e40)
        omega_raw = xp.clip(self.omega_field, -1e40, 1e40)

        # 计算应变率模
        S_mag = self.compute_strain_rate_magnitude(grad_U)

        # Kato-Launder 驻点修正（工业 RANS 标配：Fluent/OpenFOAM/STAR-CCM+）：
        # 产生项用 S*Ω 替代 S²，防止驻点/滞止区（车头、机头）k 非物理增长。
        # 驻点处 S 大但 Ω≈0 → P_k≈0；剪切层中 S≈Ω → P_k≈S²（退化为标准式）。
        Omega_mag = self.compute_vorticity_magnitude(grad_U)
        S_omega_prod = S_mag * Omega_mag  # Kato-Launder 有效应变率

        # omega 的 realizability 下限 omega_r = max(逐点 0.1 S, 0.1 omega_inf)，只用于
        # **模型项求值**（`bounds.model_evaluation_fields`），不裁剪被输运的 omega：
        #
        # * 逐点 `0.1 S`：时间尺度 realizability，防止剪切区 omega 过小导致
        #   tau = 1/(beta* omega) 过大。必须逐点（2026-09-15 真实缺陷：曾用
        #   全域 max(S)，整场 omega 被耦合到单个最差点上，79 万单元网格发散）；
        # * `0.1 omega_inf`：P0 阶段 S 恒为零、没有产生项时（2026-09-07：omega=1e-12
        #   曾是稳定不动点，cube_demo 升阶后 k 撞 k_max 的根因）给模型项一个有意义
        #   的时间尺度。环境维持项之后 omega 本身也不会塌陷到零（`ambient.py`）。
        self._omega_realizability_min = omega_realizability_floor(self, S_mag, xp)
        # 模型项求值用的有效值（realizability 只在这里施加，不裁剪被输运的量）
        k_safe, omega_safe = model_evaluation_fields(k_raw, omega_raw, self._omega_realizability_min, xp)

        # 交叉扩散项 CD_kw（F1 与 S_omega 的 CD_omega 项共用同一个量，
        # 标准做法是先算这个再算两处，避免重复计算且保证一致）
        grad_dot_product = xp.sum(grad_k * grad_omega, axis=2)  # (n_cells,n_sps)
        CD_kw = xp.maximum(
            2.0 * rho * self.sigma_w2 / omega_safe * grad_dot_product, 1e-10
        )

        # 计算 blending functions（传入钳制值，防止中间量 overflow）
        F1 = self.compute_blending_function_F1(k_safe, omega_safe, d_wall, nu, rho, CD_kw)
        F2 = self.compute_blending_function_F2(k_safe, omega_safe, d_wall, nu)

        # 暂存 F1：k / ln(omega) 的有效扩散系数在同一组 (k, omega) 上读它
        # （`transport/residual.py::turbulence_diffusivities`，与 nu_t 同一约定，不重算）
        self._last_F1 = F1

        beta = F1 * self.beta1 + (1.0 - F1) * self.beta2

        # 暂存本次求值用的混合 beta（用于 update_fields 的半隐式阻尼——
        # 见该方法文档），避免在那里重新跑一遍 F1/blending 计算。
        self._last_beta_blend = beta

        # 计算涡粘系数（传入钳制值）
        self.nu_t = self.compute_eddy_viscosity(
            k_safe, omega_safe, rho, S_mag, F2, mu
        )

        # === k 方程源项 ===
        # 产生项: P_k = μ_t * S * Ω（Kato-Launder 修正，替代标准 S²）
        P_k = self.production_factor * self.nu_t * rho * S_omega_prod

        # P_k 上限（标准 SST 要求，此前缺失）：P_k = min(P_k, 10*beta_star*rho*k*omega)，
        # 防止驻点/强剪切层附近产生项无界增长导致 k 失控。
        P_k = xp.minimum(P_k, 10.0 * self.beta_star * rho * k_safe * omega_safe)

        # 耗散项：标准 RANS 为 D_k = rho beta* k omega；DES 族为 D_k = rho k^1.5 / l_DES，长度尺度用
        # 本次求值的 F1 与 nu_t（模块文档）。耗散写成"k 的原值 x 有效 omega"：k 越过零时耗散变号、
        # 把它推回来（DES 分支同理取 k sqrt(|k|)，k >= 0 时即 k^1.5）。
        l_des = None
        if des is not None:
            l_des = xp.maximum(des(xp, k_safe, omega_safe, self.nu_t, nu, d_wall, grad_U, F1, self.beta_star), 1e-10)
            D_k = rho * k_raw * xp.sqrt(xp.abs(k_raw)) / l_des
        else:
            D_k = rho * self.beta_star * k_raw * omega_safe

        # k 方程总源项
        Sk = P_k - D_k

        # === ω 方程源项 ===
        # 产生项: P_ω = ρ * γ * S * Ω（Kato-Launder 修正，替代标准 S²）
        gamma1 = self.beta1 / self.beta_star - self.sigma_w1 * self.kappa**2 / np.sqrt(self.beta_star)
        gamma2 = self.beta2 / self.beta_star - self.sigma_w2 * self.kappa**2 / np.sqrt(self.beta_star)
        gamma = F1 * gamma1 + (1.0 - F1) * gamma2

        P_omega = self.production_factor * rho * gamma * S_omega_prod

        # 耗散项: D_ω = ρ β ω²，写成"有效 omega × omega 原值"：omega 越过下限时
        # 随原值线性下降、低于零时变号，与环境维持项一起把它推回来
        D_omega = rho * beta * omega_safe * omega_raw

        # 交叉扩散项: CD_ω = 2 * ρ * (1-F1) * σ_w2 / ω * ∇k · ∇ω（与上面
        # 算 CD_kw 用的是同一个 grad_dot_product，(1-F1) 权重是标准 SST
        # 公式要求的——CD_kw 用于 F1 判据时不带这个权重，两者不是同一个量，
        # 不能合并）。
        CD_omega = 2.0 * rho * (1.0 - F1) * self.sigma_w2 / omega_safe * grad_dot_product

        # ω 方程总源项
        S_omega = P_omega - D_omega + CD_omega

        # 环境维持项（SST-sust）：来流 (k_inf, omega_inf) 是无剪切区的精确不动点
        Sk_amb, S_omega_amb = ambient_sustaining_terms(self, rho, beta, xp, l_des)
        Sk = Sk + Sk_amb
        S_omega = S_omega + S_omega_amb

        # 源项 NaN/Inf 隔离：退化网格上 grad_k·grad_omega 等可能为 inf，导致 inf-inf=NaN 传播——替换为零并计数告警
        Sk = replace_nonfinite(self, Sk, "SST k 源项", xp)
        S_omega = replace_nonfinite(self, S_omega, "SST omega 源项", xp)

        return Sk, S_omega
