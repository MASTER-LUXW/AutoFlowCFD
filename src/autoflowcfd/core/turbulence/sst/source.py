"""AutoFlowCFD V2.0 - k/omega 源项（产生、耗散、交叉扩散、realizability 下限）

从 `src/autoflowcfd/core/turbulence/sst.py` 的 `SSTModelFR` 拆出（2026-09-24，项目「单文件不超
500 行」规范）。mixin 是本仓库既有惯例（`_SolverGeometryMixin`、
`_GPUSolverInitMixin` 等），沿用它而不是另发明一套。

**只含方法，没有状态**：全部属性由 `SSTModelFR` 的 `__init__` 建立，
这里通过 `self` 访问。
"""

import numpy as np
from typing import Tuple


class _SSTSourceMixin:
    """k/omega 源项（产生、耗散、交叉扩散、realizability 下限）"""

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

        # 钳制湍流变量防 overflow：k*omega 在 k,omega~1e155 时超 float64
        # 上限。物理上 k<1e6, omega<1e8 已远超任何工程工况，保守取 1e40
        # 确保 k*omega=1e80 后乘 rho*beta_star 仍安全
        k_safe = np.minimum(self.k_field, 1e40)
        omega_safe_raw = np.minimum(self.omega_field, 1e40)
        omega_safe = np.maximum(omega_safe_raw, 1e-10)

        # 计算应变率模
        S_mag = self.compute_strain_rate_magnitude(grad_U)

        # Kato-Launder 驻点修正（工业 RANS 标配：Fluent/OpenFOAM/STAR-CCM+）：
        # 产生项用 S*Ω 替代 S²，防止驻点/滞止区（车头、机头）k 非物理增长。
        # 驻点处 S 大但 Ω≈0 → P_k≈0；剪切层中 S≈Ω → P_k≈S²（退化为标准式）。
        Omega_mag = self.compute_vorticity_magnitude(grad_U)
        S_omega_prod = S_mag * Omega_mag  # Kato-Launder 有效应变率

        # 时间尺度 realization：动态计算 ω 下限（供 apply_positivity_limiter 使用）。
        # 约束 ω ≥ C * S，防止远场 ω 衰减到过小值导致 τ = 1/(β*ω) 过大。
        # C=0.1 是保守值（Fluent 默认时间尺度限制等价于 C≈0.1-0.3）。
        #
        # 真实 bug 修复（2026-09-07，cube_demo 791,492 单元真实网格
        # Order Continuation P0->P1 升阶后长程续算 k_mean 持续增长排查
        # 发现，是本次排查的真正根因——此前 grad_vel/omega 壁面松弛/
        # resume 误重置/wall_distance 阶数切换污染 4 个真实bug均已修复
        # 但问题依旧，逐一排除 P_k 上限失控、网格局部退化(ANSA 高质量
        # 网格，真实交叉体积比 1.27~2.28，完全正常)后，用历史 checkpoint
        # 数据（iter_001900~002300，早于本次排查所有事件、纯 P0 阶段）
        # 直接定位到：cell=81600/81552/81546 等一小撮单元的 omega_field
        # 在 P0 阶段极早期（iter~2000左右）就已经塌陷到 1e-12（`apply_
        # positivity_limiter` 的裸正性下限——只防负值，不是物理意义上的
        # 下限），k 未同步跌落，形成天文数字级的 k/omega 比值。
        #
        # 根因：**P0 阶段 grad_vel 恒为零（P0 是分片常数场，多项式导数
        # 恒为零，见 fr_solver/turbulence.py 模块文档"为什么P0阶段摩擦
        # 阻力算不出来"一节同一个事实）**，本行原公式 `0.1*max(S_mag)`
        # 在 P0 阶段因此恒为 0——这个本该防止 ω 衰减过度的 realizability
        # 下限在整个 P0 阶段是完全失效的空话，一旦某个 SP 的显式积分把
        # ω 打到裸正性下限 1e-12，P0 阶段内没有任何机制能让它恢复（此时
        # 产生项 P_omega~S*Ω=0、耗散项 D_omega=beta*rho*omega²≈0，是这套
        # ODE 系统在 P0 阶段的一个稳定不动点，会原样冻结直到阶数真正
        # 切换）。冻结在 ω≈1e-12 的这些单元升阶到 P1 后，S_mag 变为非零，
        # 巨大的 k/omega 比值被 nu_t 湍流粘性比上限(TURBULENT_VISCOSITY_
        # RATIO_MAX=1e5)钳到 ~1.47（不是零！），这个虽被钳制但依然很大
        # 的 nu_t 撑起一条异常畅通的扩散通道（Gamma_k=mu+sigma_k*rho*
        # nu_t），持续从周围真正有产生项的区域把 k 抽/扩散进来，最终把
        # 这些单元（以及被扩散波及的邻居）顶到 k_max 安全上限——这正是
        # 之前排查看到的"局部单元 k_max/nu_t_max 双双撞墙"现象的真正
        # 成因，跟 wall_distance/omega 壁面边界条件精度都无关（这两者
        # 已用真实数据决定性证伪：omega 实际值/Wilcox 目标值比值全程
        # 稳定在 0.947，未衰减；30/100 步 A/B 对照两版 wall_distance
        # 处理方式下 k_mean 轨迹几乎完全一致）。
        #
        # 修复：realizability 下限不能只依赖 S_mag（P0 下恒零、形同虚设），
        # 加一个与阶数/S_mag 无关、恒定有效的物理量纲下限——来流 omega_inf
        # 的一个保守比例（沿用同一个 C=0.1 系数，物理意义："本地湍流
        # 时间尺度不应该比来流环境值大 10 倍以上"这个 realizability 的
        # 精神在 S_mag 不可用时同样适用于来流尺度）。两者取更大值，S_mag
        # 非零时（P1+）这条新增下限通常远小于 0.1*max(S_mag)、不改变
        # 既有行为；S_mag 恒零时（P0）它是唯一起作用的下限，防止 ω 塌陷
        # 到物理上毫无意义的 1e-12。
        # **真实 bug 修复（2026-09-15）：这条下限此前用的是全域最大值
        # `np.max(S_mag)`，是一个标量，被施加到每一个单元的 omega 上。**
        # 那让"局部 realizability 约束"退化成"全场耦合到单个最差点"：
        # 79 万单元 cube_demo 真实网格 250 步对照里，一旦模态滤波器不再
        # 把 P1 内容清零（grad_vel 真正非零），max(S_mag) 由全场最差的
        # 那一个点决定，整个 omega 场被抬到同一个值上 -> nu_t = rho*k/omega
        # 全场被同比压低 -> 湍流扩散崩塌 -> 局部应变更大 -> 下限更高，
        # 正反馈。实测 om_min 10 步内从 1.28e2 跳到 1.85e4（176 倍），
        # 最终 om_min≈om_max≈1e6（整个场被钉在下限上）并发散
        # （AFCFD_FILTER_MODE=off step 103、sensor step 157）。
        #
        # 它此前一直没有暴露，恰恰是因为 P0 阶段 grad_vel 恒为零（见下方
        # 2026-09-05 那段）、而 P1/P2 阶段模态滤波器每个 RK stage 把非常数
        # 模态清零（见 fr/modal_filter.py：order=1 保留秩 1/8，P1 实际是
        # P0），两者都让 S_mag 恒等于钳位值 1e-10——也就是说这个 bug 被
        # 另外两个缺陷共同掩盖了。
        #
        # Durbin 的 realizability / Wilcox 的时间尺度约束、以及本行注释
        # 原本声称等价的 Fluent turbulence time scale limiter，**都是逐点
        # 的**，没有任何一个是"取全域最大"。改为逐点后：
        #   - P0 阶段 S_mag 恒为钳位值 1e-10 => 0.1*S_mag = 1e-11 远小于
        #     0.1*omega_inf，逐点下限**逐位等于**此前的标量下限，P0 行为
        #     完全不变（这是这次改动的安全保证，有对应回归测试）；
        #   - P1+ 阶段每个单元按**自己的**应变率定下限，不再被别处的
        #     尖峰绑架。
        self._omega_realizability_min = np.maximum(0.1 * S_mag, 0.1 * self.omega_inf)

        # 交叉扩散项 CD_kw（F1 与 S_omega 的 CD_omega 项共用同一个量，
        # 标准做法是先算这个再算两处，避免重复计算且保证一致）
        grad_dot_product = np.sum(grad_k * grad_omega, axis=2)  # (n_cells,n_sps)
        CD_kw = np.maximum(
            2.0 * rho * self.sigma_w2 / omega_safe * grad_dot_product, 1e-10
        )

        # 计算 blending functions（传入钳制值，防止中间量 overflow）
        F1 = self.compute_blending_function_F1(
            k_safe, omega_safe, d_wall, nu, S_mag, rho, CD_kw
        )
        F2 = self.compute_blending_function_F2(
            k_safe, omega_safe, d_wall, S_mag, nu
        )

        # Blending 常数
        sigma_k = F1 * self.sigma_k1 + (1.0 - F1) * self.sigma_k2
        sigma_w = F1 * self.sigma_w1 + (1.0 - F1) * self.sigma_w2
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
        P_k = np.minimum(P_k, 10.0 * self.beta_star * rho * k_safe * omega_safe)

        # 耗散项：标准 RANS 为 D_k = ρ*β**k*ω；DES/DDES 激活时（T-04）
        # 替换为 D_k = ρ*k^1.5/l_eff，用 DDES 的混合长度尺度直接替代
        # SST 隐含的 RANS 耗散长度尺度，而不是用启发式系数缩放 β*。
        if self.des_length_scale is not None:
            D_k = rho * k_safe**1.5 / np.maximum(self.des_length_scale, 1e-10)
        else:
            D_k = rho * self.beta_star * k_safe * omega_safe

        # k 方程总源项
        Sk = P_k - D_k

        # === ω 方程源项 ===
        # 产生项: P_ω = ρ * γ * S * Ω（Kato-Launder 修正，替代标准 S²）
        gamma1 = self.beta1 / self.beta_star - self.sigma_w1 * self.kappa**2 / np.sqrt(self.beta_star)
        gamma2 = self.beta2 / self.beta_star - self.sigma_w2 * self.kappa**2 / np.sqrt(self.beta_star)
        gamma = F1 * gamma1 + (1.0 - F1) * gamma2

        P_omega = self.production_factor * rho * gamma * S_omega_prod

        # 耗散项: D_ω = ρ * β * ω^2
        # omega_safe 已钳制到 [1e-10, 1e100]，平方后 1e200 仍在 float64 范围内
        D_omega = rho * beta * omega_safe**2

        # 交叉扩散项: CD_ω = 2 * ρ * (1-F1) * σ_w2 / ω * ∇k · ∇ω（与上面
        # 算 CD_kw 用的是同一个 grad_dot_product，(1-F1) 权重是标准 SST
        # 公式要求的——CD_kw 用于 F1 判据时不带这个权重，两者不是同一个量，
        # 不能合并）。
        CD_omega = 2.0 * rho * (1.0 - F1) * self.sigma_w2 / omega_safe * grad_dot_product

        # ω 方程总源项
        S_omega = P_omega - D_omega + CD_omega

        # 源项 NaN/Inf 隔离：退化网格上 grad_k·grad_omega 等可能为 inf，
        # 导致 inf-inf=NaN 传播。将非有限源项归零。
        Sk = np.where(np.isfinite(Sk), Sk, 0.0)
        S_omega = np.where(np.isfinite(S_omega), S_omega, 0.0)

        return Sk, S_omega
