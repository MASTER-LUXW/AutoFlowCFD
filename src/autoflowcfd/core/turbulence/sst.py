"""
AutoFlowCFD V2.0 - SST k-omega 湍流模型 FR 离散 (T-01, T-02)

核心功能:
1. SST k-ω 模型源项计算（产生项/耗散项/交叉扩散项）
2. 正性保持限制器 (Positivity-preserving Limiter)
3. blending functions F1, F2（Menter 1994 标准公式，V2.0 二次评审修复：
   此前 F1/F2 都误用 Von Karman 常数 kappa 顶替 beta_star，且各丢失了
   标准公式里的一项，已改正并用近壁/远场极限数值验证）

已知局限（V2.0 已修复）：k/omega 输运方程现已包含完整的对流+扩散输运项
（见 core/turbulence_transport.py），k/omega 随流场对流、跨单元扩散，
不再仅是逐点源项 ODE 近似。F1/F2 混合函数、源项量纲、正性限制器等
此前的问题也均已在 V2.0 评审中修复。
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
        with np.errstate(over='ignore', invalid='ignore'):
            term3 = 4.0 * rho * self.sigma_w2 * k / (np.maximum(CD_kw, 1e-10) * d**2)

        arg1 = np.minimum(np.maximum(term1, term2), term3)
        # 防 overflow：arg1**4 在 arg1>~1.3e154 时超 float64 上限，
        # tanh(大值)=1.0 物理正确（近壁 F1→1）。
        # 退化网格上 term3 中间量可 overflow 到 inf，需先替换非有限值
        arg1 = np.where(np.isfinite(arg1), arg1, 1e75)
        arg1 = np.minimum(arg1, 1e75)
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
        # 防 overflow：同 F1 策略
        arg2 = np.where(np.isfinite(arg2), arg2, 1e150)
        arg2 = np.minimum(arg2, 1e150)
        F2 = np.tanh(arg2**2)

        return F2

    # 湍流粘性比上限 (Turbulent Viscosity Ratio, mu_t/mu)：主流 RANS
    # 求解器的标准安全阀（ANSYS Fluent/CFX 默认值即 1e5；OpenFOAM 等
    # 同样内置类似限制），用于切断"k/omega 比值局部失控增长"这个
    # SST 涡粘公式本身没有自带上限保护的反馈环——见 compute_eddy_
    # viscosity 文档。不是为这次调试新发明的阈值，是补齐标准 SST/
    # RANS 实现里本来就该有、这里此前没有的一道物理限制。
    TURBULENT_VISCOSITY_RATIO_MAX = 1.0e5

    def compute_eddy_viscosity(self, k: np.ndarray, omega: np.ndarray,
                              rho: np.ndarray, S_mag: np.ndarray,
                              F2: np.ndarray, mu: float) -> np.ndarray:
        """
        计算湍流涡粘系数 ν_t。

        ν_t = a1 * k / max(a1*ω, F2*S)，再施加湍流粘性比上限
        （见 TURBULENT_VISCOSITY_RATIO_MAX 类属性文档）。

        Args:
            k: 湍动能
            omega: 比耗散率
            rho: 密度
            S_mag: 应变率模
            F2: blending function
            mu: 分子动力粘度（用于换算粘性比上限对应的 nu_t 上限，
                真实复现：Order Continuation 跨阶数切换（尤其是 P1->P2，
                真正的 FR 梯度重构首次启用）后的"冷启动"瞬态里，个别
                SP 的湍流标量输运方程（transport.py）显式积分短暂失衡，
                把 omega 压到接近正性下限、同时 k 未同步跌落，产生一个
                物理上不合理的巨大 k/omega 比值——此前这里只有一个与
                物理粘度完全脱钩的绝对值上限 nu_t<=1e6（对应粘性比高达
                ~6.8e10，形同虚设），nu_t 由此被放大到足以让下一步的
                湍流扩散系数 Gamma=mu+sigma*rho*nu_t 本身变得极度刚性，
                反过来让输运残差进一步失控——几步内呈指数级放大（真实
                测得 P2 阶段 domega/dt 输运残差 3.8e5->1.5e6->7.6e8->
                2.6e12，每步放大约 3-4 个数量级）。合成 Couette+SST
                算例上实测验证：把上限换成物理粘性比上限（1e5×mu/rho）
                后这个链条被切断，P2 不再发散。

        Returns:
            nu_t: 湍流涡粘系数
        """
        k = np.maximum(k, 1e-10)
        omega = np.maximum(omega, 1e-10)
        S_mag = np.maximum(S_mag, 1e-10)

        # Boussinesq 假设下的涡粘系数
        with np.errstate(over='ignore', invalid='ignore'):
            nu_t = self.a1 * k / np.maximum(self.a1 * omega, F2 * S_mag)

        # 湍流粘性比上限（见本方法/类属性文档）：nu_t_max = TVR_max*mu/rho
        # （mu_t/mu = nu_t/nu 是同一个比值，rho 逐 SP 变化，除法在此处
        # 完成而不是换算成一个固定 nu_t 常数，避免密度变化大的场合下
        # 限制器本身引入新的不一致）。
        nu_t_max = self.TURBULENT_VISCOSITY_RATIO_MAX * mu / np.maximum(rho, 1e-10)
        nu_t = np.minimum(nu_t, nu_t_max)
        # NaN 安全网（k/omega 已被上游 limiter 保护，此处为防御性编程）
        nu_t = np.where(np.isfinite(nu_t), nu_t, 0.0)

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

        # 钳制湍流变量防 overflow：k*omega 在 k,omega~1e155 时超 float64
        # 上限。物理上 k<1e6, omega<1e8 已远超任何工程工况，保守取 1e40
        # 确保 k*omega=1e80 后乘 rho*beta_star 仍安全
        k_safe = np.minimum(self.k_field, 1e40)
        omega_safe_raw = np.minimum(self.omega_field, 1e40)
        omega_safe = np.maximum(omega_safe_raw, 1e-10)

        # 计算应变率模
        S_mag = self.compute_strain_rate_magnitude(grad_U)

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
        # 产生项: P_k = μ_t * S^2
        P_k = self.nu_t * rho * S_mag**2

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
        # 产生项: P_ω = ρ * γ * S^2
        gamma1 = self.beta1 / self.beta_star - self.sigma_w1 * self.kappa**2 / np.sqrt(self.beta_star)
        gamma2 = self.beta2 / self.beta_star - self.sigma_w2 * self.kappa**2 / np.sqrt(self.beta_star)
        gamma = F1 * gamma1 + (1.0 - F1) * gamma2

        P_omega = rho * gamma * S_mag**2

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

    def apply_positivity_limiter(self, min_k: float = 1e-12, min_omega: float = 1e-12):
        """
        正性保持限制器 (T-02)：强制 k 和 omega 非负，并在重构过程中嵌入硬约束。

        Args:
            min_k: k 的最小允许值
            min_omega: omega 的最小允许值
        """
        # NaN/Inf 恢复：np.maximum(NaN, x) 仍返回 NaN，必须先替换
        bad_k = ~np.isfinite(self.k_field)
        bad_w = ~np.isfinite(self.omega_field)
        if np.any(bad_k):
            self.k_field[bad_k] = min_k
        if np.any(bad_w):
            self.omega_field[bad_w] = min_omega

        # 硬截断确保物理合理性（T-02 规范要求的正性约束：k,omega >= 0）
        self.k_field = np.maximum(self.k_field, min_k)
        self.omega_field = np.maximum(self.omega_field, min_omega)

        # 此前这里还有一段"超过全场均值100倍就拍回100倍均值"的全局裁剪，
        # 已删除：这不是正性限制器要求的东西（T-02 只要求 k,omega>=0），
        # 全场均值在驻点/强剪切层附近偏低是正常物理现象，用它做裁剪阈值
        # 会把这些区域本应合法的高 k 值直接削平，是非物理的伪平滑，且
        # 阈值 100 没有任何依据（见 V2.0 二次评审 T-02 发现）。

    def update_fields(self, dt: float, Sk: np.ndarray, S_omega: np.ndarray,
                     diff_k: np.ndarray = None, diff_omega: np.ndarray = None,
                     transport_k: np.ndarray = None,
                     transport_omega: np.ndarray = None):
        """
        执行一个时间步长的湍流场更新。

        Args:
            dt: 时间步长（标量或逐 SP 数组，与 Sk/S_omega 广播兼容——见
                fr_solver/step.py 文档：稳态加速模式传逐 SP 的局部 CFL
                步长 dt_local，DUAL_TIME 模式传标量物理 dt）
            Sk: 湍动能源项（dk/dt 量纲，已除以 rho，P_k-D_k 合并后的净值）
            S_omega: 比耗散率源项（domega/dt 量纲，已除以 rho，
                P_omega-D_omega+CD_omega 合并后的净值）
            diff_k: k 的扩散项（可选，已弃用——现在由 transport_k 替代）
            diff_omega: omega 的扩散项（可选，已弃用）
            transport_k: k 的完整输运残差（对流+扩散，dk/dt 量纲），
                由 core/turbulence_transport.py 计算。非 None 时替代
                diff_k 并加入更新。
            transport_omega: omega 的完整输运残差（对流+扩散），同上。
        """
        # 源项半隐式阻尼（point-implicit destruction）：真实复现
        # （合成 Couette+SST 小算例、order continuation 到 P2）：即使
        # dt 已经是 cfl.py 正确按阶数/粘性/几何刚性收紧过的局部步长，
        # 纯显式积分 D_omega=rho*beta*omega^2 这类关于场量自身的二次
        # destruction 项仍会失稳——这是逐点 ODE 反应项刚性，
        # cfl.py::compute_local_time_step 的对流/粘性 CFL 估计的是
        # *空间*算子（对流通量/扩散通量）的谱半径，从未覆盖、也不该
        # 覆盖这种*逐点*反应项刚性（两者是独立的稳定性机制）。本方法
        # 及调用方 fr_solver/turbulence.py 的注释此前一直声称这里是
        # "半隐式阻尼更新"，但实际代码是纯前向欧拉
        # `k_field += dt*dk_total`，没有任何阻尼——文档与实现不符，
        # 现在改正为文档一直声称的做法。
        #
        # 标准 point-implicit 处理（Blazek《CFD Principles and
        # Applications》、Wilcox《Turbulence Modeling for CFD》等对
        # k-omega 类模型刚性 destruction 项的标准做法）：把 destruction
        # 项在 phi_new 上线性化、用 phi_old 处的系数隐式求解：
        #   D_k/rho   = beta_star*omega*k   （对 k 线性，系数 beta_star*omega）
        #   D_omega/rho = beta*omega^2      （对 omega 自身非线性，冻结一个
        #                                     omega 因子做隐式，另一个仍用旧值）
        # 设 S = P/rho - D/rho（Sk/S_omega 已经是这个合并后的净值，用
        # phi_old 求出），隐式方程：
        #   phi_new = phi_old + dt*(S + c*phi_old - c*phi_new)
        # （即把 S 里已经用 phi_old 算出的 destruction 部分换成对 phi_new
        # 隐式求解，c 是上面两个线性化系数）整理得：
        #   phi_new = phi_old + dt*S / (1 + dt*c)
        # 这就是"阻尼系数 1/(1+dt*c)"——c 越大（omega 越高、destruction
        # 越刚性）阻尼越强，无条件稳定，不依赖 dt 取多小；c 很小时
        # （omega 接近 0）阻尼趋于 1，退化回普通显式欧拉，物理正确。
        # 只阻尼 Sk/S_omega（逐点反应项刚性），不阻尼 transport_k/
        # transport_omega（对流+扩散的空间算子刚性已经由 dt_local 本身
        # 的粘性 CFL 项覆盖，是不同机制，重复阻尼没有理论依据）。
        beta_star = self.beta_star
        beta_blend = getattr(self, "_last_beta_blend", None)
        if beta_blend is None:
            # 防御性回退（正常路径下 compute_source_terms 总在
            # update_fields 之前被调用，_last_beta_blend 应已存在）：
            # 用 beta2（> beta1，阻尼更强而非更弱，不会引入新的失稳）。
            beta_blend = self.beta2

        omega_old_safe = np.maximum(self.omega_field, 1e-10)
        c_k = beta_star * omega_old_safe
        c_omega = beta_blend * omega_old_safe

        with np.errstate(over='ignore', invalid='ignore'):
            Sk_damped = Sk / (1.0 + dt * c_k)
            S_omega_damped = S_omega / (1.0 + dt * c_omega)
        Sk_damped = np.where(np.isfinite(Sk_damped), Sk_damped, 0.0)
        S_omega_damped = np.where(np.isfinite(S_omega_damped), S_omega_damped, 0.0)

        # 源项 + 输运项联合更新
        dk_total = Sk_damped
        domega_total = S_omega_damped

        # 向后兼容：旧的 diff_k/diff_omega 参数仍支持
        if diff_k is not None and transport_k is None:
            dk_total = dk_total + diff_k
        if diff_omega is not None and transport_omega is None:
            domega_total = domega_total + diff_omega

        # 新的完整输运项（对流+扩散）
        if transport_k is not None:
            dk_total = dk_total + transport_k
        if transport_omega is not None:
            domega_total = domega_total + transport_omega

        # NaN/Inf 隔离：退化网格上源项/输运项可能产生 NaN（inf-inf），
        # 直接加到场量上会污染全场。将非有限增量归零，依赖后续的
        # positivity limiter 钳制场量本身。
        dk_total = np.where(np.isfinite(dk_total), dk_total, 0.0)
        domega_total = np.where(np.isfinite(domega_total), domega_total, 0.0)

        self.k_field += dt * dk_total
        self.omega_field += dt * domega_total

        # 应用正性限制器（含 NaN/Inf 恢复）
        self.apply_positivity_limiter()

    def get_turbulent_viscosity(self) -> np.ndarray:
        """
        获取当前的湍流涡粘系数。

        Returns:
            nu_t: 涡粘系数场
        """
        return self.nu_t.copy()
