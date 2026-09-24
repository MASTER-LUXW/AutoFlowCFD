"""AutoFlowCFD V2.0 - SST 混合函数 F1/F2、应变率与涡量模、涡粘度

从 `src/autoflowcfd/core/turbulence/sst.py` 的 `SSTModelFR` 拆出（2026-09-24，项目「单文件不超
500 行」规范）。mixin 是本仓库既有惯例（`_SolverGeometryMixin`、
`_GPUSolverInitMixin` 等），沿用它而不是另发明一套。

**只含方法，没有状态**：全部属性由 `SSTModelFR` 的 `__init__` 建立，
这里通过 `self` 访问。
"""

import numpy as np
from .kernels import compute_strain_and_vorticity_magnitude


class _SSTBlendingMixin:
    """SST 混合函数 F1/F2、应变率与涡量模、涡粘度"""

    def compute_strain_rate_magnitude(self, grad_u: np.ndarray) -> np.ndarray:
        """
        计算应变率张量的模 |S|。

        Args:
            grad_u: 速度梯度张量，形状 (n_cells, n_sps, 3, 3)

        Returns:
            S_mag: 应变率模，形状 (n_cells, n_sps)
        """
        # 融合 kernel（性能优化 2026-09-13，见
        # `_strain_vorticity_magnitude_kernel` 文档：原实现先物化一份
        # 456MiB 的对称化张量再 einsum 收缩，两者都是单线程）。与原路径
        # 机器精度等价（~1e-16 相对误差，非逐位相同，见该 kernel 文档）。
        return compute_strain_and_vorticity_magnitude(grad_u)[0]

    def compute_vorticity_magnitude(self, grad_u: np.ndarray) -> np.ndarray:
        """计算涡量张量的模 |Ω|（用于 Kato-Launder 驻点修正）。

        Ω_ij = 0.5 * (∂u_i/∂x_j - ∂u_j/∂x_i)
        |Ω| = sqrt(2 * Ω_ij * Ω_ij)

        Args:
            grad_u: 速度梯度张量，形状 (n_cells, n_sps, 3, 3)

        Returns:
            Omega_mag: 涡量模，形状 (n_cells, n_sps)
        """
        # 同上融合 kernel（机器精度等价，~1e-16 相对误差）
        return compute_strain_and_vorticity_magnitude(grad_u)[1]

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
