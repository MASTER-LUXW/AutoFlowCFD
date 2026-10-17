"""AutoFlowCFD V2.0 - SST 混合函数 F1/F2、应变率与涡量模、涡粘度（Menter 1994 标准式）

**只含方法，没有状态**：全部属性由 `SSTModelFR` 的 `__init__` 建立，这里通过 `self` 访问。
数组模块无关：数组模块取自入参（`core/utils/array_module.py`），CPU 与 GPU 模型共用这一份。
"""

import numpy as np

from autoflowcfd.core.turbulence.limits import TURBULENT_VISCOSITY_RATIO_MAX
from autoflowcfd.core.utils.array_module import array_module
from .kernels import compute_strain_and_vorticity_magnitude


class _SSTBlendingMixin:
    """SST 混合函数 F1/F2、应变率与涡量模、涡粘度"""

    def compute_strain_rate_magnitude(self, grad_u):
        """应变率模 |S| = sqrt(2 S_ij S_ij)，`grad_u (n_cells, n_sps, 3, 3)` -> `(n_cells, n_sps)`。"""
        return compute_strain_and_vorticity_magnitude(grad_u)[0]

    def compute_vorticity_magnitude(self, grad_u):
        """涡量模 |Omega| = sqrt(2 Omega_ij Omega_ij)（Kato-Launder 驻点修正用）。"""
        return compute_strain_and_vorticity_magnitude(grad_u)[1]

    def compute_blending_function_F1(self, k, omega, d, nu, rho, CD_kw):
        """F1 = tanh(arg1^4)，

            arg1 = min[ max( sqrt(k)/(beta* omega d), 500 nu/(d^2 omega) ), 4 rho sigma_w2 k/(CD_kw d^2) ]

        历史缺陷（ProjectFiles/V2.0/6_整体专家组二次评审.md T-01 发现16）：第一项曾写成倒数且用
        kappa 顶替 beta*，第三项（交叉扩散上界）缺失。

        Args:
            k, omega: 模型求值用的有效值（`bounds.model_evaluation_fields`）
            d: 壁距；nu: 运动粘度；rho: 密度
            CD_kw: max(2 rho sigma_w2/omega grad k . grad omega, 1e-10)
        """
        xp = array_module(k)
        omega = xp.maximum(omega, 1e-10)
        k = xp.maximum(k, 1e-10)
        d = xp.maximum(d, 1e-10)

        term1 = xp.sqrt(k) / (self.beta_star * omega * d)
        term2 = 500.0 * nu / (d ** 2 * omega)
        with np.errstate(over='ignore', invalid='ignore'):
            term3 = 4.0 * rho * self.sigma_w2 * k / (xp.maximum(CD_kw, 1e-10) * d ** 2)

        arg1 = xp.minimum(xp.maximum(term1, term2), term3)
        # arg1^4 在 arg1 > ~1.3e154 时溢出；退化网格上 term3 可能已是 inf。tanh(大值) = 1（近壁 F1 -> 1）
        arg1 = xp.where(xp.isfinite(arg1), arg1, 1e75)
        arg1 = xp.minimum(arg1, 1e75)
        return xp.tanh(arg1 ** 4)

    def compute_blending_function_F2(self, k, omega, d, nu):
        """F2 = tanh(arg2^2)，arg2 = max( 2 sqrt(k)/(beta* omega d), 500 nu/(d^2 omega) )。

        历史缺陷：同 F1 用 kappa 顶替 beta*，且缺 500 nu/(d^2 omega) 项（粘性子层内 F2 过早趋于 0，
        剪应力限制器 max(a1 omega, F2 S) 在边界层内失效）。
        """
        xp = array_module(k)
        omega = xp.maximum(omega, 1e-10)
        k = xp.maximum(k, 1e-10)
        d = xp.maximum(d, 1e-10)

        term1 = 2.0 * xp.sqrt(k) / (self.beta_star * omega * d)
        term2 = 500.0 * nu / (d ** 2 * omega)
        arg2 = xp.maximum(term1, term2)
        arg2 = xp.where(xp.isfinite(arg2), arg2, 1e150)
        arg2 = xp.minimum(arg2, 1e150)
        return xp.tanh(arg2 ** 2)

    # 湍流粘性比上限 mu_t/mu（主流 RANS 求解器的标准安全阀，Fluent/CFX 默认 1e5），切断
    # "k/omega 比值局部失控增长"的反馈环，见 `compute_eddy_viscosity`
    TURBULENT_VISCOSITY_RATIO_MAX = TURBULENT_VISCOSITY_RATIO_MAX

    def compute_eddy_viscosity(self, k, omega, rho, S_mag, F2, mu: float):
        """nu_t = a1 k / max(a1 omega, F2 S)，再施加粘性比上限 nu_t <= TVR_max mu / rho。

        上限的由来：Order Continuation 升阶后的冷启动瞬态里，个别解点 omega 被压到接近下限而 k
        未同步跌落，k/omega 巨大；此前只有与物理粘度脱钩的绝对上限 nu_t <= 1e6（粘性比 ~6.8e10，
        形同虚设），nu_t 放大让湍流扩散系数极度刚性，P2 阶段输运残差每步放大 3~4 个数量级
        （3.8e5 -> 1.5e6 -> 7.6e8 -> 2.6e12）。合成 Couette+SST 上换成物理粘性比上限后不再发散。

        Args:
            k, omega: 模型求值用的有效值；rho: 密度；S_mag: 应变率模；F2: 混合函数
            mu: 分子动力粘度
        """
        xp = array_module(k)
        k = xp.maximum(k, 1e-10)
        omega = xp.maximum(omega, 1e-10)
        S_mag = xp.maximum(S_mag, 1e-10)

        with np.errstate(over='ignore', invalid='ignore'):
            nu_t = self.a1 * k / xp.maximum(self.a1 * omega, F2 * S_mag)

        # mu_t/mu = nu_t/nu：逐解点按当地密度换算（密度变化大时不引入新的不一致）
        nu_t_max = self.TURBULENT_VISCOSITY_RATIO_MAX * mu / xp.maximum(rho, 1e-10)
        nu_t = xp.minimum(nu_t, nu_t_max)
        return xp.where(xp.isfinite(nu_t), nu_t, 0.0)
