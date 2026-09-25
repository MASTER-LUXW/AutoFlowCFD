"""AutoFlowCFD V2.0 - 正性限幅与场更新

从 `src/autoflowcfd/core/turbulence/sst.py` 的 `SSTModelFR` 拆出（2026-09-24，项目「单文件不超
500 行」规范）。mixin 是本仓库既有惯例（`_SolverGeometryMixin`、
`_GPUSolverInitMixin` 等），沿用它而不是另发明一套。

**只含方法，没有状态**：全部属性由 `SSTModelFR` 的 `__init__` 建立，
这里通过 `self` 访问。
"""

import numpy as np

from .bounds import clip_to_bounds


class _SSTUpdateMixin:
    """正性限幅与场更新"""

    def apply_positivity_limiter(self):
        """
        正性保持限制器 (T-02)：强制 k 和 omega 在物理合理范围内。

        下界：k, omega >= min（防止负值导致后续计算崩溃）
        上界：k <= k_max, omega <= omega_max（防止输运方程数值爆炸）

        上界的物理依据：
        - k_max = 0.5 * vel_inf^2：湍动能不可能超过平均流动能
        - omega_max：远大于任何工程壁面 omega 值的保守上界
        不设上界时，SST 源项+输运项的正反馈（P_k ∝ k，transport ∝ ∇k）
        会导致 k/omega 指数增长到 1e260+ 量级（实测 cube_demo 100 步内
        即达到此量级），而平均流完全不受影响（nu_t 被 SST a1 限幅保持
        合理），形成"平均流正常但湍流场完全发散"的隐蔽失效模式。

        区间的唯一定义（含 k 的来流下限、omega 的逐点 realizability 下限）见
        `bounds.py`；GPU 版与隐式 k-omega 的界约束读同一份。
        """
        clip_to_bounds(self, np)

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
