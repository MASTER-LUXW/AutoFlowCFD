"""
AutoFlowCFD V2.0 - SST k-ω 湍流模型源项 GPU 化

将 SSTModelFR 的源项计算完全迁移到 GPU（CuPy），包括：
- 应变率模 |S| 计算
- Blending functions F1, F2
- 涡粘系数 ν_t 计算
- k 方程源项 Sk（产生项 - 耗散项）
- ω 方程源项 S_omega（产生项 - 耗散项 + 交叉扩散项）
- 正性保持限制器

设计：
- 与 CPU 版 SSTModelFR 保持相同接口语义
- 所有数组为 CuPy ndarray，数据常驻 GPU
- 支持 DES 长度尺度替换（与 CPU 版一致）

使用:
    from autoflowcfd.core.gpu.turbulence.gpu_turbulence_sst import GPUTurbulenceSST
    gpu_sst = GPUTurbulenceSST(n_cells, n_sps, device_id=0)
    Sk, S_omega = gpu_sst.compute_source_terms(Q_gpu, grad_U_gpu, d_wall_gpu, mu, grad_k_gpu, grad_omega_gpu)
"""

import numpy as np
from typing import Optional, Tuple

from autoflowcfd.core.gpu import gpu_available, get_cupy


class GPUTurbulenceSST:
    """GPU 版 SST k-ω 湍流模型。

    所有场变量存储在 GPU 上，源项计算全程在 GPU 完成。

    Attributes:
        n_cells: 单元数
        n_sps: 每单元解点数
        device_id: GPU 设备 ID
        k_field: 湍动能场 (GPU)
        omega_field: 比耗散率场 (GPU)
        nu_t: 涡粘系数场 (GPU)
    """

    def __init__(self, n_cells: int, n_sps: int, device_id: int = 0,
                 k_inf: float = 1e-6, omega_inf: float = 1.0):
        """初始化 GPU SST 模型。

        Args:
            n_cells: 单元数量
            n_sps: 每单元解点数
            device_id: GPU 设备 ID
            k_inf: 来流湍动能初值（默认 1e-6，工业标准由 Tu/VR 推导）
            omega_inf: 来流比耗散率初值（默认 1.0，工业标准由 Tu/VR 推导）
        """
        if not gpu_available:
            raise RuntimeError("CuPy required for GPU turbulence model")

        cp = get_cupy()
        self.n_cells = n_cells
        self.n_sps = n_sps
        self.device_id = device_id

        with cp.cuda.Device(device_id):
            # 初始化湍流场（工业标准：从 Tu/VR 推导物理自洽的 k/omega）
            self.k_field = cp.ones((n_cells, n_sps), dtype=cp.float64) * k_inf
            self.omega_field = cp.ones((n_cells, n_sps), dtype=cp.float64) * omega_inf
            self.nu_t = cp.zeros((n_cells, n_sps), dtype=cp.float64)

        # 来流 omega/k（持久属性）：与 CPU 版 sst.py 同一处真实 bug 修复
        # 同一个理由——P0 阶段 grad_vel 恒为零导致 S_mag 恒零，omega
        # realizability 下限若只依赖 S_mag 会完全失效，需要一个不依赖
        # 阶数的物理量纲下限；k_inf 供 apply_positivity_limiter_gpu 里
        # k 的同类下限使用（2026-09-11，见该处文档）。
        self.omega_inf = omega_inf
        self.k_inf = k_inf

        # SST 模型常数（与 CPU 版一致）
        self.sigma_k1 = 0.85
        self.sigma_k2 = 1.0
        self.sigma_w1 = 0.5
        self.sigma_w2 = 0.856
        self.beta1 = 0.075
        self.beta2 = 0.0828
        self.a1 = 0.31
        self.kappa = 0.41
        self.beta_star = 0.09
        self.CD_epsilon = 1e-10

        # DES 长度尺度（可选）
        self.des_length_scale: Optional['cp.ndarray'] = None

        # k/omega 物理上界（防止输运方程数值爆炸）。k_max 这里是占位保守值，
        # GPUFRSolver 初始化湍流模型后会按 CPU 同一公式覆盖为 0.5·vel_inf²
        # （见 gpu_solver.py，与 CPU 版 fr_solver/turbulence.py::
        # _set_turbulence_bounds 一致；2026-08-25 代码审查前此处恒为 1e6，
        # 与 CPU 惯例不一致且注释误称"与 CPU 版一致"）。
        self.k_max: float = 1e6
        self.omega_max: float = 1e6

        # 湍流产项渐变因子 [0, 1]，初值 1.0，每步由
        # gpu_solver_io.py::_update_production_ramp_gpu 更新（第四次评审
        # 修复：此前 GPU 路径没有任何代码更新这个值，恒为初始的 1.0，
        # 与 CPU 版 fr_solver/turbulence.py::_update_production_ramp 的
        # "从 0 线性爬升 50 步"行为不一致）。
        self.production_factor: float = 1.0

    def compute_strain_rate_magnitude_gpu(self, grad_u: 'cp.ndarray') -> 'cp.ndarray':
        """GPU 计算应变率张量模 |S|。

        Args:
            grad_u: 速度梯度 (n_cells, n_sps, 3, 3) CuPy 数组

        Returns:
            S_mag: 应变率模 (n_cells, n_sps)
        """
        cp = get_cupy()
        # S_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)
        S_ij = 0.5 * (grad_u + cp.transpose(grad_u, (0, 1, 3, 2)))
        # |S| = sqrt(2 * S_ij * S_ij)
        S_mag = cp.sqrt(2.0 * cp.sum(S_ij * S_ij, axis=(2, 3)))
        return S_mag

    def compute_vorticity_magnitude_gpu(self, grad_u: 'cp.ndarray') -> 'cp.ndarray':
        """GPU 计算涡量张量模 |Ω|（用于 Kato-Launder 驻点修正）。

        Ω_ij = 0.5 * (∂u_i/∂x_j - ∂u_j/∂x_i)
        |Ω| = sqrt(2 * Ω_ij * Ω_ij)
        """
        cp = get_cupy()
        W_ij = 0.5 * (grad_u - cp.transpose(grad_u, (0, 1, 3, 2)))
        Omega_mag = cp.sqrt(2.0 * cp.sum(W_ij * W_ij, axis=(2, 3)))
        return Omega_mag

    def compute_blending_F1_gpu(
        self, k: 'cp.ndarray', omega: 'cp.ndarray',
        d: 'cp.ndarray', nu: 'cp.ndarray',
        rho: 'cp.ndarray', CD_kw: 'cp.ndarray'
    ) -> 'cp.ndarray':
        """GPU 计算 SST blending function F1。

        arg1 = min[ max(sqrt(k)/(β*·ω·d), 500·ν/(d²·ω)), 4·ρ·σ_w2·k/(CD_kw·d²) ]
        F1 = tanh(arg1^4)
        """
        cp = get_cupy()
        omega = cp.maximum(omega, 1e-10)
        k = cp.maximum(k, 1e-10)
        d = cp.maximum(d, 1e-10)

        sqrt_k = cp.sqrt(k)
        term1 = sqrt_k / (self.beta_star * omega * d)
        term2 = 500.0 * nu / (d**2 * omega)
        term3 = 4.0 * rho * self.sigma_w2 * k / (cp.maximum(CD_kw, 1e-10) * d**2)

        arg1 = cp.minimum(cp.maximum(term1, term2), term3)
        # 防 overflow（与 CPU 版 sst.py::compute_blending_F1 逐字对应，
        # V2.0 专家组盲审发现 GPU 版此前缺这一步）：arg1**4 在
        # arg1>~1.3e154 时超 float64 上限；退化网格上 term3 中间量可
        # overflow 到 inf，先替换非有限值再 clip，tanh(大值)=1.0 物理
        # 正确（近壁 F1→1）。
        arg1 = cp.where(cp.isfinite(arg1), arg1, 1e75)
        arg1 = cp.minimum(arg1, 1e75)
        F1 = cp.tanh(arg1**4)
        return F1

    def compute_blending_F2_gpu(
        self, k: 'cp.ndarray', omega: 'cp.ndarray',
        d: 'cp.ndarray', nu: 'cp.ndarray'
    ) -> 'cp.ndarray':
        """GPU 计算 SST blending function F2。

        arg2 = max(2·sqrt(k)/(β*·ω·d), 500·ν/(d²·ω))
        F2 = tanh(arg2^2)
        """
        cp = get_cupy()
        omega = cp.maximum(omega, 1e-10)
        k = cp.maximum(k, 1e-10)
        d = cp.maximum(d, 1e-10)

        sqrt_k = cp.sqrt(k)
        term1 = 2.0 * sqrt_k / (self.beta_star * omega * d)
        term2 = 500.0 * nu / (d**2 * omega)
        arg2 = cp.maximum(term1, term2)
        # 防 overflow：同 F1 策略（与 CPU 版逐字对应）。
        arg2 = cp.where(cp.isfinite(arg2), arg2, 1e150)
        arg2 = cp.minimum(arg2, 1e150)
        F2 = cp.tanh(arg2**2)
        return F2

    # 湍流粘性比上限（与 CPU 版 SSTModelFR.TURBULENT_VISCOSITY_RATIO_MAX
    # 保持一致，见该类属性文档：主流 RANS 求解器标准安全阀，切断
    # k/omega 比值局部失控增长的反馈环）。
    TURBULENT_VISCOSITY_RATIO_MAX = 1.0e5

    def compute_eddy_viscosity_gpu(
        self, k: 'cp.ndarray', omega: 'cp.ndarray',
        rho: 'cp.ndarray', S_mag: 'cp.ndarray', F2: 'cp.ndarray', mu: float
    ) -> 'cp.ndarray':
        """GPU 计算涡粘系数 ν_t，并施加湍流粘性比上限。

        ν_t = a1 * k / max(a1*ω, F2*S)，再钳制到
        nu_t_max = TURBULENT_VISCOSITY_RATIO_MAX * mu / rho——与 CPU 版
        SSTModelFR.compute_eddy_viscosity 完全一致（见该方法文档：此前
        这里与 CPU 版一样只有一个与物理粘度脱钩的绝对值上限 nu_t<=1e6，
        对应粘性比高达 ~6.8e10，形同虚设，是 P2 SST 发散链条的一环，
        CPU 版已修复但此 GPU 版此前遗漏，GPU 上的 DES/LES 会重新触发
        同一个已被证实、已被修复的发散问题）。

        Args:
            mu: 分子动力粘度（用于换算粘性比上限对应的 nu_t 上限）
        """
        cp = get_cupy()
        k = cp.maximum(k, 1e-10)
        omega = cp.maximum(omega, 1e-10)
        S_mag = cp.maximum(S_mag, 1e-10)

        nu_t = self.a1 * k / cp.maximum(self.a1 * omega, F2 * S_mag)
        nu_t_max = self.TURBULENT_VISCOSITY_RATIO_MAX * mu / cp.maximum(rho, 1e-10)
        nu_t = cp.minimum(nu_t, nu_t_max)
        nu_t = cp.where(cp.isfinite(nu_t), nu_t, 0.0)
        return nu_t

    def compute_source_terms_gpu(
        self,
        Q: 'cp.ndarray',
        grad_U: 'cp.ndarray',
        d_wall: 'cp.ndarray',
        mu: float,
        grad_k: 'cp.ndarray',
        grad_omega: 'cp.ndarray',
    ) -> Tuple['cp.ndarray', 'cp.ndarray']:
        """GPU 计算 SST 源项。

        Args:
            Q: 原始变量场 (rho, u, v, w, p) (n_cells, n_sps, 5)
            grad_U: 速度梯度张量 (n_cells, n_sps, 3, 3)
            d_wall: 壁面距离 (n_cells, n_sps)
            mu: 动力粘度
            grad_k: k 梯度 (n_cells, n_sps, 3)
            grad_omega: omega 梯度 (n_cells, n_sps, 3)

        Returns:
            Sk: k 方程源项 (n_cells, n_sps)
            S_omega: omega 方程源项 (n_cells, n_sps)
        """
        cp = get_cupy()

        rho = Q[:, :, 0]
        nu = mu / cp.maximum(rho, 1e-10)

        # 应变率模
        S_mag = self.compute_strain_rate_magnitude_gpu(grad_U)

        # Kato-Launder 驻点修正（与 CPU 版一致）：
        # 产生项用 S*Ω 替代 S²，防止驻点区 k 非物理增长
        Omega_mag = self.compute_vorticity_magnitude_gpu(grad_U)
        S_omega_prod = S_mag * Omega_mag

        # 时间尺度 realization：动态 ω 下限。真实 bug 修复（2026-09-07，
        # 与 CPU 版 sst.py 同一处同一个真实bug——完整推导见该文件文档）：
        # P0 阶段 grad_vel/S_mag 恒为零，只用 S_mag 会让这个下限完全
        # 失效，加一个与阶数无关的物理量纲下限（来流 omega_inf 的保守
        # 比例），两者取更大值。
        # 逐点 realizability 下限，与 CPU 端 core/turbulence/sst.py 同一处
        # 2026-09-15 真实 bug 修复逐字对应（此前是全域标量 cp.max(S_mag)，
        # 会把整个 omega 场耦合到单个最差点上，真实网格实测导致发散）。
        self._omega_realizability_min = cp.maximum(0.1 * S_mag, 0.1 * self.omega_inf)

        # 交叉扩散项
        grad_dot = cp.sum(grad_k * grad_omega, axis=2)
        omega_safe = cp.maximum(self.omega_field, 1e-10)
        CD_kw = cp.maximum(
            2.0 * rho * self.sigma_w2 / omega_safe * grad_dot, 1e-10
        )

        # Blending functions
        F1 = self.compute_blending_F1_gpu(
            self.k_field, self.omega_field, d_wall, nu, rho, CD_kw
        )
        F2 = self.compute_blending_F2_gpu(
            self.k_field, self.omega_field, d_wall, nu
        )

        # Blending 常数
        sigma_k = F1 * self.sigma_k1 + (1.0 - F1) * self.sigma_k2
        sigma_w = F1 * self.sigma_w1 + (1.0 - F1) * self.sigma_w2
        beta = F1 * self.beta1 + (1.0 - F1) * self.beta2

        # 暂存本次求值用的混合 beta（供 update_fields_gpu 的半隐式阻尼
        # 使用，见该方法文档，与 CPU 版 SSTModelFR.compute_source_terms
        # 完全一致）。
        self._last_beta_blend = beta

        # 涡粘系数（传入 mu 以施加物理粘性比上限，见 compute_eddy_
        # viscosity_gpu 文档）
        self.nu_t = self.compute_eddy_viscosity_gpu(
            self.k_field, self.omega_field, rho, S_mag, F2, mu
        )

        # === k 方程源项 ===
        # 产生项: P_k = μ_t * S * Ω（Kato-Launder 修正）
        P_k = self.production_factor * self.nu_t * rho * S_omega_prod
        P_k = cp.minimum(P_k, 10.0 * self.beta_star * rho * self.k_field * omega_safe)

        if self.des_length_scale is not None:
            D_k = rho * self.k_field**1.5 / cp.maximum(self.des_length_scale, 1e-10)
        else:
            D_k = rho * self.beta_star * self.k_field * self.omega_field

        Sk = P_k - D_k

        # === ω 方程源项 ===
        gamma1 = self.beta1 / self.beta_star - self.sigma_w1 * self.kappa**2 / cp.sqrt(self.beta_star)
        gamma2 = self.beta2 / self.beta_star - self.sigma_w2 * self.kappa**2 / cp.sqrt(self.beta_star)
        gamma = F1 * gamma1 + (1.0 - F1) * gamma2

        # 产生项: P_ω = ρ * γ * S * Ω（Kato-Launder 修正）
        P_omega = self.production_factor * rho * gamma * S_omega_prod
        D_omega = rho * beta * self.omega_field**2
        CD_omega = 2.0 * rho * (1.0 - F1) * self.sigma_w2 / omega_safe * grad_dot

        S_omega = P_omega - D_omega + CD_omega

        # 最终 isfinite 归零（与 CPU 版 sst.py:402-403 逐字对应，V2.0
        # 专家组盲审发现 GPU 版此前缺这一步）：F1/F2 的 overflow 保护
        # 只清理了这两个混合函数本身，P_k/D_k/P_omega/D_omega 各自的
        # 中间量（如 CD_omega 里的 1/omega_safe）在极端退化网格上仍可能
        # 产生局部 NaN/Inf，若不在这里兜底，会被 update_fields_gpu 的
        # 半隐式阻尼放大后写入 k_field/omega_field。
        Sk = cp.where(cp.isfinite(Sk), Sk, 0.0)
        S_omega = cp.where(cp.isfinite(S_omega), S_omega, 0.0)

        return Sk, S_omega

    def apply_positivity_limiter_gpu(
        self, min_k: float = 1e-12, min_omega: float = 1e-12
    ):
        """GPU 正性保持限制器（含物理上界）。

        真实 bug 修复（V2.0 专家组盲审发现，2026-08-27）：此前这里没有
        NaN/Inf 恢复步骤——`cp.maximum(NaN, x)` 按 IEEE754 语义仍返回
        NaN（与 `np.maximum` 完全一样），一旦退化网格（棱柱侧面法向
        失配等，见 cube_demo 相关记录）在 GPU SST 源项计算里产生 NaN，
        会直接穿透这个限制器永久污染 k_field/omega_field，且限制器
        本身给不出任何提示——与 CPU 版 `sst.py::apply_positivity_limiter`
        的 NaN/Inf 恢复逻辑逐字对应，消除这个此前更彻底的失效模式。
        """
        cp = get_cupy()
        bad_k = ~cp.isfinite(self.k_field)
        bad_w = ~cp.isfinite(self.omega_field)
        self.k_field = cp.where(bad_k, min_k, self.k_field)
        self.omega_field = cp.where(bad_w, min_omega, self.omega_field)

        self.k_field = cp.maximum(self.k_field, min_k)
        self.omega_field = cp.maximum(self.omega_field, min_omega)
        self.k_field = cp.minimum(self.k_field, self.k_max)
        self.omega_field = cp.minimum(self.omega_field, self.omega_max)

        # k 的来流下限（与 CPU 版 sst.py::apply_positivity_limiter 同一处
        # 真实 bug 修复，2026-09-11，理由/量级选取见该处完整文档）。
        k_inf = getattr(self, 'k_inf', None)
        if k_inf is not None and k_inf > 0:
            self.k_field = cp.maximum(self.k_field, 1e-3 * k_inf)

        # 时间尺度 realization（与 CPU 版一致）
        if hasattr(self, '_omega_realizability_min'):
            self.omega_field = cp.maximum(self.omega_field, self._omega_realizability_min)

    def update_fields_gpu(
        self,
        dt,
        Sk: 'cp.ndarray',
        S_omega: 'cp.ndarray',
        transport_k: Optional['cp.ndarray'] = None,
        transport_omega: Optional['cp.ndarray'] = None,
    ):
        """GPU 湍流场时间更新，含源项半隐式阻尼（point-implicit
        destruction）——与 CPU 版 SSTModelFR.update_fields 完全一致
        （见该方法文档的推导）：纯显式更新 destruction 项
        D_omega=rho*beta*omega^2 这类逐点二次反应项在真实网格上会失稳，
        这是 CPU 版已确认、已修复的 P2 SST 发散根因之一，此 GPU 版此前
        遗漏同一处修复。

        Args:
            dt: 时间步长：标量（DUAL_TIME 的物理时间步），或可广播到
                `(n_cells, n_sps)` 的逐点局部步长（稳态加速，与 CPU
                `SSTModelFR.update_fields` 同一个量）
            Sk: k 方程源项
            S_omega: omega 方程源项
            transport_k: k 输运残差（可选）
            transport_omega: omega 输运残差（可选）
        """
        cp = get_cupy()

        beta_blend = getattr(self, "_last_beta_blend", None)
        if beta_blend is None:
            # 防御性回退，理由同 CPU 版：正常路径下 compute_source_terms_gpu
            # 总在 update_fields_gpu 之前被调用。
            beta_blend = self.beta2

        omega_old_safe = cp.maximum(self.omega_field, 1e-10)
        c_k = self.beta_star * omega_old_safe
        c_omega = beta_blend * omega_old_safe

        Sk_damped = Sk / (1.0 + dt * c_k)
        S_omega_damped = S_omega / (1.0 + dt * c_omega)
        Sk_damped = cp.where(cp.isfinite(Sk_damped), Sk_damped, 0.0)
        S_omega_damped = cp.where(cp.isfinite(S_omega_damped), S_omega_damped, 0.0)

        dk_total = Sk_damped
        domega_total = S_omega_damped

        if transport_k is not None:
            dk_total = dk_total + transport_k
        if transport_omega is not None:
            domega_total = domega_total + transport_omega

        self.k_field += dt * dk_total
        self.omega_field += dt * domega_total

        self.apply_positivity_limiter_gpu()

    def get_nu_t_cpu(self) -> np.ndarray:
        """获取涡粘系数（下载到 CPU）。"""
        cp = get_cupy()
        return cp.asnumpy(self.nu_t)

    def get_fields_cpu(self) -> dict:
        """获取湍流场（下载到 CPU）。"""
        cp = get_cupy()
        return {
            'k': cp.asnumpy(self.k_field),
            'omega': cp.asnumpy(self.omega_field),
            'nu_t': cp.asnumpy(self.nu_t),
        }

    def set_fields_from_cpu(self, k: np.ndarray, omega: np.ndarray):
        """从 CPU 设置湍流场（上传到 GPU）。"""
        cp = get_cupy()
        with cp.cuda.Device(self.device_id):
            self.k_field = cp.asarray(k)
            self.omega_field = cp.asarray(omega)

    def cleanup(self):
        """释放 GPU 资源。"""
        del self.k_field
        del self.omega_field
        del self.nu_t
        if self.des_length_scale is not None:
            del self.des_length_scale
