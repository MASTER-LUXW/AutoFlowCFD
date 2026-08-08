"""ViscousRANSResidual 的 SST k-omega 湍流部分：涡粘性 + 源项。

从 fvm_viscous_residual.py 拆出来的 mixin：应变率张量（`_strain`）、涡
粘性（`_eddy_viscosity`）、F1 混合函数（`_f1_blend`）、production/
dissipation/cross-diffusion 源项（`_sst_sources`）。纯粹是为了控制单
文件行数拆出去的，不是独立的概念层——依赖宿主类 `ViscousRANSResidual`
已有的 `self.mu_lam`/`self.wall_distance`/`self._use_gpu` 等属性，不
独立维护状态。
"""

from __future__ import annotations

import numpy as np

from .fvm_inviscid_kernels import NUMBA_AVAILABLE
from .fvm_sst_kernels import _eddy_viscosity_kernel, _sst_sources_kernel
from .fvm_sst_kernels_gpu import eddy_viscosity_gpu, sst_sources_gpu
from .fvm_inviscid_kernels_gpu import CUDA_AVAILABLE

# SST k-omega 常数（Menter 2003）。
SST_A1 = 0.31
SST_BETA_STAR = 0.09
SST_KAPPA = 0.41
SST_SIGMA_K1, SST_SIGMA_K2 = 0.85, 1.0
SST_SIGMA_W1, SST_SIGMA_W2 = 0.5, 0.856
SST_BETA1, SST_BETA2 = 0.075, 0.0828
SST_GAMMA1 = SST_BETA1 / SST_BETA_STAR - SST_SIGMA_W1 * SST_KAPPA**2 / np.sqrt(SST_BETA_STAR)
SST_GAMMA2 = SST_BETA2 / SST_BETA_STAR - SST_SIGMA_W2 * SST_KAPPA**2 / np.sqrt(SST_BETA_STAR)


def _blend(f1, a1_val, a2_val):
    """SST 混合：f1*phi1 + (1-f1)*phi2。"""
    return f1 * a1_val + (1.0 - f1) * a2_val


class SSTSourceMixin:
    """提供应变率/涡粘性/F1混合/SST源项方法给 `ViscousRANSResidual`。"""

    def _strain(self, grad_vel):
        """对称应变率张量 S 及其大小 |S|=sqrt(2 SijSij)。"""
        S = 0.5 * (grad_vel + np.transpose(grad_vel, (0, 2, 1)))
        Smag = np.sqrt(2.0 * np.einsum('nij,nij->n', S, S) + 1e-30)
        return S, Smag

    # ------------------------------------------------------------------
    # SST 涡粘性  mu_t = rho a1 k / max(a1 omega, S F2)
    # ------------------------------------------------------------------
    def _eddy_viscosity(self, rho, k, omega, grad_vel):
        """SST 涡粘性：mu_t = rho*a1*k / max(a1*omega, |S|*F2)。"""
        if self._use_gpu:
            # ⚠️ 未经真实 GPU 硬件验证，见 fvm_sst_kernels_gpu.py 模块文档字符串。
            return eddy_viscosity_gpu(
                np.ascontiguousarray(rho, dtype=np.float64),
                np.ascontiguousarray(k, dtype=np.float64),
                np.ascontiguousarray(omega, dtype=np.float64),
                np.ascontiguousarray(grad_vel, dtype=np.float64),
                self.wall_distance, self.mu_lam,
            )
        if NUMBA_AVAILABLE:
            # Numba 加速路径，已验证与下方 numpy 路径完全一致（在真实
            # 网格上逐位相同）——见 fvm_sst_kernels.py 自己的模块文档
            # 字符串。
            return _eddy_viscosity_kernel(
                np.ascontiguousarray(rho, dtype=np.float64),
                np.ascontiguousarray(k, dtype=np.float64),
                np.ascontiguousarray(omega, dtype=np.float64),
                np.ascontiguousarray(grad_vel, dtype=np.float64),
                self.wall_distance, self.mu_lam,
            )

        _, Smag = self._strain(grad_vel)
        nu = self.mu_lam / rho
        d = self.wall_distance

        omega_safe = np.maximum(omega, 1e-8)  # omega 的物理下限 (1/s)

        arg2 = np.maximum(2.0 * np.sqrt(np.maximum(k, 0.0)) / (SST_BETA_STAR * omega_safe * d),
                          500.0 * nu / (d**2 * omega_safe))
        F2 = np.tanh(arg2**2)
        denom = np.maximum(SST_A1 * omega_safe, Smag * F2)
        mu_t = rho * SST_A1 * np.maximum(k, 0.0) / np.maximum(denom, 1e-12)
        return np.clip(mu_t, 0.0, 1e5 * self.mu_lam)

    def _f1_blend(self, rho, k, omega, grad_k, grad_omega):
        """SST F1 混合函数。"""
        d = self.wall_distance
        nu = self.mu_lam / rho

        # 关键修复：保护 CDkw 计算中的除零
        omega_safe = np.maximum(omega, 1e-8)  # omega 的物理下限 (1/s)

        CDkw = np.maximum(
            2.0 * rho * SST_SIGMA_W2 / omega_safe *
            np.einsum('nd,nd->n', grad_k, grad_omega),
            1e-10,
        )
        arg1 = np.minimum(
            np.maximum(np.sqrt(np.maximum(k, 0.0)) / (SST_BETA_STAR * omega_safe * d),
                       500.0 * nu / (d**2 * omega_safe)),
            4.0 * rho * SST_SIGMA_W2 * k / (CDkw * d**2),
        )
        return np.tanh(arg1**4), CDkw

    # ------------------------------------------------------------------
    # SST 源项
    # ------------------------------------------------------------------
    def _sst_sources(self, rho, k, omega, mu_t, grad_vel, grad_turb, residual):
        gk, gw = grad_turb[:, 0, :], grad_turb[:, 1, :]  # 各自 (n_cells, 3)

        if self._use_gpu:
            # ⚠️ 未经真实 GPU 硬件验证，见 fvm_sst_kernels_gpu.py 模块文档字符串。
            src = sst_sources_gpu(
                np.ascontiguousarray(rho, dtype=np.float64),
                np.ascontiguousarray(k, dtype=np.float64),
                np.ascontiguousarray(omega, dtype=np.float64),
                np.ascontiguousarray(mu_t, dtype=np.float64),
                np.ascontiguousarray(grad_vel, dtype=np.float64),
                np.ascontiguousarray(gk, dtype=np.float64),
                np.ascontiguousarray(gw, dtype=np.float64),
                self.wall_distance, self.mu_lam,
                SST_SIGMA_W1, SST_SIGMA_W2, SST_BETA1, SST_BETA2, SST_GAMMA1, SST_GAMMA2,
            )
            residual[:, 5] -= src[:, 0]
            residual[:, 6] -= src[:, 1]
            return

        if NUMBA_AVAILABLE:
            # Numba 加速路径（把 _strain/_f1_blend/production-dissipation-
            # cross-diffusion 融合进一个逐单元 kernel）——已在随机生成的
            # 单元状态上验证与下方 numpy 路径一致到 ~1e-11（这是 float64
            # 累加顺序带来的噪声，不是真实差异）——见 fvm_sst_kernels.py
            # 自己的模块文档字符串。
            src = _sst_sources_kernel(
                np.ascontiguousarray(rho, dtype=np.float64),
                np.ascontiguousarray(k, dtype=np.float64),
                np.ascontiguousarray(omega, dtype=np.float64),
                np.ascontiguousarray(mu_t, dtype=np.float64),
                np.ascontiguousarray(grad_vel, dtype=np.float64),
                np.ascontiguousarray(gk, dtype=np.float64),
                np.ascontiguousarray(gw, dtype=np.float64),
                self.wall_distance, self.mu_lam,
                SST_SIGMA_W1, SST_SIGMA_W2, SST_BETA1, SST_BETA2, SST_GAMMA1, SST_GAMMA2,
            )
            residual[:, 5] -= src[:, 0]
            residual[:, 6] -= src[:, 1]
            return

        S, Smag = self._strain(grad_vel)

        F1, CDkw = self._f1_blend(rho, k, omega, gk, gw)

        beta = _blend(F1, SST_BETA1, SST_BETA2)
        gamma = _blend(F1, SST_GAMMA1, SST_GAMMA2)
        sigma_w = _blend(F1, SST_SIGMA_W1, SST_SIGMA_W2)

        # k 的产生项，限幅到 10*beta_star*rho*k*omega（Menter 限制器）。
        Pk = mu_t * Smag**2
        Pk = np.minimum(Pk, 10.0 * SST_BETA_STAR * rho * k * omega)
        Dk = SST_BETA_STAR * rho * k * omega

        Pw = gamma * rho * Smag**2  # 用应变率形式表示 = gamma/nu_t * Pk（mu_t=rho a1 k/...）
        Dw = beta * rho * omega**2

        # 关键修复：保护 cross-diffusion 项的除零。
        # omega -> 0 时该项会发散，导致数值发散
        omega_safe = np.maximum(omega, 1e-8)  # omega 的物理下限 (1/s)
        cross = 2.0 * (1.0 - F1) * rho * sigma_w / omega_safe * np.einsum('nd,nd->n', gk, gw)

        # 额外的安全措施：限幅 cross-diffusion，防止出现极端值
        max_cross = 10.0 * np.maximum(np.abs(Pw), np.abs(Dw))
        cross = np.clip(cross, -max_cross, max_cross)

        # residual 是 dU/dt = -R；源项以相反符号加进去（加到 U 上）。
        # 对守恒形式的 rho*k、rho*omega 方程：
        residual[:, 5] -= (Pk - Dk)
        residual[:, 6] -= (Pw - Dw + cross)
