"""ViscousRANSResidual 的无粘通量部分：MUSCL 重构 + AUSM+up（HLLC 备用）。

从 fvm_viscous_residual.py 拆出来的 mixin：`_inviscid_flux`（二阶 MUSCL
重构 + 逐面通量累加）、`_ausm_up`（当前实际使用的通量格式）、`_hllc`
（未接入主流程的参考实现，保留用于对比）。纯粹是为了控制单文件行数
拆出去的，不是独立的概念层——依赖宿主类 `ViscousRANSResidual` 已有的
`self.geom`/`self._use_gpu`/`self.mach_ref` 等属性，不独立维护状态。
"""

from __future__ import annotations

import numpy as np

from .fvm_gradients import green_gauss_gradient, barth_jespersen_limiter
from .fvm_inviscid_kernels import NUMBA_AVAILABLE, _ausm_up_flux_kernel
from .fvm_inviscid_kernels_gpu import CUDA_AVAILABLE, ausm_up_flux_gpu

GAMMA = 1.4

# AUSM+up 常数（Liou 2006, JCP 214:137-170, "A sequel to AUSM, Part
# II: AUSM+-up for all speeds"）。
_AUSM_KP = 0.25      # 速度-压力耦合系数
_AUSM_KU = 0.75      # 压力-速度耦合系数
_AUSM_SIGMA = 1.0    # Mp 截断系数
_AUSM_BETA = 1.0 / 8.0  # 马赫数分裂形状参数


class InviscidFluxMixin:
    """提供 `_inviscid_flux`/`_ausm_up`/`_hllc` 给 `ViscousRANSResidual`。"""

    # ------------------------------------------------------------------
    # 无粘通量：MUSCL 重构 + HLLC
    # ------------------------------------------------------------------
    def _inviscid_flux(self, U, boundary_states, flux_accum):
        geom = self.geom

        # --- 对原始变量做二阶重构 ---
        rho, vel, p, T, k, omega = self.to_primitive(U)
        prim = np.column_stack([rho, vel[:, 0], vel[:, 1], vel[:, 2], p, k, omega])

        # 边界条件原始变量，供梯度的边界贡献使用。
        bo = geom.bnd_owner
        prim_b = None
        if bo.size:
            rb, vb, pb, tb, kb, wb = self.to_primitive(boundary_states[geom.boundary_mask])
            prim_b = np.column_stack([rb, vb[:, 0], vb[:, 1], vb[:, 2], pb, kb, wb])

        grad = green_gauss_gradient(prim, geom, prim_b, use_gpu=self._use_gpu)
        phi = barth_jespersen_limiter(prim, grad, geom, use_gpu=self._use_gpu)
        grad_lim = grad * phi[:, :, None]

        # 重构到内部面中心。
        io, ineigh = geom.int_owner, geom.int_neigh
        fc = geom.centers[geom.internal_mask]
        rL = fc - geom.cell_centroids[io]
        rR = fc - geom.cell_centroids[ineigh]
        pL = prim[io] + np.einsum('nvd,nd->nv', grad_lim[io], rL)
        pR = prim[ineigh] + np.einsum('nvd,nd->nv', grad_lim[ineigh], rR)

        # 保证重构后的 rho、p、k、omega 为正。
        for col in (0, 4):
            pL[:, col] = np.maximum(pL[:, col], 1e-6)
            pR[:, col] = np.maximum(pR[:, col], 1e-6)
        pL[:, 5:] = np.maximum(pL[:, 5:], 0.0)
        pR[:, 5:] = np.maximum(pR[:, 5:], 0.0)

        n_int = geom.normals[geom.internal_mask]
        a_int = geom.areas[geom.internal_mask]
        f_int = self._ausm_up(pL, pR, n_int) * a_int[:, None]

        # R = (1/V) * sum_outward F.nA。面法向对 owner 是朝外的，对
        # neighbour 是朝内的，所以符号相反。
        np.add.at(flux_accum, io, f_int)
        np.add.at(flux_accum, ineigh, -f_int)

        # --- 边界面：一阶（owner 状态 vs ghost 状态）---
        if bo.size:
            pOwner = prim[bo]
            # ghost 原始变量已经算好，就是 prim_b
            n_b = geom.normals[geom.boundary_mask]
            a_b = geom.areas[geom.boundary_mask]
            f_b = self._ausm_up(pOwner, prim_b, n_b) * a_b[:, None]
            np.add.at(flux_accum, bo, f_b)

    def _ausm_up(self, primL: np.ndarray, primR: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """向量化的 AUSM+up 全速域通量（Liou 2006），用于 7 方程系统——
        当前实际使用的无粘通量，取代了 HLLC（保留在下方，未使用，供参考
        对比）。

        和 HLLC 那种基于波速区间的 Riemann 求解器不同——HLLC 的中间波速
        Sstar 一旦其 SL/SR 区间被低马赫数预处理人为收窄就会变得病态，
        具体失败情形见 _hllc 自己的注释——AUSM+up 是一种通量矢量分裂
        格式，由显式的界面马赫数和显式的低马赫数缩放函数 f_a 构建而成。
        它在自己的公式里就内建了正确的 O(M^2) 低马赫数压力-速度解耦，
        没有波速区间可被破坏，并且当地马赫数接近/超过 1 时能平滑退化为
        标准迎风格式（因此局部加速区域，例如车身尖锐边缘附近，也能被
        正确处理）。

        primL/primR 各列：[rho, u, v, w, p, k, omega]。
        返回通量数组 (n, 7)。
        """
        if self._use_gpu:
            # ⚠️ 未经真实 GPU 硬件验证，见 fvm_inviscid_kernels_gpu.py 模块文档字符串。
            return ausm_up_flux_gpu(
                np.ascontiguousarray(primL, dtype=np.float64),
                np.ascontiguousarray(primR, dtype=np.float64),
                np.ascontiguousarray(normal, dtype=np.float64),
                self.mach_ref,
            )
        if NUMBA_AVAILABLE:
            # Numba 加速路径——同样的 Liou 2006 公式，翻译成显式逐面循环
            # （见 fvm_inviscid_kernels.py 自己的模块文档字符串）。已在
            # 随机生成的亚/超声速状态上验证与下方 numpy 路径一致，绝对
            # 误差 ~1e-7（相对误差 ~1e-15，即 float64 机器精度）。
            return _ausm_up_flux_kernel(
                np.ascontiguousarray(primL, dtype=np.float64),
                np.ascontiguousarray(primR, dtype=np.float64),
                np.ascontiguousarray(normal, dtype=np.float64),
                self.mach_ref,
            )

        rhoL, uL, vL, wL, pL, kL, wkL = primL.T
        rhoR, uR, vR, wR, pR, kR, wkR = primR.T
        nx, ny, nz = normal[:, 0], normal[:, 1], normal[:, 2]

        # === 数值稳定性：与 _hllc 相同的限幅处理 ===
        MAX_VELOCITY = 1e4  # 10 km/s，物理上合理的上界
        vel_mag_L = np.sqrt(uL**2 + vL**2 + wL**2)
        vel_mag_R = np.sqrt(uR**2 + vR**2 + wR**2)
        clip_factor_L = np.minimum(1.0, MAX_VELOCITY / np.maximum(vel_mag_L, 1e-12))
        clip_factor_R = np.minimum(1.0, MAX_VELOCITY / np.maximum(vel_mag_R, 1e-12))
        uL = uL * clip_factor_L; vL = vL * clip_factor_L; wL = wL * clip_factor_L
        uR = uR * clip_factor_R; vR = vR * clip_factor_R; wR = wR * clip_factor_R

        rhoL = np.maximum(rhoL, 1e-9)
        rhoR = np.maximum(rhoR, 1e-9)
        pL = np.maximum(pL, 1.0)
        pR = np.maximum(pR, 1.0)

        unL = uL * nx + vL * ny + wL * nz
        unR = uR * nx + vR * ny + wR * nz

        EL = pL / (GAMMA - 1.0) + 0.5 * rhoL * (uL**2 + vL**2 + wL**2)
        ER = pR / (GAMMA - 1.0) + 0.5 * rhoR * (uR**2 + vR**2 + wR**2)
        MAX_ENERGY = 1e12
        EL = np.minimum(EL, MAX_ENERGY)
        ER = np.minimum(ER, MAX_ENERGY)
        HL = (EL + pL) / rhoL
        HR = (ER + pR) / rhoR

        # --- 界面（临界）声速，Liou 2006 式 30-33。
        # 保证压缩和膨胀两种情形下界面声速都表现一致、行为正确，而不是
        # 简单取平均。
        a_crit_L = np.sqrt(np.maximum(2.0 * (GAMMA - 1.0) / (GAMMA + 1.0) * HL, 1e-12))
        a_crit_R = np.sqrt(np.maximum(2.0 * (GAMMA - 1.0) / (GAMMA + 1.0) * HR, 1e-12))
        a_hat_L = a_crit_L**2 / np.maximum(a_crit_L, unL)
        a_hat_R = a_crit_R**2 / np.maximum(a_crit_R, -unR)
        a_half = np.maximum(np.minimum(a_hat_L, a_hat_R), 1e-6)

        ML = unL / a_half
        MR = unR / a_half

        # --- 低马赫数参考缩放函数 f_a（AUSM+up 里的 "up"）：用
        # self.mach_ref 做正则化，使 f_a 在真正的驻点（局部 Mbar -> 0）
        # 附近保持远离零，否则下面 Mp 里的 1/f_a 项会在那里发散。
        rho_half = 0.5 * (rhoL + rhoR)
        Mbar2 = (unL**2 + unR**2) / (2.0 * a_half**2)
        M0_2 = np.clip(np.maximum(Mbar2, self.mach_ref**2), 0.0, 1.0)
        f_a = np.maximum(np.sqrt(M0_2) * (2.0 - np.sqrt(M0_2)), 1e-6)
        alpha = 3.0 / 16.0 * (-4.0 + 5.0 * f_a**2)

        # --- 马赫数分裂多项式（Liou 2006 式 19、21）。---
        def M1_plus(M): return 0.5 * (M + np.abs(M))
        def M1_minus(M): return 0.5 * (M - np.abs(M))
        def M2_plus(M): return 0.25 * (M + 1.0) ** 2
        def M2_minus(M): return -0.25 * (M - 1.0) ** 2

        subL = np.abs(ML) < 1.0
        subR = np.abs(MR) < 1.0

        M4_plus = np.where(
            subL,
            M2_plus(ML) * (1.0 - 16.0 * _AUSM_BETA * M2_minus(ML)),
            M1_plus(ML),
        )
        M4_minus = np.where(
            subR,
            M2_minus(MR) * (1.0 + 16.0 * _AUSM_BETA * M2_plus(MR)),
            M1_minus(MR),
        )

        # --- 压力分裂多项式（Liou 2006 式 24）。|M|>=1 分支
        # （M1_plus/M）只会在 M!=0 时被 np.where 实际*选中*（该区域按定义
        # 排除了 M=0），但 np.where 会对每个元素都急切地计算两个分支——
        # 这里对除法做保护，避免 M=0 的元素（它们总是走另一条 5 阶分支）
        # 因为算出又立刻被丢弃的 0/0 而触发多余的 "invalid value in
        # divide" 警告。
        ML_safe = np.where(ML != 0.0, ML, 1.0)
        MR_safe = np.where(MR != 0.0, MR, 1.0)
        P5_plus = np.where(
            subL,
            M2_plus(ML) * ((2.0 - ML) - 16.0 * alpha * ML * M2_minus(ML)),
            M1_plus(ML) / ML_safe,
        )
        P5_minus = np.where(
            subR,
            M2_minus(MR) * ((-2.0 - MR) + 16.0 * alpha * MR * M2_plus(MR)),
            M1_minus(MR) / MR_safe,
        )

        # --- 速度-压力耦合（Liou 2006 式 8、15）——这正是 AUSM+up 相比
        # 普通迎风 AUSM 能给出正确低马赫数渐近行为的关键所在。---
        Mp = (-_AUSM_KP / f_a) * np.maximum(1.0 - _AUSM_SIGMA * Mbar2, 0.0) \
            * (pR - pL) / (rho_half * a_half ** 2)
        M_half = M4_plus + M4_minus + Mp

        pu = -_AUSM_KU * P5_plus * P5_minus * (rhoL + rhoR) * f_a * a_half * (unR - unL)
        p_half = P5_plus * pL + P5_minus * pR + pu

        # --- 质量通量与迎风对流量。---
        mdot = a_half * M_half * np.where(M_half > 0, rhoL, rhoR)

        pos = mdot >= 0
        u_up = np.where(pos, uL, uR)
        v_up = np.where(pos, vL, vR)
        w_up = np.where(pos, wL, wR)
        H_up = np.where(pos, HL, HR)
        k_up = np.where(pos, kL, kR)
        wk_up = np.where(pos, wkL, wkR)

        return np.column_stack([
            mdot,
            mdot * u_up + p_half * nx,
            mdot * v_up + p_half * ny,
            mdot * w_up + p_half * nz,
            mdot * H_up,
            mdot * k_up,
            mdot * wk_up,
        ])

    def _hllc(self, primL: np.ndarray, primR: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """向量化的 HLLC 通量，用于 7 方程系统。

        **不**在当前实际求解路径中使用（见 _ausm_up，它已取代此函数作为
        _inviscid_flux 的通量函数）——保留是为了参考对比，也因为它本身
        是一个独立正确的 Riemann 求解器（其通量一致性 F(U,U)=F(U) 依然
        成立）。为什么改用 AUSM+up 见 _ausm_up 的文档字符串和
        solver_steady.py 里 mach_ref 处的注释：HLLC 的 Sstar 计算对
        SL/SR 波速区间的任何收窄（低马赫数预处理就会这样做）都异常敏感，
        而 AUSM+up 的通量矢量分裂公式完全没有这个问题。

        primL/primR 各列：[rho, u, v, w, p, k, omega]。
        返回通量数组 (n, 7)。
        """
        rhoL, uL, vL, wL, pL, kL, wkL = primL.T
        rhoR, uR, vR, wR, pR, kR, wkR = primR.T
        nx, ny, nz = normal[:, 0], normal[:, 1], normal[:, 2]

        # === 数值稳定性：限幅速度以防止动能爆炸 ===
        MAX_VELOCITY = 1e4  # 10 km/s，物理上合理的上界
        vel_mag_L = np.sqrt(uL**2 + vL**2 + wL**2)
        vel_mag_R = np.sqrt(uR**2 + vR**2 + wR**2)

        clip_factor_L = np.minimum(1.0, MAX_VELOCITY / np.maximum(vel_mag_L, 1e-12))
        clip_factor_R = np.minimum(1.0, MAX_VELOCITY / np.maximum(vel_mag_R, 1e-12))

        uL *= clip_factor_L; vL *= clip_factor_L; wL *= clip_factor_L
        uR *= clip_factor_R; vR *= clip_factor_R; wR *= clip_factor_R

        # 保证密度和压力为正
        rhoL = np.maximum(rhoL, 1e-9)
        rhoR = np.maximum(rhoR, 1e-9)
        pL = np.maximum(pL, 1.0)
        pR = np.maximum(pR, 1.0)

        unL = uL * nx + vL * ny + wL * nz
        unR = uR * nx + vR * ny + wR * nz

        # 限幅声速，避免除零或极端值
        aL = np.sqrt(np.maximum(GAMMA * pL / rhoL, 1.0))
        aR = np.sqrt(np.maximum(GAMMA * pR / rhoR, 1.0))

        EL = pL / (GAMMA - 1.0) + 0.5 * rhoL * (uL**2 + vL**2 + wL**2)
        ER = pR / (GAMMA - 1.0) + 0.5 * rhoR * (uR**2 + vR**2 + wR**2)

        # 防止能量溢出
        MAX_ENERGY = 1e12
        EL = np.minimum(EL, MAX_ENERGY)
        ER = np.minimum(ER, MAX_ENERGY)

        # 波速估计（Davis / Einfeldt）——刻意**不**做低马赫数预处理，与
        # 伪时间步长的处理方式不同（见 TimeIntegrator.local_time_step）。
        # 曾经尝试过在这里对 SL/SR 做预处理，后来撤销了：HLLC 的中间波速
        # Sstar 是
        #   Sstar = (pR-pL + rhoL*unL*(SL-unL) - rhoR*unR*(SR-unR)) / denom
        #   denom = rhoL*(SL-unL) - rhoR*(SR-unR)
        # 用原始的声学 SL/SR，(SL-unL) 和 (SR-unR) 量级为 O(a)（空气中约
        # 340 m/s），给 `denom` 留出了相当的抵消余量。预处理会人为把
        # SL/SR 向 un 收缩（这正是低马赫数预处理的目的，为了 CFL/耗散上
        # 的好处）——但这会把这同一份余量在**整个**低速流场里（不只是
        # 驻点附近）按同一个因子（~beta，例如本算例 M~0.09 时约缩小
        # 10 倍）一起压缩，使 Sstar 的分母在全场都对噪声更加敏感。
        # 实测直接印证了这一点：启用后导致比不加预处理时快得多、范围
        # 广得多的数值爆炸（200 步以内速度就撞到 1e4 m/s 的限幅），符合
        # "这是真实的条件数退化，而非改进" 的判断——所以只对时间步长做
        # 预处理，通量自身的波速估计从不预处理。
        SL = np.minimum(unL - aL, unR - aR)
        SR = np.maximum(unL + aL, unR + aR)
        denom = rhoL * (SL - unL) - rhoR * (SR - unR)
        denom = np.where(np.abs(denom) < 1e-12, np.sign(denom) * 1e-12 + 1e-12, denom)
        Sstar = (pR - pL + rhoL * unL * (SL - unL) - rhoR * unR * (SR - unR)) / denom

        def phys_flux(rho, u, v, w, p, E, kk, wk, un):
            return np.column_stack([
                rho * un,
                rho * u * un + p * nx,
                rho * v * un + p * ny,
                rho * w * un + p * nz,
                (E + p) * un,
                rho * kk * un,
                rho * wk * un,
            ])

        FL = phys_flux(rhoL, uL, vL, wL, pL, EL, kL, wkL, unL)
        FR = phys_flux(rhoR, uR, vR, wR, pR, ER, kR, wkR, unR)

        UL = np.column_stack([rhoL, rhoL*uL, rhoL*vL, rhoL*wL, EL, rhoL*kL, rhoL*wkL])
        UR = np.column_stack([rhoR, rhoR*uR, rhoR*vR, rhoR*wR, ER, rhoR*kR, rhoR*wkR])

        def star_state(rho, u, v, w, p, E, kk, wk, un, S):
            # 保护 (S - Sstar) 分母（S 和 Sstar 可能重合）。
            dS = S - Sstar
            dS = np.where(np.abs(dS) < 1e-12, np.sign(dS) * 1e-12 + 1e-12, dS)
            factor = rho * (S - un) / dS
            Ustar = np.empty((len(rho), 7))
            Ustar[:, 0] = factor
            Ustar[:, 1] = factor * (u + (Sstar - un) * nx)
            Ustar[:, 2] = factor * (v + (Sstar - un) * ny)
            Ustar[:, 3] = factor * (w + (Sstar - un) * nz)
            # 用代数上已消去的 Toro 形式表示能量：p 项里的 (S-un) 会与
            # factor 相消，避免 S ~ un 时出现 0/0。
            Ustar[:, 4] = factor * (E / rho + (Sstar - un) * Sstar) \
                + (Sstar - un) * p / dS
            Ustar[:, 5] = factor * kk
            Ustar[:, 6] = factor * wk
            return Ustar

        F = np.empty_like(FL)
        # 区域选择。
        left = SL >= 0
        right = SR <= 0
        starL = (~left) & (~right) & (Sstar >= 0)
        starR = (~left) & (~right) & (Sstar < 0)

        F[left] = FL[left]
        F[right] = FR[right]
        if np.any(starL):
            UsL = star_state(rhoL, uL, vL, wL, pL, EL, kL, wkL, unL, SL)
            F[starL] = FL[starL] + SL[starL, None] * (UsL[starL] - UL[starL])
        if np.any(starR):
            UsR = star_state(rhoR, uR, vR, wR, pR, ER, kR, wkR, unR, SR)
            F[starR] = FR[starR] + SR[starR, None] * (UsR[starR] - UR[starR])
        return F
