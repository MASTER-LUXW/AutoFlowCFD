"""ViscousRANSResidual 的粘性通量部分：应力/热传导/湍流扩散 + 壁面函数。

从 fvm_viscous_residual.py 拆出来的 mixin：边界面梯度修正
（`_boundary_face_grad`）、壁面函数（`_wall_tangential_velocity`/
`_wall_function_targets`/`wall_shear_stress`）、粘性通量本体
（`_viscous_flux`/`_stress_dot_normal`）。纯粹是为了控制单文件行数拆
出去的，不是独立的概念层——依赖宿主类 `ViscousRANSResidual` 已有的
`self.geom`/`self.mu_lam`/`self._bo`/`self._e_OB`/`self._bdist`/
`self._e_ON`/`self._dist`/`self._wall_face_mask_b`/`self.wall_distance`/
`self._use_gpu` 等属性，不独立维护状态。
"""

from __future__ import annotations

import numpy as np

from .fvm_gradients import green_gauss_gradient
from .fvm_inviscid_kernels import NUMBA_AVAILABLE
from .fvm_viscous_kernels import _viscous_internal_flux_kernel
from .fvm_viscous_kernels_gpu import viscous_internal_flux_gpu
from .fvm_inviscid_kernels_gpu import CUDA_AVAILABLE

GAMMA = 1.4
R_GAS = 287.058          # J/(kg K)，干空气气体常数
PRANDTL_LAMINAR = 0.72
PRANDTL_TURBULENT = 0.90
CP = GAMMA * R_GAS / (GAMMA - 1.0)

SST_SIGMA_K1 = 0.85
SST_SIGMA_W1 = 0.5
SST_BETA1 = 0.075
SST_BETA_STAR = 0.09
SST_KAPPA = 0.41

# Menter scalable/automatic 壁面处理常数（对数律壁面律）。E=9.8 是
# kappa=0.41 时光滑壁面对数律的截距；WALL_YPLUS_SWITCH=11.06 是与
# (kappa, E) 相容的、线性粘性底层曲线 u+=y+ 与对数律 u+=(1/kappa)ln(E y+)
# 的交点——低于它直接用底层公式，等于/高于它则用 Newton 迭代求解对数律。
# 这是标准的两段式 "scalable" 壁面函数切换（Menter），不是单一连续的
# Spalding 公式——实现/验证更简单，同时在其设计的 y+ 范围内仍能保持
# 网格无关性。
WALL_LOG_KAPPA = SST_KAPPA
WALL_LOG_E = 9.8
WALL_YPLUS_SWITCH = 11.06


class ViscousFluxMixin:
    """提供粘性通量与壁面函数相关方法给 `ViscousRANSResidual`。"""

    # ------------------------------------------------------------------
    # 边界面辅助函数（owner -> 面中心，以 ghost 状态为目标）
    # ------------------------------------------------------------------
    def _boundary_face_grad(self, cell_grad: np.ndarray, cell_val: np.ndarray,
                            ghost_val: np.ndarray) -> np.ndarray:
        """边界面上的单侧修正梯度。

        修正方式与内部面处理（见 ``_viscous_flux``）相同的过松弛修正，
        只是这里的"neighbour"是面中心处的 ghost/壁面值，而不是真实的
        相邻单元。

        Args:
            cell_grad: 逐单元梯度，标量场形状 (n_cells, 3)，矢量场形状
                (n_cells, ncomp, 3)。
            cell_val, ghost_val: 每个边界面上 owner 单元和 ghost 的值，
                形状 (n_bf,) / (n_bf,) 或 (n_bf, ncomp)。

        Returns:
            修正后的面梯度，尾部形状与 ``cell_grad`` 相同。
        """
        bo = self._bo
        g_owner = cell_grad[bo]
        d_val = ghost_val - cell_val
        if g_owner.ndim == 2:
            proj = np.einsum('nd,nd->n', g_owner, self._e_OB)
            corr = d_val / self._bdist - proj
            return g_owner + corr[:, None] * self._e_OB
        proj = np.einsum('nij,nj->ni', g_owner, self._e_OB)
        corr = d_val / self._bdist[:, None] - proj
        return g_owner + corr[:, :, None] * self._e_OB[:, None, :]

    def _wall_tangential_velocity(self, rho: np.ndarray, vel: np.ndarray,
                                  boundary_states: np.ndarray):
        """仅针对 WALL/GROUND 边界面，owner 单元相对壁面的速度分解出的
        切向分量（形状 (n_wall_faces, 3)），以及壁面切向单位向量和大小。

        壁面自身的速度通过面平均 (owner+ghost)/2 值恢复，而不需要单独
        访问 BoundaryConditionHandler 的爬升/地面速度状态：`_wall_bc`
        构造 ghost 时特意让这个平均值恰好等于给定的壁面速度（镜像构造），
        复用这一点比把壁面速度状态额外传进本类要简单。
        """
        geom = self.geom
        bo = self._bo
        wm = self._wall_face_mask_b
        n_wb = geom.normals[geom.boundary_mask][wm]
        rho_gw = np.maximum(boundary_states[geom.boundary_mask, 0][wm], 1e-9)
        vel_ghost_w = boundary_states[geom.boundary_mask, 1:4][wm] / rho_gw[:, None]
        vel_owner_w = vel[bo][wm]
        vel_wall = 0.5 * (vel_owner_w + vel_ghost_w)
        vel_rel = vel_owner_w - vel_wall
        un_rel = np.einsum('nd,nd->n', vel_rel, n_wb)
        vel_tang = vel_rel - un_rel[:, None] * n_wb
        tang_mag = np.maximum(np.linalg.norm(vel_tang, axis=1), 1e-8)
        tang_dir = vel_tang / tang_mag[:, None]
        return tang_dir, tang_mag

    def _wall_function_targets(self, rho_owner: np.ndarray, u_tang_mag: np.ndarray,
                               y_p: np.ndarray):
        """Menter scalable/automatic 壁面处理：摩擦速度 u_tau（在
        y+=WALL_YPLUS_SWITCH 处做层流底层/对数律切换求得），以及对应的
        壁面剪切大小 + 近壁 k/omega 目标值。

        Args:
            rho_owner: 每个壁面上 owner 单元的密度
            u_tang_mag: owner 单元相对壁面运动的切向速度大小（见
                _wall_tangential_velocity）
            y_p: owner 单元中心到壁面的距离

        Returns:
            (tau_w, k_wall, omega_wall)：各自形状 (n_wall_faces,)。
            tau_w 是大小（>=0）；调用方会把它施加在与切向相对速度相反
            的方向上。
        """
        rho_s = np.maximum(rho_owner, 1e-9)
        y_s = np.maximum(y_p, 1e-12)
        u_p = np.maximum(u_tang_mag, 1e-8)

        # 层流底层估计：tau_w = mu*u_p/y_p = rho*u_tau^2。
        u_tau_lam = np.sqrt(self.mu_lam * u_p / (rho_s * y_s))
        yplus_lam = rho_s * u_tau_lam * y_s / self.mu_lam

        # 对数律分支：Newton 迭代求解 u_tau*[(1/kappa)ln(E*y+)] = u_p，
        # 其中 y+ = rho*u_tau*y_p/mu（两个因子都依赖 u_tau）。
        u_tau_log = np.maximum(u_tau_lam, 1e-8)
        for _ in range(8):
            yplus = np.maximum(rho_s * u_tau_log * y_s / self.mu_lam, 1e-3)
            f = u_tau_log * (1.0 / WALL_LOG_KAPPA) * np.log(WALL_LOG_E * yplus) - u_p
            dfdu = (1.0 / WALL_LOG_KAPPA) * (np.log(WALL_LOG_E * yplus) + 1.0)
            dfdu = np.where(np.abs(dfdu) < 1e-8, 1e-8, dfdu)
            u_tau_log = np.maximum(u_tau_log - f / dfdu, 1e-8)

        u_tau = np.where(yplus_lam <= WALL_YPLUS_SWITCH, u_tau_lam, u_tau_log)
        u_tau = np.maximum(u_tau, 1e-8)

        tau_w = rho_s * u_tau ** 2

        # 混合 omega（Menter）：sqrt(omega_viscous^2 + omega_log^2)——
        # 在近壁渐近值（小 y+）和对数律值（大 y+）之间平滑过渡，不用硬
        # 切换。
        omega_vis = 6.0 * self.mu_lam / (rho_s * SST_BETA1 * y_s ** 2)
        omega_log = u_tau / (np.sqrt(SST_BETA_STAR) * SST_KAPPA * y_s)
        omega_wall = np.sqrt(omega_vis ** 2 + omega_log ** 2)

        k_wall = u_tau ** 2 / np.sqrt(SST_BETA_STAR)

        return tau_w, k_wall, omega_wall

    def wall_shear_stress(self, rho: np.ndarray, vel: np.ndarray, mu_t: np.ndarray,
                          grad_vel: np.ndarray, boundary_states: np.ndarray) -> np.ndarray:
        """粘性应力张量与外法向的点积，逐边界面：``tau . n``，形状
        (n_boundary_faces, 3)。

        由 :meth:`_viscous_flux`（残差自身的边界粘性项）与气动力积分
        （摩擦阻力）共用，保证两者用的是动量方程实际平衡的同一份力。

        对于构造时提供了壁面函数掩码（见 wall_face_mask）的 WALL/GROUND
        面，基于 CFD 解析梯度的应力会被 log-law 模型值（Menter scalable
        壁面处理）替换——解析梯度估计只有在第一层网格真正落在粘性底层
        （y+~1）时才准确；网格更粗时它会悄悄低估摩擦阻力而不报错，壁面
        函数值则无论实际 y+ 是多少都会做出修正。
        """
        geom = self.geom
        bo = self._bo
        if not bo.size:
            return np.zeros((0, 3))
        n_b = geom.normals[geom.boundary_mask]
        mu_eff = self.mu_lam + mu_t
        rho_b = np.maximum(boundary_states[geom.boundary_mask, 0], 1e-9)
        vel_ghost = boundary_states[geom.boundary_mask, 1:4] / rho_b[:, None]
        gv_face_b = self._boundary_face_grad(grad_vel, vel[bo], vel_ghost)
        tau_resolved = self._stress_dot_normal(gv_face_b, n_b, mu_eff[bo])

        wm = self._wall_face_mask_b
        if wm is None or not np.any(wm):
            return tau_resolved

        tang_dir, tang_mag = self._wall_tangential_velocity(rho, vel, boundary_states)
        y_p = self.wall_distance[bo][wm]
        tau_w_mag, _, _ = self._wall_function_targets(rho[bo][wm], tang_mag, y_p)

        tau_wf = tau_resolved.copy()
        # 壁面剪切力与流体相对壁面的切向运动方向相反（对流体的拖曳力），
        # 大小来自 log-law 模型。
        tau_wf[wm] = -tau_w_mag[:, None] * tang_dir
        return tau_wf

    # ------------------------------------------------------------------
    # 粘性通量
    # ------------------------------------------------------------------
    def _viscous_flux(self, rho, vel, T, k, omega, mu_t, grad_vel, grad_turb,
                      boundary_states, flux_accum, k_ghost_b, omega_ghost_b):
        geom = self.geom
        io, ineigh = geom.int_owner, geom.int_neigh
        n_int = geom.normals[geom.internal_mask]
        a_int = geom.areas[geom.internal_mask]

        mu_eff = self.mu_lam + mu_t                          # (n_cells,)

        # 温度梯度——下面内部面通量和更下面的边界部分都需要。
        gT = green_gauss_gradient(T[:, None], geom, use_gpu=self._use_gpu)[:, 0, :]    # (n_cells, 3)
        gk, gw = grad_turb[:, 0, :], grad_turb[:, 1, :]  # 各自 (n_cells, 3)

        if self._use_gpu:
            # ⚠️ 未经真实 GPU 硬件验证，见 fvm_viscous_kernels_gpu.py 模块文档字符串。
            fvisc = viscous_internal_flux_gpu(
                io, ineigh, a_int, n_int, self._e_ON, self._dist,
                np.ascontiguousarray(vel, dtype=np.float64),
                np.ascontiguousarray(grad_vel, dtype=np.float64),
                mu_eff, T, gT, mu_t, k, omega, gk, gw,
                self.mu_lam, CP, PRANDTL_LAMINAR, PRANDTL_TURBULENT,
                SST_SIGMA_K1, SST_SIGMA_W1,
            )
        elif NUMBA_AVAILABLE:
            # Numba 加速路径（应力张量 + 热传导 + 湍流扩散融合进一个
            # 逐面 kernel）——已在真实网格上验证与下方 numpy 路径一致，
            # 精度达到 float64 机器精度——见 fvm_viscous_kernels.py 自己
            # 的模块文档字符串（其中记录了开发过程中真实出现、被验证
            # 捕获并修复的一次并行 scratch buffer 竞争 bug）。
            fvisc = _viscous_internal_flux_kernel(
                io, ineigh, a_int, n_int, self._e_ON, self._dist,
                np.ascontiguousarray(vel, dtype=np.float64),
                np.ascontiguousarray(grad_vel, dtype=np.float64),
                mu_eff, T, gT, mu_t, k, omega, gk, gw,
                self.mu_lam, CP, PRANDTL_LAMINAR, PRANDTL_TURBULENT,
                SST_SIGMA_K1, SST_SIGMA_W1,
            )
        else:
            # 面平均梯度，沿单元连线方向做过松弛修正，提升在畸变网格上
            # 的稳健性。
            gvL, gvR = grad_vel[io], grad_vel[ineigh]
            gv_face = 0.5 * (gvL + gvR)
            # 方向导数修正
            dvel = vel[ineigh] - vel[io]                         # (nif, 3)
            proj = np.einsum('nij,nj->ni', gv_face, self._e_ON) # (nif, 3)
            corr = (dvel / self._dist[:, None] - proj)
            gv_face = gv_face + corr[:, :, None] * self._e_ON[:, None, :]

            mu_f = 0.5 * (mu_eff[io] + mu_eff[ineigh])

            tau_n = self._stress_dot_normal(gv_face, n_int, mu_f)   # (nif, 3)

            gT_face = 0.5 * (gT[io] + gT[ineigh])
            dT = T[ineigh] - T[io]
            gT_face = gT_face + (dT / self._dist - np.einsum('nd,nd->n', gT_face, self._e_ON))[:, None] * self._e_ON
            cond = CP * (self.mu_lam / PRANDTL_LAMINAR + 0.5*(mu_t[io]+mu_t[ineigh]) / PRANDTL_TURBULENT)
            qn = cond * np.einsum('nd,nd->n', gT_face, n_int)        # 热传导

            vel_face = 0.5 * (vel[io] + vel[ineigh])
            work = np.einsum('nd,nd->n', tau_n, vel_face)

            fvisc = np.zeros((len(io), 7))
            fvisc[:, 1:4] = tau_n
            fvisc[:, 4] = work + qn

            # 湍流量扩散：(mu + sigma*mu_t) grad(k 或 omega).n
            gk_face = 0.5*(gk[io]+gk[ineigh])
            gw_face = 0.5*(gw[io]+gw[ineigh])
            # sigma 用 F1 混合（面值近似）
            mut_f = 0.5*(mu_t[io]+mu_t[ineigh])
            diff_k = (self.mu_lam + SST_SIGMA_K1 * mut_f) * np.einsum('nd,nd->n', gk_face, n_int)
            diff_w = (self.mu_lam + SST_SIGMA_W1 * mut_f) * np.einsum('nd,nd->n', gw_face, n_int)
            fvisc[:, 5] = diff_k
            fvisc[:, 6] = diff_w

            fvisc *= a_int[:, None]

        # 粘性通量进入残差时符号与无粘通量相反：
        # R = (1/V)[sum F_inv.nA - sum F_visc.nA]。
        np.add.at(flux_accum, io, -fvisc)
        np.add.at(flux_accum, ineigh, fvisc)

        # --- 边界面：分子 + 湍流粘性通量 -----------
        # 以前这里完全缺失：没有这个边界贡献，固壁上的剪切应力、热传导、
        # 湍流扩散在动量/能量/k-omega 方程里都是零，摩擦阻力从未真正进入
        # 求解的方程组，尽管壁面 ghost 状态的速度镜像本来就是专门为此
        # 构造的（见 BoundaryConditionHandler._wall_bc 的文档字符串）。
        bo = self._bo
        if bo.size:
            n_b = geom.normals[geom.boundary_mask]
            a_b = geom.areas[geom.boundary_mask]

            rho_b = np.maximum(boundary_states[geom.boundary_mask, 0], 1e-9)
            vel_ghost = boundary_states[geom.boundary_mask, 1:4] / rho_b[:, None]
            tau_n_b = self.wall_shear_stress(rho, vel, mu_t, grad_vel, boundary_states)

            E_ghost = boundary_states[geom.boundary_mask, 4]
            ke_ghost = 0.5 * rho_b * np.sum(vel_ghost**2, axis=1)
            p_ghost = np.maximum((GAMMA - 1.0) * (E_ghost - ke_ghost), 1.0)
            T_ghost = p_ghost / (rho_b * R_GAS)
            gT_face_b = self._boundary_face_grad(gT, T[bo], T_ghost)
            cond_b = CP * (self.mu_lam / PRANDTL_LAMINAR + mu_t[bo] / PRANDTL_TURBULENT)
            qn_b = cond_b * np.einsum('nd,nd->n', gT_face_b, n_b)

            vel_face_b = 0.5 * (vel[bo] + vel_ghost)
            work_b = np.einsum('nd,nd->n', tau_n_b, vel_face_b)

            fvisc_b = np.zeros((len(bo), 7))
            fvisc_b[:, 1:4] = tau_n_b
            fvisc_b[:, 4] = work_b + qn_b

            # k_ghost_b/omega_ghost_b：支持壁面函数的边界 ghost 值，与
            # _turbulence_wall_ghost 共用（且由它统一计算一次）——为什么
            # 必须和 _turbulence_gradient 用的是同一份数组、而不是各自
            # 独立算一份，见它自己的文档字符串。
            gk_face_b = self._boundary_face_grad(gk, k[bo], k_ghost_b)
            gw_face_b = self._boundary_face_grad(gw, omega[bo], omega_ghost_b)
            mut_b = mu_t[bo]
            diff_k_b = (self.mu_lam + SST_SIGMA_K1 * mut_b) * np.einsum('nd,nd->n', gk_face_b, n_b)
            diff_w_b = (self.mu_lam + SST_SIGMA_W1 * mut_b) * np.einsum('nd,nd->n', gw_face_b, n_b)
            fvisc_b[:, 5] = diff_k_b
            fvisc_b[:, 6] = diff_w_b

            fvisc_b *= a_b[:, None]
            np.add.at(flux_accum, bo, -fvisc_b)

    @staticmethod
    def _stress_dot_normal(grad_vel, normal, mu):
        """tau . n，其中 tau = mu(grad u + grad u^T - 2/3 div(u) I)。"""
        divu = grad_vel[:, 0, 0] + grad_vel[:, 1, 1] + grad_vel[:, 2, 2]
        tau = mu[:, None, None] * (grad_vel + np.transpose(grad_vel, (0, 2, 1)))
        # 对角线上减去 2/3 mu divu
        for i in range(3):
            tau[:, i, i] -= (2.0/3.0) * mu * divu
        return np.einsum('nij,nj->ni', tau, normal)
