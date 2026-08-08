"""伪时间稳态求解器与瞬态计算的时间积分格式。

稳态求解器在伪时间上推进解，直到残差趋于 0。为此我们采用显式的
强稳定性保持 Runge-Kutta 格式（SSP-RK2 / SSP-RK3）——这是有限体积
CFD 里标准、可证明正确的显式积分器，配合按对流+声速+粘性 CFL 条件
确定的**局部（逐单元）时间步长**。

这套实现取代了之前的版本，那个版本：(a) 自称"backward Euler"，实际
做的却是显式前向欧拉步；(b) RK2/AB3 用的是占位的残差历史；(c) 通过对
密度/速度/通量做硬幅值限幅来掩盖发散。现在物理正定性只在数学上确实
需要的地方（rho>0，p>0）才强制施加，用一个保持速度不变的*压力下限*
实现，发散会被报告出来，而不是被掩盖。
"""

from __future__ import annotations

import numpy as np
from enum import Enum
from typing import Callable, Optional
from loguru import logger

from .low_mach_preconditioning import preconditioned_acoustic_eigs

GAMMA = 1.4


class TimeIntegrationScheme(Enum):
    """显式伪时间积分格式。"""

    FORWARD_EULER = "forward_euler"
    SSP_RK2 = "ssp_rk2"
    SSP_RK3 = "ssp_rk3"
    # 保留旧别名，使已有的 config/测试仍能正常导入。
    BACKWARD_EULER = "forward_euler"
    RUNGE_KUTTA_2 = "ssp_rk2"
    ADAMS_BASHFORTH_3 = "ssp_rk3"


# SSP-RK Shu-Osher 系数：各阶段形如
#   u^(i) = sum_k alpha[i,k] u^(k) + beta[i] dt L(u^(i-1))
# 其中 L(u) = -R(u)。这里按格式分别存储各自的阶段系数表。
_SSP_RK2 = {
    "stages": 2,
    # u1 = u0 + dt L0 ;  u2 = 1/2 u0 + 1/2 (u1 + dt L1)
    "alpha": [[1.0], [0.5, 0.5]],
    "beta": [1.0, 0.5],
}
_SSP_RK3 = {
    "stages": 3,
    "alpha": [[1.0],
              [0.75, 0.25],
              [1.0/3.0, 0.0, 2.0/3.0]],
    "beta": [1.0, 0.25, 2.0/3.0],
}
_EULER = {"stages": 1, "alpha": [[1.0]], "beta": [1.0]}

_SCHEME_TABLE = {
    TimeIntegrationScheme.FORWARD_EULER: _EULER,
    TimeIntegrationScheme.SSP_RK2: _SSP_RK2,
    TimeIntegrationScheme.SSP_RK3: _SSP_RK3,
}


def enforce_positivity(U: np.ndarray, p_floor: float = 1.0) -> np.ndarray:
    """在一次时间步更新后，对守恒变量施加物理上的边界约束。

    把密度和压力投影到正的下限，同时保持速度不变。另外还会限幅速度
    大小，防止动能爆炸。
    """
    MAX_VELOCITY = 1e4  # 10 km/s 上界

    rho = np.maximum(U[:, 0], 1e-6)
    U[:, 0] = rho

    vel = U[:, 1:4] / rho[:, None]

    # 限幅速度大小，并把限幅后的动量写回 U——下面的 ke 必须从这个
    # 最终写回 U 的同一份 vel 推导，否则压力下限会针对一个与实际写回
    # 的动量不匹配的动能来计算（以前这里有一次重复的重新限幅——算出来
    # 了却从未写回 U——可能引入的正是这种虽然通常很小、但确实存在的
    # 物理不一致）。
    vel_mag = np.sqrt(np.sum(vel**2, axis=1))
    clip_mask = vel_mag > MAX_VELOCITY
    if np.any(clip_mask):
        clip_factor = MAX_VELOCITY / vel_mag[clip_mask]
        vel[clip_mask] *= clip_factor[:, None]
        U[clip_mask, 1:4] = (rho[clip_mask, None] * vel[clip_mask])

    ke = 0.5 * rho * np.sum(vel**2, axis=1)
    p = (GAMMA - 1.0) * (U[:, 4] - ke)
    low = p < p_floor
    if np.any(low):
        U[low, 4] = p_floor / (GAMMA - 1.0) + ke[low]
    U[:, 5] = np.maximum(U[:, 5], 0.0)      # rho*k >= 0
    U[:, 6] = np.maximum(U[:, 6], 1e-8)     # rho*omega > 0
    return U


class TimeIntegrator:
    """带局部时间步长的显式 SSP Runge-Kutta 积分器。"""

    def __init__(
        self,
        scheme: TimeIntegrationScheme = TimeIntegrationScheme.SSP_RK3,
        dt: float = 1e-5,
        cfl_target: float = 1.0,
    ):
        # 把任何旧别名映射到规范的枚举成员。
        self.scheme = TimeIntegrationScheme(scheme.value) if isinstance(scheme, TimeIntegrationScheme) \
            else TimeIntegrationScheme(scheme)
        self.dt = dt
        self.cfl_target = cfl_target
        self.n_steps = 0
        self.current_time = 0.0
        self._table = _SCHEME_TABLE[self.scheme]

    # ------------------------------------------------------------------
    def local_time_step(
        self, U: np.ndarray, geom, mu_eff: Optional[np.ndarray] = None,
        omega: Optional[np.ndarray] = None,
        mach_ref: Optional[float] = None,
    ) -> np.ndarray:
        """逐单元的稳定伪时间步长 dt_i = CFL * V_i / sum_f (|u.n|+a) A_f。

        提供 ``mu_eff`` 时额外施加粘性限制；提供 ``omega`` 时额外施加
        SST 湍流源项刚性限制。

        k/omega 的耗散项（Dk = beta_star*rho*k*omega，
        Dw = beta*rho*omega^2）是形如 dy/dt ~ -c*omega*y 的常微分方程，
        其显式欧拉稳定性上界是 dt < ~1/(c*omega)——与上面的对流/粘性限制
        完全独立。在近壁区域（或任何 omega 较大的地方——例如很薄的边界
        层单元，或者已经开始漂移的解），这个限制可能比前两者严格得多；
        没有它，即便对平均流场而言 CFL 很安全，湍流方程用的时间步长仍
        可能悄悄地过大——这是一种可信的失稳机制：残差表面上"卡住"很多
        迭代，实际上是一个从未被限制过的刚性模态，某一步突然发散。

        Args:
            mach_ref: 低马赫数预处理用的参考（自由来流）马赫数（见
                low_mach_preconditioning.py）。默认 None 表示直接用原始
                的物理声速 `a`，与之前的行为完全一致。给定时，会把
                `|u.n|+a` 这一声学贡献换成对应的预处理形式，在局部流速
                远低于声速时放松 CFL 限制——否则对任何真正低速（M << 1）
                的算例，声学与对流刚性的差异都会强制要求非常小的 CFL。
        """
        rho = np.maximum(U[:, 0], 1e-9)
        vel = U[:, 1:4] / rho[:, None]
        ke = 0.5 * rho * np.sum(vel**2, axis=1)
        p = np.maximum((GAMMA - 1.0) * (U[:, 4] - ke), 1.0)
        a = np.sqrt(GAMMA * p / rho)

        n_cells = geom.n_cells
        spectral = np.zeros(n_cells)

        owner = geom.owner
        neigh = geom.neigh
        normals = geom.normals
        areas = geom.areas
        bmask = geom.boundary_mask
        imask = geom.internal_mask

        def _face_spectral(cell_idx, n_face):
            """某个面上，某侧单元自身状态贡献的 |lambda|_max。"""
            un_signed = np.einsum('nd,nd->n', vel[cell_idx], n_face)
            if mach_ref is not None:
                lam_plus, lam_minus, _ = preconditioned_acoustic_eigs(
                    un_signed, a[cell_idx], mach_ref
                )
                return np.maximum(np.abs(lam_plus), np.abs(lam_minus))
            return np.abs(un_signed) + a[cell_idx]

        # 内部面对两侧单元都有贡献
        io, ineigh = geom.int_owner, geom.int_neigh
        n_int = normals[imask]
        a_int = areas[imask]
        un_o = _face_spectral(io, n_int)
        un_n = _face_spectral(ineigh, n_int)
        np.add.at(spectral, io, un_o * a_int)
        np.add.at(spectral, ineigh, un_n * a_int)

        # 边界面只贡献给 owner
        bo = geom.bnd_owner
        if bo.size:
            n_b = normals[bmask]
            a_b = areas[bmask]
            un_b = _face_spectral(bo, n_b)
            np.add.at(spectral, bo, un_b * a_b)

        spectral = np.maximum(spectral, 1e-30)
        dt = self.cfl_target * geom.cell_volumes / spectral

        if mu_eff is not None:
            # 粘性稳定性：dt_visc ~ CFL * rho V^{5/3} / mu
            Lc2 = geom.cell_volumes ** (2.0 / 3.0)
            dt_visc = 0.25 * self.cfl_target * rho * Lc2 / np.maximum(mu_eff, 1e-30)
            dt = np.minimum(dt, dt_visc)

        if omega is not None:
            # SST 源项刚性限制（见上方文档字符串）。用 beta_star（0.09）
            # 作为单一保守常数，因为它是 SST 三个耗散系数里最大的一个
            # （beta_star=0.09 > beta2=0.0828 > beta1=0.075），给出三者中
            # 最紧（最安全）的界限。
            SST_BETA_STAR = 0.09
            dt_turb = self.cfl_target / np.maximum(SST_BETA_STAR * omega, 1e-30)
            dt = np.minimum(dt, dt_turb)

        return dt

    # ------------------------------------------------------------------
    def step(
        self,
        solution: np.ndarray,
        residual_func: Callable[[np.ndarray], np.ndarray],
        dt_local: np.ndarray,
        p_floor: float = 1.0,
        residual0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """用配置好的 SSP-RK 格式推进一个伪时间步。

        Args:
            solution: 当前守恒状态 (n_cells, n_vars)。
            residual_func: 可调用对象 U -> R(U)，已经除以单元体积，即
                dU/dt = -R(U)。
            dt_local: 逐单元伪时间步长 (n_cells,)。
            p_floor: 正定性投影用的最小压力。
            residual0: 可选的预先算好的 R(solution)——这里每个格式的
                第 0 阶段都是 Ui=U0=solution，所以如果调用方已经有
                R(solution)（例如用于收敛监控），传进来可以避免把残差
                （MUSCL + HLLC + 粘性 + SST 源项——一次迭代里最贵的部分）
                再多算一遍，却得不到任何新信息。

        Returns:
            更新后的守恒状态。
        """
        alpha = self._table["alpha"]
        beta = self._table["beta"]
        dt = dt_local[:, None]

        U0 = solution
        stages = [U0]
        for i in range(self._table["stages"]):
            Ui = stages[-1]
            if i == 0 and residual0 is not None:
                L = -residual0                 # dU/dt，复用调用方的 R(U0)
            else:
                L = -residual_func(Ui)         # dU/dt
            combo = np.zeros_like(U0)
            for k, a_ik in enumerate(alpha[i]):
                if a_ik != 0.0:
                    combo += a_ik * stages[k]
            Unew = combo + beta[i] * dt * L
            Unew = enforce_positivity(Unew, p_floor)
            stages.append(Unew)

        self.n_steps += 1
        return stages[-1]

    def reset(self) -> None:
        self.n_steps = 0
        self.current_time = 0.0
