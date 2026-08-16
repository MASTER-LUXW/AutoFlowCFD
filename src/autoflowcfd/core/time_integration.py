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
from typing import Callable, Optional, Tuple
from loguru import logger

from .low_mach_preconditioning import preconditioned_acoustic_eigs

GAMMA = 1.4


class TimeIntegrationScheme(Enum):
    """时间积分格式枚举。"""
    FORWARD_EULER = "forward_euler"
    SSP_RK2 = "ssp_rk2"
    SSP_RK3 = "ssp_rk3"
    IMEX_EULER = "imex_euler"  # 新增：一阶 IMEX
    DUAL_TIME = "dual_time"    # 新增：双时间步长
    
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
    if U.shape[1] > 5:
        U[:, 5] = np.maximum(U[:, 5], 0.0)      # rho*k >= 0
    if U.shape[1] > 6:
        U[:, 6] = np.maximum(U[:, 6], 1e-8)     # rho*omega > 0
    return U


class TimeIntegrator:
    """带局部时间步长的显式 SSP Runge-Kutta 积分器。"""

    def __init__(
        self,
        scheme: TimeIntegrationScheme = TimeIntegrationScheme.SSP_RK3,
        dt: float = 1e-5,
        cfl_target: float = 1.0,
        dual_time_steps: int = 20,  # 每个物理步内的伪时间迭代次数
        # 默认值从 3 提高到 20：真实测得 BDF1 单物理步（受控衰减算例，
        # dt_physical=0.05）在默认 cfl 起点=下限=0.1 的保守步长策略下，
        # 3 次内迭代只完成 28% 的目标收敛量，需要约 60 次才能收敛到
        # BDF1 精度（见 time_integration.py::step_dual_time 文档 cfl
        # 起点选择的说明）。20 不是"证明足够"的精确值，是在"默认值必须
        # 明显好于 3"与"不无谓拖慢每个物理步"之间的工程折衷，通过
        # --dual-time-inner-iter（CLI）/FRSolver(dual_time_inner_iter=...)
        # 暴露给需要更严格收敛的场景调整，而不是像此前那样完全没有
        # 途径设置。
    ):
        # 把任何旧别名映射到规范的枚举成员。
        self.scheme = TimeIntegrationScheme(scheme.value) if isinstance(scheme, TimeIntegrationScheme) \
            else TimeIntegrationScheme(scheme)
        self.dt = dt
        self.cfl_target = cfl_target
        self.dual_time_steps = dual_time_steps
        self.n_steps = 0
        self.current_time = 0.0
        self._table = _SCHEME_TABLE.get(self.scheme, _EULER)

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
        filter_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> np.ndarray:
        """根据配置的方案，推进一个时间步。
        
        对于显式SSP-RK格式，严格按照Shu-Osher形式实现三阶段计算，
        每个阶段都重新计算残差以确保时间精度。
        
        Args:
            solution: 当前解 U^n
            residual_func: 残差计算函数 R(U)
            dt_local: 局部时间步长数组
            p_floor: 压力下限
            residual0: 预计算的初始残差（可选优化）
            filter_func: 可选的模态滤波回调（见 core/fr_solver_filter.py），
                每个 RK stage 的正定性投影之后立即施加一次——不能只在最终
                组合结果上滤波一次：真实复现，坍缩坐标节点配置法的混叠
                噪声会在*中间* stage（Stage1/Stage2 各自重新求值残差时）
                就已经放大到 NaN，等不到最终组合完成，见
                _ssp_rk_stage_step 里各 stage 后的调用点

        Returns:
            U_new: 更新后的解 U^{n+1}
        """
        if self.scheme == TimeIntegrationScheme.IMEX_EULER:
            # IMEX 需要把残差拆成显式对流项/隐式粘性+源项两部分分别求值
            # （见 step_imex 文档），这个拆分只有调用方（fr_solver.py）知道
            # 怎么做——通用的单一 residual_func 接口表达不了。此前这里直接
            # 把同一个组合残差函数传了两遍，代数上退化成 total_res=2*R(U)，
            # 不是真正的 IMEX。做法与下面 DUAL_TIME 分支一致：拒绝走这条
            # 通用入口，调用方须直接调用 step_imex(...)。
            raise ValueError(
                "IMEX_EULER scheme 需要拆分的显式(对流)/隐式(粘性+源项)残差函数，"
                "请直接调用 step_imex(solution, residual_explicit, residual_implicit, ...)，"
                "不要通过通用的 step(...) 入口（该入口只有一个组合残差函数，无法拆分）"
            )

        elif self.scheme == TimeIntegrationScheme.DUAL_TIME:
            # DUAL_TIME 需要真正的物理时间步长 dt_physical 与上一物理时间层
            # 的解 solution_prev（BDF2 时间导数项必需，见 step_dual_time
            # 文档），这两个概念在这个通用 dt_local 数组接口里表达不了，
            # 调用方（fr_solver.py::step）须直接调用 step_dual_time，不能
            # 经过这个通用分发入口。
            raise ValueError(
                "DUAL_TIME scheme 需要 dt_physical/solution_prev，请直接调用 "
                "step_dual_time(...)，不要通过通用的 step(...) 入口"
            )

        else:
            U_new = self._ssp_rk_stage_step(
                solution, residual_func, dt_local, p_floor, residual0, filter_func=filter_func
            )
            self.n_steps += 1
            return U_new

    def _ssp_rk_stage_step(
        self,
        solution: np.ndarray,
        residual_func: Callable[[np.ndarray], np.ndarray],
        dt_local: np.ndarray,
        p_floor: float = 1.0,
        residual0: Optional[np.ndarray] = None,
        table: Optional[dict] = None,
        filter_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> np.ndarray:
        """SSP-RK2/RK3 的 Shu-Osher stage 推进本体，不含 scheme 分发/计步——
        从 step() 拆出来，供 step_dual_time 的内层伪时间迭代复用（见该方法
        文档：内层迭代此前用的是纯前向欧拉，但 dt_local/pseudo_dt 是按这套
        SSP-RK 格式的稳定性域标定的 CFL 步长，前向欧拉的稳定性域明显更小，
        直接复用同一个 CFL 数会失稳——真实复现：Couette 层流验证算例第一次
        内层迭代残差就从 5.5e6 暴涨到 2e27，两步内 NaN）。

        Args:
            table: 显式指定要用的 Shu-Osher 系数表；None 时用 self._table
                （即 self.scheme 对应的表）。step_dual_time 调用时必须显式
                传入 _SSP_RK3——self.scheme 此时是 DUAL_TIME 本身，
                self._table 会回退成 1-stage 的 _EULER 表（_SCHEME_TABLE
                里没有 DUAL_TIME 这个键），伪时间迭代想用的是 RK3，不是
                self.scheme 这个外层枚举。
        """
        tbl = table if table is not None else self._table
        alpha = tbl["alpha"]
        beta = tbl["beta"]
        n_stages = tbl["stages"]
        dt = dt_local[:, None]

        # Stage 0: 初始状态
        U0 = solution.copy()

        # 如果提供了预计算的残差，直接使用；否则计算
        if residual0 is not None:
            L0 = -residual0  # dU/dt = -R(U)
        else:
            L0 = -residual_func(U0)

        # === Stage 1 ===
        # U^(1) = U^0 + dt * L(U^0)
        U_stage1 = U0 + dt * L0
        enforce_positivity(U_stage1, p_floor)
        if filter_func is not None:
            U_stage1 = filter_func(U_stage1)

        # FORWARD_EULER 只有 1 个 stage：_EULER 表里 alpha 只有 alpha[0]，
        # 下面 Stage 2/3 无条件访问 alpha[1] 会越界 IndexError（已实测复现）。
        # 单级前向欧拉的结果就是 Stage 1 本身，直接返回。
        if n_stages == 1:
            return U_stage1

        # 重新计算Stage 1的残差（关键：不能省略）
        L1 = -residual_func(U_stage1)

        # === Stage 2 ===
        # U^(2) = alpha[1,0]*U^0 + alpha[1,1]*U^(1) + beta[1]*dt*L(U^(1))
        U_stage2 = (alpha[1][0] * U0 +
                   alpha[1][1] * U_stage1 +
                   beta[1] * dt * L1)
        enforce_positivity(U_stage2, p_floor)
        if filter_func is not None:
            U_stage2 = filter_func(U_stage2)

        # 重新计算Stage 2的残差（关键：不能省略）
        L2 = -residual_func(U_stage2)

        # === Stage 3 (如果是RK3) ===
        if n_stages >= 3:
            # U^(3) = alpha[2,0]*U^0 + alpha[2,1]*U^(1) + alpha[2,2]*U^(2) + beta[2]*dt*L(U^(2))
            U_stage3 = (alpha[2][0] * U0 +
                       alpha[2][1] * U_stage1 +
                       alpha[2][2] * U_stage2 +
                       beta[2] * dt * L2)
            enforce_positivity(U_stage3, p_floor)
            if filter_func is not None:
                U_stage3 = filter_func(U_stage3)

            # 对于RK3，最终解就是U^(3)
            U_new = U_stage3
        else:
            # 对于RK2，最终解是U^(2)
            U_new = U_stage2

        return U_new
    
    def step_imex(
        self,
        solution: np.ndarray,
        residual_explicit: Callable[[np.ndarray], np.ndarray],
        residual_implicit: Callable[[np.ndarray], np.ndarray],
        dt_local: np.ndarray,
        p_floor: float = 1.0,
    ) -> np.ndarray:
        """执行一步 IMEX Euler 推进 (S-05)。实现见
        time_integration_imex.py::step_imex（从本文件拆出，控制单文件
        行数），文档字符串也在那里。"""
        from .time_integration_imex import step_imex as _step_imex

        return _step_imex(self, solution, residual_explicit, residual_implicit, dt_local, p_floor=p_floor)

    def step_dual_time(
        self,
        solution: np.ndarray,
        spatial_residual: Callable[[np.ndarray], np.ndarray],
        pseudo_dt: np.ndarray,
        dt_physical: float,
        solution_prev: Optional[np.ndarray] = None,
        max_inner_iter: int = 5,
        tol: float = 1e-4,
        filter_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> np.ndarray:
        """执行一步 Dual-Time Stepping (S-05)。实现见
        time_integration_dual.py::step_dual_time（从本文件拆出，控制
        单文件行数），文档字符串也在那里。"""
        from .time_integration_dual import step_dual_time as _step_dual_time

        return _step_dual_time(
            self, solution, spatial_residual, pseudo_dt, dt_physical,
            solution_prev=solution_prev, max_inner_iter=max_inner_iter, tol=tol, filter_func=filter_func,
        )

    def reset(self) -> None:
        self.n_steps = 0
        self.current_time = 0.0
