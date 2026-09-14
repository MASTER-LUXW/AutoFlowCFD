"""
AutoFlowCFD V2.0 - GPU 版时间积分

与 core/time_integration.py 对应的 CuPy 版本。
所有操作在 GPU 上完成，数据常驻显存，避免 CPU↔GPU 传输。

包含：
- GPU 版 SSP-RK2/RK3 Shu-Osher stage 推进
- GPU 版前向 Euler
- GPU 版正定性强制（rho>0, p>0）
- GPU 版局部 CFL 时间步长计算

设计：
- 所有数组都是 CuPy 数组，常驻 GPU
- 与 CPU 版公式完全一致，只是 np.* → cp.*
- 残差函数由 GPUFRSolver 提供，内部全程 GPU
"""

import numpy as np
from enum import Enum
from typing import Callable, Optional

from autoflowcfd.core.gpu import get_cupy

GAMMA = 1.4

# SSP-RK Shu-Osher 系数（与 CPU 版一致）
_SSP_RK2 = {
    "stages": 2,
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
    "forward_euler": _EULER,
    "ssp_rk2": _SSP_RK2,
    "ssp_rk3": _SSP_RK3,
}


def enforce_positivity_gpu(U, p_floor: float = 1.0):
    """GPU 版正定性强制。

    与 core/time_integration.py::enforce_positivity 公式完全一致。
    在 GPU 上原地修改 U。

    Args:
        U: CuPy 数组 (N, n_vars)，守恒变量
        p_floor: 压力下限

    Returns:
        U: 修改后的 CuPy 数组（同一引用）
    """
    cp = get_cupy()
    MAX_VELOCITY = 1e4

    rho = cp.maximum(U[:, 0], 1e-6)
    U[:, 0] = rho

    vel = U[:, 1:4] / rho[:, None]

    # 限幅速度
    vel_mag = cp.sqrt(cp.sum(vel**2, axis=1))
    clip_mask = vel_mag > MAX_VELOCITY
    if cp.any(clip_mask):
        clip_factor = MAX_VELOCITY / vel_mag[clip_mask]
        vel[clip_mask] *= clip_factor[:, None]
        U[clip_mask, 1:4] = rho[clip_mask, None] * vel[clip_mask]

    ke = 0.5 * rho * cp.sum(vel**2, axis=1)
    p = (GAMMA - 1.0) * (U[:, 4] - ke)
    low = p < p_floor
    if cp.any(low):
        U[low, 4] = p_floor / (GAMMA - 1.0) + ke[low]

    # 湍流量
    if U.shape[1] > 5:
        U[:, 5] = cp.maximum(U[:, 5], 0.0)
    if U.shape[1] > 6:
        U[:, 6] = cp.maximum(U[:, 6], 1e-8)

    return U


def compute_local_cfl_step_gpu(
    U, cell_volumes, owner_cell, neighbor_cell, is_boundary,
    normals, areas, cell_owner, cell_areas,
    cfl: float = 1.0, mu_eff=None, poly_order: int = 0,
    det_jacs_sp=None, metric_flux_scale_sp=None,
    mach_ref=None, return_physical_too: bool = False,
):
    """GPU 版局部 CFL 时间步长计算。

    dt_i = CFL * order_factor * V_i / sum_f (|u.n|+a) A_f

    谱半径用物理声速 a（2026-08-25 代码审查修复）：此前这里用
    Weiss-Smith 预处理声速 c_precond 替代物理声速，低马赫数下（Mach 0.1
    时 c_precond≈36 m/s vs a≈340 m/s）dt 被高估 ~10 倍，有效 CFL 远超
    SSP-RK3 稳定极限——与 CPU 侧 cfl.py::compute_local_time_step 的
    2026-08-24 修复完全同因（该次修复只改了 CPU 路径，GPU 路径被遗漏，
    此处把 GPU 拉回与 CPU 一致的物理声速）。原理：显式积分的是物理通量，
    稳定性由物理通量谱半径（|u_n|+a）决定；预处理只改善条件数，不改变
    稳定极限。order_factor 与 CPU 侧同一公式 1/(2p+1)。

    Args:
        U: CuPy 数组 (n_cells, 1, n_vars)——**必须是单 SP 切片**，见下方
            函数体开头的校验与说明
        cell_volumes: CuPy 数组 (n_cells,)
        owner_cell, neighbor_cell, is_boundary: 面连接关系
        normals: CuPy 数组 (n_faces, 3)
        areas: CuPy 数组 (n_faces,)
        cell_owner: 边界面 → cell 映射
        cell_areas: 边界面面积
        cfl: CFL 数
        mu_eff: 有效粘度（可选）
        poly_order: 当前多项式阶数，用于 1/(2p+1) 的阶数相关收紧，
            与 CPU 侧 cfl.py::compute_local_time_step 的
            order_factor_advective/order_factor_viscous 同一公式。
        det_jacs_sp: 本次调用对应 SP 的 |det(J)|，CuPy 数组 (n_cells,)
            （可选）。与 metric_flux_scale_sp 一起提供时，额外施加第三个
            几何/度量 CFL 限制——与 CPU 侧 cfl.py::compute_local_time_step
            的 dt_geometric 同一机制：坍缩坐标下同一单元内不同 SP 的
            det(J) 天然可以相差几百倍（Duffy 坍缩变换在 P>=2 时的固有
            性质），此前 GPU 路径完全没有这一限制，可能重新触发 CPU 侧
            已经修复过的同一类发散（项目记忆 "Tet collapsed-coord
            anisotropy"）。
        metric_flux_scale_sp: 本次调用对应 SP 的度量"通量面积"标度
            sum_m||adj(J)[:,m,:]||，CuPy 数组 (n_cells,)（可选，与
            det_jacs_sp 一起提供时才生效）。
        mach_ref: 不为 None 时，额外算一份**预处理后**的平均流 dt——
            把对流项与几何项里的声速换成有效声速 sqrt(beta^2)*a
            （beta^2 按速度模取，见 gpu_preconditioning.py）。这一份
            **只有在调用方真的把 Gamma 作用到平均流残差上时才可以使用**
            （见上方"谱半径用物理声速"那段记录的 2026-08-25 事故：
            只改 CFL 不改方程必然失稳）。粘性限制与声速无关，原样复用。
        return_physical_too: True 时返回 (dt_mean_flow, dt_physical)。
            湍流标量（k/omega）必须用后者——它们是被动输运量、不含声学
            模态，但其显式更新刻意没有 point-implicit 阻尼，保守起见不跟着
            放大步长（与 CPU 侧 cfl.py 同一处理）。

    Returns:
        return_physical_too=False（默认）: dt_local，CuPy 数组 (n_cells,)
        return_physical_too=True: (dt_mean_flow, dt_physical)；未启用
            预处理（mach_ref is None）时两者是同一个数组对象。
    """
    cp = get_cupy()
    n_cells = U.shape[0]

    # **本函数按约定只处理单个 SP**（2026-09-14 更正）：
    # 此前这里写"使用 SP0 的值计算时间步长（简化：取每个 cell 第一个
    # SP）"——那个标注已不成立。两条生产调用方（单机
    # `gpu_solver.py::_compute_local_time_step_gpu` 与多 GPU
    # `gpu_distributed.py` 同名方法）都在**逐 SP 循环**里传
    # `U[:, sp:sp+1, :]` 这样的单 SP 切片、并对结果取单元内最小值，
    # 所以 `U[:, 0, ...]` 取到的就是当前那个 SP，不是"只用第一个 SP"
    # 的近似。（多 GPU 那条原先确实只传 SP0，那是真实简化，已在同一批
    # 改动里随该方法的重写一起修掉。）
    #
    # 下面这道校验把这条隐式契约变成显式失败：如果有人传完整的
    # (n_cells, n_sps, n_vars) 数组，旧代码会**静默**只用 SP0 算步长——
    # 逐 SP 的几何/度量 CFL 限制（dt_geometric，专门用来防坍缩坐标下
    # 同一单元内 det(J) 相差几百倍导致的局部刚性失稳）会因此大部分
    # 失效，而且不报任何错。宁可直接失败。
    if U.ndim != 3 or U.shape[1] != 1:
        raise ValueError(
            f"compute_local_cfl_step_gpu 期望单 SP 切片 (n_cells, 1, n_vars)，"
            f"实际收到 {tuple(U.shape)}——调用方必须按 SP 循环、逐 SP 调用"
            f"并对结果取单元内最小值（见 gpu_solver.py/gpu_distributed.py 的"
            f"_compute_local_time_step_gpu）。传完整数组会静默只用 SP0、"
            f"让逐 SP 的几何/度量 CFL 保护失效。")
    rho = cp.maximum(U[:, 0, 0], 1e-9)
    vel = U[:, 0, 1:4] / rho[:, None]
    ke = 0.5 * rho * cp.sum(vel**2, axis=1)
    p = cp.maximum((GAMMA - 1.0) * (U[:, 0, 4] - ke), 1.0)
    a = cp.sqrt(GAMMA * p / rho)

    # 阶数相关收紧（与 CPU 侧 cfl.py 同一公式）：显式 FR 格式的对流/
    # 粘性稳定极限随阶数衰减，对流 ~1/(2p+1)、粘性 ~1/(2p+1)^2。
    order_factor_advective = 1.0 / (2 * poly_order + 1)
    order_factor_viscous = order_factor_advective ** 2

    # 谱半径累加
    spectral = cp.zeros(n_cells, dtype=cp.float64)

    # 内部面贡献
    int_mask = ~is_boundary
    io = owner_cell[int_mask]
    ineigh = neighbor_cell[int_mask]
    n_int = normals[int_mask]
    a_int = areas[int_mask]

    un_o = cp.abs(cp.einsum('nd,nd->n', vel[io], n_int)) + a[io]
    un_n = cp.abs(cp.einsum('nd,nd->n', vel[ineigh], n_int)) + a[ineigh]

    cp.scatter_add(spectral, io, un_o * a_int)
    cp.scatter_add(spectral, ineigh, un_n * a_int)

    # 边界面贡献
    bnd_mask = is_boundary
    bo = owner_cell[bnd_mask]
    if bo.size > 0:
        n_b = normals[bnd_mask]
        a_b = areas[bnd_mask]
        un_b = cp.abs(cp.einsum('nd,nd->n', vel[bo], n_b)) + a[bo]
        cp.scatter_add(spectral, bo, un_b * a_b)

    spectral = cp.maximum(spectral, 1e-30)
    dt = cfl * order_factor_advective * cell_volumes / spectral

    # 粘性限制（阶数收紧与 CPU 侧 cfl.py 同一公式）
    if mu_eff is not None:
        Lc2 = cell_volumes ** (2.0 / 3.0)
        dt_visc = 0.25 * cfl * order_factor_viscous * rho * Lc2 / cp.maximum(mu_eff, 1e-30)
        dt = cp.minimum(dt, dt_visc)

    # 几何/度量 CFL 限制（与 CPU 侧 cfl.py::compute_local_time_step 的
    # dt_geometric 同一公式，见上方参数文档）：用该 SP 自己的 det(J) 当作
    # 局部"体积"，metric_flux_scale 当作局部"总通量面积"。
    if det_jacs_sp is not None and metric_flux_scale_sp is not None:
        wave_speed = cp.maximum(cp.sqrt(cp.sum(vel**2, axis=1)) + a, 1e-10)
        dt_geometric = cfl * cp.abs(det_jacs_sp) / cp.maximum(
            metric_flux_scale_sp * wave_speed, 1e-300
        )
        dt = cp.minimum(dt, dt_geometric)

    if mach_ref is None:
        return (dt, dt) if return_physical_too else dt

    # === 预处理后的平均流 dt（与 CPU 侧 cfl.py 的同名段落逐项对应）===
    from .gpu_preconditioning import preconditioned_sound_speed_gpu
    vel_mag = cp.sqrt(cp.sum(vel ** 2, axis=1))
    c_pre = preconditioned_sound_speed_gpu(vel_mag, a, float(mach_ref))

    spectral_p = cp.zeros(n_cells, dtype=cp.float64)
    un_o_p = cp.abs(cp.einsum('nd,nd->n', vel[io], n_int)) + c_pre[io]
    un_n_p = cp.abs(cp.einsum('nd,nd->n', vel[ineigh], n_int)) + c_pre[ineigh]
    cp.scatter_add(spectral_p, io, un_o_p * a_int)
    cp.scatter_add(spectral_p, ineigh, un_n_p * a_int)
    if bo.size > 0:
        un_b_p = cp.abs(cp.einsum('nd,nd->n', vel[bo], n_b)) + c_pre[bo]
        cp.scatter_add(spectral_p, bo, un_b_p * a_b)
    spectral_p = cp.maximum(spectral_p, 1e-30)
    dt_mean = cfl * order_factor_advective * cell_volumes / spectral_p

    if mu_eff is not None:
        Lc2 = cell_volumes ** (2.0 / 3.0)
        dt_mean = cp.minimum(
            dt_mean,
            0.25 * cfl * order_factor_viscous * rho * Lc2 / cp.maximum(mu_eff, 1e-30),
        )
    if det_jacs_sp is not None and metric_flux_scale_sp is not None:
        wave_speed_p = cp.maximum(vel_mag + c_pre, 1e-10)
        dt_mean = cp.minimum(
            dt_mean,
            cfl * cp.abs(det_jacs_sp) / cp.maximum(
                metric_flux_scale_sp * wave_speed_p, 1e-300),
        )

    return (dt_mean, dt) if return_physical_too else dt_mean


class GPUTimeIntegrator:
    """GPU 版显式 SSP Runge-Kutta 时间积分器。

    与 CPU 版 TimeIntegrator 接口一致，但所有操作在 GPU 上完成。

    Attributes:
        scheme: 时间积分方案名称
        n_steps: 已执行的步数
        current_time: 当前物理时间
    """

    def __init__(self, scheme: str = "ssp_rk3", cfl: float = 1.0):
        """初始化 GPU 时间积分器。

        Args:
            scheme: "forward_euler" / "ssp_rk2" / "ssp_rk3"
            cfl: CFL 数
        """
        self.scheme = scheme
        self.cfl = cfl
        self.n_steps = 0
        self.current_time = 0.0
        self._table = _SCHEME_TABLE.get(scheme, _EULER)

    def step(
        self,
        solution,
        residual_func,
        dt_local,
        p_floor: float = 1.0,
        residual0=None,
        filter_func=None,
    ):
        """执行一个时间步（GPU 版 SSP-RK / DUAL_TIME / IMEX）。

        Args:
            solution: CuPy 数组 (N, n_vars) 当前解
            residual_func: 残差函数 R(U)，返回 CuPy 数组 (N, n_vars)
            dt_local: CuPy 数组 (N,) 局部时间步长
            p_floor: 压力下限
            residual0: 预计算的初始残差（可选）
            filter_func: 可选的模态滤波回调函数

        Returns:
            U_new: CuPy 数组 (N, n_vars) 更新后的解
        """
        cp = get_cupy()

        if self.scheme == "imex_euler":
            raise ValueError(
                "IMEX_EULER scheme 需要拆分的显式(对流)/隐式(粘性+源项)残差函数，"
                "请直接调用 step_imex(solution, residual_explicit, residual_implicit, ...)，"
                "不要通过通用的 step(...) 入口"
            )
        elif self.scheme == "dual_time":
            raise ValueError(
                "DUAL_TIME scheme 需要 dt_physical/solution_prev，请直接调用 "
                "step_dual_time(...)，不要通过通用的 step(...) 入口"
            )
        else:
            # SSP-RK2/RK3 or Forward Euler
            return self._ssp_rk_stage_step_gpu(
                solution, residual_func, dt_local, p_floor, residual0, filter_func=filter_func
            )

    def _ssp_rk_stage_step_gpu(
        self,
        solution,
        residual_func,
        dt_local,
        p_floor: float = 1.0,
        residual0=None,
        table=None,
        filter_func=None,
    ):
        """GPU 版 SSP-RK2/RK3 的 Shu-Osher stage 推进本体。

        Args:
            table: 显式指定要用的 Shu-Osher 系数表；None 时用 self._table
        """
        cp = get_cupy()
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
        enforce_positivity_gpu(U_stage1, p_floor)
        if filter_func is not None:
            U_stage1 = filter_func(U_stage1)

        # FORWARD_EULER 只有 1 个 stage
        if n_stages == 1:
            self.n_steps += 1
            return U_stage1

        # 重新计算Stage 1的残差（关键：不能省略）
        L1 = -residual_func(U_stage1)

        # === Stage 2 ===
        # U^(2) = alpha[1,0]*U^0 + alpha[1,1]*U^(1) + beta[1]*dt*L(U^(1))
        U_stage2 = (alpha[1][0] * U0 +
                   alpha[1][1] * U_stage1 +
                   beta[1] * dt * L1)
        enforce_positivity_gpu(U_stage2, p_floor)
        if filter_func is not None:
            U_stage2 = filter_func(U_stage2)

        # 重新计算Stage 2的残差（关键：不能省略）
        L2 = -residual_func(U_stage2)

        # === Stage 3 (RK3) ===
        # U^(3) = alpha[2,0]*U^0 + alpha[2,1]*U^(1) + alpha[2,2]*U^(2) + beta[2]*dt*L(U^(2))
        if n_stages >= 3:
            U_stage3 = (alpha[2][0] * U0 +
                       alpha[2][1] * U_stage1 +
                       alpha[2][2] * U_stage2 +
                       beta[2] * dt * L2)
            enforce_positivity_gpu(U_stage3, p_floor)
            if filter_func is not None:
                U_stage3 = filter_func(U_stage3)

            # 对于RK3，最终解就是U^(3)
            U_new = U_stage3
        else:
            # 对于RK2，最终解是U^(2)
            U_new = U_stage2

        self.n_steps += 1
        return U_new

    def step_imex(
        self,
        solution,
        residual_explicit,
        residual_implicit,
        dt_local,
        p_floor: float = 1.0,
    ):
        """执行一步 GPU 版 IMEX Euler 推进。

        Args:
            solution: CuPy 数组 (N, n_vars)
            residual_explicit: 显式残差函数 R_exp(U)
            residual_implicit: 隐式残差函数 R_imp(U)
            dt_local: CuPy 数组 (N,) 局部时间步长
            p_floor: 压力下限

        Returns:
            U_new: CuPy 数组 (N, n_vars)
        """
        from autoflowcfd.core.gpu.gpu_time_integration_imex import step_imex_gpu
        return step_imex_gpu(self, solution, residual_explicit, residual_implicit, dt_local, p_floor)

    def step_dual_time(
        self,
        solution,
        spatial_residual,
        pseudo_dt,
        dt_physical: float,
        solution_prev=None,
        max_inner_iter: int = 5,
        tol: float = 1e-4,
        filter_func=None,
    ):
        """执行一步 GPU 版 Dual-Time Stepping。

        Args:
            solution: CuPy 数组 (N, n_vars) 物理时间层 n 的状态
            spatial_residual: 纯空间残差函数 R_spatial(U)
            pseudo_dt: CuPy 数组 (N,) 伪时间迭代用的局部步长
            dt_physical: 真正的物理时间步长（标量）
            solution_prev: CuPy 数组 (N, n_vars) 物理时间层 n-1 的状态
            max_inner_iter: 最大内层迭代次数
            tol: 绝对收敛容差
            filter_func: 可选的模态滤波回调函数

        Returns:
            U_tau: CuPy 数组 (N, n_vars) 收敛后的伪时间解
        """
        from autoflowcfd.core.gpu.gpu_time_integration_dual import step_dual_time_gpu
        return step_dual_time_gpu(
            self, solution, spatial_residual, pseudo_dt, dt_physical,
            solution_prev=solution_prev, max_inner_iter=max_inner_iter, tol=tol, filter_func=filter_func,
        )

    def reset(self):
        """重置积分器状态。"""
        self.n_steps = 0
        self.current_time = 0.0
