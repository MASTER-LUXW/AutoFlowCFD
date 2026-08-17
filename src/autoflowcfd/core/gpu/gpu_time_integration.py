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
    cfl: float = 1.0, mu_eff=None,
):
    """GPU 版局部 CFL 时间步长计算。

    dt_i = CFL * V_i / sum_f (|u.n|+a) A_f

    Args:
        U: CuPy 数组 (n_cells, n_sps, n_vars)
        cell_volumes: CuPy 数组 (n_cells,)
        owner_cell, neighbor_cell, is_boundary: 面连接关系
        normals: CuPy 数组 (n_faces, 3)
        areas: CuPy 数组 (n_faces,)
        cell_owner: 边界面 → cell 映射
        cell_areas: 边界面面积
        cfl: CFL 数
        mu_eff: 有效粘度（可选）

    Returns:
        dt_local: CuPy 数组 (n_cells,)
    """
    cp = get_cupy()
    n_cells = U.shape[0]

    # 使用 SP0 的值计算时间步长（简化：取每个 cell 第一个 SP）
    rho = cp.maximum(U[:, 0, 0], 1e-9)
    vel = U[:, 0, 1:4] / rho[:, None]
    ke = 0.5 * rho * cp.sum(vel**2, axis=1)
    p = cp.maximum((GAMMA - 1.0) * (U[:, 0, 4] - ke), 1.0)
    a = cp.sqrt(GAMMA * p / rho)

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
    dt = cfl * cell_volumes / spectral

    # 粘性限制
    if mu_eff is not None:
        Lc2 = cell_volumes ** (2.0 / 3.0)
        dt_visc = 0.25 * cfl * rho * Lc2 / cp.maximum(mu_eff, 1e-30)
        dt = cp.minimum(dt, dt_visc)

    return dt


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
