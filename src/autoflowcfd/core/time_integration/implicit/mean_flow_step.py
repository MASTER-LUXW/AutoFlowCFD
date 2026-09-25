"""AutoFlowCFD V2.0 - 平均流的一个 PTC-Newton-Krylov 步（全部后端共用）。

单机 CPU（`fr_solver/step.py`）与单机 GPU（`gpu/solver/gpu_solver/step.py`）
的 NEWTON_KRYLOV 分支都调用本文件的 `step_mean_flow_newton`；它们只提供
残差函数、状态、伪时间步长、归约对象与面相邻关系，隐式步本身的一切
（跨步状态、只解平均流 5 个变量、块 Jacobi 缓存、诊断日志）只有这一份。

## 一次 step() 做一个 Newton 步

外层 `solve()` 已经在做残差监控、自适应 CFL、checkpoint、Order
Continuation，Newton 外迭代放在它里面让这些机制原样生效（见 `jfnk.py`
模块文档）。

## 模态滤波与隐式步不能同时启用

模态滤波是显式 RK 逐 stage 的去噪手段；Newton 步里的"解"是线性系统
的解、没有 stage 的概念，在 Newton 步之后滤一次会改变被求解的不动点方程
本身（`R(U)=0` 变成另一个问题），让残差与收敛判据失去意义。默认档
`FILTER_MODE=off` 本来就不构造滤波回调；显式指定了非 off 档时这里明确
报错，不静默忽略。

## 未知量只有平均流 5 个守恒变量

湍流模型开启时状态向量带两个 k/omega 槽位（`U[..., 5:7]`），它们全仓库
无人读取、残差恒为零（2026-09-25 实测 inv/visc 残差这两列 max = 0.0），
k/omega 真正的值在湍流模型对象上、由隐式湍流步更新
（`fr_solver/turbulence/implicit.py`）。带着它们解会让 Krylov 向量、块
Jacobi 的块尺寸与装配次数都白白多出 7/5 倍（plate_demo P1+SST：196 次 vs
140 次残差求值）。

## 跨步状态（挂在求解器对象上，换阶时由 Order Continuation 清空）

    _newton_forcing         inexact-Newton 的 forcing term（Eisenstat-Walker）
    _newton_dtau_scale      PTC 的 dtau 缩放（步内缩档的剩余档数跨步延续）
    _newton_block_precond   单元块 Jacobi 缓存（冻结 J_cc，按判据重装配）
    _newton_last_info       上一步的诊断
"""

from typing import Callable

import numpy as np
from loguru import logger

from .block_jacobi import BlockJacobiCache
from .forcing import EisenstatWalkerForcing
from .jfnk import step_newton_krylov
from .reductions import LocalReductions

#: 平均流守恒变量个数（rho, rho*u, rho*v, rho*w, rho*E）
N_MEAN_FLOW_VARS = 5


class ResidualVariableSlice:
    """把 `(N, n_vars)` 的残差函数限制到前 `n` 个变量上，其余列固定在步前值。

    做成类而不是闭包，理由同 `jacobian_vector.py::MatrixFreeJacobian`（在
    整个 Krylov 求解期间存活，只持有需要的字段）。数组模块由状态决定。
    """

    __slots__ = ("_residual", "_full", "_n")

    def __init__(self, residual: Callable, u_full, n: int):
        self._residual = residual
        self._full = u_full.copy()
        self._n = n

    def __call__(self, u_sub):
        if self._n == self._full.shape[1]:
            return self._residual(u_sub)
        u = self._full.copy()
        u[:, :self._n] = u_sub
        return self._residual(u)[:, :self._n]


def newton_step_ok(info: dict) -> bool:
    """本步 Newton 是否被**完整**接受（SER 律据此区分物理暂态与方向不可信）。"""
    return info["theta"] >= 1.0 and info["n_dtau_cuts"] == 0


def step_mean_flow_newton(
    solver, residual: Callable, u_flat, dtau_flat, scales: np.ndarray, *,
    red: LocalReductions, cell_is_prism: np.ndarray, cell_colors: Callable[[], np.ndarray],
    order: int, filter_active: bool,
):
    """平均流的一个 PTC-Newton-Krylov 步，返回 `(U_new_flat, info)`。

    Args:
        solver: 持有跨步状态的求解器对象（见模块文档那四个属性）。
        residual: `R(U_flat) -> (N, n_vars)`（启用低马赫预处理时是 `Gamma R`）。
        u_flat: `(N, n_vars)` 当前状态（`red.xp` 上）。
        dtau_flat: `(N,)` 逐 SP 伪时间步长（天花板，见 `jfnk.py`）。
        scales: `(n_vars,)` 守恒变量参考量级（`residual_diagnostics._reference_scales`）。
        red: 全局归约（单进程 numpy / cupy，或分布式子类）。
        cell_is_prism: `(n_cells,)` 主机端布尔掩码（单机"棱柱在前"，分布式
            local 排列里棱柱/四面体交错，所以不能用"前 n_prism 个"表达）。
        cell_colors: 返回 `(n_cells,)` 块 Jacobi 着色（主机端 numpy），只在
            首次构造缓存时调用。单机：按面相邻关系贪心着色；分布式：全局
            一致着色里本 rank 那一段（同色单元跨 rank 也不相邻，见
            `block_jacobi.py::CellBlockJacobian` 文档）。
        order: 当前阶数（决定每类单元的真实解点数）。
        filter_active: 是否构造了模态滤波回调（为真时报错，见模块文档）。
    """
    from autoflowcfd.fr.native_padding import real_sps_per_cell

    if filter_active:
        raise ValueError(
            "NEWTON_KRYLOV（隐式稳态）与模态滤波不能同时启用："
            "滤波会改变被求解的不动点方程本身（R(U)=0 变成另一个"
            "问题），使残差与收敛判据失去意义。请用 "
            "AFCFD_FILTER_MODE=off（默认值），或改用显式格式。"
        )
    n_vars = u_flat.shape[1]
    n_mf = min(n_vars, N_MEAN_FLOW_VARS)
    if solver._newton_forcing is None:
        solver._newton_forcing = EisenstatWalkerForcing()
    if solver._newton_block_precond is None:
        cell_is_prism = np.asarray(cell_is_prism, dtype=bool)
        n_cells = cell_is_prism.size
        n_real_prism, n_real_tet = real_sps_per_cell(int(order))
        if u_flat.shape[0] % n_cells:
            raise ValueError(f"状态行数 {u_flat.shape[0]} 不是单元数 {n_cells} 的整数倍")
        solver._newton_block_precond = BlockJacobiCache(
            cell_is_prism=cell_is_prism, colors=cell_colors(),
            n_sps=u_flat.shape[0] // n_cells, n_real_prism=n_real_prism,
            n_real_tet=n_real_tet, n_var=n_mf, red=red)

    u_new_mf, info = step_newton_krylov(
        ResidualVariableSlice(residual, u_flat, n_mf), u_flat[:, :n_mf], dtau_flat,
        np.asarray(scales)[:n_mf],
        forcing=solver._newton_forcing, dtau_scale=solver._newton_dtau_scale,
        block_precond=solver._newton_block_precond, red=red)
    u_new = u_flat.copy()
    u_new[:, :n_mf] = u_new_mf
    solver._newton_last_info = info
    solver._newton_dtau_scale = info["dtau_scale"]
    if info["theta"] <= 0.0:
        logger.warning(
            "Newton 步未能前进（theta=0, gmres_info=%s, gmres_iters=%d, dtau_scale=%.3e, "
            "本步已缩 %d 档）——dtau 缩到下限仍拿不到被接受的步，那不再是步长问题"
            "（dtau->0 即显式前向 Euler、必然被接受），检查残差求值在当前状态上是否"
            "已经非物理" % (info["gmres_info"], info["gmres_iters"], info["dtau_scale"],
                           info["n_dtau_cuts"]))
    elif info["n_dtau_cuts"] > 0:
        logger.info("Newton 步缩 %d 档 dtau 后被接受（dtau_scale=%.3e, theta=%.3f）"
                    % (info["n_dtau_cuts"], info["dtau_scale"], info["theta"]))
    return u_new, info

