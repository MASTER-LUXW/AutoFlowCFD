"""AutoFlowCFD V2.0 - 矩阵自由 Newton-Krylov 稳态求解（JFNK）+ 伪瞬态延拓。

## 它解决什么

显式推进求稳态解的代价由**最小**单元的稳定性极限决定，与要走完的物理
时标无关。本项目真实网格上的量级：`plate_demo` 在 CFL 0.03 下走完绕板
特征时间需要约 2 万步（实测 350 步只走完 1.8%，见项目记忆
`stagnation_face_overpressure_localized`）。这不是"再等等"的问题 ——
显式格式的步长与收敛所需步数都由同一个 `h_min` 决定，加网格只会更糟。

隐式稳态求解把这条约束去掉：Newton 迭代的收敛步数由**非线性程度**决定，
不由 `h_min` 决定。工业稳态求解器（SU2、FUN3D、CFL3D）全部走这条路。

## 结构

每次调用 `step_newton_krylov` 做**一个** Newton 步：

    ( I/dtau + J ) dU = -R(U)        <- Krylov（GMRES）矩阵自由求解
    U <- U + theta * dU              <- theta 由物理性限幅给出

* `J v` 用 Fréchet 差分，一次 `J v` = 一次残差求值
  （`jacobian_vector.py`，含"必须逐变量按参考量级无量纲化"的说明）；
* `I/dtau` 是伪瞬态延拓项，`dtau` 直接用显式路径已经算好的逐 SP
  `dt_local`，于是 `--cfl-start/--cfl-max` 那套自适应控制器原样复用：
  CFL 小 -> 接近显式、鲁棒；CFL 大 -> 接近纯 Newton、快
  （`preconditioner.py`）；
* 线性求解容差由 inexact-Newton 的 forcing term 自适应给出
  （`forcing.py`），不把线性系统解到机器精度；
* 物理性限幅是逐单元松弛（`PHYSICALITY_MAX_RELATIVE_CHANGE` 的说明），
  `theta` 是其后残差接受判据的回溯比例（`globalization.py::accept_step`）；
* 一步不被接受（`theta = 0`）时**当场缩小 `dtau` 重试**，而不是原地
  不动等外层控制器 —— 那条交接被真实运行证明从来没有发生过（残差逐位
  不变让按残差历史工作的控制器看不到停滞），完整证据与修法见
  `dtau_control.py`。

**一次调用一个 Newton 步**是刻意的：外层的 `solver.solve()` 已经在做
残差监控、自适应 CFL、checkpoint、Order Continuation，把 Newton 外迭代
放在它里面（而不是在这里再套一层循环）让这些机制原样生效，不需要为隐式
路径复制一份。

## 为什么不组装 Jacobian

P2 下 79 万单元的一个 `(n_cells, n_sps, n_var)` 场就是 1.2 GB；稀疏
Jacobian 每单元一个 `(27*5)^2` 的块加上面耦合块，在同一网格上是不可
接受的量级。矩阵自由的内存占用与显式推进相同。代价是预处理只能用便宜
的对角形式（见 `preconditioner.py` 里如实说明的局限）。
"""

from typing import Callable, Optional, Tuple


import numpy as np

from .dtau_control import MIN_SCALE as DTAU_MIN_SCALE, PtcDtauScale
from .forcing import EisenstatWalkerForcing
from .jacobian_vector import MatrixFreeJacobian
from .block_jacobi import BlockJacobiCache
from .gmres import basis_itemsize
from .linear_solve import GMRES_MAX_ITER, _RealRows, _solve_direction, krylov_restart
from .physicality import _cellwise_relaxation, density_pressure_row_limits, update_local_dtau_scale
from .preconditioner import PseudoTransientDiagonal
from .globalization import LINEAR_SOLVE_MAX_REL_RESIDUAL, ResidualNorm, accept_step
from .reductions import LocalReductions

#: 单次 `step_newton_krylov` 调用里允许缩小 `dtau` 重试的档数
#: （每档 `dtau_control.FAIL_SHRINK`，即 4 倍）。
#:
#: 3 档 = `dtau` 最小到 1/64，按 `dtau_control.py` 里的论证足以穿过
#: "方向不可信"的区间（实测停滞出现在 CFL=10、CFL=1 没有，跨度约一个
#: 数量级）。每次重试要重解一次 GMRES，所以不能无上限；用不完的档由
#: **下一次调用**继续（`dtau_scale` 跨步持久化），于是极端情形下总档数
#: 不受这个值限制、单步成本却受它限制。
DTAU_MAX_CUTS_PER_STEP = 3


def step_newton_krylov(
    residual: Callable[[np.ndarray], np.ndarray],
    u0_flat: np.ndarray,
    dtau_flat: np.ndarray,
    scales: np.ndarray,
    *,
    forcing: Optional[EisenstatWalkerForcing] = None,
    tol_nonlinear: float = 1e-10,
    gmres_restart: Optional[int] = None,
    gmres_max_iter: int = GMRES_MAX_ITER,
    dtau_scale: float = 1.0,
    block_precond: Optional[BlockJacobiCache] = None,
    physicality: Callable = density_pressure_row_limits,
    rows_per_cell: int = 1,
    red: Optional[LocalReductions] = None,
    local_dtau_scale=None,
    norm_weights=None,
    real_rows=None,
    physical_time=None,
    norm_scales=None,
) -> Tuple[object, dict]:
    """做**一个** PTC-Newton-Krylov 步，返回 `(U_new_flat, info)`。

    Args:
        residual: `R(U_flat) -> (N, n_var)`，与 `fr_solver/step.py` 里
            `mean_flow_residual` 同一个约定（`dU/dt = -R`）。
        u0_flat: `(N, n_var)` 当前守恒变量。
        dtau_flat: `(N,)` 逐 SP 伪时间步长（`cfl.py` 的 `dt_local`）。
            它是**天花板**：本函数只会在它之下用 `dtau_scale` 缩放。
        scales: `(n_var,)` 守恒变量参考量级（Fréchet 差分的无量纲化，
            见 `jacobian_vector.py`）。
        forcing: 跨 Newton 步复用的 forcing-term 状态；`None` 时本步用
            固定的保守容差。
        tol_nonlinear: 外迭代目标绝对容差，只用来给 forcing term 定
            安全下限。
        gmres_restart: 重启长度；None 时按内存预算自适应（`krylov_restart`）。
        gmres_max_iter: 见模块级常量。
        dtau_scale: 上一次调用返回的 `info["dtau_scale"]`，把 PTC 的
            `dtau` 缩放状态跨步带过来（见 `dtau_control.py`）。调用方
            持久化它即可，不需要知道缩放策略。
        block_precond: 跨 Newton 步持有单元块 Jacobian 的缓存
            （`block_jacobi.py`）；`None` 时用逐 SP 对角预处理。
        physicality: `(U0, dU, red) -> alpha_rows` 逐行物理性限值。默认按
            守恒变量约束密度与压力；湍流标量方程传 `ScaledFieldRowLimits`。
        rows_per_cell: 每个单元占几行（解点数）：逐行限值在单元内取最小，
            作为该单元更新的松弛因子（见 `PHYSICALITY_MAX_RELATIVE_CHANGE`）。
        norm_weights: `(N,)` 残差范数的逐行权重（见 `ResidualNorm`）；None 时逐点 RMS。
            返回的 `res_norm`/`res_norm_new` 与接受判据都用这一个范数。
        local_dtau_scale: `(N,)` 上一次调用返回的 `info["local_dtau_scale"]`（逐行
            局部伪时间步长缩放，见 `LOCAL_DTAU_CUT_MIN`）；None 时全为 1。调用方持久化。
        real_rows: `(N,)` 布尔，真实解点行（原生基零填充槽位为 False）。给出时 Krylov
            向量只存真实行（`_RealRows`）；零填充行的残差必须恒为零，否则报错。
        physical_time: 双时间步的 BDF 物理时间项（`physical_time.py::BdfTerm`）。给出时 `residual`
            仍是空间残差 `R`，本函数解 `(I/dtau + a0/dt + J_R) dU = -G`，`G = R + (a0 U - b)/dt`：
            右端、接受判据与返回的残差范数都按 `G`，Krylov 算子与预处理用 `J_R` 加等效步长
            `1/(1/dtau + a0/dt)`。
        red: 全局归约（`reductions.py`）。`None` 时为单进程 numpy；GPU 传
            `LocalReductions(cupy)`、分布式传跨 rank 归约的子类——本函数
            里一切"对整个解向量取标量"的操作都经过它。

    Returns:
        `(U_new_flat, info)`。`info` 含 `res_norm`（步前的 `||R||` RMS）、
        `eta`、`gmres_iters`（本次调用**累计**的 Krylov 迭代数，含重试）、
        `n_matvec`、`theta`、`gmres_info`、`dtau_scale`（下一次调用要
        带回来的缩放因子）、`n_dtau_cuts`（本步缩了几档）、
        `theta_physicality`（逐单元松弛因子的全局最小值）、
        `limited_fraction`（被松弛的单元占比）与 `linear_rel_residual`
        （最后一次 GMRES 实际达到的相对线性残差）。

        `theta == 0.0` 表示这一步**没有前进**，但 `dtau` 还有档可缩
        （本次调用的档数用完了，`dtau_scale` 带着已缩小的值返回，下一次
        调用继续缩）。

    Raises:
        RuntimeError: `dtau_scale` 已经缩到 `dtau_control.MIN_SCALE` 仍
            拿不到一个被接受的步。那不再是步长问题 —— `dtau -> 0` 退化
            成显式前向 Euler、必然被残差接受判据接受（见
            `dtau_control.py` 的论证），所以走到这里只可能是残差求值
            本身在当前状态上已经失效（非有限值、或状态已非物理）。
            **不静默原地打转**：那正是本次修复要消灭的停滞形态，只是
            换了个位置（每步照样烧掉一次完整的 GMRES 求解）。
    """
    red = red if red is not None else LocalReductions()
    xp = red.xp
    u0_flat = xp.ascontiguousarray(u0_flat, dtype=xp.float64)
    n_var = u0_flat.shape[1]
    ctrl = PtcDtauScale(dtau_scale)
    local_scale = (xp.ones(u0_flat.shape[0]) if local_dtau_scale is None
                   else xp.asarray(local_dtau_scale, dtype=xp.float64).ravel())

    # `spatial`/`r0_spatial`：线性化与单元块装配用的空间残差；`residual`/`r0`：Newton 实际在解的方程
    # （双时间步时多一个物理时间项）
    spatial = residual
    r0_spatial = xp.ascontiguousarray(spatial(u0_flat), dtype=xp.float64)
    if r0_spatial.shape != u0_flat.shape:
        raise ValueError(
            f"残差形状 {r0_spatial.shape} 与状态形状 {u0_flat.shape} 不符")
    time_shift = 0.0
    if physical_time is not None:
        residual = physical_time.bind(spatial, None if real_rows is None else xp.asarray(real_rows, dtype=bool))
        time_shift = physical_time.shift
        r0 = xp.ascontiguousarray(residual(u0_flat), dtype=xp.float64)
    else:
        r0 = r0_spatial
    # 残差范数的逐列参考量级：默认与差分步长同一组；`norm_scales` 给出时用它（紧耦合步的湍流列，
    # 见 `coupled_step.py::residual_reference_scales`）
    norm_scales = scales if norm_scales is None else norm_scales
    for s in (scales, norm_scales):
        if np.asarray(s).size != u0_flat.shape[1]:
            raise ValueError(f"参考量级长度 {np.asarray(s).size} 与变量数 {u0_flat.shape[1]} 不符 —— 残差范数与差分步长"
                             f"都逐变量无量纲化，形状不符不能静默按广播处理")
    norm = ResidualNorm(norm_weights, red, norm_scales)
    res_norm = norm(r0)
    if res_norm == 0.0:
        return u0_flat, dict(res_norm=0.0, res_norm_new=0.0, eta=0.0,
                             gmres_iters=0, n_matvec=0, n_residual_extra=0,
                             theta=0.0, theta_physicality=0.0, limited_fraction=0.0,
                             gmres_info=0, linear_rel_residual=0.0, linear_growth=0.0,
                             dtau_scale=ctrl.scale, n_dtau_cuts=0,
                             local_dtau_scale=local_scale, local_dtau_min=red.min(local_scale))

    rows = None
    if real_rows is not None:
        rows = _RealRows(real_rows, u0_flat.shape[0], n_var, xp)
        pad = ~xp.asarray(real_rows, dtype=bool)
        if red.sum(xp.abs(r0[pad])) != 0.0:
            raise ValueError("零填充槽位的残差不为零：Krylov 向量不能只存真实行（见 _RealRows）")
    jac = MatrixFreeJacobian(spatial, u0_flat, r0_spatial, scales, red=red)
    dtau_base = xp.ascontiguousarray(dtau_flat, dtype=xp.float64).ravel() * local_scale

    def _linear_dtau(dtau):
        # 线性系统对角上的 1/dtau + a0/dt 写成一个等效步长，交给预处理（它同时提供 Krylov 算子的对角项）
        return dtau if time_shift == 0.0 else dtau / (1.0 + time_shift * dtau)

    if block_precond is not None:
        block_precond.begin_step(spatial, u0_flat, r0_spatial, scales, _linear_dtau(dtau_base * ctrl.scale))
    eta = (forcing.next_eta(res_norm, tol_nonlinear)
           if forcing is not None else 0.1)
    if gmres_restart is None:
        gmres_restart = krylov_restart(u0_flat.size if rows is None else rows.idx.size * n_var, red,
                                       flexible=0.0 if block_precond is None else block_precond.flexible_fraction,
                                       itemsize=basis_itemsize(eta),
                                       cap=None if block_precond is None else block_precond.restart_cap)

    u_new = u0_flat
    theta = 0.0
    theta_phys = 0.0
    limited_frac = 0.0
    res_norm_new = res_norm
    gmres_info = 0
    linear_rel = float("nan")
    linear_growth = float("nan")
    iters_total = 0
    iters_since_build = 0
    n_extra_total = 0
    n_cuts = 0

    # `dtau` 逐档缩小重试：一步不被接受说明当前 `dtau` 下的方向不可信，
    # 缩小 `dtau` 让 PTC 系统更接近对角主导（`dtau -> 0` 即显式前向
    # Euler，必然被接受）。为什么必须在这里重试、而不能像原先那样返回
    # `theta=0` 交给外层自适应 CFL：见 `dtau_control.py` 模块文档里的
    # 真实停滞记录（残差逐位不变 -> 按残差历史工作的控制器看不到它）。
    while True:
        dtau_try = dtau_base * ctrl.scale
        dtau_lin = _linear_dtau(dtau_try)
        prec = None          # 上一档的逆先释放，再按本档 dtau 构造（不同时持有两份）
        prec = (block_precond.preconditioner(dtau_lin, n_var) if block_precond is not None
                else PseudoTransientDiagonal(dtau_lin, n_var))
        budget = block_precond.stale_budget() if block_precond is not None else None
        # 复核点（`gmres.py`）：复用的块在过时预算处；刚装配的块在还有补救可做（对角位移未启用）时取一个
        # 重启周期——外推到容差还要更久就不把预算烧完，直接补救
        review = budget if budget is not None or block_precond is None or not block_precond.recoverable             else gmres_restart
        du, iters, ginfo, linear_rel = _solve_direction(
            jac, prec, dtau_lin, r0, n_var, eta, gmres_restart, gmres_max_iter, red, rows, norm,
            review_at=None if review is None or review >= gmres_max_iter else review)
        iters_total += iters
        if ginfo > 0 and linear_rel > LINEAR_SOLVE_MAX_REL_RESIDUAL and block_precond is not None:
            # 方向不可信（没到容差、线性残差也没减半）：复用的块当场按本步基态重装配、预处理启用对角位移
            # （`BlockJacobiCache.recover`），有动作就再解一次。没到容差但线性残差已减半的方向是可用的
            # inexact-Newton 方向，直接用：cube_demo SA P1 稳态 CFL 3e4~1e5 的十几步里第一次求解 200 次
            # 到 0.14~0.3 后仍重装配再解 200 次，一步 37 分钟里一半白费。ginfo、linear_rel 是全局量，
            # 各 rank 在这里的分支一致。
            prec = None          # 释放持有旧块的预处理对象，重装配期间不留两份（见 block_jacobi._build）
            if block_precond.recover(spatial, u0_flat, r0_spatial, scales, dtau_lin):
                prec = block_precond.preconditioner(dtau_lin, n_var)
                du, iters, ginfo, linear_rel = _solve_direction(
                    jac, prec, dtau_lin, r0, n_var, eta, gmres_restart, gmres_max_iter, red, rows, norm)
                iters_total += iters
        iters_since_build = iters
        gmres_info = ginfo
        if du is not None and ginfo > 0 and linear_rel > LINEAR_SOLVE_MAX_REL_RESIDUAL:
            # 没解到容差、线性残差也没减半：方向不可信，不走出去（`LINEAR_SOLVE_MAX_REL_RESIDUAL` 文档），
            # 与残差判据不接受同样处理——缩 dtau 让系统更接近对角主导后重解
            du = None
            theta = 0.0
        if du is not None:
            # 线性系统自己预测的残差放大倍数：(I/dtau + J) dU = -R 解出的 dU 满足 R + J dU = -dU/dtau
            # （双时间步时 R、J 换成 G、dG/dU，dtau 仍是伪时间步），
            # 所以 ||dU/dtau|| / ||R|| 就是"走完这一步后线性化残差是原来的几倍"（见
            # `mean_flow_step.py::note_linear_growth`）。
            linear_growth = norm(du / dtau_try[:, None]) / res_norm
            alpha, theta_phys, limited_frac = _cellwise_relaxation(
                physicality(u0_flat, du, red), rows_per_cell, red)
            u_try, theta, res_norm_new, n_extra = accept_step(
                residual, u0_flat, du * alpha[:, None], 1.0, res_norm, red, norm)
            n_extra_total += n_extra
            if theta > 0.0:
                u_new = u_try
                ctrl.reward(theta)
                local_scale = update_local_dtau_scale(local_scale, alpha, xp)
                break
        if ctrl.scale <= DTAU_MIN_SCALE:
            raise RuntimeError(
                "隐式稳态步在 dtau 缩到下限（scale=%.3e，相对自适应 CFL "
                "给出的天花板）之后仍拿不到一个被残差判据接受的步。"
                "dtau->0 时 PTC 退化成显式前向 Euler、必然被接受（前提"
                "是 R 在该状态附近连续），所以这不是步长问题：要么残差"
                "求值本身已失效（非有限值/非物理状态），要么 R 在这一点"
                "附近不连续。"
                "诊断：||R||=%.6e，R 全有限=%s，GMRES info=%s，"
                "物理性限幅 theta=%.3e。"
                % (ctrl.scale, res_norm, red.all_finite(r0),
                   gmres_info, theta_phys))
        if n_cuts >= DTAU_MAX_CUTS_PER_STEP or not ctrl.cut():
            # 本次调用的档数用完：如实返回 `theta=0`，`dtau_scale` 带着
            # 已经缩小的值回去，下一次调用接着缩（见 `dtau_control.py`）。
            break
        n_cuts += 1

    if forcing is not None:
        forcing.record_step(linear_rel, full_step=theta >= 1.0 and limited_frac == 0.0)
    if block_precond is not None:
        # 刷新判据的基线用最后一次求解（刚装配时即新块的迭代数），不含过时那一次
        block_precond.record(iters_since_build, accepted=theta > 0.0)
    return u_new, dict(res_norm=res_norm, res_norm_new=res_norm_new, update_norm=norm(u_new - u0_flat),
                       eta=eta, gmres_iters=iters_total,
                       n_matvec=jac.n_matvec,
                       n_residual_extra=n_extra_total,
                       theta=theta, theta_physicality=theta_phys,
                       limited_fraction=limited_frac,
                       gmres_info=gmres_info, linear_rel_residual=linear_rel, linear_growth=linear_growth,
                       dtau_scale=ctrl.scale, n_dtau_cuts=n_cuts,
                       local_dtau_scale=local_scale,
                       local_dtau_min=red.min(local_scale))
