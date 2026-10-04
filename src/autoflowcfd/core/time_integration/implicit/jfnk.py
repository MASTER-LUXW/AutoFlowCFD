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
from .gmres import gmres_right
from .physicality import _cellwise_relaxation, density_pressure_row_limits, update_local_dtau_scale
from .preconditioner import PseudoTransientDiagonal
from .globalization import ResidualNorm, accept_step
from .reductions import LocalReductions

#: GMRES 重启长度的**下限**。Krylov 基向量按 `(N, n_var)` 存，重启长度直接决定
#: 峰值内存：P2 下 79 万单元一个基向量 1.2 GB，大问题只能取小值。
GMRES_RESTART = 30

#: 重启长度的上限（正交化代价随 m 线性增长，m=200 时每次迭代约 200 次向量更新）。
GMRES_RESTART_MAX = 200

#: Krylov 基允许的内存（每个 rank / 每块设备）。重启长度按它自适应
#: （`krylov_restart`），夹在 `[GMRES_RESTART, GMRES_RESTART_MAX]` 之间。
#:
#: **为什么不是固定 30**（2026-09-26）：大 CFL 下迭代数上百时 GMRES(30) 反复
#: 重启、丢掉 Krylov 子空间，会停滞。plate_demo P0（17.9 万单元，一个基向量
#: 7 MB）CFL 1000 的同一线性系统、同一块 ILU、`rtol=0.1`：GMRES(30) 600 次仍停在
#: 0.136，GMRES(100) 181 次、GMRES(200) 121 次达到。固定 30 是按 P2 大网格的内存
#: 定的，小问题完全没必要受它限制。
KRYLOV_BASIS_BYTES = 2 * 2 ** 30

#: 单个 Newton 步允许的最大 Krylov 迭代数（= 最大残差求值次数）。
#:
#: **60 -> 200（2026-09-19，被真实运行修正）**：`dtau` 大的时候预处理后
#: 的算子是 `I + dtau*J`，特征值随 `dtau` 线性增长，GMRES 需要的迭代数
#: 跟着涨。实测 Blasius P1 在 CFL 100 下每步用满 60 次仍未达到 `eta`，
#: 于是拿到一个**半收敛**的方向 —— 而半收敛的方向既不是 Newton 方向也
#: 不是显式方向，是最坏的情形。
#:
#: 放开到 200 的同时加了两道闸，所以"成本失控"这件事由它们承担而不是
#: 由这个上限承担：
#:   * `gmres_info > 0`（未达容差）会被记录并交给残差接受判据；
#:   * 残差接受判据（`globalization.py::accept_step`）不接受让残差恶化的步，回溯到接受
#:     或者放弃这一步（`theta=0`，外层自适应 CFL 随之缩小 dtau）。
GMRES_MAX_ITER = 200

#: 单次 `step_newton_krylov` 调用里允许缩小 `dtau` 重试的档数
#: （每档 `dtau_control.FAIL_SHRINK`，即 4 倍）。
#:
#: 3 档 = `dtau` 最小到 1/64，按 `dtau_control.py` 里的论证足以穿过
#: "方向不可信"的区间（实测停滞出现在 CFL=10、CFL=1 没有，跨度约一个
#: 数量级）。每次重试要重解一次 GMRES，所以不能无上限；用不完的档由
#: **下一次调用**继续（`dtau_scale` 跨步持久化），于是极端情形下总档数
#: 不受这个值限制、单步成本却受它限制。
DTAU_MAX_CUTS_PER_STEP = 3

def krylov_restart(n_local_entries: int, red: LocalReductions, flexible: bool = False) -> int:
    """按 Krylov 基内存预算给出的重启长度（见 `KRYLOV_BASIS_BYTES`）。

    分布式下各 rank 的 GMRES 必须走同样多步（内积是集体操作），所以取全局最小。
    灵活模式每步多存一个 `M^{-1} v_j`（`gmres.py`），同一预算下重启长度减半。
    """
    per_step = 8 * max(int(n_local_entries), 1) * (2 if flexible else 1)
    m_local = KRYLOV_BASIS_BYTES // per_step - 1
    m = int(red.min(red.xp.asarray([float(m_local)])))
    return int(min(GMRES_RESTART_MAX, max(GMRES_RESTART, m)))


class _RealRows:
    """Krylov 向量只存真实解点行（原生基的零填充槽位不进基向量）。

    零填充行的残差恒为零（调用方保证、`step_newton_krylov` 入口核查）、Jacobian
    行列也为零，`A` 在那里就是 `I/dtau`、右端项为零，解在那里恒为零——在真实行
    上求解与全尺寸求解数学上等价。P3 原生基真实行只占约 42%（棱柱 40/64、四面体
    20/64），plate_demo P3 一个全尺寸基向量 0.46 GB、紧凑后 0.16 GB。
    做成类而不是闭包：在整个 Newton 步存活（项目规范）。
    """

    __slots__ = ("idx", "n_dof", "n_var")

    def __init__(self, real_rows, n_dof: int, n_var: int, xp):
        self.idx = xp.nonzero(xp.asarray(real_rows, dtype=bool))[0]
        self.n_dof, self.n_var = int(n_dof), int(n_var)

    def expand(self, x_1d, xp):
        full = xp.zeros((self.n_dof, self.n_var), dtype=x_1d.dtype)
        full[self.idx] = x_1d.reshape(-1, self.n_var)
        return full

    def compact(self, y):
        return y[self.idx].reshape(-1)


def _solve_direction(jac: MatrixFreeJacobian, prec, r0, n_var: int, eta: float,
                     gmres_restart: int, gmres_max_iter: int, red: LocalReductions,
                     rows: Optional[_RealRows] = None, norm: Optional[ResidualNorm] = None):
    """解 `(I/dtau + J) dU = -R`，返回 `(du, gmres_iters, gmres_info, linear_rel_residual)`。

    `prec` 同时提供 PTC 对角项（`add_ptc_term`）与预处理作用（`apply`），
    即 `PseudoTransientDiagonal` 或其子类 `CellBlockJacobiPreconditioner`。
    线性求解用后端无关的 `gmres.py::gmres_right`（右预处理：它的收敛判据
    就是真实线性残差 `||R + (I/dtau+J) dU|| <= eta ||R||`，正是 inexact
    Newton 要求的量）。

    **线性残差在全局化的同一个范数里最小化**（2026-10-03）：`norm` 带逐行权重 `w`
    时（`ResidualNorm`），对 `D A M^{-1} D^{-1}`（`D = diag(sqrt(w))`）做 GMRES，解
    `x = M^{-1} D^{-1} y`。它的残差就是 `D (b - A x)`，即接受判据与 SER 所用的加权范数；
    `D A M^{-1} D^{-1}` 与 `A M^{-1}` 相似，预处理后算子的谱不变。此前 GMRES 在未加权
    2 范数里最小化：残差被体积小 1e6 倍的单元主导，GMRES 在那里把线性残差压到
    `eta` 以下，在占全域体积的大单元上却让它变大——湍流平板层流 P2（`tests/validation/
    _flat_plate_case.py`）被回溯的步上未加权线性残差 0.03~0.10、体积加权 1.1~2.5 倍
    （非线性误差只有 0.001~0.003），线性模型本身就让全局度量恶化，接受判据回溯、SER
    收缩，CFL 卡在 ~60 近 200 步。

    `du` 为 `None` 表示线性求解产出了非有限方向（调用方据此缩小 `dtau`
    重试或放弃这一步，**不**静默用一个截断后的方向凑一步）。

    `jac` 由调用方构造并跨重试复用：它只依赖基态 `(U0, R0)` 与参考量级，
    与 `dtau` 无关，所以缩小 `dtau` 重试时不需要重算基残差、也不需要
    重建 Krylov 算子的基态部分（`n_matvec` 在它内部累计）。
    """
    n_dof = r0.shape[0]
    xp = red.xp

    if rows is None:
        def _apply_A(x_1d):
            v = x_1d.reshape(n_dof, n_var)
            jv = jac.matvec(v)
            return prec.add_ptc_term(jv, v).reshape(-1)

        def _apply_Minv(x_1d):
            return prec.apply(x_1d.reshape(n_dof, n_var)).reshape(-1)

        b = (-r0).reshape(-1)
    else:
        def _apply_A(x_1d):
            v = rows.expand(x_1d, xp)
            return rows.compact(prec.add_ptc_term(jac.matvec(v), v))

        def _apply_Minv(x_1d):
            return rows.compact(prec.apply(rows.expand(x_1d, xp)))

        b = rows.compact(-r0)

    if norm is not None and norm.weights is not None:
        d_full = xp.broadcast_to(xp.sqrt(norm.weights)[:, None], (n_dof, n_var))
        d = xp.ascontiguousarray(d_full).reshape(-1) if rows is None else rows.compact(d_full)
        if not bool(red.all_finite(d)) or red.min(d) <= 0.0:
            raise ValueError("残差范数权重在真实行上必须为正有限值（加权线性求解需要 D 可逆）")
        unscaled_A, unscaled_Minv = _apply_A, _apply_Minv

        def _apply_A(x_1d):
            return d * unscaled_A(x_1d)

        def _apply_Minv(y_1d):
            return unscaled_Minv(y_1d / d)

        b = d * b

    du_1d, iters, info, rel = gmres_right(
        _apply_A, b, _apply_Minv, rtol=eta,
        restart=gmres_restart, max_iter=gmres_max_iter, red=red, flexible=prec.flexible)
    if info < 0 or not red.all_finite(du_1d):
        return None, iters, -1, float("nan")
    du = du_1d.reshape(n_dof, n_var) if rows is None else rows.expand(du_1d, xp)
    return du, iters, int(info), float(rel)


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

    r0 = xp.ascontiguousarray(residual(u0_flat), dtype=xp.float64)
    if r0.shape != u0_flat.shape:
        raise ValueError(
            f"残差形状 {r0.shape} 与状态形状 {u0_flat.shape} 不符")
    norm = ResidualNorm(norm_weights, red)
    res_norm = norm(r0)
    if res_norm == 0.0:
        return u0_flat, dict(res_norm=0.0, res_norm_new=0.0, eta=0.0,
                             gmres_iters=0, n_matvec=0, n_residual_extra=0,
                             theta=0.0, theta_physicality=0.0, limited_fraction=0.0,
                             gmres_info=0, linear_rel_residual=0.0,
                             dtau_scale=ctrl.scale, n_dtau_cuts=0,
                             local_dtau_scale=local_scale, local_dtau_min=red.min(local_scale))

    rows = None
    if real_rows is not None:
        rows = _RealRows(real_rows, u0_flat.shape[0], n_var, xp)
        pad = ~xp.asarray(real_rows, dtype=bool)
        if red.sum(xp.abs(r0[pad])) != 0.0:
            raise ValueError("零填充槽位的残差不为零：Krylov 向量不能只存真实行（见 _RealRows）")
    jac = MatrixFreeJacobian(residual, u0_flat, r0, scales, red=red)
    dtau_base = xp.ascontiguousarray(dtau_flat, dtype=xp.float64).ravel() * local_scale
    if block_precond is not None:
        block_precond.begin_step(residual, u0_flat, r0, scales, dtau_base * ctrl.scale)
    if gmres_restart is None:
        gmres_restart = krylov_restart(u0_flat.size if rows is None else rows.idx.size * n_var, red,
                                       flexible=block_precond is not None and block_precond.flexible)
    eta = (forcing.next_eta(res_norm, tol_nonlinear)
           if forcing is not None else 0.1)

    u_new = u0_flat
    theta = 0.0
    theta_phys = 0.0
    limited_frac = 0.0
    res_norm_new = res_norm
    gmres_info = 0
    linear_rel = float("nan")
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
        prec = None          # 上一档的逆先释放，再按本档 dtau 构造（不同时持有两份）
        prec = (block_precond.preconditioner(dtau_try, n_var) if block_precond is not None
                else PseudoTransientDiagonal(dtau_try, n_var))
        budget = block_precond.stale_budget() if block_precond is not None else None
        du, iters, ginfo, linear_rel = _solve_direction(
            jac, prec, r0, n_var, eta, gmres_restart,
            gmres_max_iter if budget is None else min(budget, gmres_max_iter), red, rows, norm)
        iters_total += iters
        if ginfo > 0 and budget is not None and budget < gmres_max_iter:
            # 复用的 J_cc 在过时预算内解不到容差：当场按本步基态重装配再解
            # （见 block_jacobi.py 模块文档"复用与刷新"）。ginfo 是全局量，各 rank
            # 在这里的分支一致。
            prec = None          # 释放持有旧块的预处理对象，重装配期间不留两份（见 block_jacobi._build）
            block_precond.refresh(residual, u0_flat, r0, scales, dtau_try)
            prec = block_precond.preconditioner(dtau_try, n_var)
            du, iters, ginfo, linear_rel = _solve_direction(
                jac, prec, r0, n_var, eta, gmres_restart, gmres_max_iter, red, rows, norm)
            iters_total += iters
        iters_since_build = iters
        gmres_info = ginfo
        if du is not None:
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

    if block_precond is not None:
        # 刷新判据的基线用最后一次求解（刚装配时即新块的迭代数），不含过时那一次
        block_precond.record(iters_since_build, accepted=theta > 0.0)
    return u_new, dict(res_norm=res_norm, res_norm_new=res_norm_new,
                       eta=eta, gmres_iters=iters_total,
                       n_matvec=jac.n_matvec,
                       n_residual_extra=n_extra_total,
                       theta=theta, theta_physicality=theta_phys,
                       limited_fraction=limited_frac,
                       gmres_info=gmres_info, linear_rel_residual=linear_rel,
                       dtau_scale=ctrl.scale, n_dtau_cuts=n_cuts,
                       local_dtau_scale=local_scale,
                       local_dtau_min=red.min(local_scale))
