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
  `theta` 是其后残差接受判据的回溯比例（`_accept_step`）；
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
from .preconditioner import PseudoTransientDiagonal
from .reductions import LocalReductions

#: GMRES 重启长度。Krylov 基向量按 `(N, n_var)` 存，重启长度直接决定
#: 峰值内存：P2 下 79 万单元一个基向量 1.2 GB，所以这个值必须小。
#: 30 是 GMRES(m) 的常用取值；配上对角预处理与 inexact-Newton 容差，
#: 实测在 PTC 的鲁棒档（dtau 小）下通常几步就满足容差、根本不到重启。
GMRES_RESTART = 30

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
#:   * 残差接受判据（`_accept_step`）不接受让残差恶化的步，回溯到接受
#:     或者放弃这一步（`theta=0`，外层自适应 CFL 随之缩小 dtau）。
GMRES_MAX_ITER = 200

#: 残差接受判据：允许一步之后 `||R||` 相对恶化的上限。
#:
#: 伪瞬态延拓不是对某个 merit function 做线搜索，所以**不能**要求残差
#: 严格单调下降 —— 真实瞬态本身会让它先升后降（Blasius 从均匀初场起步
#: 就是这样）。但也不能什么都接受：那正是"CFL 10 跑到 6.5e19"的来源。
#:
#: 取 1.5 的含义：一步之内残差恶化超过 50% 就认为这一步的方向不可信，
#: 回溯 `theta`。它足够松以容纳真实瞬态的上升（实测显式路径上单步的
#: 残差增幅远小于此），又足够紧以立刻拦住发散。
RESIDUAL_ACCEPT_GROWTH = 1.5

#: 回溯次数上限。每次回溯把 `theta` 减半并重算一次残差，所以代价是
#: 最多这么多次额外残差求值。4 次即 `theta` 最小到 1/16。
MAX_BACKTRACK = 4

#: 单次 `step_newton_krylov` 调用里允许缩小 `dtau` 重试的档数
#: （每档 `dtau_control.FAIL_SHRINK`，即 4 倍）。
#:
#: 3 档 = `dtau` 最小到 1/64，按 `dtau_control.py` 里的论证足以穿过
#: "方向不可信"的区间（实测停滞出现在 CFL=10、CFL=1 没有，跨度约一个
#: 数量级）。每次重试要重解一次 GMRES，所以不能无上限；用不完的档由
#: **下一次调用**继续（`dtau_scale` 跨步持久化），于是极端情形下总档数
#: 不受这个值限制、单步成本却受它限制。
DTAU_MAX_CUTS_PER_STEP = 3

#: 物理性限幅允许的单步最大相对**下降**（对 `rho` 与 `p`，湍流的 `k`/`omega`）；
#: 增长按对数对称给出：单步 `u_new/u in [1-c, 1/(1-c)]`，c=0.5 即"最多减半、
#: 最多加倍"。这些量都是跨多个量级的正值量，对数对称才是同一个"变化幅度"。
#: 这是 SU2/FUN3D 那类"非物理点上收紧步长"的标准做法 —— Newton 方向在
#: 远离解时可以指向 `rho<0`，直接走过去会让下一次残差求值算在非物理态
#: 上、整个迭代失去意义。
#:
#: **逐单元松弛，不是全局取最小**（2026-09-25）：每个单元按自己全部解点
#: 的限值取一个因子 `alpha_c in [0,1]`，只缩放该单元的更新（单元内各解点
#: 共用一个因子，保持单元内更新的多项式形状）。此前是全场取最小的一个
#: 标量 `theta`：plate_demo P0+SST（17.9 万单元，NK）第 14 步湍流 Newton
#: 的 `theta_phys = 7.9e-5`——一个单元想把 omega 降一半以上，整个湍流场
#: 就只能走万分之一步；湍流冻住后平均流 `R_new/R = 0.9995`，残差此后
#: 逐位不动。SU2 的 `ComputeUnderRelaxationFactor` 同样是逐点的。
PHYSICALITY_MAX_RELATIVE_CHANGE = 0.5


def _pressure(u_flat: np.ndarray) -> np.ndarray:
    """`p = (gamma-1)(rho_E - |m|^2 / (2 rho))`，形状 `(N,)`。

    这里刻意**不**做 `state._update_primitives()` 里那套 `rho>=1e-10` /
    `p>=1.0` 的钳制：本函数的用途正是**检测**非物理，钳制会把要检测的
    东西抹掉。
    """
    rho = u_flat[:, 0]
    m2 = u_flat[:, 1] ** 2 + u_flat[:, 2] ** 2 + u_flat[:, 3] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return 0.4 * (u_flat[:, 4] - 0.5 * m2 / rho)


def density_pressure_row_limits(u0_flat, du_flat, red: LocalReductions):
    """逐行（逐解点）`alpha_i in [0, 1]`，使 `U0_i + alpha_i*dU_i` 保持物理。

    只约束两个真正会破坏残差求值的量：

    * **密度**：解析给出 `rho_new/rho in [1-c, 1/(1-c)]`；
    * **压力**：`p` 是 `U` 的非线性函数
      （`p = (gamma-1)(rho_E - |m|^2/(2 rho))`），所以不解那个二次
      不等式，而是先按密度定 `alpha`、再对 `p` 做一次**保守回缩**：
      若 `U0 + alpha*dU` 处的 `p` 越出同一个比例区间，按实际超出比例把
      `alpha` 再缩一次。只往"缩小"方向走。

    `c = PHYSICALITY_MAX_RELATIVE_CHANGE`。逐单元取最小由调用方
    （`step_newton_krylov`）统一做。

    **为什么不是"只要不变负就行"**：`rho` 掉到原值的 1e-6 虽然还是正数，
    但那一点的温度/声速会离谱到让下一次残差求值毫无意义、并污染整个
    Krylov 基。限制**相对变化**才是有效的护栏。
    """
    c = PHYSICALITY_MAX_RELATIVE_CHANGE
    xp = red.xp
    alpha = xp.clip(_relative_change_limits(xp, u0_flat[:, 0], du_flat[:, 0], c), 0.0, 1.0)

    p0 = _pressure(u0_flat)
    p1 = _pressure(u0_flat + alpha[:, None] * du_flat)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = (p1 - p0) / xp.maximum(xp.abs(p0), 1e-300)
        # 超出比例区间时按线性化回缩（NaN 即非物理试探点上 rho->0：不参与，
        # 那一点的 alpha 已由密度约束）
        shrink = xp.where(rel < -c, c / -rel, xp.where(rel > _up(c), _up(c) / rel, 1.0))
    return alpha * xp.where(xp.isnan(shrink), 1.0, shrink)


def _up(c):
    """与最大相对下降 `c` 对数对称的最大相对增长 `1/(1-c) - 1`。"""
    return c / (1.0 - c)


def _relative_change_limits(xp, u0, du, c):
    """逐点允许的最大步长比例，使 `(u0 + alpha*du)/u0 in [1-c, 1/(1-c)]`
    （正值量；`du == 0` 处 `+inf`）。

    **增长也要约束**（2026-09-25）：此前只约束下降。全局取最小 theta 的年代
    这一点被掩盖了——一个单元的下降限值把全场都冻住，增长也跟着被冻住；
    改成逐单元松弛后，plate_demo P0+SST 第 15 步 k 的最大值一步从 0.13 跳到
    114（未被约束的单元里 Newton 方向把 k 放大近千倍）。取对数对称而不是
    `|du|/u <= c`：后者把增长也压在 +50%，棱柱通道 NK+SST 算例上湍流发展期
    50~75% 的单元被松弛、120 步内收敛不了（对数对称下正常收敛）。"""
    up, down = du > 0.0, du < 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        lim_up = _up(c) * u0 / xp.where(up, du, 1.0)
        lim_down = c * u0 / xp.where(down, -du, 1.0)
    return xp.where(up, lim_up, xp.where(down, lim_down, xp.inf))


def positive_fields_row_limits(u0_flat, du_flat, red: LocalReductions):
    """标量正值场（湍流 `k`/`omega`）的逐行限值：每一列都满足
    `u_new/u in [1-c, 1/(1-c)]`，`c = PHYSICALITY_MAX_RELATIVE_CHANGE`。

    与守恒变量那一版同一个理由：不是"不变负就行"，而是单步相对变化不超过
    `c`——`omega` 掉到原值的 1e-6 仍是正数，但涡粘 `nu_t = k/omega` 会被放大
    六个数量级、下一次残差求值毫无意义；反方向同理（见 `_relative_change_limits`）。
    """
    lim = _relative_change_limits(red.xp, u0_flat, du_flat, PHYSICALITY_MAX_RELATIVE_CHANGE).min(axis=1)
    return red.xp.clip(lim, 0.0, 1.0)


def _cellwise_relaxation(alpha_rows, rows_per_cell: int, red: LocalReductions):
    """逐行限值 -> 逐单元因子（单元内取最小）按行展开，返回
    `(alpha_rows_cellwise, alpha_min, limited_cell_fraction)`（后两个是全局量）。"""
    xp = red.xp
    alpha_cell = alpha_rows.reshape(-1, rows_per_cell).min(axis=1)
    alpha_min = red.min(alpha_cell)
    limited = red.sum(alpha_cell < 1.0) / max(red.count(alpha_cell), 1.0)
    return xp.repeat(alpha_cell, rows_per_cell), float(alpha_min), float(limited)


def _accept_step(residual: Callable, u0_flat, du_flat, theta0: float, res_norm0: float,
                 red: LocalReductions) -> Tuple[object, float, float, int]:
    """按残差接受判据回溯 `theta`，返回
    `(U_new, theta, res_norm_new, n_extra_residual_eval)`。

    从 `theta0`（物理性限幅给出的上界）开始，每次不被接受就减半，最多
    `MAX_BACKTRACK` 次。接受条件是

        ||R(U0 + theta*dU)||  <=  RESIDUAL_ACCEPT_GROWTH * ||R(U0)||

    全部回溯都不被接受时返回 `theta = 0`（**这一步不前进**）。调用方
    `step_newton_krylov` 据此**当场缩小 `dtau` 重解一次**（见
    `dtau_control.py`），而不是把状态原样交出去 —— 后者被真实运行证明
    会永久停滞。不"硬着头皮走一步"是刻意的：本项目已经吃过一次"越界
    之后收缩救不回来"的亏（项目记忆
    `adaptive_cfl_four_defects_and_soft_ceiling` 第 12 条）。

    为什么不用标准线搜索（Armijo）：那需要一个 merit function
    （通常 `0.5||R||^2`）与它的方向导数，而伪瞬态解的不是
    `min ||R||^2` 而是 `(I/dtau + J) dU = -R`；在 `dtau` 小的时候
    `dU` 根本不是 `||R||^2` 的下降方向（它是时间推进方向）。所以这里用的
    是"不允许显著恶化"这个更弱、但与 PTC 语义相容的判据。
    """
    theta = float(theta0)
    n_eval = 0
    for _ in range(MAX_BACKTRACK + 1):
        if theta <= 0.0:
            break
        u_try = u0_flat + theta * du_flat
        r_try = residual(u_try)
        n_eval += 1
        if red.all_finite(r_try):
            rn = red.rms(r_try)
            if rn <= RESIDUAL_ACCEPT_GROWTH * res_norm0:
                return u_try, theta, rn, n_eval
        theta *= 0.5
    return u0_flat, 0.0, res_norm0, n_eval


def _solve_direction(jac: MatrixFreeJacobian, prec, r0, n_var: int, eta: float,
                     gmres_restart: int, gmres_max_iter: int, red: LocalReductions):
    """解 `(I/dtau + J) dU = -R`，返回 `(du, gmres_iters, gmres_info, linear_rel_residual)`。

    `prec` 同时提供 PTC 对角项（`add_ptc_term`）与预处理作用（`apply`），
    即 `PseudoTransientDiagonal` 或其子类 `CellBlockJacobiPreconditioner`。
    线性求解用后端无关的 `gmres.py::gmres_right`（右预处理：它的收敛判据
    就是真实线性残差 `||R + (I/dtau+J) dU|| <= eta ||R||`，正是 inexact
    Newton 要求的量）。

    `du` 为 `None` 表示线性求解产出了非有限方向（调用方据此缩小 `dtau`
    重试或放弃这一步，**不**静默用一个截断后的方向凑一步）。

    `jac` 由调用方构造并跨重试复用：它只依赖基态 `(U0, R0)` 与参考量级，
    与 `dtau` 无关，所以缩小 `dtau` 重试时不需要重算基残差、也不需要
    重建 Krylov 算子的基态部分（`n_matvec` 在它内部累计）。
    """
    n_dof = r0.shape[0]

    def _apply_A(x_1d):
        v = x_1d.reshape(n_dof, n_var)
        jv = jac.matvec(v)
        return prec.add_ptc_term(jv, v).reshape(-1)

    def _apply_Minv(x_1d):
        return prec.apply(x_1d.reshape(n_dof, n_var)).reshape(-1)

    du_1d, iters, info, rel = gmres_right(
        _apply_A, (-r0).reshape(-1), _apply_Minv, rtol=eta,
        restart=gmres_restart, max_iter=gmres_max_iter, red=red)
    if info < 0 or not red.all_finite(du_1d):
        return None, iters, -1, float("nan")
    return du_1d.reshape(n_dof, n_var), iters, int(info), float(rel)


def step_newton_krylov(
    residual: Callable[[np.ndarray], np.ndarray],
    u0_flat: np.ndarray,
    dtau_flat: np.ndarray,
    scales: np.ndarray,
    *,
    forcing: Optional[EisenstatWalkerForcing] = None,
    tol_nonlinear: float = 1e-10,
    gmres_restart: int = GMRES_RESTART,
    gmres_max_iter: int = GMRES_MAX_ITER,
    dtau_scale: float = 1.0,
    block_precond: Optional[BlockJacobiCache] = None,
    physicality: Callable = density_pressure_row_limits,
    rows_per_cell: int = 1,
    red: Optional[LocalReductions] = None,
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
        gmres_restart / gmres_max_iter: 见模块级常量。
        dtau_scale: 上一次调用返回的 `info["dtau_scale"]`，把 PTC 的
            `dtau` 缩放状态跨步带过来（见 `dtau_control.py`）。调用方
            持久化它即可，不需要知道缩放策略。
        block_precond: 跨 Newton 步持有单元块 Jacobian 的缓存
            （`block_jacobi.py`）；`None` 时用逐 SP 对角预处理。
        physicality: `(U0, dU, red) -> alpha_rows` 逐行物理性限值。默认按
            守恒变量约束密度与压力；湍流标量方程传 `positive_fields_row_limits`。
        rows_per_cell: 每个单元占几行（解点数）：逐行限值在单元内取最小，
            作为该单元更新的松弛因子（见 `PHYSICALITY_MAX_RELATIVE_CHANGE`）。
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

    r0 = xp.ascontiguousarray(residual(u0_flat), dtype=xp.float64)
    if r0.shape != u0_flat.shape:
        raise ValueError(
            f"残差形状 {r0.shape} 与状态形状 {u0_flat.shape} 不符")
    res_norm = red.rms(r0)
    if res_norm == 0.0:
        return u0_flat, dict(res_norm=0.0, res_norm_new=0.0, eta=0.0,
                             gmres_iters=0, n_matvec=0, n_residual_extra=0,
                             theta=0.0, theta_physicality=0.0, limited_fraction=0.0,
                             gmres_info=0, linear_rel_residual=0.0,
                             dtau_scale=ctrl.scale, n_dtau_cuts=0)

    jac = MatrixFreeJacobian(residual, u0_flat, r0, scales, red=red)
    if block_precond is not None:
        block_precond.begin_step(residual, u0_flat, r0, scales)
    dtau_base = xp.ascontiguousarray(dtau_flat, dtype=xp.float64).ravel()
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
        prec = (block_precond.preconditioner(dtau_try, n_var) if block_precond is not None
                else PseudoTransientDiagonal(dtau_try, n_var))
        budget = block_precond.stale_budget() if block_precond is not None else None
        du, iters, ginfo, linear_rel = _solve_direction(
            jac, prec, r0, n_var, eta, gmres_restart,
            gmres_max_iter if budget is None else min(budget, gmres_max_iter), red)
        iters_total += iters
        if ginfo > 0 and budget is not None and budget < gmres_max_iter:
            # 复用的 J_cc 在过时预算内解不到容差：当场按本步基态重装配再解
            # （见 block_jacobi.py 模块文档"复用与刷新"）。ginfo 是全局量，各 rank
            # 在这里的分支一致。
            block_precond.refresh(residual, u0_flat, r0, scales)
            prec = block_precond.preconditioner(dtau_try, n_var)
            du, iters, ginfo, linear_rel = _solve_direction(
                jac, prec, r0, n_var, eta, gmres_restart, gmres_max_iter, red)
            iters_total += iters
        iters_since_build = iters
        gmres_info = ginfo
        if du is not None:
            alpha, theta_phys, limited_frac = _cellwise_relaxation(
                physicality(u0_flat, du, red), rows_per_cell, red)
            u_try, theta, res_norm_new, n_extra = _accept_step(
                residual, u0_flat, du * alpha[:, None], 1.0, res_norm, red)
            n_extra_total += n_extra
            if theta > 0.0:
                u_new = u_try
                ctrl.reward(theta)
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
                       dtau_scale=ctrl.scale, n_dtau_cuts=n_cuts)
