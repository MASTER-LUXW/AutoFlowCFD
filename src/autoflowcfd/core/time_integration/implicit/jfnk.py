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
* `theta` 是物理性限幅，见 `_physicality_limited_step`。

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

from .forcing import EisenstatWalkerForcing
from .jacobian_vector import MatrixFreeJacobian
from .preconditioner import PseudoTransientDiagonal

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

#: 物理性限幅允许的单步最大相对变化（对 `rho` 与 `p`）。
#: 0.5 的含义：一个 Newton 步不允许把任何一点的密度或压力改变超过 50%。
#: 这是 SU2/FUN3D 那类"非物理点上收紧步长"的标准做法 —— Newton 方向在
#: 远离解时可以指向 `rho<0`，直接走过去会让下一次残差求值算在非物理态
#: 上、整个迭代失去意义。
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


def _physicality_limited_step(u0_flat: np.ndarray, du_flat: np.ndarray
                              ) -> float:
    """返回 `theta in [0, 1]`，使 `U0 + theta*dU` 处处保持物理。

    只约束两个真正会破坏残差求值的量：

    * **密度**：解析给出 `rho + theta*drho >= (1-c) * rho`；
    * **压力**：`p` 是 `U` 的非线性函数
      （`p = (gamma-1)(rho_E - |m|^2/(2 rho))`），所以不解那个二次
      不等式，而是先按密度定一个 `theta`、再对 `p` 做一次**保守回缩**：
      若 `U0 + theta*dU` 处的 `p` 相对变化超过 `c`，按实际超出比例把
      `theta` 再缩一次。因为只往"缩小"方向走，永远偏安全。

    `c = PHYSICALITY_MAX_RELATIVE_CHANGE`。

    **为什么不是"只要不变负就行"**：`rho` 掉到原值的 1e-6 虽然还是正数，
    但那一点的温度/声速会离谱到让下一次残差求值毫无意义、并污染整个
    Krylov 基。限制**相对变化**才是有效的护栏。
    """
    c = PHYSICALITY_MAX_RELATIVE_CHANGE
    theta = 1.0

    rho0 = u0_flat[:, 0]
    drho = du_flat[:, 0]
    down = drho < 0.0
    if np.any(down):
        # rho0 + theta*drho >= (1-c)*rho0  ->  theta <= c*rho0 / (-drho)
        limit = c * rho0[down] / (-drho[down])
        theta = min(theta, float(np.min(limit)))
    theta = max(theta, 0.0)
    if theta <= 0.0:
        return 0.0

    p0 = _pressure(u0_flat)
    p1 = _pressure(u0_flat + theta * du_flat)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(p1 - p0) / np.maximum(np.abs(p0), 1e-300)
    rel_max = float(np.nanmax(rel)) if rel.size else 0.0
    if rel_max > c:
        theta *= c / rel_max
    return float(theta)


def _accept_step(residual: Callable[[np.ndarray], np.ndarray],
                 u0_flat: np.ndarray, du_flat: np.ndarray,
                 theta0: float, res_norm0: float
                 ) -> Tuple[np.ndarray, float, float, int]:
    """按残差接受判据回溯 `theta`，返回
    `(U_new, theta, res_norm_new, n_extra_residual_eval)`。

    从 `theta0`（物理性限幅给出的上界）开始，每次不被接受就减半，最多
    `MAX_BACKTRACK` 次。接受条件是

        ||R(U0 + theta*dU)||  <=  RESIDUAL_ACCEPT_GROWTH * ||R(U0)||

    全部回溯都不被接受时返回 `theta = 0`（**这一步不前进**），把处置交给
    外层自适应 CFL —— 缩小 `dtau` 会让 PTC 系统更接近对角主导、方向更
    可信。这比"硬着头皮走一步"正确：本项目已经吃过一次"越界之后收缩救
    不回来"的亏（项目记忆 `adaptive_cfl_four_defects_and_soft_ceiling`
    第 12 条），所以宁可原地不动也不要走出一步坏的。

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
        r_try = np.asarray(residual(u_try), dtype=np.float64)
        n_eval += 1
        if np.all(np.isfinite(r_try)):
            rn = float(np.linalg.norm(r_try) / np.sqrt(max(r_try.size, 1)))
            if rn <= RESIDUAL_ACCEPT_GROWTH * res_norm0:
                return u_try, theta, rn, n_eval
        theta *= 0.5
    return u0_flat, 0.0, res_norm0, n_eval


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
) -> Tuple[np.ndarray, dict]:
    """做**一个** PTC-Newton-Krylov 步，返回 `(U_new_flat, info)`。

    Args:
        residual: `R(U_flat) -> (N, n_var)`，与 `fr_solver/step.py` 里
            `mean_flow_residual` 同一个约定（`dU/dt = -R`）。
        u0_flat: `(N, n_var)` 当前守恒变量。
        dtau_flat: `(N,)` 逐 SP 伪时间步长（`cfl.py` 的 `dt_local`）。
        scales: `(n_var,)` 守恒变量参考量级（Fréchet 差分的无量纲化，
            见 `jacobian_vector.py`）。
        forcing: 跨 Newton 步复用的 forcing-term 状态；`None` 时本步用
            固定的保守容差。
        tol_nonlinear: 外迭代目标绝对容差，只用来给 forcing term 定
            安全下限。
        gmres_restart / gmres_max_iter: 见模块级常量。

    Returns:
        `(U_new_flat, info)`。`info` 含 `res_norm`（步前的 `||R||` RMS）、
        `eta`、`gmres_iters`、`n_matvec`、`theta`、`gmres_info`。
        `theta == 0.0` 表示这一步**没有前进**（线性求解失败或物理性限幅
        归零），调用方应当把它当成"需要缩小 dtau"的信号。
    """
    from scipy.sparse.linalg import LinearOperator, gmres

    u0_flat = np.ascontiguousarray(u0_flat, dtype=np.float64)
    n_dof, n_var = u0_flat.shape

    r0 = np.ascontiguousarray(residual(u0_flat), dtype=np.float64)
    if r0.shape != u0_flat.shape:
        raise ValueError(
            f"残差形状 {r0.shape} 与状态形状 {u0_flat.shape} 不符")
    res_norm = float(np.linalg.norm(r0) / np.sqrt(max(r0.size, 1)))
    if res_norm == 0.0:
        return u0_flat, dict(res_norm=0.0, res_norm_new=0.0, eta=0.0,
                             gmres_iters=0, n_matvec=0, n_residual_extra=0,
                             theta=0.0, theta_physicality=0.0, gmres_info=0)

    jac = MatrixFreeJacobian(residual, u0_flat, r0, scales)
    prec = PseudoTransientDiagonal(dtau_flat, n_var)

    def _apply_A(x_1d):
        v = x_1d.reshape(n_dof, n_var)
        jv = jac.matvec(v)
        return prec.add_ptc_term(jv, v).reshape(-1)

    def _apply_M(x_1d):
        return prec.apply(x_1d.reshape(n_dof, n_var)).reshape(-1)

    size = n_dof * n_var
    a_op = LinearOperator((size, size), matvec=_apply_A, dtype=np.float64)
    m_op = LinearOperator((size, size), matvec=_apply_M, dtype=np.float64)

    eta = (forcing.next_eta(res_norm, tol_nonlinear)
           if forcing is not None else 0.1)

    # GMRES 迭代计数：scipy 不返回它，`callback_type="pr_norm"` 下
    # callback 每次迭代调用一次，正好用来数。
    iters = [0]

    def _count(_pr_norm):
        iters[0] += 1

    rhs = (-r0).reshape(-1)
    n_cycles = max(1, int(np.ceil(gmres_max_iter / max(1, gmres_restart))))
    du_1d, gmres_info = gmres(
        a_op, rhs, rtol=eta, atol=0.0, restart=gmres_restart,
        maxiter=n_cycles, M=m_op, callback=_count,
        callback_type="pr_norm",
    )
    du = du_1d.reshape(n_dof, n_var)

    if not np.all(np.isfinite(du)):
        # 线性求解产出非有限方向：**不**走这一步，把处置交给外层自适应
        # CFL（缩小 dtau 让系统更接近对角主导）。不静默用一个截断后的
        # 方向凑一步 —— 那会让"这一步没能解出来"这件事消失在日志里。
        return u0_flat, dict(res_norm=res_norm, res_norm_new=res_norm,
                             eta=eta, gmres_iters=iters[0],
                             n_matvec=jac.n_matvec, n_residual_extra=0,
                             theta=0.0, theta_physicality=0.0,
                             gmres_info=-1)

    theta_phys = _physicality_limited_step(u0_flat, du)
    u_new, theta, res_norm_new, n_extra = _accept_step(
        residual, u0_flat, du, theta_phys, res_norm)
    return u_new, dict(res_norm=res_norm, res_norm_new=res_norm_new,
                       eta=eta, gmres_iters=iters[0],
                       n_matvec=jac.n_matvec, n_residual_extra=n_extra,
                       theta=theta, theta_physicality=theta_phys,
                       gmres_info=int(gmres_info))
