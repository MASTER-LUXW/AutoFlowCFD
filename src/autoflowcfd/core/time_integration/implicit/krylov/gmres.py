"""AutoFlowCFD V2.0 - 重启、右预处理 GMRES（后端无关）。

## 为什么不用 `scipy.sparse.linalg.gmres`

它只能在单进程 numpy 上跑：内积是本地的、向量必须是 numpy。隐式稳态
路径要在 GPU（cupy）与分布式（内积跨 rank 归约）上用同一份算法，就必须
自己持有 Krylov 循环，把"向量运算"交给数组模块、把"内积"交给
`reductions.py` 的归约对象。

## 算法

标准的 GMRES(m)（Saad & Schultz 1986）：

* **右预处理**：解 `A M^{-1} y = b`、`x = M^{-1} y`。Krylov 最小化的是
  `||b - A x||` 本身，所以这里判收敛用的就是真实线性残差——与
  inexact-Newton 的 forcing term（`forcing.py`）要求的
  `||R + J dU|| <= eta ||R||` 是同一个量，不需要换算；
* 正交化用修正 Gram-Schmidt（每个基向量一次全局内积）；
* 最小二乘用 Givens 旋转增量求解，每次迭代直接拿到当前残差范数；
* 重启之间用 `b - A x` 重算真实残差（多一次 `A` 作用），不累积舍入。

`A`、`M^{-1}` 都只以作用的形式给出（矩阵自由）。返回的迭代数是 `A` 的
作用次数（不含重启时那一次重算残差）。

## 灵活模式（FGMRES，Saad 1993）

`M^{-1}` 每次作用不是同一个线性算子时（多层预处理的 K 循环在粗层上做固定
步数的内层 Krylov，`coarse/multilevel.py`），`x = M^{-1}(V y)` 不再成立。
灵活模式存下每个 `z_j = M^{-1} v_j`，用 `x = x + Z y` 更新——最小化的仍是
真实残差，代价是多存一组与 Krylov 基同样多的向量。预处理对象带
`flexible = True` 时由调用方打开（`linear_solve.py::_solve_direction`）；线性预处理
不打开，不白占这份内存。

## 部分灵活（2026-10-15）

预处理是若干列块的块对角组合、只有一部分块非线性时（紧耦合 Newton 步里平均流用块 Jacobi、湍流用多层 K 循环），
只为非线性列存 `z_j`（`FlexibleSplit.mask` 为真的分量）：线性列的 `z_j` 是固定线性算子 `M_lin^{-1}` 作用在
`v_j` 上，`sum_j y_j z_j` 在那些列上等于 `M_lin^{-1}(sum_j y_j v_j)`，更新解时现算一次（`FlexibleSplit.apply_linear`）。
Arnoldi 关系里 `A` 作用的 `z_j` 与更新解用的是同一组值：非线性列是存下来（可能舍入到单精度）的值，线性列是
双精度的 `M_lin^{-1} v_j`。cube_demo 粗网格 SA P2（6 列里 1 列非线性）灵活模式那一份从 3.85 GiB 降到 0.64 GiB。

## 基向量单精度存储（2026-10-12）

线性容差不低于 `SINGLE_BASIS_MIN_RTOL` 时 Krylov 基（灵活模式下连同 `z_j`）按 float32 存，同样的内存
放两倍的向量；算子作用、内积、正交化仍按双精度做。`A` 作用在**存下来的** `z_j`（舍入后再升回双精度）上，
`x = x + Z y` 用的是同一组 `z_j`。`V` 的舍入让 Arnoldi 关系差 `eps32` 量级、真实残差比 Givens 递推值多出
`eps32 * ||y||`（`y` 是 Krylov 系数）：条件数 1e6、不带预处理的系统上 `rtol = 1e-4` 时递推报告已到容差而
真实残差是 3.5e-3。所以单精度模式下递推判定收敛后**重算一次真实残差核实**，不达标就接着迭代（迭代改进），
返回的 `rel_residual` 是真实值；代价是每次求解多一次 `A` 作用。

为什么需要：cube_demo SA P1（37.4 万单元，一个基向量 89 MB、灵活模式两份）重启长度被内存压在下限 30，
CFL 1500 的线性系统残差曲线前 30 次迭代只降到 0.61、之后才加速（第 40 次 0.28、第 46 次到 0.1）——
GMRES(30) 恰好在加速之前丢掉子空间，198 次迭代、1068 s 才到容差；GMRES(60) 46 次、245 s。同一步、
同一预处理。

## 复核点（2026-10-12）

调用方复用着过时的预处理块时，"重装配后重新求解"的代价是已知的迭代数 `review_at`（刚装配时的基线迭代数 +
一次装配折合的迭代数，`block_jacobi.py::stale_budget`）。迭代到这个次数还没到容差时，按最近
`PROJECTION_WINDOW` 次的下降率外推剩余迭代数：比 `review_at` 多就返回让调用方重装配，否则继续——继续比
重来便宜。此前到这个次数一律截断重来：cube_demo SA P1（CFL 1650）残差曲线先平后陡（第 30 次 0.82、
第 40 次 0.58、第 50 次 0.23），预算 57 次恰好截在离容差只差几次的地方，重装配 60 s 后重新求解 64 次
（比用旧块还多），一步 766 s，其中一半以上白费。

## 停滞提前终止（2026-10-10）

最近 `STAGNATION_WINDOW` 次迭代残差下降不到 `1 - STAGNATION_RATE`，并且按这个速率用完剩余迭代数也到
不了容差时，立即按"用满"返回（`info > 0`）。残差由 Givens 递推逐次给出，判据不需要额外的算子作用。
系统不定（伪瞬态项与物理时间项压不住 Jacobian 的不稳定模态）时 GMRES 一点都降不动：cube_demo SA P1
双时间步 dt=1e-3 / 5e-4 的第一次内迭代 200 次后线性残差 1.0 / 0.99，一次失败的求解 1220 s；提前终止后
是 50 次。能收敛的求解不受影响：判据只在"按当前速率到不了"时触发。
"""

from typing import Any, Callable, Optional, Tuple

import numpy as np

from .. import vector_ops
from ..reductions import LocalReductions


#: Krylov 基按单精度存的线性容差下限（模块文档"基向量单精度存储"）：float32 的舍入约 6e-8，容差比它高
#: 三个数量级以上才用；Newton 的 forcing 容差在 [1e-4, 0.1]，粗层内层求解与要求更严的调用仍是双精度。
SINGLE_BASIS_MIN_RTOL = 1.0e-5


def basis_itemsize(rtol: float) -> int:
    """容差为 `rtol` 的求解里每个基向量元素占的字节数（重启长度按它定，`linear_solve.py::krylov_restart`）。"""
    return 4 if rtol >= SINGLE_BASIS_MIN_RTOL else 8


class FlexibleSplit:
    """部分灵活（模块文档）：`mask` 是与求解向量同形状的布尔数组，真为预处理作用非线性的分量；`apply_linear(y)`
    给出线性部分的作用（`mask` 为假的分量有效）。"""

    __slots__ = ("mask", "apply_linear")

    def __init__(self, mask, apply_linear):
        self.mask, self.apply_linear = mask, apply_linear


#: 复核点外推剩余迭代数用的回看窗口（迭代数）。
PROJECTION_WINDOW = 10

#: 停滞判据的回看窗口（迭代数）与窗口内的残差比值下限（见模块文档"停滞提前终止"）
STAGNATION_WINDOW = 50
STAGNATION_RATE = 0.98


def _stagnated(history, rtol: float, max_iter: int) -> bool:
    """`history[i]` 是第 `i + 1` 次迭代后的相对残差。"""
    n = len(history)
    if n <= STAGNATION_WINDOW or n >= max_iter:
        return False
    rate = history[-1] / history[-1 - STAGNATION_WINDOW]
    if rate < STAGNATION_RATE:
        return False
    return history[-1] * rate ** ((max_iter - n) / STAGNATION_WINDOW) > rtol


def projected_remaining(history, rtol: float) -> float:
    """按最近 `PROJECTION_WINDOW` 次迭代的平均下降率外推：还要多少次迭代到 `rtol`（不降时为 inf）。"""
    w = min(PROJECTION_WINDOW, len(history) - 1)
    if w < 1 or history[-1] <= rtol:
        return 0.0 if history and history[-1] <= rtol else float("inf")
    rate = history[-1] / history[-1 - w]
    if not rate < 1.0:
        return float("inf")
    return float(w * np.log(rtol / history[-1]) / np.log(rate))


def gmres_right(apply_A: Callable, b, apply_Minv: Callable, *, rtol: float,
                restart: int, max_iter: int, red: Optional[LocalReductions] = None,
                flexible: bool = False, review_at: Optional[int] = None,
                split: "Optional[FlexibleSplit]" = None) -> Tuple[Any, int, int, float]:
    """解 `A x = b`，返回 `(x, iterations, info, rel_residual)`。

    `info == 0` 表示达到 `||b - A x|| <= rtol * ||b||`；`info > 0` 表示
    用满 `max_iter`、或停滞提前终止仍未达到（`x` 是当时的最好解）；`info < 0` 表示出现
    非有限值（`x` 不可用，调用方不得静默使用它）。

    `rel_residual` 是返回的 `x` 实际达到的 `||b - A x|| / ||b||`（Givens 递推
    给出的值，精确算术下即真实残差；重启处是重算的真实残差）。用满迭代数
    时它说明方向还有多少可信度——自适应 CFL 据此判断线性求解是否失败
    （`adaptive_cfl/ser.py`）。

    `flexible`：灵活模式（见模块文档），`M^{-1}` 可以每次作用都不同。`split`：灵活模式下只有一部分分量的作用
    非线性时给出（模块文档"部分灵活"），只为那些分量存 `z_j`。

    `review_at`：迭代到这个次数仍未达到容差时复核一次（模块文档"复核点"）：按残差曲线外推，剩余迭代数
    超过 `review_at` 就按"未达到"返回（`info > 0`），否则继续到 `max_iter`。
    """
    red = red if red is not None else LocalReductions()
    xp = red.xp
    x = xp.zeros_like(b)
    bnorm = red.norm(b)
    if bnorm == 0.0:
        return x, 0, 0, 0.0
    target = rtol * bnorm
    r = b.copy()
    beta = bnorm
    total = 0
    history = []        # 逐次迭代后的相对残差（停滞判据用）
    history_start = 0   # `history` 起算处的迭代数（单精度基核实不达标后重新起算）
    stalled = False
    # Krylov 基**按需**逐个分配、重启之间复用：修正 Gram-Schmidt 的 `w -= h*V[i]`
    # 原地做，不再每次新分配一个整长向量（见 vector_ops.py）。不按重启长度整块预分配
    # （2026-09-29）：Windows 上 `empty` 立即计入提交量，plate_demo P3 一个向量 0.16 GB、
    # 重启 30 就是 4.8 GB，而块 Jacobi 下实际只迭代 13 次——未用到的一半多把第 2 个
    # Newton 步挤到 OOM。
    V: list = []
    Z: list = []    # 灵活模式下的 M^{-1} v_j（预处理作用本来就返回新数组，直接持有它）
    single = basis_itemsize(rtol) == 4 and b.dtype == xp.float64
    store = xp.float32 if single else b.dtype

    def basis(j):
        if j == len(V):
            V.append(xp.empty(b.shape, dtype=store))
        return V[j]

    def working(v):
        """存储的向量升回运算精度（双精度存储时原样）。"""
        return v.astype(b.dtype) if single else v

    partial = flexible and split is not None

    def precond(j):
        z = apply_Minv(working(V[j]))
        if not flexible:
            return z
        kept = z[split.mask] if partial else z
        if single:
            kept = kept.astype(store)
        if j == len(Z):
            Z.append(kept)
        else:
            Z[j] = kept
        if not partial:
            return working(kept)    # A 作用在存下来的 z_j 上（模块文档"基向量单精度存储"）
        z[split.mask] = working(kept)
        return z

    def _combine(W, y, k_used):
        out = xp.multiply(W[0], y[0], dtype=b.dtype)
        for i in range(1, k_used):
            vector_ops.axpy_(xp, out, y[i], W[i])
        return out

    def _update(y, k_used):
        """解的增量 `M^{-1}(V y)`（灵活模式 `Z y`；部分灵活时非线性分量 `Z y`、线性分量现算）。"""
        if not flexible:
            return apply_Minv(_combine(V, y, k_used))
        if not partial:
            return _combine(Z, y, k_used)
        upd = split.apply_linear(_combine(V, y, k_used))
        upd[split.mask] = _combine(Z, y, k_used)
        return upd

    while True:
        m = restart
        xp.divide(r, beta, out=basis(0))
        H = np.zeros((m + 1, m))
        cs = np.zeros(m)
        sn = np.zeros(m)
        g = np.zeros(m + 1)
        g[0] = beta
        k_used = 0
        converged = False
        for j in range(m):
            w = apply_A(precond(j))
            total += 1
            w = xp.ascontiguousarray(w)
            for i in range(j + 1):
                h = red.dot(w, V[i])
                H[i, j] = h
                vector_ops.axpy_(xp, w, -h, V[i])
            h_next = red.norm(w)
            H[j + 1, j] = h_next
            if not (np.isfinite(h_next) and np.all(np.isfinite(H[: j + 2, j]))):
                return x, total, -1, float("nan")
            # 之前的 Givens 旋转作用到新列上
            for i in range(j):
                t = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = t
            denom = np.hypot(H[j, j], H[j + 1, j])
            if denom == 0.0:
                cs[j], sn[j] = 1.0, 0.0
            else:
                cs[j], sn[j] = H[j, j] / denom, H[j + 1, j] / denom
            H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
            H[j + 1, j] = 0.0
            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]
            k_used = j + 1
            if abs(g[j + 1]) <= target:
                converged = True
                break
            history.append(abs(g[j + 1]) / bnorm)
            if _stagnated(history, rtol, max_iter - history_start) or (
                    total == review_at and projected_remaining(history, rtol) > review_at):
                stalled = True
                break
            if total >= max_iter or h_next == 0.0:     # 用满 / 幸运击穿
                converged = h_next == 0.0
                break
            xp.divide(w, h_next, out=basis(j + 1))
        # 回代 y，更新 x = x + M^{-1} (V y)（灵活模式：x = x + Z y）
        y = np.linalg.solve(np.triu(H[:k_used, :k_used]), g[:k_used]) if k_used else np.zeros(0)
        x = x + _update(y, k_used)
        rel = abs(g[k_used]) / bnorm
        exhausted = total >= max_iter or stalled
        if converged and not single:
            return x, total, 0, rel
        if exhausted and not converged:
            return x, total, total, rel
        # 重启，或单精度基下对"递推判定已收敛"的核实：重算真实残差。单精度基的舍入让真实残差比递推值多出
        # eps32 * ||y|| 量级（y 是 Krylov 系数，系统越病态越大），不达标就接着迭代（即迭代改进）
        r = b - apply_A(x)
        beta = red.norm(r)
        if not np.isfinite(beta):
            return x, total, -1, float("nan")
        if beta <= target:
            return x, total, 0, beta / bnorm
        if exhausted:
            return x, total, total, beta / bnorm
        if converged:
            # 核实不达标：真实残差高于递推值，之前的历程不再代表下降趋势，停滞判据从这里重新起算
            history, history_start = [], total
