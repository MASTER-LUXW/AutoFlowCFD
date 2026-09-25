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
"""

from typing import Callable, Tuple

import numpy as np

from . import vector_ops
from .reductions import LocalReductions


def gmres_right(apply_A: Callable, b, apply_Minv: Callable, *, rtol: float,
                restart: int, max_iter: int, red: LocalReductions = None
                ) -> Tuple[object, int, int, float]:
    """解 `A x = b`，返回 `(x, iterations, info, rel_residual)`。

    `info == 0` 表示达到 `||b - A x|| <= rtol * ||b||`；`info > 0` 表示
    用满 `max_iter` 仍未达到（`x` 是当时的最好解）；`info < 0` 表示出现
    非有限值（`x` 不可用，调用方不得静默使用它）。

    `rel_residual` 是返回的 `x` 实际达到的 `||b - A x|| / ||b||`（Givens 递推
    给出的值，精确算术下即真实残差；重启处是重算的真实残差）。用满迭代数
    时它说明方向还有多少可信度——自适应 CFL 据此判断线性求解是否失败
    （`adaptive_cfl/ser.py`）。
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
    # Krylov 基预先整块分配（重启之间复用）：修正 Gram-Schmidt 的 `w -= h*V[i]`
    # 原地做，不再每次新分配一个整长向量（见 vector_ops.py）
    V = xp.empty((restart + 1,) + b.shape, dtype=b.dtype)
    while True:
        m = restart
        V[0] = r / beta
        H = np.zeros((m + 1, m))
        cs = np.zeros(m)
        sn = np.zeros(m)
        g = np.zeros(m + 1)
        g[0] = beta
        k_used = 0
        converged = False
        for j in range(m):
            w = apply_A(apply_Minv(V[j]))
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
            if total >= max_iter or h_next == 0.0:     # 用满 / 幸运击穿
                converged = h_next == 0.0
                break
            V[j + 1] = w / h_next
        # 回代 y，更新 x = x + M^{-1} (V y)
        y = np.linalg.solve(np.triu(H[:k_used, :k_used]), g[:k_used]) if k_used else np.zeros(0)
        upd = y[0] * V[0]
        for i in range(1, k_used):
            vector_ops.axpy_(xp, upd, y[i], V[i])
        x = x + apply_Minv(upd)
        rel = abs(g[k_used]) / bnorm
        if converged:
            return x, total, 0, rel
        if total >= max_iter:
            return x, total, total, rel
        r = b - apply_A(x)
        beta = red.norm(r)
        if not np.isfinite(beta):
            return x, total, -1, float("nan")
        if beta <= target:
            return x, total, 0, beta / bnorm
