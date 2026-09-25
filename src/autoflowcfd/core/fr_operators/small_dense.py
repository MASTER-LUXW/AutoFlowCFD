"""AutoFlowCFD V2.0 - 并行核内的小矩阵乘（显式循环，不调 BLAS）。

## 为什么不在 `prange` 里用 `@` / `np.dot`（2026-09-25 实测）

界面核逐面做"外插矩阵 × 单元解点值"、"提升矩阵 × 加权跳跃"这类小矩阵乘
（尺寸 ~10×6×5）。numba 把 `@` 编译成对 BLAS 的调用；在 `prange` 的多个线程里
逐面调用 OpenBLAS，线程越多越慢（BLAS 每次调用都有全局的缓冲区加锁与调度开销）。
同一个 20 万次 9×6 @ 6×5 的微基准：

| 写法                         | 1 线程   | 16 线程  | 加速比 |
|---|---|---|---|
| `E @ X`（每次分配输出）       | 37.0 ms  | 152.7 ms | 0.24 |
| `np.dot(E, X, out)`          | 25.8 ms  | 162.3 ms | 0.16 |
| 显式三重循环                  | 7.3 ms   | 2.2 ms   | 3.25 |

plate_demo P1+SST（17.9 万单元）一次平均流残差求值里，两个界面核在 16 线程下只有
1.30 倍（粘性）与 0.99 倍（无粘）加速、占总时间 76%，整次求值 4 线程最快（2.4 s）、
16 线程反而 3.6 s。

求和顺序固定为 `sum_k A[i,k] B[k,j]` 按 k 递增：与 BLAS 的内部顺序不同，结果与改动前
差浮点重结合量级（相对 1e-16），这是唯一的数值差别。
"""

import numpy as np
from numba import njit


@njit(cache=True, inline='always')
def matmul_small(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """`A (m,n) @ B (n,k)` -> `(m,k)`。"""
    m, n = A.shape
    k = B.shape[1]
    C = np.empty((m, k))
    for i in range(m):
        for j in range(k):
            acc = 0.0
            for s in range(n):
                acc += A[i, s] * B[s, j]
            C[i, j] = acc
    return C


@njit(cache=True, inline='always')
def matvec_small(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """`A (m,n) @ x (n,)` -> `(m,)`。"""
    m, n = A.shape
    y = np.empty(m)
    for i in range(m):
        acc = 0.0
        for s in range(n):
            acc += A[i, s] * x[s]
        y[i] = acc
    return y


@njit(cache=True, inline='always')
def extrap_tensor3x3(field_cell: np.ndarray, E: np.ndarray) -> np.ndarray:
    """`(n_sps,3,3)` 场按外插矩阵 `E (n_fp,n_sps)` 外插到 `(n_fp,3,3)`。"""
    n_fp, n_sps = E.shape
    out = np.empty((n_fp, 3, 3))
    for i in range(n_fp):
        for a in range(3):
            for b in range(3):
                acc = 0.0
                for s in range(n_sps):
                    acc += E[i, s] * field_cell[s, a, b]
                out[i, a, b] = acc
    return out
