"""AutoFlowCFD V2.0 - 隐式求解的长向量运算（内积、原地 axpy）。

GMRES 的修正 Gram-Schmidt 每个 Arnoldi 步要做 j 次内积与 j 次 `w -= h*V[i]`。
P1 棱柱网格 17.9 万单元时一个向量有 720 万个 float64（57 MB）：numpy 的
`w - h*V[i]` 每次新分配一个临时数组、`np.vdot` 在求解期间被限成单线程 BLAS
（`fr_solver/solver/threads.py::blas_threads_limited`），plate_demo P1+SST 一个隐式步
里 GMRES 自身的向量运算约 4 s（整步约 30 s）。numpy 后端改用 numba 并行、原地的
版本；cupy 后端仍用数组表达式（GPU 上是并行的）。

内积改为分块并行求和后，求和顺序与 BLAS 不同，只差浮点重结合——对 Krylov 求解
这是正常的舍入扰动，不改变收敛判据的含义。
"""

import numpy as np
from numba import njit, prange

#: 小于这个长度的向量直接用 numpy（并行启动开销不划算）。
_PARALLEL_MIN = 65536


@njit(cache=True, parallel=True)
def _par_dot(a, b):
    n = a.shape[0]
    acc = 0.0
    for i in prange(n):
        acc += a[i] * b[i]
    return acc


@njit(cache=True, parallel=True)
def _par_axpy(y, alpha, x):
    for i in prange(y.shape[0]):
        y[i] += alpha * x[i]


def dot(xp, a, b) -> float:
    """本进程部分的内积 `sum(a*b)`（全局归约由 `reductions.py` 负责）。"""
    if xp is np and a.size >= _PARALLEL_MIN:
        return float(_par_dot(np.ascontiguousarray(a).reshape(-1),
                              np.ascontiguousarray(b).reshape(-1)))
    return float(xp.vdot(a.ravel(), b.ravel()))


def axpy_(xp, y, alpha: float, x) -> None:
    """原地 `y += alpha * x`（`y` 必须是可写的连续数组）。"""
    if xp is np and y.size >= _PARALLEL_MIN:
        if not y.flags.c_contiguous:
            raise ValueError("axpy_ 的 y 必须是连续数组（否则 reshape 得到拷贝、原地写入丢失）")
        _par_axpy(y.reshape(-1), float(alpha), np.ascontiguousarray(x).reshape(-1))
    else:
        y += alpha * x
