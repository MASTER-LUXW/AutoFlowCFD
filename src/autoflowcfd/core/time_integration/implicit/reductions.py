"""AutoFlowCFD V2.0 - 隐式求解里的全局归约（单进程 / 分布式共用一个接口）。

Newton-Krylov 里凡是"对整个解向量取一个标量"的地方——GMRES 的内积与
范数、残差 RMS、Fréchet 差分步长里的 RMS、物理性限幅的全局最小 theta、
非有限值检查——在分布式后端上都必须是**跨 rank 的全局量**，否则每个
rank 会各自解一个不同的线性系统、走不同长度的步。

此前 `jfnk.py` 直接调用 `np.linalg.norm` / `scipy.sparse.linalg.gmres`，
于是隐式路径只能在单进程 numpy 上跑。本模块把这些归约收成一个对象：
算法代码只调用它，后端差异（numpy / cupy / MPI allreduce）只在这里。

`LocalReductions(xp)`：单进程，`xp` 为 numpy 或 cupy。分布式后端传一个
覆盖 `_allreduce_*` 的子类（见 `core/mpi/`），算法代码一字不改。
"""

import math

import numpy as np


class LocalReductions:
    """单进程归约。所有返回值都是 Python 标量（跨后端可比）。"""

    __slots__ = ("xp",)

    def __init__(self, xp=np):
        self.xp = xp

    # ---- 分布式子类覆盖这三个：把本进程的部分结果变成全局结果 ----
    def _allreduce_sum(self, value: float) -> float:
        return value

    def _allreduce_min(self, value: float) -> float:
        return value

    def _allreduce_max(self, value: float) -> float:
        return value

    # ---- 算法代码用的接口 ----
    def sum(self, x) -> float:
        return self._allreduce_sum(float(self.xp.sum(x)))

    def count(self, x) -> float:
        return self._allreduce_sum(float(x.size))

    def min(self, x) -> float:
        local = float(self.xp.min(x)) if x.size else math.inf
        return self._allreduce_min(local)

    def max(self, x) -> float:
        local = float(self.xp.max(x)) if x.size else -math.inf
        return self._allreduce_max(local)

    def all_finite(self, x) -> bool:
        local = 1.0 if bool(self.xp.all(self.xp.isfinite(x))) else 0.0
        return self._allreduce_min(local) > 0.0

    def dot(self, a, b) -> float:
        return self._allreduce_sum(float(self.xp.vdot(a.ravel(), b.ravel())))

    def norm(self, x) -> float:
        return math.sqrt(max(self.dot(x, x), 0.0))

    def rms(self, x) -> float:
        n = self.count(x)
        return math.sqrt(self.dot(x, x) / n) if n > 0 else 0.0
