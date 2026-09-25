"""AutoFlowCFD V2.0 - FR 求解器主类（包）。

从单文件 `core/fr_solver/solver.py`（1244 行）拆成子包（2026-09-25，项目「单文件
不超 500 行」规范）：`core`（类与装配顺序）、`setup`（`__init__` 的装配阶段）、
`residuals`（残差与委托）、`solve_loop`（求解循环）、`threads`（BLAS/numba 线程）。
`mach_ref` 的计算在同级 `core/fr_solver/mach_ref.py`（CLI 的分布式分支也要用）。
"""

from .core import FRSolver  # noqa: F401
from .threads import _limit_blas_threads, blas_threads_limited, configure_numba_threads  # noqa: F401

__all__ = ["FRSolver", "blas_threads_limited", "configure_numba_threads"]
