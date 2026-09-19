"""AutoFlowCFD V2.0 - 隐式时间推进（矩阵自由 Newton-Krylov + 伪瞬态延拓）。

    jfnk.py               Newton 外步 + GMRES 线性求解 + 物理性限幅
    jacobian_vector.py    Fréchet 差分的矩阵自由 `J v`
    preconditioner.py     伪瞬态（PTC）对角预处理
    forcing.py            inexact-Newton 的 Eisenstat-Walker forcing term

设计背景、为什么不组装 Jacobian、以及当前预处理的局限，见 `jfnk.py`
与各模块的文档。
"""

from .forcing import EisenstatWalkerForcing  # noqa: F401
from .jfnk import (  # noqa: F401
    GMRES_MAX_ITER,
    GMRES_RESTART,
    PHYSICALITY_MAX_RELATIVE_CHANGE,
    step_newton_krylov,
)

__all__ = [
    "EisenstatWalkerForcing",
    "GMRES_MAX_ITER",
    "GMRES_RESTART",
    "PHYSICALITY_MAX_RELATIVE_CHANGE",
    "step_newton_krylov",
]
