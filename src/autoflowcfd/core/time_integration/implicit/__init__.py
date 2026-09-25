"""AutoFlowCFD V2.0 - 隐式时间推进（矩阵自由 Newton-Krylov + 伪瞬态延拓）。

    jfnk.py               Newton 外步 + GMRES 线性求解 + 物理性限幅
    jacobian_vector.py    Fréchet 差分的矩阵自由 `J v`
    preconditioner.py     伪瞬态（PTC）对角预处理
    block_jacobi.py       单元块 Jacobi 预处理（着色有限差分装配、跨步复用）
    forcing.py            inexact-Newton 的 Eisenstat-Walker forcing term
    dtau_control.py       `dtau` 缩放（由 Newton 步成败驱动，治停滞）

设计背景、为什么不组装 Jacobian、以及当前预处理的局限，见 `jfnk.py`
与各模块的文档。
"""

from .block_jacobi import BlockJacobiCache  # noqa: F401
from .dtau_control import PtcDtauScale  # noqa: F401
from .forcing import EisenstatWalkerForcing  # noqa: F401
from .jfnk import (  # noqa: F401
    DTAU_MAX_CUTS_PER_STEP,
    GMRES_MAX_ITER,
    GMRES_RESTART,
    PHYSICALITY_MAX_RELATIVE_CHANGE,
    step_newton_krylov,
)

__all__ = [
    "BlockJacobiCache",
    "DTAU_MAX_CUTS_PER_STEP",
    "EisenstatWalkerForcing",
    "GMRES_MAX_ITER",
    "GMRES_RESTART",
    "PHYSICALITY_MAX_RELATIVE_CHANGE",
    "PtcDtauScale",
    "step_newton_krylov",
]
