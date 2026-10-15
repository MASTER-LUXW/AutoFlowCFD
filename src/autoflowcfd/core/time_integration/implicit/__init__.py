"""AutoFlowCFD V2.0 - 隐式时间推进（矩阵自由 Newton-Krylov + 伪瞬态延拓）。

    jfnk.py               Newton 外步（物理性限幅、残差接受判据、dtau 缩档）
    linear_solve.py       Newton 步里的 GMRES 线性求解（重启长度、真实行压缩、加权范数）
    jacobian_vector.py    Fréchet 差分的矩阵自由 `J v`
    preconditioner.py     伪瞬态（PTC）对角预处理
    block_jacobi.py       单元块预处理缓存（装配、档位、刷新与复用）
    cell_block_inverse.py 单元块逐块求逆与块 Jacobi 预处理作用
    forcing.py            inexact-Newton 的 Eisenstat-Walker forcing term
    dtau_control.py       `dtau` 缩放（由 Newton 步成败驱动，治停滞）

设计背景、为什么不组装 Jacobian、以及当前预处理的局限，见 `jfnk.py`
与各模块的文档。
"""

from .block_jacobi import BlockJacobiCache  # noqa: F401
from .dtau_control import PtcDtauScale  # noqa: F401
from .forcing import EisenstatWalkerForcing  # noqa: F401
from .jfnk import DTAU_MAX_CUTS_PER_STEP, step_newton_krylov  # noqa: F401
from .linear_solve import GMRES_MAX_ITER, GMRES_RESTART  # noqa: F401
from .physicality import PHYSICALITY_MAX_RELATIVE_CHANGE  # noqa: F401

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
