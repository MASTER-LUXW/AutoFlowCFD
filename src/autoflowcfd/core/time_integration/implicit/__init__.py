"""AutoFlowCFD V2.0 - 隐式时间推进（矩阵自由 Newton-Krylov + 伪瞬态延拓）。

子包：

    newton/   Newton 外步与伪时间控制（单步 / 紧耦合步 / 双时间步内迭代、接受判据、物理性限幅、dtau 缩档、
              Eisenstat-Walker forcing）
    krylov/   线性求解（右预处理 GMRES、重启长度与真实行压缩、矩阵自由 `J v`）
    blocks/   单元块预处理（着色差分 / 解析装配、块 Jacobi、块 ILU、内存预算与分档、对角位移、刷新判据）
    coarse/   多层预处理的粗层（聚合、Galerkin 粗矩阵、K 循环、分布式全局粗校正）

顶层模块（各子包共用）：

    reductions.py   全局归约（单进程 / 分布式共用一个接口）
    vector_ops.py   长向量运算（内积、原地 axpy）

设计背景、为什么不组装全局 Jacobian、以及预处理的局限，见 `newton/jfnk.py` 与各模块的文档。
"""

from .blocks.block_jacobi import BlockJacobiCache  # noqa: F401
from .krylov.linear_solve import GMRES_MAX_ITER, GMRES_RESTART  # noqa: F401
from .newton.dtau_control import PtcDtauScale  # noqa: F401
from .newton.forcing import EisenstatWalkerForcing  # noqa: F401
from .newton.jfnk import DTAU_MAX_CUTS_PER_STEP, step_newton_krylov  # noqa: F401
from .newton.physicality import PHYSICALITY_MAX_RELATIVE_CHANGE  # noqa: F401

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
