"""AutoFlowCFD V2.0 - 块预处理的粗空间（聚合层次、Galerkin 粗算子、K 循环多层预处理）。

    aggregation.py   单元耦合图上的分层贪心聚合（拓扑量）
    galerkin.py      Galerkin 粗算子 P^T A P、转移算子、装配矩阵的矩阵-向量乘
    multilevel.py    多层预处理子（细层块 ILU 光滑 + 粗层 K 循环 + 最粗层直接解）
    global_coarse.py 分布式的全局粗校正（跨 rank 粗矩阵，各 rank 冗余直接解）
    selection.py     块 ILU 档的预处理构造（本地多层 / 块 ILU，分布式叠全局粗校正）
"""

from .aggregation import build_hierarchy  # noqa: F401
from .global_coarse import CoarseCommContext  # noqa: F401
from .multilevel import COARSEST_DOF, MultilevelPreconditioner, preconditioner_hierarchy  # noqa: F401
from .selection import CoarsePreconditionerFactory  # noqa: F401

__all__ = ["build_hierarchy", "COARSEST_DOF", "CoarseCommContext", "CoarsePreconditionerFactory",
           "MultilevelPreconditioner", "preconditioner_hierarchy"]
