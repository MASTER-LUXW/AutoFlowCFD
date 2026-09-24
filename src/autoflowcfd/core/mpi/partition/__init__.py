"""
AutoFlowCFD V2.0 - 网格分区（METIS 接口）

将全局网格分割为若干子域，每个 MPI rank 负责一个子域。

分区算法:
    1. 从 FRFaceConnectivity 构建单元邻接图（cell → cell via shared face）
    2. 调用 METIS 的 part_graph 将图分成 n_parts 个分区
    3. 输出 DistributedPartition 数据结构

分区质量:
    METIS 的图分区算法最小化分区间的切割边数（= 跨 rank halo 交换量），
    同时保持各分区大小近似均衡（负载平衡）。对结构良好的非结构网格，
    切割边数通常为 O(n_faces^(2/3))，远优于随机分区。

面分类:
    分区后每个 rank 的面分为四类：
    - interior: owner 和 neighbor 都是 local cell
    - partition_boundary: owner 是 local，neighbor 在另一个 rank
    - physical_boundary: 原始物理边界面（is_boundary=True）
    - halo: neighbor 是 local，owner 在另一个 rank（用于接收校正）

    注意：物理边界面优先级高于 partition_boundary——如果一个面既是
    物理边界又是分区边界，它被归类为 physical_boundary（因为 neighbor=-1，
    不需要 halo 交换）。


## 文件分工(2026-09-24 拆包, 原 612 行)

    types.py             分区结果的数据结构
    build.py             邻接图构造与分区
    halo.py              面通量点跨引用所需的 halo 扩展

本 `__init__.py` re-export 全部既有名, 所以全仓库导入一字不改。
"""

from .types import (  # noqa: F401
    DistributedPartition,
    FaceClassification,
)
from .build import (  # noqa: F401
    build_cell_adjacency_graph,
    build_distributed_partition,
    partition_mesh,
)
from .halo import (  # noqa: F401
    extend_halo_for_flux_point_cross_references,
)

__all__ = [
    "DistributedPartition",
    "FaceClassification",
    "build_cell_adjacency_graph",
    "build_distributed_partition",
    "extend_halo_for_flux_point_cross_references",
    "partition_mesh",
]
