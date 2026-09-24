"""
AutoFlowCFD V2.0 - 分布式面几何

将 FlatFaceGeometry 改造为分布式版本：每个 rank 只持有 owner 为 local cell 的面。
面分类为 interior / partition_boundary / physical_boundary。

关键设计:
- 不重排全局面序（保持与原始面序一致，满足退化 Jacobian 敏感性约束）
- 使用 mask 索引分类面（interior_mask, partition_boundary_mask 等）
- partition_boundary 面的 neighbor 数据来自 halo cell（通过 halo 交换获取）
- 面几何数组的索引空间从全局 cell 转为 local+halo 扩展索引

扩展索引约定:
- [0, n_local_cells): local cells
- [n_local_cells, n_total_cells): halo cells
- FlatFaceGeometry 中的 owner_cell/neighbor_cell 使用扩展索引


## 文件分工(2026-09-24 拆包, 原 552 行)

    types.py             分布式扁平面几何的数据类与紧凑 src1 展开
    build.py             从局部网格 + 分区构造分布式扁平面几何

本 `__init__.py` re-export 全部既有名, 所以全仓库导入一字不改。
"""

from .types import (  # noqa: F401
    DistributedFlatFaceGeometry,
    _expand_compact_src1,
)
from .build import (  # noqa: F401
    build_distributed_flat_face,
)

__all__ = [
    "DistributedFlatFaceGeometry",
    "build_distributed_flat_face",
]
