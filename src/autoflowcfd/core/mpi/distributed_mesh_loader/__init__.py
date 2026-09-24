"""
AutoFlowCFD V2.0 - 完全分布式网格加载

优化内存使用：只有 root rank 加载完整网格，然后通过 MPI 将每个 rank 的
局部数据分发出去。非 root rank 不需要持有完整网格数据。

流程:
1. Root rank 加载完整网格 → 构建面连接关系 → 分区
2. Root rank 提取每个 rank 的局部数据（mesh geometry + face connectivity）
3. Root rank 通过 MPI 发送各 rank 的局部数据
4. 非 root rank 接收并构建局部网格对象

关键设计:
- 非 root rank 不再调用 load_mesh_for_solver()
- 局部网格使用重映射的 cell 索引（从 0 开始）
- 面连接关系的 owner_cell/neighbor_cell 使用局部索引
- halo cell 信息通过 partition 的 recv_lists 获取

## 文件分工（2026-09-24 拆包，原 1030 行）

    package.py            root 端预切的紧凑包（`PrecompactedMeshData` +
                          `build_fully_distributed_rank_package`）
    fully_distributed.py  完全分布式加载入口与跨阶数重分发
    legacy.py             CPU MPI"传统模式"分发（每 rank 各自加载完整网格）

本 `__init__.py` re-export 全部既有公开名，所以全仓库
`from autoflowcfd.core.mpi.distributed_mesh_loader import ...` 一个字
都不用改。
"""

from .package import (  # noqa: F401
    PrecompactedMeshData,
    build_fully_distributed_rank_package,
)
from .fully_distributed import (  # noqa: F401
    distributed_mesh_load_v2,
    redistribute_fully_distributed_for_new_order,
)
from .legacy import (  # noqa: F401
    build_local_mesh_from_data,
    distribute_mesh_data,
    distributed_mesh_load,
    extract_local_mesh_data,
)

__all__ = [
    "PrecompactedMeshData",
    "build_fully_distributed_rank_package",
    "build_local_mesh_from_data",
    "distribute_mesh_data",
    "distributed_mesh_load",
    "distributed_mesh_load_v2",
    "extract_local_mesh_data",
    "redistribute_fully_distributed_for_new_order",
]
