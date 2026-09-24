"""
AutoFlowCFD V2.0 - MultiGPUDistributedSolver "完全分布式加载"构造入口
（2026-09-02，问题清单 #1：多 GPU "完全分布式加载"/内存最优模式此前
从未实现，只支持"传统模式"——每个 rank 独立加载完整全局网格）。

镜像 CPU `DistributedFRSolver.from_fully_distributed_package`（见
`core/mpi/distributed_mesh_loader.py`/`core/mpi/distributed_solver.py`
模块文档）同一套设计：root rank 预先用完整全局网格算好每个 rank 的
紧凑（local+halo 压缩索引空间）数据包（`build_fully_distributed_rank_
package`/`distributed_mesh_load_v2`，两者本身与后端无关，CPU/GPU 共用
同一套 package 构造逻辑），本 rank 只接收这份紧凑包，从未持有、也不
需要持有完整全局网格——这是"完全分布式加载"名副其实的内存优化。

与 CPU 版本的关键区别只在于：紧凑几何/状态数据构造好之后需要一次性
上传到 GPU（`GPUArrayManager.upload_mesh_data`/`GPUFlatFaceGeometry`/
`cp.asarray`），且湍流模型用 GPU 版类（`GPUTurbulenceSST`/`GPUDDESModel`/
`GPUIDDESModel`/`GPUWALEModel`）而不是 CPU 版——除此之外，package 的
构造、字段含义、范围边界（支持 turbulence_model='none'/'sst'/'ddes'/
'iddes'/'wmles'/'les'，DUAL_TIME 已接入）与 CPU 版完全一致，直接复用
同一个 `build_fully_distributed_rank_package`/`distributed_mesh_load_v2`，
不重新实现一遍。

Order Continuation 支持：`redistribute_multi_gpu_fully_distributed_for_
new_order`（本模块）与 CPU 版 `redistribute_fully_distributed_for_new_
order`（`distributed_mesh_loader.py`）同一套协议——root 用持续持有的
`solver._root_context`（`distributed_mesh_load_v2` 返回的第二个值）
重新计算 + 重新分发新阶数的紧凑包，本 rank 用它替换全部 compact 相关
属性（不重新构造 solver 实例本身）。由
`gpu_distributed_order_continuation.py::gpu_interpolate_to_new_order`
按 `solver._is_fully_distributed` 分派到这里。


## 文件分工(2026-09-24 拆包, 原 557 行)

    upload.py            把紧凑索引空间的壁面几何上传到设备
    build.py             从完全分布式包构造多 GPU solver
    redistribute.py      阶数切换时重新分发（Order Continuation 分布式路径）

本 `__init__.py` re-export 全部既有名, 所以全仓库导入一字不改。
"""

from .upload import (  # noqa: F401
    _upload_wall_geometry_compact,
)
from .build import (  # noqa: F401
    build_multi_gpu_solver_from_fully_distributed_package,
)
from .redistribute import (  # noqa: F401
    redistribute_multi_gpu_fully_distributed_for_new_order,
)

__all__ = [
    "build_multi_gpu_solver_from_fully_distributed_package",
    "redistribute_multi_gpu_fully_distributed_for_new_order",
]
