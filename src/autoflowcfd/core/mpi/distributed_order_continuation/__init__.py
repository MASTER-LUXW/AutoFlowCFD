"""
AutoFlowCFD V2.0 - 分布式 Order Continuation。

三条分布式路径（CPU MPI"传统模式"、CPU"完全分布式加载"、多 GPU 分布式）与单机一样在目标阶数 >= 2 时自动
执行 Order Continuation（P0 -> P1 -> ... -> 目标阶数逐步升阶，迭代循环是同一份
`core/utils/order_continuation/run.py::run_order_continuation`）。

架构难点：架构核心难点（此前如实汇报未完成的原因，本次真正解决）：
`DistributedFRSolver.partition`/`.dist_flat_face`（`DistributedFlatFace
Geometry`）都是**按阶数固化**的对象——阶数一变，`n_sps`/FR Flux Point
多源交叉插值依赖关系都变了，必须整体重建，不能像单机 `HighOrderMesh.
set_order` 那样只换个缓存条目。三条后端的重建成本各不相同：

- **CPU MPI"传统模式"**（每个 rank 独立持有完整全局网格）：`mesh.
  face_connectivity`（拓扑）与阶数无关（见 `grid/high_order/high_order_mesh_order.py` 的
  set_order 文档——只有 `face_flux_points` 随阶数重建），`cell_
  partition`（每个 cell 属于哪个 rank）同样是纯拓扑量、阶数无关——
  只需要用同一个 `cell_partition` 重新调用 `build_distributed_
  partition`/`build_distributed_flat_face`（**全新对象**，不是原地
  复用旧的——旧对象已经为旧阶数的 FP 交叉引用需求扩展过 halo，见下面
  `cpu_traditional_interpolate_to_new_order` 文档"为什么重建而不是
  复用"一节），不需要任何新的 MPI 通信。

- **多 GPU 分布式**：`MultiGPUDistributedSolver` 同样在"传统模式"下
  持有完整全局 `mesh` + `cell_partition`（见 `core/gpu/distributed/gpu_distributed/`），重建逻辑与 CPU
  完全对称，唯一区别是重建后的紧凑几何量
  需要重新上传到 GPU（对应模块的 `_upload_geometry_to_gpu`/等价逻辑）。

- **CPU"完全分布式加载"**：本 rank 从未持有完整全局网格（内存优化的
  核心卖点），阶数切换必须由 root 重新计算+重新分发紧凑包——一个新的
  运行时协议（`redistribute_fully_distributed_for_new_order`），root
  端需要在初次分发之后继续持有 `mesh`/`ops`/`cell_partition`/
  `face_connectivity`/`boundary_ghost_provider_global` 等（见
  `core/mpi/distributed_mesh_loader/fully_distributed.py::distributed_mesh_load_v2` 的
  `root_context` 返回值），非 root rank 参与对应的 Recv 一侧。

迭代循环与单机后端是同一份（`core/utils/order_continuation/run.py::run_order_continuation`），只是阶数切换时调用各自的
`solver._interpolate_to_new_order(target_p)`（在
`DistributedFRSolver._interpolate_to_new_order`/
`MultiGPUDistributedSolver._interpolate_to_new_order` 里按构造方式
分派到本模块对应的重建函数）。

## 文件分工

    rebuild.py           阶数切换时重建 CPU 传统模式的分区与状态、逐 rank 插值

"""

from .rebuild import cpu_traditional_interpolate_to_new_order  # noqa: F401
# resume 钳制检测是全部后端共用的唯一实现（`core/utils/order_continuation`）。
from autoflowcfd.core.utils.order_continuation import (  # noqa: F401
    check_resumed_turbulence,
)

__all__ = [
    "cpu_traditional_interpolate_to_new_order",
]
