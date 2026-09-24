"""
AutoFlowCFD V2.0 - 分布式 Order Continuation（2026-09-02）

补齐用户明确要求（"除 cube_demo 网格质量门、VTK 高阶导出外其余都优化"、
"此前就是明确的范围边界"这种表述绝对不允许出现）后排查确认的最后一项
真实缺口：单机 `FRSolver.solve()` 在目标阶数 >= 2 时**自动**执行 Order
Continuation（P0 -> P1 -> ... -> 目标阶数逐步升阶，见
`core/utils/order_continuation.py::run_order_continuation`），但三条
分布式路径（CPU MPI"传统模式"、CPU"完全分布式加载"、多 GPU 分布式）
此前完全没有这个机制——`solve_steady_command.py` 甚至显式 fail-fast
拒绝 `--phase-max-iter`/`--residual-drop-threshold` 搭配这些后端
（"求解器没有 Order Continuation 机制"），意味着 `solve steady --order 2
--n-ranks 4`（或 `--multi-gpu`）此前直接从均匀自由流场在目标阶数
（P2/P3）上开始求解——这正是 Order Continuation 本来要规避的高风险
数值起步方式（见 order_continuation.py 模块文档"resume/P0 重置"一节
的历史教训）。

架构核心难点（此前如实汇报未完成的原因，本次真正解决）：
`DistributedFRSolver.partition`/`.dist_flat_face`（`DistributedFlatFace
Geometry`）都是**按阶数固化**的对象——阶数一变，`n_sps`/FR Flux Point
多源交叉插值依赖关系都变了，必须整体重建，不能像单机 `HighOrderMesh.
set_order` 那样只换个缓存条目。三条后端的重建成本各不相同：

- **CPU MPI"传统模式"**（每个 rank 独立持有完整全局网格）：`mesh.
  face_connectivity`（拓扑）与阶数无关（见 `high_order_mesh_order.py::
  set_order` 文档——只有 `face_flux_points` 随阶数重建），`cell_
  partition`（每个 cell 属于哪个 rank）同样是纯拓扑量、阶数无关——
  只需要用同一个 `cell_partition` 重新调用 `build_distributed_
  partition`/`build_distributed_flat_face`（**全新对象**，不是原地
  复用旧的——旧对象已经为旧阶数的 FP 交叉引用需求扩展过 halo，见下面
  `cpu_traditional_interpolate_to_new_order` 文档"为什么重建而不是
  复用"一节），不需要任何新的 MPI 通信。

- **多 GPU 分布式**：`MultiGPUDistributedSolver` 同样在"传统模式"下
  持有完整全局 `mesh` + `cell_partition`（见 `gpu_distributed.py::
  __init__`），重建逻辑与 CPU 完全对称，唯一区别是重建后的紧凑几何量
  需要重新上传到 GPU（对应模块的 `_upload_geometry_to_gpu`/等价逻辑）。

- **CPU"完全分布式加载"**：本 rank 从未持有完整全局网格（内存优化的
  核心卖点），阶数切换必须由 root 重新计算+重新分发紧凑包——一个新的
  运行时协议（`redistribute_fully_distributed_for_new_order`），root
  端需要在初次分发之后继续持有 `mesh`/`ops`/`cell_partition`/
  `face_connectivity`/`boundary_ghost_provider_global` 等（见
  `distributed_mesh_loader.py::distributed_mesh_load_v2` 的
  `root_context` 返回值），非 root rank 参与对应的 Recv 一侧。

三条路径共用同一套残差-下降判据/checkpoint 回调/打印格式的迭代循环
（`run_distributed_order_continuation`），只是阶数切换时调用各自的
`solver._interpolate_to_new_order(target_p)`（在
`DistributedFRSolver._interpolate_to_new_order`/
`MultiGPUDistributedSolver._interpolate_to_new_order` 里按构造方式
分派到本模块对应的重建函数）。


## 文件分工(2026-09-24 拆包, 原 640 行)

    p0_residual.py       P0 阶的分布式无粘残差（阶数延拓起点需要它单独一条路径）
    rebuild.py           阶数切换时重建 CPU 传统模式的分区与状态、逐 rank 插值
    run.py               顶层编排：按 phase 推进各阶数

本 `__init__.py` re-export 全部既有名, 所以全仓库导入一字不改。
"""

from .p0_residual import (  # noqa: F401
    compute_distributed_p0_inviscid_residual,
)
from .rebuild import (  # noqa: F401
    _interp_state_and_turbulence_local,
    _rebuild_cpu_traditional_partition_and_state,
    cpu_traditional_interpolate_to_new_order,
)
# resume 钳制检测是全部后端共用的**唯一实现**，住在后端中立的
# `core/utils/order_continuation`（2026-09-24 合并：此前单机内联一份、
# 这里再写一份）。re-export 让既有导入一字不改。
from autoflowcfd.core.utils.order_continuation import (  # noqa: F401
    _reset_turbulence_if_resumed_field_exploded,
)
from .run import (  # noqa: F401
    run_distributed_order_continuation,
)

__all__ = [
    "compute_distributed_p0_inviscid_residual",
    "cpu_traditional_interpolate_to_new_order",
    "run_distributed_order_continuation",
]
