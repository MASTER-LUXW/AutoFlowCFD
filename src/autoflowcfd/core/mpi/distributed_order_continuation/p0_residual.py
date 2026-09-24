"""AutoFlowCFD V2.0 - P0 阶的分布式无粘残差（阶数延拓起点需要它单独一条路径）

从 `src/autoflowcfd/core/mpi/distributed_order_continuation.py`(原 640 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""

import numpy as np






def compute_distributed_p0_inviscid_residual(solver, U_local_p0: np.ndarray) -> np.ndarray:
    """P0（1 SP/cell）分布式无粘残差——CPU"传统模式"专用（见
    `DistributedFRSolver._p0_global_boundary_ghost_provider` 文档）。

    为什么不能复用 P1+ 路径的 `distributed_compute_inviscid_residual`：
    单机 `compute_inviscid_residual_fr` 在 `mesh.n_points_1d==1` 时
    短路到一条完全独立的有限体积实现（`inviscid_p0.py`），直接读取
    `mesh.face_connectivity` 的**原始三角化半面几何**
    （`fc.normal`/`fc.area`，未经"flat face"压缩抽象，用于 multi-source
    棱柱四边形侧面拆分面的 dedup 回退，见该文件模块文档"关于棱柱四边形
    侧面拆分的处理"一节）——P1+ 路径共用的 `DistributedFlatFaceGeometry`/
    `DistributedMeshAdapter` compact 索引空间抽象根本不携带这套原始
    半面几何，这不是"多传一个参数"就能接上的缺口，而是 P0 有限体积
    kernel 本身在设计上就是**全局**的（对 `mesh.n_cells`/
    `mesh.face_connectivity.n_faces` 逐面/逐单元 scatter-add），从未
    考虑过按 rank 拆分。

    解决方式（"传统模式"下每个 rank 已经持有完整全局 `mesh`，这是该
    模式"不是内存最优"的既有设计取舍——见 `DistributedFRSolver.
    __init__` 文档——的自然延伸）：各 rank 把自己的 local U 通过
    gather+broadcast 组装成全局 U，各自独立调用单机的
    `compute_inviscid_residual_fr(U_global, solver.mesh, ...)` 算出
    全局残差（与单机路径逐位一致，因为就是同一个函数、同一份完整
    网格），再只取自己 `local_cells` 那部分。P0 阶段只是 Order
    Continuation 爬坡最初、最短暂的一段（典型 `stage_iter_budget`
    只有几十步），这个额外的 gather+broadcast 通信开销可以接受——
    不是长期热路径。

    "完全分布式加载"模式（`solver._is_fully_distributed is True`）用
    同一个思路的变体：本 rank 没有完整全局网格，但 root（`solver.
    _root_context`，见 `distributed_mesh_loader.py::distributed_mesh_
    load_v2`/`redistribute_fully_distributed_for_new_order` 文档）
    持续持有——gather 全局 U 到 root 之后，只有 root 能算
    `compute_inviscid_residual_fr(U_global, root_context['mesh'], ...)`，
    算完的全局残差再 broadcast 给全部 rank（不是只 broadcast 回
    root_context 本身——非 root rank 从始至终不需要、也不会拿到完整
    网格，只需要最终的残差数值）。`root_context` 为 None（未提供）时
    fail-fast，不静默产生错误结果。
    """
    from autoflowcfd.core.mpi.distributed_checkpoint import gather_global_state
    from autoflowcfd.core.mpi.comm import bcast_from_root
    from autoflowcfd.core.mpi import get_rank
    from autoflowcfd.core.fr_residual.inviscid import compute_inviscid_residual_fr

    n_global = solver.partition.n_global_cells
    local_cells = solver.partition.local_cells
    U_global = gather_global_state(U_local_p0, local_cells, n_global)

    if getattr(solver, '_is_fully_distributed', False):
        if get_rank() == 0:
            root_context = getattr(solver, '_root_context', None)
            if root_context is None:
                raise NotImplementedError(
                    "compute_distributed_p0_inviscid_residual: root rank 没有 "
                    "_root_context（本实例不是通过 distributed_mesh_load_v2 "
                    "返回的 root_context 构造的），P0 阶段的 Order "
                    "Continuation 在'完全分布式加载'模式下不支持（如实报告，"
                    "不是假装能用）。"
                )
            residual_global = compute_inviscid_residual_fr(
                U_global, root_context['mesh'], root_context['ops'],
                root_context['boundary_ghost_provider_global'],
                mach_ref=root_context['mach_ref'],
            )
        else:
            residual_global = None
        residual_global = bcast_from_root(residual_global)
        return residual_global[local_cells]

    provider = getattr(solver, '_p0_global_boundary_ghost_provider', None)
    if provider is None:
        raise NotImplementedError(
            "compute_distributed_p0_inviscid_residual: 本 solver 实例没有"
            "全局边界幽灵态提供者。"
        )

    U_global = bcast_from_root(U_global)
    mach_ref = solver.local_solver.freestream["mach_ref"]
    residual_global = compute_inviscid_residual_fr(
        U_global, solver.mesh, solver.ops, provider, mach_ref=mach_ref,
    )
    return residual_global[local_cells]
