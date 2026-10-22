"""AutoFlowCFD V2.0 - 阶数切换时重建 CPU 传统模式的分区与状态、逐 rank 插值"""

import numpy as np

from autoflowcfd.core.utils.order_continuation.order_switch import (
    install_turbulence, reset_order_dependent_state, switch_mesh_order, switched_fields,
)


def _rebuild_cpu_traditional_partition_and_state(solver, target_p: int, new_local_U: np.ndarray) -> None:
    """"传统模式"阶数切换的共用收尾：mesh/ops 切换 + 全新
    partition/dist_flat_face 重建 + state 构造 + halo 交换器重建 +
    wall_distance/IDDES 几何重新切片 + boundary_ghost_provider 失效。

    调用方（`cpu_traditional_interpolate_to_new_order`/
    `_reset_cpu_traditional_to_p0`）负责在调用**之前**把
    `solver.turb_model.k_field`/`.omega_field`/`.nu_t`/
    `solver.sgs_model.nu_t` 处理成新阶数的形状
    （插值或重置，两条路径的处理方式不同）——这些字段只按
    `n_local`（阶数切换不变，见下）存储，与本函数要重建的
    partition/dist_flat_face 无关，不需要等这里的重建完成。

    为什么每次都整体重建 partition/dist_flat_face，而不是原地复用/
    扩展旧对象：`build_distributed_flat_face` 内部的 `extend_halo_
    for_flux_point_cross_references` 会原地扩展传入的 `partition`
    以覆盖 FR Flux Point 多源交叉插值（src0/src1）依赖——这个依赖集合
    随阶数变化（阶数越高，每个面的 Flux Point 越多，交叉引用的邻居
    单元集合可能不同）。如果复用一个已经为旧阶数扩展过 halo 的
    `partition` 对象再喂给新阶数的 `build_distributed_flat_face`，
    扩展逻辑是纯粹的"发现即添加"（不会移除不再需要的旧扩展条目），
    虽然不会产生错误的结果（旧扩展条目对新阶数而言至多是"多余但无害"
    的额外 halo），但会让 halo 层随阶数切换单调膨胀、且 `dist_flat_
    face.compact_global_ids` 的排列会依赖"调用历史"而不是"当前阶数"
    本身，不透明也不必要——用同一个 `cell_partition`（`cell_partition`
    本身是纯拓扑量，见模块文档，阶数切换不变）每次重新调用
    `build_distributed_partition`（全新对象，local_cells/初始 halo_
    cells 完全由 `cell_partition` 决定，与调用历史无关）更清晰、更
    可预测。

    Args:
        new_local_U: (n_local, new_n_sps, n_vars)，调用方已经算好的
            新阶数 local 守恒变量（插值结果或均匀自由流场）。
    """
    from autoflowcfd.core.mpi.partition import build_distributed_partition
    from autoflowcfd.core.mpi.distributed_flat_face import build_distributed_flat_face
    from autoflowcfd.core.mpi.distributed_state import DistributedFRState
    from autoflowcfd.core.mpi.halo import HaloExchange
    from autoflowcfd.core.mpi.distributed_turbulence import (
        apply_distributed_wall_distance, compute_distributed_wall_distance,
    )
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive

    cell_partition = solver._oc_cell_partition

    # mesh 是"传统模式"下每个 rank 都持有的完整全局网格——阶数切换与
    # 单机 run_order_continuation 同一套流程（先清理旧阶数几何缓存，
    # 再 set_order，避免大网格同时驻留多份完整阶数几何 OOM，见该函数
    # 文档"性能说明"一节，同一个理由在分布式"传统模式"下依然成立，
    # 每个 rank 各自持有一份完整网格，OOM 风险不小于单机）。
    solver.ops = switch_mesh_order(solver.mesh, target_p)

    new_partition = build_distributed_partition(
        solver.mesh.face_connectivity, cell_partition, solver.rank, solver.n_ranks
    )
    new_dist_fc = build_distributed_flat_face(
        solver.mesh, solver.ops, new_partition, cell_partition=cell_partition
    )

    n_local = new_partition.n_local_cells
    new_n_sps = solver.ops.n_sps
    n_vars = solver.state.n_vars
    new_state = DistributedFRState(new_partition, new_n_sps)
    new_state.U[:n_local] = new_local_U
    new_state.Q[:n_local] = conserved_to_primitive(new_local_U[..., :5])

    solver.partition = new_partition
    solver.dist_flat_face = new_dist_fc
    solver.state = new_state
    solver.halo_exchange = HaloExchange(new_partition, new_n_sps, n_vars)

    if solver.turb_model is not None:
        # wall_distance_compact：compact 索引空间随 dist_flat_face 重建
        # 而变化，与其尝试插值一个"索引空间都不一样"的旧数组，直接用
        # 与 __init__ 同一套函数重新算——这是纯几何量（KDTree 最近邻），
        # 对同一个完整全局 mesh 重算一次的开销与初次构造时相同量级，
        # 阶数切换本身就是低频事件（每次 Order Continuation 只发生
        # len(orders)-1 次），可以接受。
        # 与构造时同一个来源在新阶数的解点上重查（壁距是纯几何量，不插值）
        solver.wall_distance_compact = compute_distributed_wall_distance(
            new_dist_fc, solver.mesh, getattr(solver, "_wall_distance_source", None))
        # 换阶后贴壁解点更靠近壁面：omega 上界随之重定
        apply_distributed_wall_distance(solver.turb_model, solver.wall_distance_compact, solver)

        if solver.ddes_model is not None:
            # 真实 bug 修复（2026-09-02，DDES 补齐 max_edge 网格尺度时
            # 发现，与本次 Order Continuation 改动无关）：此前这里只要
            # `_iddes_h_max` 存在就无条件读 `solver._iddes_h_wn`，隐含
            # 假设"有 h_max 就一定是 IDDES、一定有 h_wn"——DDES（非
            # IDDES）现在也会设置 `_iddes_h_max`（只需要 h_max，不需要
            # h_wn，见 `init_turbulence_models` DDES 分支文档），DDES +
            # Order Continuation 组合会在这里 AttributeError。
            if getattr(solver, '_iddes_h_max', None) is not None:
                # h_max/h_wn 是全局、阶数无关的逐单元几何量（见
                # des/grid_scale.py::compute_h_max_and_h_wn 文档），只需要按新的
                # compact_global_ids 重新切片，不需要重算。
                compact_global_ids = new_dist_fc.compact_global_ids
                solver.iddes_h_max_compact = solver._iddes_h_max[compact_global_ids]
                if getattr(solver, '_iddes_h_wn', None) is not None:
                    solver.iddes_h_wn_compact = solver._iddes_h_wn[compact_global_ids]

    # boundary_ghost_provider：SEM 入口幽灵态按当时阶数的 FP 几何预存了
    # 每面 FP 物理坐标，必须随阶数重建——`local_solver` 是懒加载属性，
    # 失效后下次访问会用 `self.mesh`（已经 set_order 到 target_p）+
    # 更新后的 `solver_kwargs['order']` 重新构造一个真正的单机 FRSolver
    # 取得新阶数的 boundary_ghost_provider，再按新 partition.local_faces
    # 重新切片 group_code（与 `local_solver` property 文档同一套逻辑）。
    solver.solver_kwargs['order'] = target_p
    solver._local_solver = None

    reset_order_dependent_state(solver, solver.sgs_model)

    solver.current_order = target_p


def cpu_traditional_interpolate_to_new_order(solver, target_p: int) -> None:
    """CPU MPI"传统模式"（`DistributedFRSolver` 主 `__init__` 构造）
    的阶数切换。

    两种情形（`target_p` 相对 `solver.current_order` 的大小）：
    - **升阶**（`target_p > current_order`，Order Continuation 爬坡的
      正常情形）：local 守恒变量/湍流场做精确的多项式延拓（`fr/order_interp.py`：旧阶数多项式在新解点上
      求值——这个精确性只在升阶方向成立）。
    - **降阶到目标阶数以下**（`target_p < current_order`，唯一真实
      发生的场景是 `run_order_continuation`（四个后端共用）开始爬坡前把
      solver 从"CLI 直接在目标阶数构造"重置回 P0，见该函数文档）：
      不调用插值（对降阶而言不是精确延拓，只是走样的粗化投影，没有
      意义），直接重置为均匀自由流场——与单机 `run_order_continuation`
      "状态不在 P0 就重置回 P0 均匀流场"分支同一个处理方式。
    """

    old_order = solver.current_order
    if old_order == target_p:
        return

    n_local = solver.partition.n_local_cells
    # 只换 local 单元（halo 由下一步 step() 的 halo 交换重新取得）；场在 `order_switch.switched_fields` 里换阶，与其余
    # 后端同一份
    turb = solver.turb_model
    switched = switched_fields(
        old_order, target_p, int(solver.mesh.n_prism_cells), solver.freestream, solver.state.get_local_U()[:n_local],
        turb, turbulence=None if turb is None else [f[:n_local] for f in turb.transported_fields()],
        nu_t=None if turb is None or turb.nu_t is None else turb.nu_t[:n_local])
    install_turbulence(turb, switched)

    _rebuild_cpu_traditional_partition_and_state(solver, target_p, switched.U)
