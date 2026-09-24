"""AutoFlowCFD V2.0 - 阶数切换时重建 CPU 传统模式的分区与状态、逐 rank 插值

从 `src/autoflowcfd/core/mpi/distributed_order_continuation.py`(原 640 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""

import numpy as np






def _interp_state_and_turbulence_local(solver, old_order: int,
                                       target_p: int,
                                       n_local: int) -> np.ndarray:
    """对 local cells（不含 halo）的守恒变量 + 湍流场做精确延拓插值
    （复用 `fr/order_interp.py::apply_order_interp` 这一份实现，只是作用
    范围限定在 `[0, n_local)`——halo 部分在下面几何重建之后由 halo 交换
    在下一步 `step()` 里重新获取，不需要在这里插值一份很快就会被覆盖
    的值）。

    **按基分派（2026-09-20 修复）**：此前这里收一个一维 Gauss 张量积
    Lagrange 矩阵 `W`，而那只对坍缩棱柱基的解点成立 —— native 四面体
    （自 2026-09-03 起是四面体唯一实现）与 native 棱柱（2026-09-20 起是
    默认）线性场 P1->P2 的相对误差实测 7.0e-01 / 1.4e-01。改成传阶数、
    由 `apply_order_interp` 按当前基与单元类型分派。local 索引空间同样是
    "棱柱在前"（见 `distributed_mesh_loader` 里 `n_prism_cells` 的来源）。

    Returns:
        (n_local, new_n_sps, n_vars) 插值后的 local 守恒变量，湍流场
        （若存在）直接原地写回 `solver.turb_model.k_field`/
        `.omega_field`/`.nu_t`。
    """
    from autoflowcfd.fr.order_interp import apply_order_interp

    n_prism_local = int(solver.mesh.n_prism_cells)

    def _lift(field):
        return apply_order_interp(field, n_prism_local, old_order, target_p)

    old_local_U = solver.state.get_local_U()[:n_local]
    new_local_U = _lift(old_local_U)

    turb_model = getattr(solver, 'turb_model', None)
    if turb_model is not None and hasattr(turb_model, 'k_field'):
        turb_model.k_field = _lift(turb_model.k_field[:n_local])
        turb_model.omega_field = _lift(turb_model.omega_field[:n_local])
        # nu_t：同 order_continuation.py 文档说明，只有形状匹配旧阶数时
        # 才插值（可能在 compute_source 刷新前已经是别的形状/尚未构造）。
        if getattr(turb_model, 'nu_t', None) is not None and turb_model.nu_t.shape[1] == old_local_U.shape[1]:
            turb_model.nu_t = _lift(turb_model.nu_t[:n_local])
        # des_length_scale：同 order_continuation.py 文档"DDES 有效长度
        # 尺度"一节的处理——清空而不是插值，下一步 compute_source 会
        # 用新阶数维度重新算出。
        if hasattr(turb_model, 'des_length_scale'):
            turb_model.des_length_scale = None

    sgs_model = getattr(solver, 'sgs_model', None)
    if sgs_model is not None and hasattr(sgs_model, 'nu_t'):
        sgs_model.nu_t = None

    return new_local_U


def _rebuild_cpu_traditional_partition_and_state(solver, target_p: int, new_local_U: np.ndarray) -> None:
    """"传统模式"阶数切换的共用收尾：mesh/ops 切换 + 全新
    partition/dist_flat_face 重建 + state 构造 + halo 交换器重建 +
    wall_distance/IDDES 几何重新切片 + boundary_ghost_provider 失效。

    调用方（`cpu_traditional_interpolate_to_new_order`/
    `_reset_cpu_traditional_to_p0`）负责在调用**之前**把
    `solver.turb_model.k_field`/`.omega_field`/`.nu_t`/
    `.des_length_scale`/`solver.sgs_model.nu_t` 处理成新阶数的形状
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
    from autoflowcfd.fr.operators import generate_fr_operators
    from autoflowcfd.core.mpi.partition import build_distributed_partition
    from autoflowcfd.core.mpi.distributed_flat_face import build_distributed_flat_face
    from autoflowcfd.core.mpi.distributed_state import DistributedFRState
    from autoflowcfd.core.mpi.halo import HaloExchange
    from autoflowcfd.core.mpi.distributed_turbulence import compute_distributed_wall_distance
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive

    cell_partition = getattr(solver, '_oc_cell_partition', None)
    if cell_partition is None:
        raise NotImplementedError(
            "distributed_order_continuation: 本 solver 实例的 partition "
            "不是通过 DistributedFRSolver 主 __init__ 的 face_connectivity "
            "（'兼容旧接口'，即 CLI 生产路径 --n-ranks>1 不加 "
            "--fully-distributed）构造的——用 partition_info 直接构造的"
            "路径从未被 CLI 使用过、也从未验证过是否持有完整全局网格，"
            "Order Continuation 在这条未验证路径上不支持（如实报告，"
            "不是假装能用）。"
        )

    # mesh 是"传统模式"下每个 rank 都持有的完整全局网格——阶数切换与
    # 单机 run_order_continuation 同一套流程（先清理旧阶数几何缓存，
    # 再 set_order，避免大网格同时驻留多份完整阶数几何 OOM，见该函数
    # 文档"性能说明"一节，同一个理由在分布式"传统模式"下依然成立，
    # 每个 rank 各自持有一份完整网格，OOM 风险不小于单机）。
    stale_orders = [o for o in list(solver.mesh._order_geometry_cache) if o != target_p]
    for o in stale_orders:
        del solver.mesh._order_geometry_cache[o]
    solver.mesh.set_order(target_p)
    solver.ops = generate_fr_operators(target_p)

    new_partition = build_distributed_partition(
        solver.mesh.face_connectivity, cell_partition, solver.rank, solver.n_ranks
    )
    new_dist_fc = build_distributed_flat_face(
        solver.mesh, solver.ops, new_partition, cell_partition=cell_partition
    )

    n_local = new_partition.n_local_cells
    new_n_sps = solver.ops.D_3d.shape[0]
    n_vars = solver.state.n_vars
    new_state = DistributedFRState(new_partition, new_n_sps, n_vars)
    new_state.U[:n_local] = new_local_U
    new_state.Q[:n_local] = conserved_to_primitive(new_local_U[..., :5])

    solver.partition = new_partition
    solver.dist_flat_face = new_dist_fc
    solver.state = new_state
    solver.halo_exchange = HaloExchange(new_partition, new_n_sps, n_vars)

    if solver.turb_model is not None:
        solver.turb_halo_exchange = HaloExchange(new_partition, new_n_sps, 2)

        # wall_distance_compact：compact 索引空间随 dist_flat_face 重建
        # 而变化，与其尝试插值一个"索引空间都不一样"的旧数组，直接用
        # 与 __init__ 同一套函数重新算——这是纯几何量（KDTree 最近邻），
        # 对同一个完整全局 mesh 重算一次的开销与初次构造时相同量级，
        # 阶数切换本身就是低频事件（每次 Order Continuation 只发生
        # len(orders)-1 次），可以接受。
        wall_node_indices = None
        boundary_groups = getattr(solver.mesh, 'boundary_groups', None)
        if boundary_groups is not None:
            for bg_name, bg in boundary_groups.items():
                if 'WALL' in bg_name.upper() or bg.get('type', '').upper() == 'WALL':
                    wall_node_indices = bg.get('node_indices')
                    break
        solver.wall_distance_compact = compute_distributed_wall_distance(
            new_partition, new_dist_fc, solver.mesh, wall_node_indices,
        )

        if solver.ddes_model is not None:
            solver.des_length_scale_halo_exchange = HaloExchange(new_partition, new_n_sps, 1)
            # 真实 bug 修复（2026-09-02，DDES 补齐 max_edge 网格尺度时
            # 发现，与本次 Order Continuation 改动无关）：此前这里只要
            # `_iddes_h_max` 存在就无条件读 `solver._iddes_h_wn`，隐含
            # 假设"有 h_max 就一定是 IDDES、一定有 h_wn"——DDES（非
            # IDDES）现在也会设置 `_iddes_h_max`（只需要 h_max，不需要
            # h_wn，见 `init_turbulence_models` DDES 分支文档），DDES +
            # Order Continuation 组合会在这里 AttributeError。
            if getattr(solver, '_iddes_h_max', None) is not None:
                # h_max/h_wn 是全局、阶数无关的逐单元几何量（见
                # des.py::compute_h_max_and_h_wn 文档），只需要按新的
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

    # DUAL_TIME 上一物理时间层历史随阶数切换失效（见
    # order_continuation.py::interpolate_to_new_order_checked 同名
    # 处理文档——形状不再匹配，且严格来说也不再是同一离散空间下的解），
    # 否则下一步 BDF2 会静默用一份形状不匹配的历史层。
    if hasattr(solver, '_dual_time_U_prev'):
        solver._dual_time_U_prev = None

    solver.current_order = target_p


def cpu_traditional_interpolate_to_new_order(solver, target_p: int) -> None:
    """CPU MPI"传统模式"（`DistributedFRSolver` 主 `__init__` 构造）
    的阶数切换。

    两种情形（`target_p` 相对 `solver.current_order` 的大小）：
    - **升阶**（`target_p > current_order`，Order Continuation 爬坡的
      正常情形）：local 守恒变量/湍流场做精确 Lagrange 延拓插值（见
      `_build_linear_interp_matrix_3d` 文档"只会从低阶向高阶单调推进"
      一节——这个精确性只在升阶方向成立）。
    - **降阶到目标阶数以下**（`target_p < current_order`，唯一真实
      发生的场景是 `run_distributed_order_continuation` 开始爬坡前把
      solver 从"CLI 直接在目标阶数构造"重置回 P0，见该函数文档）：
      不调用插值（对降阶而言不是精确延拓，只是走样的粗化投影，没有
      意义），直接重置为均匀自由流场——与单机 `run_order_continuation`
      "状态不在 P0 就重置回 P0 均匀流场"分支同一个处理方式。
    """
    from autoflowcfd.core.fr_solver.turbulence import _set_freestream_turbulence

    old_order = solver.current_order
    if old_order == target_p:
        return

    n_local = solver.partition.n_local_cells
    new_n_sps = (target_p + 1) ** 3

    if target_p > old_order:
        new_local_U = _interp_state_and_turbulence_local(
            solver, old_order, target_p, n_local)
    else:
        rho_inf = solver.solver_kwargs.get('rho_inf', 1.225)
        vel_inf = solver.solver_kwargs.get('vel_inf', 33.33)
        p_inf = solver.solver_kwargs.get('p_inf', 101325.0)
        gamma = 1.4
        e = p_inf / ((gamma - 1.0) * rho_inf) + 0.5 * vel_inf ** 2
        new_local_U = np.zeros((n_local, new_n_sps, solver.state.n_vars))
        new_local_U[:, :, 0] = rho_inf
        new_local_U[:, :, 1] = rho_inf * vel_inf
        new_local_U[:, :, 4] = rho_inf * e

        if solver.turb_model is not None:
            k_inf, omega_inf = _set_freestream_turbulence(solver)
            solver.turb_model.k_field = np.ones((n_local, new_n_sps)) * k_inf
            solver.turb_model.omega_field = np.ones((n_local, new_n_sps)) * omega_inf
            if hasattr(solver.turb_model, 'nu_t'):
                solver.turb_model.nu_t = np.zeros((n_local, new_n_sps))
            if hasattr(solver.turb_model, 'des_length_scale'):
                solver.turb_model.des_length_scale = None
        if getattr(solver, 'sgs_model', None) is not None and hasattr(solver.sgs_model, 'nu_t'):
            solver.sgs_model.nu_t = None

    _rebuild_cpu_traditional_partition_and_state(solver, target_p, new_local_U)
