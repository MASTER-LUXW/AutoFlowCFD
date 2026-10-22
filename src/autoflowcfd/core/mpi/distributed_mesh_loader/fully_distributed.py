"""AutoFlowCFD V2.0 - 完全分布式加载入口与跨阶数重分发。"""

from typing import Any, Callable, Optional, Tuple

from loguru import logger

from autoflowcfd.core.mpi import get_rank
from autoflowcfd.core.mpi.comm import recv_pickled, send_pickled
from .package import RUN_OPTION_NAMES, build_fully_distributed_rank_package

from autoflowcfd.core.time_integration.base import TimeIntegrationScheme
from autoflowcfd.core.turbulence.wall_model.treatment import needs_wall_model
from autoflowcfd.core.utils.order_continuation.order_switch import (
    install_turbulence, reset_order_dependent_state, switch_mesh_order, switched_fields,
)


#: 初次加载分发紧凑包的消息标签（换阶分发由调用方给出自己的一对）
_LOAD_TAGS = (310, 311)


def distributed_mesh_load_v2(
    load_root_mesh: Callable[[], Tuple[Any, Any]],
    order: int,
    n_ranks: int,
    freestream: dict,
    mu_molecular: float,
    mach_ref: float,
    *,
    enable_viscous: bool,
    turb_model_name: str,
    turbulence_intensity: float,
    viscosity_ratio: float,
    time_scheme: TimeIntegrationScheme,
    dual_time_inner_iter: int,
    cfl_start: Optional[float],
    cfl_max: Optional[float],
    cfl_min: Optional[float],
    time_accurate: bool,
    adaptive_cfl: bool,
    artificial_viscosity_enabled: bool,
    artificial_viscosity_alpha: float,
    wall_treatment: str,
    low_mach_precond: bool,
    sem_num_eddies: Optional[int],
    bc_overrides: Optional[dict] = None,
):
    """真正的完全分布式网格加载（2026-09-02）——只有 root rank 加载
    完整网格并对每个 rank 分别调用 `build_fully_distributed_rank_
    package`，其余 rank 只通过 MPI 接收自己的那一份紧凑包，从未持有
    完整全局网格。

    Order Continuation 支持（2026-09-02 续接，见 core/mpi/
    distributed_order_continuation/ 模块文档"完全分布式加载"一节）：
    返回值从单一的 `my_package` 改为 `(my_package, root_context)` 二元
    组——`root_context` 只有 root rank 非 None（其余 rank 恒为
    None，它们从未持有、也不需要持有完整全局网格），持有本函数内部
    算好的 `mesh`/`ops`/`fc`/`cell_partition`/`boundary_ghost_provider_
    global`/`wall_distance_source`/`h_max_global`/`h_wn_global` 等，供
    `redistribute_fully_distributed_for_new_order` 在阶数切换时复用
    （重新构造完整网格/重新分区都是不必要的重复开销——这些量本身除了
    `boundary_ghost_provider_global`（阶数相关的 FP 几何）之外全部与
    阶数无关）。调用方（CLI）需要把 `root_context` 原样传给
    `DistributedFRSolver.from_fully_distributed_package(package,
    n_ranks=n_ranks, root_context=root_context)`。

    Args:
        load_root_mesh: 只在 root 上调用，返回 `(HighOrderMesh, VolumeMeshData)`（CLI 传
            `cli/solve/mesh_loader.py::load_mesh_for_solver`，含质量门）。2026-10-21（I-106）以前本函数自己
            导入 CLI 层的加载函数，核心层反向依赖 CLI。
        turb_model_name: 规范化的大写模型名（`core/turbulence/registry.py::SUPPORTED_MODELS`）。root 用真实全局网格算
            wall_distance/h_max/h_wn
            （`compute_distributed_wall_distance`/`compute_h_max_and_
            h_wn` 都只需要完整网格，root 有，只算一次，见下方调用点；
            LES/WMLES 不需要任何 root 预计算的几何量），非 root rank 从未需要这些几何
            量的计算依赖。
        enable_viscous: 是否计入粘性项（仅限关键字、必传）
        turbulence_intensity, viscosity_ratio, time_scheme, dual_time_inner_iter, cfl_start /
            cfl_max / cfl_min, time_accurate, adaptive_cfl, artificial_viscosity_enabled / _alpha, wall_treatment,
            low_mach_precond, sem_num_eddies: 运行配置（透传给 `build_fully_distributed_rank_package`），仅限关键字
            且必传（与单机 FRSolver 同名参数同义；`time_scheme` 是 `TimeIntegrationScheme` 枚举值）。
            2026-10-21（I-104）以前带缺省值（`time_scheme=None` 静默取 SSP_RK3、Tu / VR 取 0.01 / 5.0 等），
            调用方漏传一个就在分布式上换成另一种格式或物理而不报错。

    Returns:
        (my_package, root_context)：`my_package` 是本 rank 的紧凑包
        （见 `build_fully_distributed_rank_package` Returns 文档，
        字段完全一致），`root_context` 见上方"Order Continuation 支持"
        一节。
    """
    from autoflowcfd.core.mpi.partition import partition_mesh
    from autoflowcfd.fr.operators import generate_fr_operators

    if get_rank() != 0:
        return _distribute_packages(None, order, n_ranks, _LOAD_TAGS), None

    logger.info("Root rank loading full mesh (fully-distributed mode)...")
    mesh, volume_data = load_root_mesh()
    ops = generate_fr_operators(order)
    fc = mesh.face_connectivity
    cell_partition = partition_mesh(fc, n_ranks, n_cells=mesh.n_cells)

    # 输运模型（SST 族、SA-neg）：wall_distance_source/h_max/h_wn 只依赖完整
    # 全局网格，只需要算一次（不随 rank 变化），见
    # build_fully_distributed_rank_package 文档对应参数说明。
    from autoflowcfd.core.turbulence.registry import has_transport_equations
    wall_distance_source = None
    h_max_global = h_wn_global = None
    if has_transport_equations(turb_model_name):
        # 壁面距离来源与单机同一个构造（WALL 组边界面，见 core/utils/wall_distance；
        # 此前这里把 BoundaryMap 的数组当字典读，真实网格上直接崩溃）
        from autoflowcfd.core.utils.wall_distance import WallDistanceSource
        wall_distance_source = WallDistanceSource.from_volume_data(volume_data)
        if turb_model_name in ("DDES", "IDDES"):
            # 逐单元网格尺度（DDES 用 h_max，IDDES 还用 h_wn），见 build_fully_distributed_rank_package
            from autoflowcfd.core.turbulence.des import compute_h_max_and_h_wn
            h_max_global, h_wn_global = compute_h_max_and_h_wn(mesh)

    # 隐式步（稳态 Newton-Krylov、双时间步内迭代）的块 Jacobi 着色必须全局一致（见 core/mpi/distributed_implicit.py），
    # 只有 root 持有全局面连接关系，在这里算一次
    global_cell_colors = global_cell_colors_d2 = None
    from autoflowcfd.core.time_integration.base import IMPLICIT_SCHEMES, scheme_from_name
    if scheme_from_name(time_scheme) in IMPLICIT_SCHEMES:
        from autoflowcfd.core.mpi.distributed_implicit import (
            global_cell_colors as _colors, global_cell_colors_d2 as _colors_d2,
        )
        global_cell_colors = _colors(fc, len(cell_partition))
        global_cell_colors_d2 = _colors_d2(fc, len(cell_partition))

    # root_context（见函数文档"Order Continuation 支持"一节）：只有 root 保留，供阶数切换重新计算、重新分发
    # 紧凑包，不随包发送给非 root rank（它们不需要、也不应该持有完整全局网格）。
    root_context = {
        'mesh': mesh, 'ops': ops, 'fc': fc, 'cell_partition': cell_partition,
        # 体网格数据：收尾的气动系数在 root 上按它估算参考面积（`cli/solve/aero_coefficients.py`）
        'volume_data': volume_data,
        'freestream': freestream, 'mu_molecular': mu_molecular, 'mach_ref': mach_ref,
        'enable_viscous': enable_viscous, 'turb_model_name': turb_model_name,
        'wall_distance_source': wall_distance_source,
        'h_max_global': h_max_global, 'h_wn_global': h_wn_global,
        'turbulence_intensity': turbulence_intensity, 'viscosity_ratio': viscosity_ratio,
        'sem_num_eddies': sem_num_eddies,
        'bc_overrides': bc_overrides or {}, 'n_ranks': n_ranks,
        'time_scheme': time_scheme, 'dual_time_inner_iter': dual_time_inner_iter,
        'cfl_start': cfl_start, 'cfl_max': cfl_max, 'cfl_min': cfl_min,
        'time_accurate': time_accurate, 'adaptive_cfl': adaptive_cfl,
        'artificial_viscosity_enabled': artificial_viscosity_enabled,
        'artificial_viscosity_alpha': artificial_viscosity_alpha,
        'wall_treatment': wall_treatment,
        'low_mach_precond': low_mach_precond,
        'global_cell_colors': global_cell_colors,
        'global_cell_colors_d2': global_cell_colors_d2,
    }
    root_context['boundary_ghost_provider_global'] = _root_boundary_provider(root_context)
    return _distribute_packages(root_context, order, n_ranks, _LOAD_TAGS), root_context


def _root_boundary_provider(root_context):
    """root 用完整全局网格构建的边界幽灵态提供者（全部 rank 共享底层配置，各包按自己的面表切片）。物理设置全部
    取用户设置：湍流模型决定速度入口的合成湍流，近壁处理决定 WALL 组是否为滑移镜像，Tu / VR / 分子粘度 / 涡核数定
    合成湍流的尺度与强度。"""
    from autoflowcfd.core.fr_solver.boundary import BoundaryPhysics, build_boundary_ghost_provider

    rc = root_context
    physics = BoundaryPhysics(rc['freestream'], rc['mu_molecular'], rc['turb_model_name'],
                              needs_wall_model(rc['turb_model_name'], rc['wall_treatment']),
                              rc['turbulence_intensity'], rc['viscosity_ratio'], rc['sem_num_eddies'])
    return build_boundary_ghost_provider(rc['mesh'], rc['ops'], physics, rc['bc_overrides'])


def _distribute_packages(root_context, order: int, n_ranks: int, tags) -> dict:
    """本 rank 的紧凑包（集体操作；初次加载与换阶共用，换阶前 root 已把 root_context 的网格、算子与边界提供者
    切到新阶数）。

    root（`root_context` 非 None）逐个 rank 构建、发出即释放，最后构建自己的：root 峰值内存是全局网格加一个包。
    2026-10-21（I-106）以前先把 n_ranks 份包全部建在内存里再逐个发送，且整包 pickle 字节一条消息发出（单包
    超过 2 GiB 时 MPI 计数溢出，见 `core/mpi/comm.py::send_pickled`）。其余 rank 接收。
    """
    if root_context is None:
        return recv_pickled(0, tags)
    rc = root_context

    def package_for(rank: int) -> dict:
        return build_fully_distributed_rank_package(
            rc['mesh'], rc['ops'], rc['fc'], rc['cell_partition'], rank, n_ranks,
            rc['boundary_ghost_provider_global'], rc['freestream'], rc['mu_molecular'], rc['mach_ref'],
            order, rc['enable_viscous'],
            wall_distance_source=rc['wall_distance_source'],
            h_max_global=rc['h_max_global'], h_wn_global=rc['h_wn_global'],
            # 着色只依赖拓扑，换阶沿用同一份
            global_cell_colors=rc['global_cell_colors'], global_cell_colors_d2=rc['global_cell_colors_d2'],
            **{name: rc[name] for name in RUN_OPTION_NAMES})

    for rank in range(1, n_ranks):
        send_pickled(package_for(rank), rank, tags)
    return package_for(0)


def redistribute_fully_distributed_for_new_order(solver, target_p: int) -> None:
    """"完全分布式加载"模式的阶数切换（2026-09-02，见 core/mpi/
    distributed_order_continuation/ 模块文档"完全分布式加载"一节）。

    与"传统模式"（`cpu_traditional_interpolate_to_new_order`）的关键
    区别：本 rank 从未持有完整全局网格，`PrecompactedMeshData`/
    `DistributedFlatFaceGeometry` 都是 root 预先按 compact 索引空间
    切好、通过 MPI 发过来的——阶数切换后这些几何量的形状/内容整体
    改变，必须由 root 用它持续持有的完整全局网格（`solver._root_
    context`，见 `distributed_mesh_load_v2` 文档）重新计算、重新分发
    一份新的紧凑包，不是本 rank 自己能算出来的。

    协议（与 `distributed_mesh_load_v2` 的初次分发同一套 Send/Recv
    机制，只是复用已有的 `root_context` 而不是重新加载/重新分区）：
    1. 每个 rank 独立在本地插值/重置自己的 local U + 湍流场（纯本地
       操作，`n_local`——同一个 `cell_partition`——阶数切换不变，不
       需要通信，见下方"local 状态"一节）。
    2. Root：`root_context['mesh'].set_order(target_p)` + 重新生成
       `ops` + 重新构建 `boundary_ghost_provider_global`（阶数相关的
       FP 几何）+ 对每个 rank 调用 `build_fully_distributed_rank_
       package` 算出新紧凑包，Send 给对应 rank（root 自己直接用）。
    3. 每个 rank：接收新包，把上一步算好的插值/重置后 local U 写入
       新包对应 partition 布局的 state，替换 `solver` 的全部 compact
       相关属性（partition/dist_flat_face/mesh/ops/wall_distance_
       compact/iddes 字段/boundary_ghost_provider），不重新构造
       `solver` 实例本身（保留 `turb_model`/`sgs_model`/`wall_model`
       对象引用，只替换它们的 k_field/omega_field/nu_t 数组）。

    只支持通过 `root_context` 非 None 构造的实例（`DistributedFRSolver.
    from_fully_distributed_package(package, n_ranks, root_context=...)`
    ——root_context 未提供时 fail-fast，不静默产生错误结果）。
    """
    from autoflowcfd.core.mpi import get_rank
    from autoflowcfd.fr.operators import generate_fr_operators
    from autoflowcfd.core.mpi.distributed_state import DistributedFRState
    from autoflowcfd.core.mpi.halo import HaloExchange
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive

    old_order = solver.current_order
    if old_order == target_p:
        return

    is_root_rank = get_rank() == 0
    root_context = getattr(solver, '_root_context', None) if is_root_rank else None
    if is_root_rank and root_context is None:
        raise NotImplementedError(
            "redistribute_fully_distributed_for_new_order: root rank 没有 "
            "_root_context（本实例不是通过 distributed_mesh_load_v2 返回的 "
            "root_context 构造的）——Order Continuation 在'完全分布式加载' "
            "模式下要求 root 持续持有完整全局网格用于重新分发，不支持从 "
            "缺少这份上下文的实例继续爬坡（如实报告，不是假装能用）。"
        )

    n_local = solver.partition.n_local_cells
    n_vars = solver.state.n_vars

    # 只换 local 单元（halo 由下一步 step() 的 halo 交换重新取得）；场在 `order_switch.switched_fields` 里换阶，与其余
    # 后端同一份
    turb = solver.turb_model
    switched = switched_fields(
        old_order, target_p, int(solver.mesh.n_prism_cells), solver.freestream, solver.state.get_local_U()[:n_local],
        turb, turbulence=None if turb is None else [f[:n_local] for f in turb.transported_fields()],
        nu_t=None if turb is None or turb.nu_t is None else turb.nu_t[:n_local])
    install_turbulence(turb, switched)

    # --- 2. Root 重新计算 + 分发新紧凑包 ---
    my_package = exchange_packages_for_new_order(
        root_context if is_root_rank else None, target_p, solver.n_ranks, tags=(320, 321))

    # --- 3. 应用新包：替换 compact 相关属性，保留 turb_model/sgs_model/
    # wall_model 对象本身（只是上一步已经替换过它们的数组）---
    solver.ops = generate_fr_operators(target_p)
    solver.mesh = my_package['precompacted_mesh']
    solver.partition = my_package['partition']
    solver.dist_flat_face = my_package['dist_fc']

    new_n_sps = solver.mesh.n_sps_per_cell
    new_state = DistributedFRState(solver.partition, new_n_sps)
    new_state.U[:n_local] = switched.U
    new_state.Q[:n_local] = conserved_to_primitive(switched.U[..., :5])
    solver.state = new_state
    solver.halo_exchange = HaloExchange(solver.partition, new_n_sps, n_vars)

    solver.wall_distance_compact = my_package.get('wall_distance_compact')
    # 换阶后贴壁解点更靠近壁面：omega 上界随之重定（与传统模式同一函数）
    from autoflowcfd.core.mpi.distributed_turbulence import apply_distributed_wall_distance
    apply_distributed_wall_distance(solver.turb_model, solver.wall_distance_compact, solver)
    solver.iddes_h_max_compact = my_package.get('iddes_h_max_compact')
    solver.iddes_h_wn_compact = my_package.get('iddes_h_wn_compact')

    # `_local_solver` 在"完全分布式加载"模式下从构造起就是这个
    # `types.SimpleNamespace` 替身（不是"传统模式"的懒加载真实
    # `FRSolver`，见 `from_fully_distributed_package` 构造处同一段
    # 代码）——阶数切换后 `boundary_ghost_provider`/`enable_viscous`
    # 都是新阶数的值，必须真正重新赋值，不能保留旧引用。
    import types
    solver._local_solver = types.SimpleNamespace(
        config=types.SimpleNamespace(
            physics=types.SimpleNamespace(enable_viscous=my_package['enable_viscous'])
        ),
        mu_molecular=my_package['mu_molecular'],
        boundary_ghost_provider=my_package['boundary_ghost_provider'],
        freestream={**my_package['freestream'], "mach_ref": my_package['mach_ref']},
    )

    reset_order_dependent_state(solver, solver.sgs_model)

    solver.current_order = target_p


def exchange_packages_for_new_order(root_context, target_p: int, n_ranks: int, *, tags) -> dict:
    """换阶时的紧凑包重算与分发（CPU 与多 GPU 的"完全分布式加载"共用），返回本
    rank 的包。

    root（`root_context` 非 None）：清掉旧阶数的几何缓存、把全局网格切到新阶数、
    重建边界幽灵态提供者，按 root_context 里**全部**构造参数重建每个 rank 的包
    并发送；其余 rank 按 `tags` 接收。2026-09-25 以前 CPU 与 GPU 各抄一份，GPU
    那份漏了 cfl_start/cfl_max/cfl_min（升阶之后 CFL 边界静默退回默认值——CPU
    那份 2026-09-15 修过的同一个问题）。
    """
    if root_context is not None:
        mesh = root_context['mesh']
        root_context['ops'] = switch_mesh_order(mesh, target_p)
        root_context['boundary_ghost_provider_global'] = _root_boundary_provider(root_context)
    return _distribute_packages(root_context, target_p, n_ranks, tags)
