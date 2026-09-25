"""AutoFlowCFD V2.0 - 完全分布式加载入口与跨阶数重分发。

从 `core/mpi/distributed_mesh_loader.py` 拆出（2026-09-24）。纯搬家，
逻辑未改。
"""

import numpy as np
from typing import Optional

from loguru import logger

from autoflowcfd.core.mpi import get_comm, get_rank
from .package import build_fully_distributed_rank_package


def distributed_mesh_load_v2(
    input_file: str,
    order: int,
    surface_mesh: Optional[str],
    n_ranks: int,
    freestream: dict,
    mu_molecular: float,
    mach_ref: float,
    enable_viscous: bool = True,
    bc_overrides: Optional[dict] = None,
    skip_quality_check: bool = False,
    turb_model_name: str = "NONE",
    turbulence_intensity: float = 0.01,
    viscosity_ratio: float = 5.0,
    time_scheme=None,
    dual_time_inner_iter: int = 20,
    cfl_start: Optional[float] = None,
    cfl_max: Optional[float] = None,
    cfl_min: Optional[float] = None,
):
    """真正的完全分布式网格加载（2026-09-02）——只有 root rank 加载
    完整网格并对每个 rank 分别调用 `build_fully_distributed_rank_
    package`，其余 rank 只通过 MPI 接收自己的那一份紧凑包，从未持有
    完整全局网格。取代此前从未真正跑通过的 `distributed_mesh_load`
    （见该函数/`build_local_mesh_from_data` 文档"注意"一节——那条路径
    产出的 `local_mesh` 缺 `face_connectivity`/`face_flux_points`，
    `DistributedMeshAdapter` 需要完整全局网格重新切片 jacobians，两个
    前提在那条路径下都不成立，构造期必然出错，此前从未被 CLI 实际
    调用过，见 `solve_steady_command.py` 对应注释）。

    Order Continuation 支持（2026-09-02 续接，见 core/mpi/
    distributed_order_continuation.py 模块文档"完全分布式加载"一节）：
    返回值从单一的 `my_package` 改为 `(my_package, root_context)` 二元
    组——`root_context` 只有 root rank 非 None（其余 rank 恒为
    None，它们从未持有、也不需要持有完整全局网格），持有本函数内部
    算好的 `mesh`/`ops`/`fc`/`cell_partition`/`boundary_ghost_provider_
    global`/`wall_node_indices`/`h_max_global`/`h_wn_global` 等，供
    `redistribute_fully_distributed_for_new_order` 在阶数切换时复用
    （重新构造完整网格/重新分区都是不必要的重复开销——这些量本身除了
    `boundary_ghost_provider_global`（阶数相关的 FP 几何）之外全部与
    阶数无关）。调用方（CLI）需要把 `root_context` 原样传给
    `DistributedFRSolver.from_fully_distributed_package(package,
    n_ranks=n_ranks, root_context=root_context)`。

    Args:
        turb_model_name: "NONE"/"SST"/"DDES"/"IDDES"/"WMLES"/"LES"
            （大写）。root 用真实全局网格算 wall_distance/h_max/h_wn
            （`compute_distributed_wall_distance`/`compute_h_max_and_
            h_wn` 都只需要完整网格，root 有，只算一次，见下方调用点；
            WMLES 同样需要 wall_distance——y+ 计算依赖它，LES 不需要
            任何 root 预计算的几何量），非 root rank 从未需要这些几何
            量的计算依赖。
        time_scheme, dual_time_inner_iter: DUAL_TIME 支持（2026-09-02
            续接）——透传给 `build_fully_distributed_rank_package`，
            `time_scheme` 是 `TimeIntegrationScheme` 枚举值（不是
            字符串），None（默认）时回退到 `TimeIntegrationScheme.
            SSP_RK3`。

    Returns:
        (my_package, root_context)：`my_package` 是本 rank 的紧凑包
        （见 `build_fully_distributed_rank_package` Returns 文档，
        字段完全一致），`root_context` 见上方"Order Continuation 支持"
        一节。
    """
    from autoflowcfd.cli.solve.helpers import load_mesh_for_solver
    from autoflowcfd.core.mpi.partition import partition_mesh
    from autoflowcfd.fr.operators import generate_fr_operators
    from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider

    rank = get_rank()
    comm = get_comm()

    if rank == 0:
        logger.info("Root rank loading full mesh (fully-distributed mode)...")
        mesh, _volume_data = load_mesh_for_solver(
            input_file, order, surface_mesh=surface_mesh,
            skip_quality_check=skip_quality_check,
        )
        ops = generate_fr_operators(order)
        fc = mesh.face_connectivity

        cell_partition = partition_mesh(fc, n_ranks, n_cells=mesh.n_cells)

        # 边界幽灵态提供者只需要完整全局网格构建一次（root 独有）——用
        # 一个最小鸭子类型对象满足 build_boundary_ghost_provider 的接口
        # 要求（`.mesh`/`.freestream`/`.turb_model_name`/`.wmles_model`，
        # 见该函数文档），不需要构造真正的 FRSolver。
        import types
        # 真实 bug 修复（2026-09-02，接入 SST/DDES/IDDES 时发现）：此前
        # 这里硬编码 `turb_model_name="NONE"`——`build_boundary_ghost_
        # provider` 用它判断 `use_sem`（DDES/IDDES 应该在 VELOCITY_INLET
        # 自动接入 BD-02 合成湍流入口，见 core/fr_solver/boundary.py
        # 文档），硬编码 "NONE" 会让"完全分布式加载"模式下的 DDES/IDDES
        # 永远拿不到 SEM 入口，与单机/CPU MPI 传统模式行为不一致。
        # 真实 bug 修复（2026-09-02，接入 WMLES 时发现，与上面 DDES/IDDES
        # 的 turb_model_name 硬编码是同一类问题）：`wmles_model` 也硬编码
        # None——`build_boundary_ghost_provider` 用 `getattr(solver,
        # "wmles_model",None) is None` 判断 WALL 组是否要切换
        # `is_no_slip=False`（见该函数文档 #9 修复说明），WMLES 激活时
        # 若这里恒为 None 会让"完全分布式加载"模式下的 WMLES 恒得到
        # `is_no_slip=True`，与单机/CPU MPI 传统模式行为不一致（这里只
        # 需要一个非 None 的哨兵值，不需要真正的 WMLESModel 实例——
        # `build_boundary_ghost_provider` 只检查 is None）。
        root_solver_stub = types.SimpleNamespace(
            mesh=mesh, freestream=freestream, turb_model_name=turb_model_name,
            wmles_model=(object() if turb_model_name == "WMLES" else None),
        )
        boundary_ghost_provider_global = build_boundary_ghost_provider(
            root_solver_stub, bc_overrides=bc_overrides or {},
        )

        # SST/DDES/IDDES/WMLES：wall_node_indices/h_max/h_wn 只依赖完整
        # 全局网格，只需要算一次（不随 rank 变化），见
        # build_fully_distributed_rank_package 文档对应参数说明。
        wall_node_indices = None
        h_max_global = h_wn_global = None
        if turb_model_name in ("SST", "DDES", "IDDES", "WMLES"):
            boundary_groups = getattr(mesh, 'boundary_groups', None)
            if boundary_groups is not None:
                for bg_name, bg in boundary_groups.items():
                    if 'WALL' in bg_name.upper() or bg.get('type', '').upper() == 'WALL':
                        wall_node_indices = bg.get('node_indices')
                        break
            if turb_model_name in ("DDES", "IDDES"):
                # DDES（2026-09-02 补齐）：apply_to_sst_model 现在优先用
                # h_max（max_edge 网格尺度）而不是 cube_root(V)，见
                # build_fully_distributed_rank_package 对应分支文档。
                from autoflowcfd.core.turbulence.des import compute_h_max_and_h_wn
                h_max_global, h_wn_global = compute_h_max_and_h_wn(mesh)

        packages = [
            build_fully_distributed_rank_package(
                mesh, ops, fc, cell_partition, r, n_ranks,
                boundary_ghost_provider_global, freestream, mu_molecular, mach_ref,
                order, enable_viscous,
                turb_model_name=turb_model_name, wall_node_indices=wall_node_indices,
                h_max_global=h_max_global, h_wn_global=h_wn_global,
                turbulence_intensity=turbulence_intensity, viscosity_ratio=viscosity_ratio,
                time_scheme=time_scheme, dual_time_inner_iter=dual_time_inner_iter,
                cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min,
            )
            for r in range(n_ranks)
        ]
        my_package = packages[0]

        if n_ranks > 1:
            import pickle
            for r in range(1, n_ranks):
                buf = pickle.dumps(packages[r])
                buf_size = np.array([len(buf)], dtype=np.int64)
                comm.Send(buf_size, dest=r, tag=310)
                comm.Send(buf, dest=r, tag=311)

        # root_context（见函数文档"Order Continuation 支持"一节）：
        # 只有 root 保留，供后续阶数切换重新计算+重新分发紧凑包用，
        # 不随本次分发的 packages 一起发送给非 root rank（它们不需要，
        # 也不应该持有完整全局网格）。
        root_context = {
            'mesh': mesh, 'ops': ops, 'fc': fc, 'cell_partition': cell_partition,
            'boundary_ghost_provider_global': boundary_ghost_provider_global,
            'freestream': freestream, 'mu_molecular': mu_molecular, 'mach_ref': mach_ref,
            'enable_viscous': enable_viscous, 'turb_model_name': turb_model_name,
            'wall_node_indices': wall_node_indices,
            'h_max_global': h_max_global, 'h_wn_global': h_wn_global,
            'turbulence_intensity': turbulence_intensity, 'viscosity_ratio': viscosity_ratio,
            'bc_overrides': bc_overrides or {}, 'n_ranks': n_ranks,
            'time_scheme': time_scheme, 'dual_time_inner_iter': dual_time_inner_iter,
            'cfl_start': cfl_start, 'cfl_max': cfl_max, 'cfl_min': cfl_min,
        }
    else:
        import pickle
        buf_size = np.empty(1, dtype=np.int64)
        comm.Recv(buf_size, source=0, tag=310)
        buf = np.empty(int(buf_size[0]), dtype=np.uint8)
        comm.Recv(buf, source=0, tag=311)
        my_package = pickle.loads(buf.tobytes())
        root_context = None

    return my_package, root_context


def redistribute_fully_distributed_for_new_order(solver, target_p: int) -> None:
    """"完全分布式加载"模式的阶数切换（2026-09-02，见 core/mpi/
    distributed_order_continuation.py 模块文档"完全分布式加载"一节）。

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
       `solver` 实例本身（保留 `turb_model`/`sgs_model`/`wmles_model`
       对象引用，只替换它们的 k_field/omega_field/nu_t 数组）。

    只支持通过 `root_context` 非 None 构造的实例（`DistributedFRSolver.
    from_fully_distributed_package(package, n_ranks, root_context=...)`
    ——root_context 未提供时 fail-fast，不静默产生错误结果）。
    """
    from autoflowcfd.core.mpi.comm import get_comm
    from autoflowcfd.core.mpi import get_rank
    from autoflowcfd.fr.operators import generate_fr_operators
    from autoflowcfd.core.mpi.distributed_state import DistributedFRState
    from autoflowcfd.core.mpi.halo import HaloExchange
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
    from autoflowcfd.core.fr_solver.turbulence import _set_freestream_turbulence

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

    # --- 1. 每个 rank 独立插值/重置自己的 local U + 湍流场（纯本地，
    # 数学与 CPU"传统模式"完全一致）---
    #
    # 自由来流条件：`self.freestream` 只在 turb_model_name 为
    # SST/DDES/IDDES 时才被 `from_fully_distributed_package` 设置（见
    # 该方法对应代码），NONE/LES/WMLES 分支没有这个属性——`self.
    # _package_freestream` 是本函数与 `from_fully_distributed_package`
    # 共同维护的、对全部 turb_model_name 都存在的通用存储点（见
    # `from_fully_distributed_package` 里的同名赋值、以及本函数末尾
    # "应用新包"一节的同名更新）。
    # 不给兜底（2026-09-24）：此前两个来源都取不到时会静默造一个
    # `{'rho_inf': 1.225, 'vel_inf': 33.33, 'p_inf': 101325.0}` —— 那是与
    # 构造函数默认值并存的第二份事实来源，而且会让一个真实的"来流丢失"
    # 缺陷以"用了一个看似合理的来流"的形式静默通过。`solver.freestream`
    # 自 2026-09-18 起对全部湍流模型无条件设置（见 distributed_solver/
    # core.py 同名注释），取不到就是真缺陷，应当直接 AttributeError。
    freestream = getattr(solver, '_package_freestream', None) or solver.freestream

    if target_p > old_order:
        # 延拓算子按基分派（2026-09-20，理由见 `fr/order_interp.py`）
        from autoflowcfd.fr.order_interp import apply_order_interp

        n_prism_local = int(solver.mesh.n_prism_cells)

        def _lift(field):
            return apply_order_interp(field, n_prism_local, old_order,
                                      target_p)

        old_local_U = solver.state.get_local_U()[:n_local]
        new_local_U = _lift(old_local_U)

        if solver.turb_model is not None and hasattr(solver.turb_model, 'k_field'):
            solver.turb_model.k_field = _lift(
                solver.turb_model.k_field[:n_local])
            solver.turb_model.omega_field = _lift(
                solver.turb_model.omega_field[:n_local])
            old_nu_t = getattr(solver.turb_model, 'nu_t', None)
            if old_nu_t is not None and old_nu_t.shape[1] == old_local_U.shape[1]:
                solver.turb_model.nu_t = _lift(old_nu_t[:n_local])
            if hasattr(solver.turb_model, 'des_length_scale'):
                solver.turb_model.des_length_scale = None
        if getattr(solver, 'sgs_model', None) is not None and hasattr(solver.sgs_model, 'nu_t'):
            solver.sgs_model.nu_t = None
    else:
        from autoflowcfd.core.utils.flow_direction import (
            freestream_conservative_state,
        )

        new_n_sps = (target_p + 1) ** 3
        # 速度方向必须取自 aoa/aos（2026-09-24 修复）：此前写死 (vel_inf, 0, 0)，
        # 而边界 Q_free 用的是正确方向。9 处同类写法已统一到
        # `freestream_conservative_state`（见其文档）。
        new_local_U = np.empty((n_local, new_n_sps, n_vars))
        new_local_U[:] = freestream_conservative_state(freestream, n_vars)

        if solver.turb_model is not None and hasattr(solver.turb_model, 'k_field'):
            k_inf, omega_inf = _set_freestream_turbulence(solver)
            solver.turb_model.k_field = np.ones((n_local, new_n_sps)) * k_inf
            solver.turb_model.omega_field = np.ones((n_local, new_n_sps)) * omega_inf
            if hasattr(solver.turb_model, 'nu_t'):
                solver.turb_model.nu_t = np.zeros((n_local, new_n_sps))
            if hasattr(solver.turb_model, 'des_length_scale'):
                solver.turb_model.des_length_scale = None
        if getattr(solver, 'sgs_model', None) is not None and hasattr(solver.sgs_model, 'nu_t'):
            solver.sgs_model.nu_t = None

    # --- 2. Root 重新计算 + 分发新紧凑包 ---
    comm = get_comm()
    n_ranks = solver.n_ranks

    if is_root_rank:
        from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider
        import types

        mesh = root_context['mesh']
        stale_orders = [o for o in list(mesh._order_geometry_cache) if o != target_p]
        for o in stale_orders:
            del mesh._order_geometry_cache[o]
        mesh.set_order(target_p)
        ops = generate_fr_operators(target_p)
        root_context['ops'] = ops

        turb_model_name = root_context['turb_model_name']
        root_solver_stub = types.SimpleNamespace(
            mesh=mesh, freestream=root_context['freestream'], turb_model_name=turb_model_name,
            wmles_model=(object() if turb_model_name == "WMLES" else None),
        )
        boundary_ghost_provider_global = build_boundary_ghost_provider(
            root_solver_stub, bc_overrides=root_context.get('bc_overrides', {}),
        )
        root_context['boundary_ghost_provider_global'] = boundary_ghost_provider_global

        packages = [
            build_fully_distributed_rank_package(
                mesh, ops, root_context['fc'], root_context['cell_partition'], r, n_ranks,
                boundary_ghost_provider_global, root_context['freestream'],
                root_context['mu_molecular'], root_context['mach_ref'],
                target_p, root_context['enable_viscous'],
                turb_model_name=turb_model_name, wall_node_indices=root_context['wall_node_indices'],
                h_max_global=root_context['h_max_global'], h_wn_global=root_context['h_wn_global'],
                turbulence_intensity=root_context['turbulence_intensity'],
                viscosity_ratio=root_context['viscosity_ratio'],
                time_scheme=root_context.get('time_scheme'),
                dual_time_inner_iter=root_context.get('dual_time_inner_iter', 20),
                # 阶数切换重分发时同样要带上（2026-09-15）：否则升阶之后
                # CFL 边界参数会悄悄退回硬编码默认值，是一个只在 Order
                # Continuation 路径上出现的静默回退。
                cfl_start=root_context.get('cfl_start'),
                cfl_max=root_context.get('cfl_max'),
                cfl_min=root_context.get('cfl_min'),
            )
            for r in range(n_ranks)
        ]
        my_package = packages[0]
        if n_ranks > 1:
            import pickle
            for r in range(1, n_ranks):
                buf = pickle.dumps(packages[r])
                buf_size = np.array([len(buf)], dtype=np.int64)
                comm.Send(buf_size, dest=r, tag=320)
                comm.Send(buf, dest=r, tag=321)
    else:
        import pickle
        buf_size = np.empty(1, dtype=np.int64)
        comm.Recv(buf_size, source=0, tag=320)
        buf = np.empty(int(buf_size[0]), dtype=np.uint8)
        comm.Recv(buf, source=0, tag=321)
        my_package = pickle.loads(buf.tobytes())

    # --- 3. 应用新包：替换 compact 相关属性，保留 turb_model/sgs_model/
    # wmles_model 对象本身（只是上一步已经替换过它们的数组）---
    solver.ops = generate_fr_operators(target_p)
    solver.mesh = my_package['precompacted_mesh']
    solver.partition = my_package['partition']
    solver.dist_flat_face = my_package['dist_fc']

    new_n_sps = solver.mesh.n_sps_per_cell
    new_state = DistributedFRState(solver.partition, new_n_sps, n_vars)
    new_state.U[:n_local] = new_local_U
    new_state.Q[:n_local] = conserved_to_primitive(new_local_U[..., :5])
    solver.state = new_state
    solver.halo_exchange = HaloExchange(solver.partition, new_n_sps, n_vars)

    if solver.turb_model is not None:
        solver.turb_halo_exchange = HaloExchange(solver.partition, new_n_sps, 2)
    solver.wall_distance_compact = my_package.get('wall_distance_compact')
    solver.iddes_h_max_compact = my_package.get('iddes_h_max_compact')
    solver.iddes_h_wn_compact = my_package.get('iddes_h_wn_compact')
    if solver.ddes_model is not None:
        solver.des_length_scale_halo_exchange = HaloExchange(solver.partition, new_n_sps, 1)

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
    solver._package_freestream = my_package['freestream']

    if hasattr(solver, '_dual_time_U_prev'):
        solver._dual_time_U_prev = None

    solver.current_order = target_p
