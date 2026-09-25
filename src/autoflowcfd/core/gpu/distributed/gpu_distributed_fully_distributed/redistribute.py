"""AutoFlowCFD V2.0 - 阶数切换时重新分发（Order Continuation 分布式路径）

从 `src/autoflowcfd/core/gpu/distributed/gpu_distributed_fully_distributed.py`(原 557 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""

import types

import pickle

import numpy as np


from autoflowcfd.core.gpu import get_cupy

from autoflowcfd.core.mpi import get_rank

from autoflowcfd.core.mpi.comm import get_comm

from autoflowcfd.core.mpi.distributed_state import DistributedFRState

from autoflowcfd.core.gpu.distributed.gpu_halo_exchange import GPUHaloExchange
from .upload import _upload_wall_geometry_compact


def redistribute_multi_gpu_fully_distributed_for_new_order(solver, target_p: int) -> None:
    """"完全分布式加载"模式的多 GPU 阶数切换（Order Continuation）。

    与 CPU 版 `distributed_mesh_loader.py::redistribute_fully_
    distributed_for_new_order` 同一套协议（该函数文档的"协议"一节
    逐字适用，这里只重复 GPU 特有的部分）：
    1. 每个 rank 独立在 CPU（numpy）上插值/重置自己的 local U + 湍流场
       （从 GPU 下载 → 插值 → 稍后重新上传，理由与 `gpu_distributed_
       order_continuation.py::gpu_interpolate_to_new_order`"传统模式"
       版本一致：CuPy 数组不支持 `np.einsum`）。
    2. Root：用 `solver._root_context` 持有的完整全局网格重新计算 +
       重新分发新阶数的紧凑包（`build_fully_distributed_rank_package`，
       与初次构造同一个函数，只是阶数/mesh 状态不同）。
    3. 每个 rank：接收新包，把上一步插值/重置好的 local U 写入新
       partition 布局的 state，替换全部 compact 相关 GPU 属性（mesh_data/
       flat_face_gpu/halo 交换器/wall_distance/boundary_ghost_provider
       等），不重新构造 solver 实例本身（保留 turb_model_gpu/
       sgs_model_gpu/wmles_model 对象引用，只替换它们的 k_field/
       omega_field/nu_t 数组——与"传统模式" `gpu_interpolate_to_new_
       order` 同一个设计）。

    只支持通过 `root_context` 非 None 构造的实例（与 CPU 版同一个
    fail-fast 护栏）。
    """
    from autoflowcfd.fr.operators import generate_fr_operators
    from autoflowcfd.core.gpu.gpu_face_geometry import build_gpu_flat_face
    from autoflowcfd.core.fr_solver.turbulence import _set_freestream_turbulence

    cp = get_cupy()
    old_order = solver.current_order
    if old_order == target_p:
        return

    is_root_rank = get_rank() == 0
    root_context = getattr(solver, '_root_context', None) if is_root_rank else None
    if is_root_rank and root_context is None:
        raise NotImplementedError(
            "redistribute_multi_gpu_fully_distributed_for_new_order: root rank 没有 "
            "_root_context（本实例不是通过 distributed_mesh_load_v2 返回的 "
            "root_context 构造的）——Order Continuation 在'完全分布式加载' "
            "模式下要求 root 持续持有完整全局网格用于重新分发，不支持从 "
            "缺少这份上下文的实例继续爬坡（如实报告，不是假装能用）。"
        )

    n_local = solver.partition.n_local_cells

    # --- 1. 每个 rank 独立插值/重置自己的 local U + 湍流场（CPU 上做，
    # 理由见函数文档）---
    freestream = getattr(solver, '_package_freestream', None) or solver.freestream
    # 不给兜底值，理由同 build.py 同名注释。

    new_k_np = new_omega_np = new_nu_t_np = None
    if target_p > old_order:
        # 延拓算子**按基分派**（2026-09-20 修复的真实缺陷）：一维 Gauss
        # 张量积 Lagrange 只对坍缩棱柱基的解点成立；native 四面体
        # （自 2026-09-03 起是四面体唯一实现）与 native 棱柱（2026-09-20
        # 起是默认）都不在那个网格上，线性场 P1->P2 实测相对误差
        # 7.0e-01 / 1.4e-01。compact/local 索引空间同样是"棱柱在前"
        # （见 `base_flat.n_prism` 文档），所以直接用局部棱柱数。
        # 完整依据见 `fr/order_interp.py`。
        from autoflowcfd.fr.order_interp import apply_order_interp

        n_prism_local = int(solver.mesh.n_prism_cells)

        def _lift(field):
            return apply_order_interp(field, n_prism_local, old_order,
                                      target_p)

        old_U_np = cp.asnumpy(solver.U_gpu)
        new_U_np = _lift(old_U_np)

        if solver.turb_model_gpu is not None:
            new_k_np = _lift(cp.asnumpy(solver.turb_model_gpu.k_field))
            new_omega_np = _lift(cp.asnumpy(solver.turb_model_gpu.omega_field))
            old_nu_t = getattr(solver.turb_model_gpu, 'nu_t', None)
            if old_nu_t is not None:
                old_nu_t_np = cp.asnumpy(old_nu_t)
                if old_nu_t_np.shape[1] == old_U_np.shape[1]:
                    new_nu_t_np = _lift(old_nu_t_np)
    else:
        from autoflowcfd.core.utils.flow_direction import (
            freestream_conservative_state,
        )

        new_n_sps = (target_p + 1) ** 3
        # 速度方向必须取自 aoa/aos（2026-09-24 修复）：此前这里写死 (vel_inf, 0, 0)，
        # 而边界 Q_free 用的是正确方向，`--aoa` 非零时初场与边界不一致。
        # 8 处同类写法已统一到 `freestream_conservative_state`（见其文档）。
        new_U_np = np.empty((n_local, new_n_sps, 5))
        new_U_np[:] = freestream_conservative_state(freestream, 5)

        if solver.turb_model_gpu is not None:
            k_inf, omega_inf = _set_freestream_turbulence(solver)
            new_k_np = np.ones((n_local, new_n_sps)) * k_inf
            new_omega_np = np.ones((n_local, new_n_sps)) * omega_inf
            new_nu_t_np = np.zeros((n_local, new_n_sps))

    # des_length_scale：与"传统模式"同一处理，清空而不是插值（依赖 nu_t，
    # 要到本阶数第一次 compute_source 调用后才会被重新算出）。
    if solver.ddes_model_gpu is not None and hasattr(solver.turb_model_gpu, 'des_length_scale'):
        solver.turb_model_gpu.des_length_scale = None

    # --- 2. Root 重新计算 + 分发新紧凑包 ---
    comm = get_comm()
    n_ranks = solver.n_ranks

    if is_root_rank:
        from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider
        from autoflowcfd.core.mpi.distributed_mesh_loader import build_fully_distributed_rank_package

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
                turb_model_name=turb_model_name,
                wall_distance_source=root_context['wall_distance_source'],
                h_max_global=root_context['h_max_global'], h_wn_global=root_context['h_wn_global'],
                turbulence_intensity=root_context['turbulence_intensity'],
                viscosity_ratio=root_context['viscosity_ratio'],
                time_scheme=root_context.get('time_scheme'),
                dual_time_inner_iter=root_context.get('dual_time_inner_iter', 20),
            )
            for r in range(n_ranks)
        ]
        my_package = packages[0]
        if n_ranks > 1:
            for r in range(1, n_ranks):
                buf = pickle.dumps(packages[r])
                buf_size = np.array([len(buf)], dtype=np.int64)
                comm.Send(buf_size, dest=r, tag=340)
                comm.Send(buf, dest=r, tag=341)
    else:
        buf_size = np.empty(1, dtype=np.int64)
        comm.Recv(buf_size, source=0, tag=340)
        buf = np.empty(int(buf_size[0]), dtype=np.uint8)
        comm.Recv(buf, source=0, tag=341)
        my_package = pickle.loads(buf.tobytes())

    # --- 3. 应用新包：替换 compact 相关属性，保留 turb_model_gpu/
    # sgs_model_gpu/wmles_model 对象本身（只是上一步已经替换过它们的
    # 数组）---
    solver.ops = generate_fr_operators(target_p)
    precompacted_mesh = my_package['precompacted_mesh']
    solver.mesh = precompacted_mesh
    solver.partition = my_package['partition']
    solver.dist_flat_face = my_package['dist_fc']

    new_n_sps = precompacted_mesh.n_sps_per_cell
    solver.flat_face_gpu = build_gpu_flat_face(solver.dist_flat_face.base_flat, solver.device_id)
    solver.mesh_data = solver.array_mgr.upload_mesh_data(precompacted_mesh, solver.ops)
    solver.n_compact = precompacted_mesh.n_cells
    solver._compact_mesh_view = precompacted_mesh
    solver.ops_data = {k: v for k, v in solver.mesh_data.items()}

    with cp.cuda.Device(solver.device_id):
        solver._perm_gpu = cp.asarray(solver.dist_flat_face.perm)
        solver._inv_perm_gpu = cp.asarray(solver.dist_flat_face.inv_perm)
        solver.U_gpu = cp.asarray(new_U_np)

    solver.gpu_halo = GPUHaloExchange(solver.partition, n_sps=new_n_sps, n_vars=5, device_id=solver.device_id)
    solver.state = DistributedFRState(solver.partition, new_n_sps, 5)

    if solver.turb_model_gpu is not None:
        with cp.cuda.Device(solver.device_id):
            solver.turb_model_gpu.k_field = cp.asarray(new_k_np)
            solver.turb_model_gpu.omega_field = cp.asarray(new_omega_np)
            if new_nu_t_np is not None:
                solver.turb_model_gpu.nu_t = cp.asarray(new_nu_t_np)
        solver.turb_halo_gpu = GPUHaloExchange(
            solver.partition, n_sps=new_n_sps, n_vars=2, device_id=solver.device_id
        )
        if solver.ddes_model_gpu is not None:
            solver.des_length_scale_halo_gpu = GPUHaloExchange(
                solver.partition, n_sps=new_n_sps, n_vars=1, device_id=solver.device_id
            )
        _upload_wall_geometry_compact(solver, my_package, cp, solver.device_id)

    if solver.sgs_model_gpu is not None:
        cell_volumes_compact = solver.mesh_data.get('cell_volumes')
        delta = cp.abs(cell_volumes_compact) ** (1.0 / 3.0)
        solver._grid_scale_compact = cp.tile(delta[:, None], (1, new_n_sps))

    if solver.wmles_model is not None:
        wall_distance_compact = my_package.get('wall_distance_compact')
        with cp.cuda.Device(solver.device_id):
            solver.wall_distance_gpu = (
                cp.asarray(wall_distance_compact) if wall_distance_compact is not None else None
            )

    solver.filter_func_gpu = None
    solver._init_modal_filter_distributed()

    solver.boundary_ghost_provider = my_package['boundary_ghost_provider']
    solver._wall_mask_k_gpu = None
    solver._open_mask_gpu = None
    if solver.turb_model_gpu is not None:
        from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport import compute_turbulence_face_masks_gpu
        compact_mesh_stub = types.SimpleNamespace(
            face_connectivity=types.SimpleNamespace(n_faces=solver.dist_flat_face.base_flat.n_faces)
        )
        wall_mask_np, open_mask_np = compute_turbulence_face_masks_gpu(compact_mesh_stub, solver.boundary_ghost_provider)
        with cp.cuda.Device(solver.device_id):
            solver._wall_mask_k_gpu = cp.asarray(wall_mask_np)
            solver._open_mask_gpu = cp.asarray(open_mask_np)

    solver.mu_molecular = my_package['mu_molecular']
    solver.freestream = {**my_package['freestream'], "mach_ref": my_package['mach_ref']}
    solver._package_freestream = my_package['freestream']

    if hasattr(solver, '_dual_time_U_prev'):
        solver._dual_time_U_prev = None

    solver.current_order = target_p
