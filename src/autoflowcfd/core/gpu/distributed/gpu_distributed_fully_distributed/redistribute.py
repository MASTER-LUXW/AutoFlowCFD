"""AutoFlowCFD V2.0 - 阶数切换时重新分发（Order Continuation 分布式路径）"""

import types

from autoflowcfd.core.gpu import get_cupy

from autoflowcfd.core.mpi import get_rank

from autoflowcfd.core.mpi.distributed_state import DistributedFRState

from autoflowcfd.core.gpu.distributed.gpu_halo_exchange import GPUHaloExchange
from .upload import _upload_wall_geometry_compact
from autoflowcfd.core.utils.order_continuation.order_switch import (
    install_turbulence, reset_order_dependent_state, switched_fields,
)


def redistribute_multi_gpu_fully_distributed_for_new_order(solver, target_p: int) -> None:
    """"完全分布式加载"模式的多 GPU 阶数切换（Order Continuation）。

    与 CPU 版 `core/mpi/distributed_mesh_loader/fully_distributed.py::redistribute_fully_distributed_for_new_order`
    同一套协议（该函数文档的"协议"一节
    逐字适用，这里只重复 GPU 特有的部分）：
    1. 每个 rank 独立在 CPU（numpy）上插值/重置自己的 local U + 湍流场
       （从 GPU 下载 → 插值 → 稍后重新上传，理由与 `gpu_distributed_order_continuation.py::
       gpu_interpolate_to_new_order`"传统模式"
       版本一致：CuPy 数组不支持 `np.einsum`）。
    2. Root：用 `solver._root_context` 持有的完整全局网格重新计算 +
       重新分发新阶数的紧凑包（`build_fully_distributed_rank_package`，
       与初次构造同一个函数，只是阶数/mesh 状态不同）。
    3. 每个 rank：接收新包，把上一步插值/重置好的 local U 写入新
       partition 布局的 state，替换全部 compact 相关 GPU 属性（mesh_data/
       flat_face_gpu/halo 交换器/wall_distance/boundary_ghost_provider
       等），不重新构造 solver 实例本身（保留 turb_model_gpu/
       sgs_model_gpu/wall_model 对象引用，只替换它们的 k_field/
       omega_field/nu_t 数组——与"传统模式" `gpu_interpolate_to_new_
       order` 同一个设计）。

    只支持通过 `root_context` 非 None 构造的实例（与 CPU 版同一个
    fail-fast 护栏）。
    """
    from autoflowcfd.fr.operators import generate_fr_operators
    from autoflowcfd.core.gpu.gpu_face_geometry import build_gpu_flat_face

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

    # 场在主机上换阶（`order_switch.switched_fields`，与其余后端同一份）
    turb = solver.turb_model_gpu
    switched = switched_fields(
        old_order, target_p, int(solver.mesh.n_prism_cells), solver.freestream, cp.asnumpy(solver.U_gpu), turb,
        turbulence=None if turb is None else [cp.asnumpy(f) for f in turb.transported_fields()],
        nu_t=None if turb is None or turb.nu_t is None else cp.asnumpy(turb.nu_t))

    # --- 2. Root 重新计算 + 分发新紧凑包（与 CPU 共用，见该函数文档）---
    from autoflowcfd.core.mpi.distributed_mesh_loader.fully_distributed import (
        exchange_packages_for_new_order,
    )

    my_package = exchange_packages_for_new_order(
        root_context if is_root_rank else None, target_p, solver.n_ranks, tags=(340, 341))

    # --- 3. 应用新包：替换 compact 相关属性，保留 turb_model_gpu/
    # sgs_model_gpu/wall_model 对象本身（只是上一步已经替换过它们的
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
        solver.U_gpu = cp.asarray(switched.U)

    solver.gpu_halo = GPUHaloExchange(solver.partition, n_sps=new_n_sps, n_vars=5, device_id=solver.device_id)
    solver.state = DistributedFRState(solver.partition, new_n_sps)

    with cp.cuda.Device(solver.device_id):
        install_turbulence(turb, switched, cp.asarray)
    if turb is not None:
        _upload_wall_geometry_compact(solver, my_package, cp, solver.device_id)

    solver.boundary_ghost_provider = my_package['boundary_ghost_provider']
    solver._wall_mask_gpu = None
    solver._open_mask_gpu = None
    if solver.turb_model_gpu is not None:
        from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport import compute_turbulence_face_masks_gpu
        compact_mesh_stub = types.SimpleNamespace(
            face_connectivity=types.SimpleNamespace(n_faces=solver.dist_flat_face.base_flat.n_faces)
        )
        wall_mask_np, open_mask_np = compute_turbulence_face_masks_gpu(compact_mesh_stub,
                                                                       solver.boundary_ghost_provider)
        with cp.cuda.Device(solver.device_id):
            solver._wall_mask_gpu = cp.asarray(wall_mask_np)
            solver._open_mask_gpu = cp.asarray(open_mask_np)

    solver.mu_molecular = my_package['mu_molecular']
    solver.freestream = {**my_package['freestream'], "mach_ref": my_package['mach_ref']}

    reset_order_dependent_state(solver, solver.sgs_model_gpu)

    solver.current_order = target_p
