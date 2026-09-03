"""
AutoFlowCFD V2.0 - 多 GPU 分布式 Order Continuation（2026-09-02）

`MultiGPUDistributedSolver`（"传统模式"：每个 rank 持有完整全局
`mesh`，见 `gpu_distributed.py::__init__` 的 `partition_info is None`
分支）的阶数切换重建——与 CPU `DistributedFRSolver`
（`core/mpi/distributed_order_continuation.py`）同一套设计，唯一区别
是需要在 CPU（numpy）上做插值/重置运算后再上传 GPU（CuPy 数组不支持
`np.einsum`，且插值矩阵构造本身是一次性的小矩阵运算，没有必要为它
单独写一个 CuPy 版本）。

"完全分布式加载"（`MultiGPUDistributedSolver.from_fully_distributed_
package` 构造，`solver._is_fully_distributed is True`，#1，2026-09-02
补齐）：委托给 `gpu_distributed_fully_distributed.py::redistribute_
multi_gpu_fully_distributed_for_new_order`（root 用持续持有的
`solver._root_context` 重新计算 + 重新分发新阶数紧凑包，见该函数
文档），不复用下面"传统模式"的逻辑（那需要每个 rank 持有的完整全局
`mesh`，这条路径下不存在）。

`__init__` 的 `partition_info` 参数构造模式（与"完全分布式加载"是两个
不同的东西——`partition_info` 是该构造函数里从未被 CLI 真正使用过的
遗留/未验证分支，`self.cell_partition` 恒为 None 且不设
`_is_fully_distributed`）仍然不支持——`solver._is_fully_distributed`
不为 True 且 `solver.cell_partition is None` 时 fail-fast。
"""

import numpy as np
from typing import Optional

from autoflowcfd.core.utils.order_continuation import _build_linear_interp_matrix_3d


def gpu_interpolate_to_new_order(solver, target_p: int) -> None:
    """`MultiGPUDistributedSolver` 的阶数切换（升阶精确 Lagrange 延拓 /
    降阶到 P0 时重置为均匀自由流场，两种情形见 CPU 版
    `distributed_order_continuation.py::cpu_traditional_interpolate_to_
    new_order` 同名文档，此处只重复 GPU 特有的部分）。
    """
    from autoflowcfd.core.gpu import get_cupy
    from autoflowcfd.fr.operators import generate_fr_operators
    from autoflowcfd.fr.quadrature_points import gauss_legendre
    from autoflowcfd.core.mpi.partition import build_distributed_partition
    from autoflowcfd.core.mpi.distributed_state import DistributedFRState
    from autoflowcfd.core.gpu.distributed.gpu_halo_exchange import GPUHaloExchange
    from autoflowcfd.core.gpu.distributed.gpu_distributed import _CompactMeshDataView
    from autoflowcfd.core.fr_solver.turbulence import _set_freestream_turbulence
    from autoflowcfd.core.fr_solver.boundary import build_boundary_ghost_provider

    cp = get_cupy()

    old_order = solver.current_order
    if old_order == target_p:
        return

    if getattr(solver, '_is_fully_distributed', False):
        from autoflowcfd.core.gpu.distributed.gpu_distributed_fully_distributed import (
            redistribute_multi_gpu_fully_distributed_for_new_order,
        )
        redistribute_multi_gpu_fully_distributed_for_new_order(solver, target_p)
        return

    if getattr(solver, 'cell_partition', None) is None:
        raise NotImplementedError(
            "gpu_interpolate_to_new_order: 本 solver 实例是通过 "
            "partition_info（'分布式加载'）构造的——本 rank 没有完整"
            "全局网格，Order Continuation 在这条路径上不支持（如实"
            "报告，见 MultiGPUDistributedSolver.__init__ 文档"
            "'partition_info' 分支说明）。"
        )

    n_local = solver.partition.n_local_cells
    rho_inf = solver.freestream['rho_inf']
    vel_inf = solver.freestream['vel_inf']
    p_inf = solver.freestream['p_inf']

    # --- 1. local U / 湍流场：CPU 上插值或重置（不依赖几何重建）---
    new_k_np = new_omega_np = new_nu_t_np = None
    if target_p > old_order:
        old_sps_1d, _ = gauss_legendre(old_order + 1)
        new_sps_1d, _ = gauss_legendre(target_p + 1)
        W = _build_linear_interp_matrix_3d(old_sps_1d, new_sps_1d)

        old_U_np = cp.asnumpy(solver.U_gpu)
        new_U_np = np.einsum('ab,cbv->cav', W, old_U_np)

        if solver.turb_model_gpu is not None:
            old_k_np = cp.asnumpy(solver.turb_model_gpu.k_field)
            old_omega_np = cp.asnumpy(solver.turb_model_gpu.omega_field)
            new_k_np = np.einsum('ab,cb->ca', W, old_k_np)
            new_omega_np = np.einsum('ab,cb->ca', W, old_omega_np)
            old_nu_t = getattr(solver.turb_model_gpu, 'nu_t', None)
            if old_nu_t is not None:
                old_nu_t_np = cp.asnumpy(old_nu_t)
                if old_nu_t_np.shape[1] == old_U_np.shape[1]:
                    new_nu_t_np = np.einsum('ab,cb->ca', W, old_nu_t_np)
    else:
        new_n_sps = (target_p + 1) ** 3
        gamma = 1.4
        e = p_inf / ((gamma - 1.0) * rho_inf) + 0.5 * vel_inf ** 2
        new_U_np = np.zeros((n_local, new_n_sps, 5))
        new_U_np[:, :, 0] = rho_inf
        new_U_np[:, :, 1] = rho_inf * vel_inf
        new_U_np[:, :, 4] = rho_inf * e

        if solver.turb_model_gpu is not None:
            k_inf, omega_inf = _set_freestream_turbulence(solver)
            new_k_np = np.ones((n_local, new_n_sps)) * k_inf
            new_omega_np = np.ones((n_local, new_n_sps)) * omega_inf
            new_nu_t_np = np.zeros((n_local, new_n_sps))

    if solver.ddes_model_gpu is not None:
        # DDES/IDDES 有效长度尺度：与 CPU 版同一处理，清空而不是插值
        # （依赖 nu_t，nu_t 要到本阶数第一次 compute_source 调用后才会
        # 被重新算出，见 order_continuation.py::interpolate_to_new_order
        # 对应文档）。
        if hasattr(solver, 'des_length_scale_gpu'):
            solver.des_length_scale_gpu = None
    if getattr(solver, 'sgs_model_gpu', None) is not None:
        # WALE 没有跨步持久 nu_t 状态（每步现算），无需处理。
        pass

    # --- 2. mesh/ops 切换（mesh 是"传统模式"下每个 rank 都持有的完整
    # 全局网格）---
    stale_orders = [o for o in list(solver.mesh._order_geometry_cache) if o != target_p]
    for o in stale_orders:
        del solver.mesh._order_geometry_cache[o]
    solver.mesh.set_order(target_p)
    solver.ops = generate_fr_operators(target_p, flux_point_type=getattr(solver, 'flux_type', 'radau'))

    solver.partition = build_distributed_partition(
        solver.mesh.face_connectivity, solver.cell_partition, solver.rank, solver.n_ranks
    )
    # 重建分布式面几何（CPU 侧 dist_flat_face + GPU 侧 flat_face_gpu）
    # ——复用 __init__ 已有的实例方法，同一套逻辑。
    solver._init_distributed_face_geometry()

    compact_mesh_view = _CompactMeshDataView(
        solver.mesh, solver.dist_flat_face.compact_global_ids,
        solver.dist_flat_face.base_flat.n_prism,
    )
    solver._compact_mesh_view = compact_mesh_view
    solver.mesh_data = solver.array_mgr.upload_mesh_data(compact_mesh_view, solver.ops)
    solver.n_compact = compact_mesh_view.n_cells
    solver.ops_data = {k: v for k, v in solver.mesh_data.items()}

    n_sps = solver.ops.D_3d.shape[0]
    with cp.cuda.Device(solver.device_id):
        solver._perm_gpu = cp.asarray(solver.dist_flat_face.perm)
        solver._inv_perm_gpu = cp.asarray(solver.dist_flat_face.inv_perm)
        solver.U_gpu = cp.asarray(new_U_np)

    solver.gpu_halo = GPUHaloExchange(solver.partition, n_sps=n_sps, n_vars=5, device_id=solver.device_id)
    solver.state = DistributedFRState(solver.partition, n_sps, 5)

    # --- 3. 湍流场对象（k_field/omega_field/nu_t 数组替换 + halo 交换器
    # 重建；模型本身——GPUTurbulenceSST/GPUDDESModel/GPUIDDESModel/
    # GPUWALEModel 实例——不含任何阶数相关内部状态，不需要重新构造）---
    if solver.turb_model_gpu is not None:
        with cp.cuda.Device(solver.device_id):
            solver.turb_model_gpu.k_field = cp.asarray(new_k_np)
            solver.turb_model_gpu.omega_field = cp.asarray(new_omega_np)
            if new_nu_t_np is not None:
                solver.turb_model_gpu.nu_t = cp.asarray(new_nu_t_np)
        solver.turb_halo_gpu = GPUHaloExchange(
            solver.partition, n_sps=n_sps, n_vars=2, device_id=solver.device_id
        )
        if solver.ddes_model_gpu is not None:
            solver.des_length_scale_halo_gpu = GPUHaloExchange(
                solver.partition, n_sps=n_sps, n_vars=1, device_id=solver.device_id
            )
            # 真实 bug 修复（2026-09-02，DDES 补齐 max_edge 网格尺度时
            # 发现——与本次 Order Continuation 改动无关）：此前这里
            # 只要 `_iddes_h_max_global` 存在就无条件读
            # `solver._iddes_h_wn_global`，隐含假设"有 h_max 就一定是
            # IDDES、一定有 h_wn"——DDES（非 IDDES）现在也会设置
            # `_iddes_h_max_global`（只需要 h_max，不需要 h_wn，见
            # `MultiGPUDistributedSolver.__init__` DDES 分支文档），
            # DDES + Order Continuation 组合会在这里 AttributeError。
            h_max_global = getattr(solver, '_iddes_h_max_global', None)
            if h_max_global is not None:
                compact_global_ids = solver.dist_flat_face.compact_global_ids
                with cp.cuda.Device(solver.device_id):
                    solver.iddes_h_max_compact = cp.asarray(h_max_global[compact_global_ids])
                h_wn_global = getattr(solver, '_iddes_h_wn_global', None)
                if h_wn_global is not None:
                    with cp.cuda.Device(solver.device_id):
                        solver.iddes_h_wn_compact = cp.asarray(h_wn_global[compact_global_ids])

    if solver.sgs_model_gpu is not None:
        cell_volumes_compact = solver.mesh_data.get('cell_volumes')
        delta = cp.abs(cell_volumes_compact) ** (1.0 / 3.0)
        solver._grid_scale_compact = cp.tile(delta[:, None], (1, n_sps))

    # 分布式模态滤波：新阶数的 filter_prism/filter_tet 形状不同，必须
    # 重建（复用 __init__ 同一个实例方法）。
    solver._init_modal_filter_distributed()

    # --- 4. boundary_ghost_provider（SEM 入口幽灵态按当时阶数的 FP
    # 几何预存了每面 FP 物理坐标，必须重建）---
    solver.boundary_ghost_provider = build_boundary_ghost_provider(solver, bc_overrides={})
    if solver.boundary_ghost_provider is not None and hasattr(solver.boundary_ghost_provider, 'group_code'):
        solver.boundary_ghost_provider.group_code = (
            solver.boundary_ghost_provider.group_code[solver.partition.local_faces]
        )

    # WALL 边界面拓扑掩码（compact 索引空间，SST k/omega 输运 Dirichlet
    # BC 用）——与 __init__ 同一套构造，压缩面数已随上面的重建改变。
    if solver.turb_model_gpu is not None:
        import types
        from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport import compute_wall_dirichlet_mask_gpu
        compact_mesh_stub = types.SimpleNamespace(
            face_connectivity=types.SimpleNamespace(n_faces=solver.dist_flat_face.base_flat.n_faces)
        )
        wall_mask_np = compute_wall_dirichlet_mask_gpu(compact_mesh_stub, solver.boundary_ghost_provider)
        with cp.cuda.Device(solver.device_id):
            solver._wall_mask_k_gpu = cp.asarray(wall_mask_np)

    # 壁面距离（compact 索引空间，y+/SST 输运需要）——复用 __init__ 同一个
    # 实例方法，内部按 solver.dist_flat_face.compact_global_ids/
    # solver.mesh.sps_coords（已经 set_order 到 target_p）重新算。
    if solver.turb_model_gpu is not None or solver.wmles_model is not None:
        solver._init_wall_distance_distributed()

    # DUAL_TIME 上一物理时间层历史随阶数切换失效——与 CPU 版
    # distributed_order_continuation.py 同一处处理，理由同该文档。
    if hasattr(solver, '_dual_time_U_prev'):
        solver._dual_time_U_prev = None

    solver.current_order = target_p
