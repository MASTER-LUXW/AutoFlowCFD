"""
AutoFlowCFD V2.0 - 单 GPU（`GPUFRSolver`）Order Continuation（2026-09-02）

补齐三条分布式路径（`core/mpi/distributed_order_continuation.py`/
`core/gpu/distributed/gpu_distributed_order_continuation.py`）之外
最后一个仍然完全没有 Order Continuation 机制的后端——`GPUFRSolver`
是单机（非分布式）GPU 求解器，没有 partition/halo/dist_flat_face 这层
复杂度，本质上和单机 CPU `FRSolver` 同构（`self.mesh` 全程都是完整
真实网格），比三条分布式路径都更接近单机 CPU 的直接镜像：阶数切换
只需要 `mesh.set_order`/`ops` 重建 + 相应 GPU 常驻数组（`mesh_data`/
`flat_face_gpu`/`boundary_ghost_provider`/湍流模型 GPU 数组/壁面距离/
模态滤波）重新计算+重新上传，不需要任何跨 rank 通信或紧凑索引空间
重排。

P0（`mesh.n_points_1d==1`）在单 GPU 路径上不是问题——`compute_
inviscid_residual_gpu` 本身就有 `self.mesh.n_points_1d == 1` 的专用
分支（见 `gpu_solver.py` 该方法文档），直接读取 `self.mesh`（完整真实
网格，从不是 compact/分布式视图），与 CPU 分布式"传统模式"当年发现的
P0 架构缺口完全不适用于这里。
"""

import numpy as np
from typing import Optional

from autoflowcfd.core.utils.order_continuation import _build_linear_interp_matrix_3d


def gpu_solver_interpolate_to_new_order(solver, target_p: int) -> None:
    """`GPUFRSolver` 的阶数切换（升阶精确 Lagrange 延拓 / 降阶到 P0
    时重置为均匀自由流场，两种情形与 CPU/GPU 分布式版本同一处理，见
    `core/mpi/distributed_order_continuation.py::cpu_traditional_
    interpolate_to_new_order` 同名文档）。
    """
    from autoflowcfd.core.gpu import get_cupy
    from autoflowcfd.fr.operators import generate_fr_operators
    from autoflowcfd.fr.quadrature_points import gauss_legendre
    from autoflowcfd.core.fr_solver.turbulence import _set_freestream_turbulence

    cp = get_cupy()

    old_order = solver.current_order
    if old_order == target_p:
        return

    n_cells = solver.mesh.n_cells
    n_vars = solver.n_vars
    rho_inf = solver.freestream['rho_inf']
    vel_inf = solver.freestream['vel_inf']
    p_inf = solver.freestream['p_inf']

    # --- 1. U / 湍流场：CPU（numpy）上插值或重置（CuPy 数组不支持
    # np.einsum，插值矩阵构造本身也是一次性小矩阵运算，没必要为它单独
    # 写 CuPy 版本）---
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
        new_U_np = np.zeros((n_cells, new_n_sps, n_vars))
        new_U_np[:, :, 0] = rho_inf
        new_U_np[:, :, 1] = rho_inf * vel_inf
        new_U_np[:, :, 4] = rho_inf * e

        if solver.turb_model_gpu is not None:
            k_inf, omega_inf = _set_freestream_turbulence(solver)
            new_k_np = np.ones((n_cells, new_n_sps)) * k_inf
            new_omega_np = np.ones((n_cells, new_n_sps)) * omega_inf
            new_nu_t_np = np.zeros((n_cells, new_n_sps))

    if solver.ddes_model_gpu is not None and hasattr(solver, 'des_length_scale_gpu'):
        # DDES/IDDES 有效长度尺度：清空而不是插值，理由同 CPU/GPU 分布式
        # 版本文档——依赖 nu_t，nu_t 要到本阶数第一次 compute_source
        # 调用后才会被重新算出。
        solver.des_length_scale_gpu = None

    # --- 2. mesh/ops 切换（self.mesh 是完整真实网格，无分布式/compact
    # 索引空间这层复杂度）---
    stale_orders = [o for o in list(solver.mesh._order_geometry_cache) if o != target_p]
    for o in stale_orders:
        del solver.mesh._order_geometry_cache[o]
    solver.mesh.set_order(target_p)
    solver.ops = generate_fr_operators(target_p, flux_point_type=getattr(solver, 'flux_type', 'radau'))

    # --- 3. GPU 常驻网格数据 + 面几何重新上传 ---
    solver.mesh_data = solver.array_mgr.upload_mesh_data(solver.mesh, solver.ops)
    solver.ops_data = {k: v for k, v in solver.mesh_data.items()}
    solver._init_face_geometry()

    n_sps = solver.mesh.n_sps_per_cell
    with cp.cuda.Device(solver.device_id):
        solver.U_gpu = cp.asarray(new_U_np)
    solver._update_primitives_gpu()

    # --- 4. 湍流场对象（k_field/omega_field/nu_t 数组替换；模型本身
    # ——GPUTurbulenceSST/GPUDDESModel/GPUIDDESModel/GPUWALEModel 实例
    # ——不含任何阶数相关内部状态，不需要重新构造，与 GPU 分布式版本
    # 同一处理）---
    if solver.turb_model_gpu is not None:
        with cp.cuda.Device(solver.device_id):
            solver.turb_model_gpu.k_field = cp.asarray(new_k_np)
            solver.turb_model_gpu.omega_field = cp.asarray(new_omega_np)
            if new_nu_t_np is not None:
                solver.turb_model_gpu.nu_t = cp.asarray(new_nu_t_np)

    if solver.sgs_model_gpu is not None:
        cell_volumes = solver.mesh.get_all_cell_volumes()
        delta_cpu = np.power(np.abs(cell_volumes), 1.0 / 3.0)
        delta_cpu = np.tile(delta_cpu[:, np.newaxis], (1, n_sps))
        with cp.cuda.Device(solver.device_id):
            solver._grid_scale_gpu = cp.asarray(delta_cpu)

    # --- 5. boundary_ghost_provider（SEM 入口幽灵态按当时阶数的 FP
    # 几何预存了每面 FP 物理坐标，必须重建）---
    solver.boundary_ghost_provider = solver._build_boundary_ghost_provider(solver.bc_overrides)

    # WALL 边界面拓扑掩码：与 __init__ 同一套构造，压缩面数已随上面的
    # 重建改变。iddes_h_max/h_wn（`self._iddes_h_max_gpu`/`_wn_gpu`）
    # 是纯逐单元几何量（与阶数/SPs 无关），不需要重算，不在这里处理。
    if solver.turb_model_gpu is not None:
        from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport import compute_wall_dirichlet_mask_gpu
        wall_mask_np = compute_wall_dirichlet_mask_gpu(solver.mesh, solver.boundary_ghost_provider)
        with cp.cuda.Device(solver.device_id):
            solver._wall_mask_k_gpu = cp.asarray(wall_mask_np)

    # 壁面距离（compact/阶数相关，y+/SST 输运需要）——复用 __init__ 同一个
    # 实例方法。
    if solver.turb_model_gpu is not None or solver.sgs_model_gpu is not None:
        solver._init_wall_distance_gpu()

    # 分布式模态滤波：新阶数的 filter_prism/filter_tet 形状不同，必须
    # 重建（复用 __init__ 同一个实例方法）。
    solver._init_modal_filter_gpu()

    # DUAL_TIME 上一物理时间层历史随阶数切换失效——与 CPU/GPU 分布式
    # 版本同一处处理，理由同该文档。
    if hasattr(solver, '_dual_time_U_prev'):
        solver._dual_time_U_prev = None

    solver.current_order = target_p
