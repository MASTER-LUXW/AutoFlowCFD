"""
AutoFlowCFD V2.0 - 单 GPU（`GPUFRSolver`）Order Continuation（2026-09-02）

补齐三条分布式路径（`core/mpi/distributed_order_continuation/`/
`core/gpu/distributed/gpu_distributed_order_continuation.py`）之外
最后一个仍然完全没有 Order Continuation 机制的后端——`GPUFRSolver`
是单机（非分布式）GPU 求解器，没有 partition/halo/dist_flat_face 这层
复杂度，本质上和单机 CPU `FRSolver` 同构（`self.mesh` 全程都是完整
真实网格），比三条分布式路径都更接近单机 CPU 的直接镜像：阶数切换
只需要 `mesh.set_order`/`ops` 重建 + 相应 GPU 常驻数组（`mesh_data`/
`flat_face_gpu`/`boundary_ghost_provider`/湍流模型 GPU 数组/壁面距离）重新计算+重新上传，不需要任何跨 rank 通信或紧凑索引空间
重排。

P0（`mesh.n_points_1d==1`）在单 GPU 路径上不是问题——`compute_
inviscid_residual_gpu` 本身就有 `self.mesh.n_points_1d == 1` 的专用
分支（见 `gpu_solver/` 该方法文档），直接读取 `self.mesh`（完整真实
网格，从不是 compact/分布式视图），与 CPU 分布式"传统模式"当年发现的
P0 架构缺口完全不适用于这里。
"""

from autoflowcfd.core.utils.order_continuation.order_switch import (
    install_turbulence, reset_order_dependent_state, switch_mesh_order, switched_fields,
)


def gpu_solver_interpolate_to_new_order(solver, target_p: int) -> None:
    """`GPUFRSolver` 的阶数切换（升阶精确 Lagrange 延拓 / 降阶到 P0
    时重置为均匀自由流场，两种情形与 CPU/GPU 分布式版本同一处理，见
    `core/mpi/distributed_order_continuation/rebuild.py::cpu_traditional_interpolate_to_new_order` 同名文档）。
    """
    from autoflowcfd.core.gpu import get_cupy
    cp = get_cupy()

    old_order = solver.current_order
    if old_order == target_p:
        return

    # 场在主机上换阶（`order_switch.switched_fields`，与其余后端同一份）
    turb = solver.turb_model_gpu
    switched = switched_fields(
        old_order, target_p, int(solver.mesh.n_prism_cells), solver.freestream, cp.asnumpy(solver.U_gpu), turb,
        turbulence=None if turb is None else [cp.asnumpy(f) for f in turb.transported_fields()],
        nu_t=None if turb is None or turb.nu_t is None else cp.asnumpy(turb.nu_t))
    solver.ops = switch_mesh_order(solver.mesh, target_p)

    # --- 3. GPU 常驻网格数据 + 面几何重新上传 ---
    solver.mesh_data = solver.array_mgr.upload_mesh_data(solver.mesh, solver.ops)
    solver.ops_data = {k: v for k, v in solver.mesh_data.items()}
    solver._init_face_geometry()

    with cp.cuda.Device(solver.device_id):
        solver.U_gpu = cp.asarray(switched.U)
    solver._update_primitives_gpu()

    with cp.cuda.Device(solver.device_id):
        install_turbulence(turb, switched, cp.asarray)

    # --- 5. boundary_ghost_provider（SEM 入口幽灵态按当时阶数的 FP
    # 几何预存了每面 FP 物理坐标，必须重建）---
    solver.boundary_ghost_provider = solver._build_boundary_ghost_provider(solver.bc_overrides)

    # WALL 边界面拓扑掩码：与 __init__ 同一套构造，压缩面数已随上面的
    # 重建改变。iddes_h_max/h_wn（`self._iddes_h_max_gpu`/`_wn_gpu`）
    # 是纯逐单元几何量（与阶数/SPs 无关），不需要重算，不在这里处理。
    if solver.turb_model_gpu is not None:
        from autoflowcfd.core.gpu.turbulence.gpu_scalar_transport import compute_turbulence_face_masks_gpu
        wall_mask_np, open_mask_np = compute_turbulence_face_masks_gpu(solver.mesh, solver.boundary_ghost_provider)
        with cp.cuda.Device(solver.device_id):
            solver._wall_mask_gpu = cp.asarray(wall_mask_np)
            solver._open_mask_gpu = cp.asarray(open_mask_np)

    # 壁面距离（compact/阶数相关，y+/SST 输运需要）——复用 __init__ 同一个
    # 实例方法。
    if solver.turb_model_gpu is not None:
        solver._init_wall_distance_gpu()

    reset_order_dependent_state(solver, solver.sgs_model_gpu)

    solver.current_order = target_p
