"""`solve transient --n-ranks`/`--multi-gpu`/`--fully-distributed` 分布式
瞬态求解分支——从 `solve_transient_command.py` 拆出（控制单文件行数，
>400 行硬性拆分阈值），镜像 `solve_steady_command.py` 已有的分布式
构造模式（2026-09-02 补齐——此前本命令完全没有分布式支持，DUAL_TIME/
DES/LES 瞬态仿真只能单机跑，与 `solve steady` 已有的分布式覆盖不
一致；同一批还补齐了 `DistributedFRSolver`/`MultiGPUDistributedSolver`
本身对 DUAL_TIME 的支持，见 `core/mpi/distributed_solver.py`/
`core/gpu/distributed/gpu_distributed.py` 对应说明——没有这里的 CLI
入口，那两处修复无法被真正用到）。

`--fully-distributed` + `--time-method dual-time`（2026-09-02 续接）：
`build_fully_distributed_rank_package`/`distributed_mesh_load_v2`
已接入 `time_scheme`/`dual_time_inner_iter` 字段（见
core/mpi/distributed_mesh_loader.py 模块文档），此前的拒绝已放开——
不再是"设计上不支持"，只是此前没人把这两个参数接上。
"""

import click

from autoflowcfd.core.time_integration.base import TimeIntegrationScheme
from autoflowcfd.cli.solve_aero_coefficients import _report_aerodynamic_coefficients


def _solve_transient_distributed(
    input_file, order, surface_mesh, skip_quality_check,
    time_scheme, dual_time_inner_iter,
    turbulence_model, max_iter, dt, use_eikonal, output_dir,
    reference_area, threads, turbulence_intensity, viscosity_ratio,
    mu_molecular, rho_inf, vel_inf, p_inf,
    n_ranks, multi_gpu, fully_distributed, gpu_device, backend,
    checkpoint_interval, phase_max_iter=None, residual_drop_threshold=100.0,
    init_checkpoint=None,
):
    """`solve transient` 的分布式分支实现。

    `init_checkpoint`（`--init-from`，2026-09-02 续接）：从稳态
    checkpoint 恢复状态作为瞬态初场，三条分布式路径都已接入，见
    `core/mpi/distributed_checkpoint.py::restore_distributed_state_
    from_checkpoint` 文档。
    """
    from autoflowcfd.core.mpi import is_root

    print(f"\nInput Grid : {input_file}")
    print(f"Backend    : {'gpu (multi-GPU)' if multi_gpu else 'cpu (MPI)'} | "
          f"Order: P{order} | Ranks: {n_ranks}")
    print(f"Turbulence : {turbulence_model} | dt: {dt:.2e} | Iterations: {max_iter}\n")

    if multi_gpu:
        _solve_transient_multi_gpu(
            input_file, order, time_scheme, dual_time_inner_iter,
            turbulence_model, max_iter, dt, output_dir, threads,
            turbulence_intensity, viscosity_ratio, mu_molecular, rho_inf, vel_inf, p_inf,
            n_ranks, gpu_device, surface_mesh, skip_quality_check, checkpoint_interval,
            phase_max_iter, residual_drop_threshold, init_checkpoint,
            fully_distributed,
        )
        return

    if fully_distributed:
        _solve_transient_fully_distributed(
            input_file, order, surface_mesh, skip_quality_check,
            time_scheme, dual_time_inner_iter,
            turbulence_model, max_iter, dt, output_dir, backend,
            turbulence_intensity, viscosity_ratio, mu_molecular, rho_inf, vel_inf, p_inf,
            n_ranks, checkpoint_interval, phase_max_iter, residual_drop_threshold,
            init_checkpoint,
        )
        return

    _solve_transient_cpu_traditional(
        input_file, order, surface_mesh, skip_quality_check,
        time_scheme, dual_time_inner_iter,
        turbulence_model, max_iter, dt, use_eikonal, output_dir, backend, threads,
        turbulence_intensity, viscosity_ratio, mu_molecular, rho_inf, vel_inf, p_inf,
        n_ranks, checkpoint_interval, reference_area, phase_max_iter, residual_drop_threshold,
        init_checkpoint,
    )


def _solve_transient_cpu_traditional(
    input_file, order, surface_mesh, skip_quality_check,
    time_scheme, dual_time_inner_iter,
    turbulence_model, max_iter, dt, use_eikonal, output_dir, backend, threads,
    turbulence_intensity, viscosity_ratio, mu_molecular, rho_inf, vel_inf, p_inf,
    n_ranks, checkpoint_interval, reference_area,
    phase_max_iter=None, residual_drop_threshold=100.0,
    init_checkpoint=None,
):
    """CPU MPI"传统模式"：每个 rank 独立加载完整网格（与 `solve steady`
    的对应分支同一套构造方式，见该文件 `elif n_ranks > 1:` 分支文档）。"""
    from autoflowcfd.core.mpi import mpi_available, is_root
    if not mpi_available:
        print("\n❌ MPI not available. Please install mpi4py and run with mpirun.")
        print("   pip install mpi4py")
        raise click.Abort()

    from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
    from autoflowcfd.core.mpi.distributed_checkpoint import (
        distributed_save_results, distributed_save_checkpoint,
    )
    from autoflowcfd.cli.solve_mesh_loader import load_mesh_for_solver
    from autoflowcfd.fr.operators import generate_fr_operators

    mesh, volume_data = load_mesh_for_solver(
        input_file, order, surface_mesh=surface_mesh, skip_quality_check=skip_quality_check,
    )
    ops = generate_fr_operators(order)

    # 壁面距离场（SST/DDES/IDDES/WMLES 需要）已在 DistributedFRSolver.
    # __init__ 内部按 turb_model_name 自动计算好（self.wall_distance_
    # compact，见该构造函数对应分支），不需要在这里再调用单机版
    # compute_wall_distance_for_solver——与 solve_steady_command.py 的
    # "传统模式"分支同一个既定分工（该分支同样不调用这个函数）。
    solver = DistributedFRSolver(
        mesh=mesh, ops=ops, face_connectivity=mesh.face_connectivity,
        n_ranks=n_ranks, backend=backend, order=order,
        turb_model_name=turbulence_model, time_scheme=time_scheme,
        dual_time_inner_iter=dual_time_inner_iter, n_threads=threads,
        turbulence_intensity=turbulence_intensity, viscosity_ratio=viscosity_ratio,
        mu_molecular=mu_molecular, rho_inf=rho_inf, vel_inf=vel_inf, p_inf=p_inf,
    )

    if is_root():
        print(f"[Distributed] Initialized with {n_ranks} ranks (traditional mode)")
        print(f"[Distributed] {solver.partition.n_local_cells} local cells, "
              f"{solver.partition.n_halo} halo cells")

    if init_checkpoint:
        from autoflowcfd.core.mpi.distributed_checkpoint import (
            restore_distributed_state_from_checkpoint,
        )
        if is_root():
            print(f"\n🔄 从 checkpoint 加载稳态结果作为瞬态初场...")
        ckpt_iter = restore_distributed_state_from_checkpoint(init_checkpoint, solver)
        if is_root():
            print(f"   源 checkpoint 迭代数: {ckpt_iter}\n")

    def _checkpoint_cb(solver_ref, iteration):
        if iteration % checkpoint_interval != 0:
            return
        try:
            # order/target_order 分离：见 solve_steady_command.py 的
            # _distributed_checkpoint_cb 同一处修复文档——Order
            # Continuation 接入分布式路径后，爬坡阶段中途的 checkpoint
            # 必须记录 solver_ref.current_order（U_sps 实际形状），不是
            # 固定的目标 order。
            saved_path = distributed_save_checkpoint(
                solver_ref, output_dir, iteration, input_file,
                solver_ref.current_order, turbulence_model, backend,
                target_order=solver_ref.order,
            )
            if saved_path and is_root():
                print(f"   [Checkpoint] iter {iteration} saved: {saved_path}")
        except Exception as e:
            if is_root():
                print(f"   [Checkpoint] Warning: save failed at iter {iteration}: {e}")

    try:
        solver.solve(n_steps=max_iter, dt=dt, output_interval=checkpoint_interval,
                     checkpoint_callback=_checkpoint_cb,
                     phase_max_iter=phase_max_iter, residual_drop_threshold=residual_drop_threshold)
        if is_root():
            print(f"\n✅ Distributed transient simulation finished: iterations={max_iter}")

        distributed_save_results(solver, output_dir)
        distributed_save_checkpoint(
            solver, output_dir, max_iter, input_file,
            solver.current_order, turbulence_model, backend,
            target_order=solver.order,
        )
    except Exception as e:
        print(f"\n❌ Distributed transient simulation failed: {e}")
        raise click.Abort()


def _solve_transient_fully_distributed(
    input_file, order, surface_mesh, skip_quality_check,
    time_scheme, dual_time_inner_iter,
    turbulence_model, max_iter, dt, output_dir, backend,
    turbulence_intensity, viscosity_ratio, mu_molecular, rho_inf, vel_inf, p_inf,
    n_ranks, checkpoint_interval,
    phase_max_iter=None, residual_drop_threshold=100.0,
    init_checkpoint=None,
):
    """"完全分布式加载"：只有 root rank 加载完整网格（与 `solve steady`
    的 `if fully_distributed:` 分支同一套构造方式）。DUAL_TIME
    （2026-09-02 续接）：`time_scheme`/`dual_time_inner_iter` 随
    `distributed_mesh_load_v2` 一起塞进 package，见该函数模块文档。"""
    import math
    from autoflowcfd.core.mpi import mpi_available, is_root
    if not mpi_available:
        print("\n❌ MPI not available. Please install mpi4py and run with mpirun.")
        raise click.Abort()

    from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
    from autoflowcfd.core.mpi.distributed_mesh_loader import distributed_mesh_load_v2
    from autoflowcfd.core.mpi.distributed_checkpoint import (
        distributed_save_results, distributed_save_checkpoint,
    )
    from autoflowcfd.core.fr_solver.solver import _MACH_REF_FLOOR

    freestream = {"rho_inf": rho_inf, "vel_inf": vel_inf, "p_inf": p_inf}
    mach_ref = max(
        vel_inf / math.sqrt(max(1.4 * p_inf / max(rho_inf, 1e-10), 1e-10)),
        _MACH_REF_FLOOR,
    )
    package, root_context = distributed_mesh_load_v2(
        input_file, order, surface_mesh, n_ranks,
        freestream=freestream, mu_molecular=mu_molecular, mach_ref=mach_ref,
        enable_viscous=True, skip_quality_check=skip_quality_check,
        turb_model_name=turbulence_model.upper(),
        turbulence_intensity=turbulence_intensity, viscosity_ratio=viscosity_ratio,
        time_scheme=time_scheme, dual_time_inner_iter=dual_time_inner_iter,
    )
    solver = DistributedFRSolver.from_fully_distributed_package(
        package, n_ranks=n_ranks, root_context=root_context,
    )

    if is_root():
        print(f"[Distributed] Initialized with {n_ranks} ranks (fully-distributed mode)")
        print(f"[Distributed] {solver.partition.n_local_cells} local cells, "
              f"{solver.partition.n_halo} halo cells")

    if init_checkpoint:
        from autoflowcfd.core.mpi.distributed_checkpoint import (
            restore_distributed_state_from_checkpoint,
        )
        if is_root():
            print(f"\n🔄 从 checkpoint 加载稳态结果作为瞬态初场...")
        ckpt_iter = restore_distributed_state_from_checkpoint(init_checkpoint, solver)
        if is_root():
            print(f"   源 checkpoint 迭代数: {ckpt_iter}\n")

    def _checkpoint_cb(solver_ref, iteration):
        if iteration % checkpoint_interval != 0:
            return
        try:
            saved_path = distributed_save_checkpoint(
                solver_ref, output_dir, iteration, input_file,
                solver_ref.current_order, turbulence_model, backend,
                target_order=solver_ref.order,
            )
            if saved_path and is_root():
                print(f"   [Checkpoint] iter {iteration} saved: {saved_path}")
        except Exception as e:
            if is_root():
                print(f"   [Checkpoint] Warning: save failed at iter {iteration}: {e}")

    try:
        solver.solve(n_steps=max_iter, dt=dt, output_interval=checkpoint_interval,
                     checkpoint_callback=_checkpoint_cb,
                     phase_max_iter=phase_max_iter, residual_drop_threshold=residual_drop_threshold)
        if is_root():
            print(f"\n✅ Distributed transient simulation finished: iterations={max_iter}")

        distributed_save_results(solver, output_dir)
        distributed_save_checkpoint(
            solver, output_dir, max_iter, input_file,
            solver.current_order, turbulence_model, backend,
            target_order=solver.order,
        )
    except Exception as e:
        print(f"\n❌ Distributed transient simulation failed: {e}")
        raise click.Abort()


def _solve_transient_multi_gpu(
    input_file, order, time_scheme, dual_time_inner_iter,
    turbulence_model, max_iter, dt, output_dir, threads,
    turbulence_intensity, viscosity_ratio, mu_molecular, rho_inf, vel_inf, p_inf,
    n_ranks, gpu_device, surface_mesh, skip_quality_check, checkpoint_interval,
    phase_max_iter=None, residual_drop_threshold=100.0,
    init_checkpoint=None,
    fully_distributed=False,
):
    """多 GPU + MPI 分布式（与 `solve steady` 的 `--multi-gpu` 分支同一套
    构造方式）。`fully_distributed`（#1，2026-09-02 补齐）：走
    `MultiGPUDistributedSolver.from_fully_distributed_package`（只有
    root rank 加载完整网格），与 `--multi-gpu` 不加 `--fully-distributed`
    的"传统模式"（每个 rank 独立加载完整网格）二选一。"""
    from autoflowcfd.core.gpu import gpu_available
    if not gpu_available:
        print("\n❌ CuPy not available. Install with: pip install cupy-cuda12x")
        raise click.Abort()

    from autoflowcfd.core.mpi import mpi_available, is_root
    if not mpi_available:
        print("\n❌ MPI not available. Install mpi4py and run with mpirun.")
        raise click.Abort()

    from autoflowcfd.core.gpu.distributed.gpu_distributed import MultiGPUDistributedSolver

    time_scheme_str = {
        TimeIntegrationScheme.SSP_RK3: "ssp_rk3",
        TimeIntegrationScheme.IMEX_EULER: "imex_euler",
        TimeIntegrationScheme.DUAL_TIME: "dual_time",
    }.get(time_scheme, "ssp_rk3")

    if fully_distributed:
        import math
        from autoflowcfd.core.mpi.distributed_mesh_loader import distributed_mesh_load_v2
        from autoflowcfd.core.fr_solver.solver import _MACH_REF_FLOOR

        freestream = {"rho_inf": rho_inf, "vel_inf": vel_inf, "p_inf": p_inf}
        mach_ref = max(
            vel_inf / math.sqrt(max(1.4 * p_inf / max(rho_inf, 1e-10), 1e-10)),
            _MACH_REF_FLOOR,
        )
        package, root_context = distributed_mesh_load_v2(
            input_file, order, surface_mesh, n_ranks,
            freestream=freestream, mu_molecular=mu_molecular, mach_ref=mach_ref,
            enable_viscous=True, skip_quality_check=skip_quality_check,
            turb_model_name=turbulence_model.upper(),
            turbulence_intensity=turbulence_intensity, viscosity_ratio=viscosity_ratio,
            time_scheme=time_scheme, dual_time_inner_iter=dual_time_inner_iter,
        )
        solver = MultiGPUDistributedSolver.from_fully_distributed_package(
            package, n_ranks=n_ranks, device_id=gpu_device, root_context=root_context,
        )
    else:
        from autoflowcfd.cli.solve_mesh_loader import load_mesh_for_solver
        from autoflowcfd.fr.operators import generate_fr_operators

        mesh, volume_data = load_mesh_for_solver(
            input_file, order, surface_mesh=surface_mesh, skip_quality_check=skip_quality_check,
        )
        ops = generate_fr_operators(order)

        solver = MultiGPUDistributedSolver(
            mesh=mesh, ops=ops, n_ranks=n_ranks, device_id=gpu_device,
            mu_molecular=mu_molecular, rho_inf=rho_inf, vel_inf=vel_inf, p_inf=p_inf,
            turb_model=turbulence_model.upper(), time_scheme=time_scheme_str,
            turbulence_intensity=turbulence_intensity, viscosity_ratio=viscosity_ratio,
        )
    solver.time_integrator.dual_time_steps = dual_time_inner_iter

    if init_checkpoint:
        from autoflowcfd.core.mpi.distributed_checkpoint import (
            restore_distributed_state_from_checkpoint,
        )
        from autoflowcfd.core.gpu import get_cupy
        if is_root():
            print(f"\n🔄 从 checkpoint 加载稳态结果作为瞬态初场...")
        ckpt_iter = restore_distributed_state_from_checkpoint(init_checkpoint, solver)
        # `restore_distributed_state_from_checkpoint` 只写 `solver.
        # state.U`（numpy，与 CPU 路径共用的接口）——GPU 路径真正参与
        # 计算的是 `self.U_gpu`，必须显式同步，与 `load_checkpoint_
        # distributed`/`save_checkpoint_distributed` 同一个既有模式
        # （见 gpu_distributed_init.py 对应方法文档）。
        cp = get_cupy()
        n_local = solver.partition.n_local_cells
        with cp.cuda.Device(solver.device_id):
            solver.U_gpu = cp.asarray(solver.state.U[:n_local])
        if is_root():
            print(f"   源 checkpoint 迭代数: {ckpt_iter}\n")

    def _checkpoint_cb(solver_ref, iteration):
        if iteration % checkpoint_interval != 0:
            return
        try:
            # order/target_order 分离：见 solve_steady_command.py 的
            # _multi_gpu_checkpoint_cb 同一处修复文档。
            saved_path = solver_ref.save_checkpoint_distributed(
                output_dir, iteration, input_file,
                solver_ref.current_order, turbulence_model, backend="gpu",
                target_order=solver_ref.order,
            )
            if saved_path and is_root():
                print(f"   [Checkpoint] iter {iteration} saved: {saved_path}")
        except Exception as e:
            if is_root():
                print(f"   [Checkpoint] Warning: save failed at iter {iteration}: {e}")

    try:
        result = solver.solve(max_iter=max_iter, dt=dt, tol=0.0,
                              checkpoint_callback=_checkpoint_cb,
                              phase_max_iter=phase_max_iter,
                              residual_drop_threshold=residual_drop_threshold)
        if is_root():
            print(f"\n✅ Multi-GPU transient simulation finished: iterations={max_iter}, "
                  f"Residual={result['final_residual']:.6e}")
        saved_path = solver.save_checkpoint_distributed(
            output_dir, max_iter, input_file,
            solver.current_order, turbulence_model, backend="gpu",
            target_order=solver.order,
        )
        if saved_path and is_root():
            print(f"   Checkpoint saved: {saved_path}")
        solver.cleanup()
    except Exception as e:
        print(f"\n❌ Multi-GPU transient simulation failed: {e}")
        raise click.Abort()
