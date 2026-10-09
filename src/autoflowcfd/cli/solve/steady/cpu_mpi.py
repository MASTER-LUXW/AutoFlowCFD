"""AutoFlowCFD V2.0 - `solve steady` 的CPU MPI 分布式（传统模式 / 完全分布式加载）后端分支。

从 `cli/solve/steady.py` 拆出（2026-09-25，项目「单文件不超 500 行」规范）；参数由
`command.py` 在公共前段（物理常数解析与范围校验）之后逐名传入。
"""

import click

from autoflowcfd.cli.solve.distributed_checkpoint_io import distributed_periodic_checkpoint_callback
from autoflowcfd.cli.solve.wall_distance import wall_distance_source_if_needed
from autoflowcfd.core.time_integration.base import scheme_from_name
from autoflowcfd.cli.solve.mesh_loader import load_mesh_for_solver
from autoflowcfd.cli.solve.aero_coefficients import distributed_reference_area, report_distributed_aerodynamic_coefficients
from autoflowcfd.core.time_integration.base import STEADY_DT


def _run_cpu_mpi(
    *,
    aoa_deg, aos_deg, artificial_viscosity_alpha, artificial_viscosity_enabled, cfl_max, cfl_min, cfl_start, checkpoint_interval,
    fully_distributed, input_file, max_iter, mu_molecular, n_ranks, order, reference_area,
    output_dir, p_inf, phase_max_iter, residual_drop_threshold, rho_inf,
    skip_quality_check, surface_mesh, threads, time_scheme, tol, turbulence_intensity,
    turbulence_model, vel_inf, viscosity_ratio, sem_num_eddies,
):
    """`solve steady` 的CPU MPI 分布式（传统模式 / 完全分布式加载）路径。"""
    # 分布式求解器路径。
    #
    # CPU MPI "传统模式"：每个 rank 独立加载完整网格、进程内分区（root 用 METIS 分区后广播）。
    # 只有 root 持有完整网格的加载方式是 --fully-distributed（`distributed_mesh_load_v2`）。
    from autoflowcfd.core.mpi import mpi_available
    if not mpi_available:
        print("\n❌ MPI not available. Please install mpi4py and run with mpirun.")
        print("   pip install mpi4py")
        print("   mpirun -np {n_ranks} autoflowcfd solve steady ...")
        raise click.Abort()

    from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
    from autoflowcfd.fr.operators import generate_fr_operators

    if fully_distributed:
        # 完全分布式网格加载：只有 root 持有完整网格（`distributed_mesh_load_v2`）。全部湍流模型（与单机相同，
        # `core/turbulence/registry.py`）、全部时间格式、checkpoint 续算与 Order Continuation 均已接入
        volume_data = None          # 只在 root 的 `_root_context` 里
        from autoflowcfd.core.mpi.distributed_mesh_loader import distributed_mesh_load_v2
        from autoflowcfd.core.fr_solver.mach_ref import resolve_mach_ref

        freestream = {"rho_inf": rho_inf, "vel_inf": vel_inf,
                      "p_inf": p_inf,
                      "aoa_deg": aoa_deg, "aos_deg": aos_deg}
        # 与 FRSolver.__init__ 同一个公式（见该文件 mach_ref 计算处），
        # 不是本处新发明的近似——AUSM+up Weiss-Smith 预处理要求分区
        # 两侧用同一个真实值，公式本身也必须与单机路径逐字一致。
        mach_ref = resolve_mach_ref(rho_inf, vel_inf, p_inf)
        # root_context（2026-09-02，Order Continuation 支持）：只有
        # root rank 非 None，持有完整全局网格供后续阶数切换重新
        # 分发用，见 distributed_mesh_load_v2/redistribute_fully_
        # distributed_for_new_order 文档。
        package, root_context = distributed_mesh_load_v2(
            input_file, order, surface_mesh, n_ranks,
            freestream=freestream, mu_molecular=mu_molecular,
            mach_ref=mach_ref,
            enable_viscous=True, skip_quality_check=skip_quality_check,
            turb_model_name=turbulence_model.upper(),
            turbulence_intensity=turbulence_intensity, viscosity_ratio=viscosity_ratio, sem_num_eddies=sem_num_eddies,
            cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min,
            artificial_viscosity_enabled=artificial_viscosity_enabled,
            artificial_viscosity_alpha=artificial_viscosity_alpha,
            time_scheme=scheme_from_name(time_scheme),
        )
        solver = DistributedFRSolver.from_fully_distributed_package(
            package, n_ranks=n_ranks, root_context=root_context,
        )

        print(f"[Distributed] Initialized with {n_ranks} ranks (fully-distributed mode: "
              f"only root rank loaded the full mesh)")
        print(f"[Distributed] {solver.partition.n_local_cells} local cells, "
              f"{solver.partition.n_halo} halo cells")
    else:
        # 传统模式：每个 rank 独立加载完整网格（与单机/--multi-gpu 路径
        # 同一个 load_mesh_for_solver 入口）。
        mesh, volume_data = load_mesh_for_solver(
            input_file, order, surface_mesh=surface_mesh, skip_quality_check=skip_quality_check
        )
        ops = generate_fr_operators(order)

        # 创建分布式求解器（传入 face_connectivity 触发"传统模式"分区，
        # 见 DistributedFRSolver.__init__ 的"兼容旧接口"分支——本次修复
        # 之前这条分支就存在，只是 CLI 从未真正用过它，见上方说明）。
        solver = DistributedFRSolver(
            mesh=mesh,
            ops=ops,
            face_connectivity=mesh.face_connectivity,
            n_ranks=n_ranks,
            order=order,
            turb_model_name=turbulence_model,
            wall_distance_source=wall_distance_source_if_needed(turbulence_model, volume_data),
            time_scheme=scheme_from_name(time_scheme),
            n_threads=threads,
            turbulence_intensity=turbulence_intensity,
            viscosity_ratio=viscosity_ratio, sem_num_eddies=sem_num_eddies,
            mu_molecular=mu_molecular,
            rho_inf=rho_inf, vel_inf=vel_inf, p_inf=p_inf,
            aoa_deg=aoa_deg, aos_deg=aos_deg,
            # 见多 GPU 传统模式同一处注释：CFL 边界参数此前在分布式
            # 路径上被静默丢弃。DistributedFRSolver 从 solver_kwargs
            # 读这三个键（见其 _cfl_controller 构造处）。
            cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min,
            artificial_viscosity_enabled=artificial_viscosity_enabled,
            artificial_viscosity_alpha=artificial_viscosity_alpha,
        )

        # 初始化状态
        print(f"[Distributed] Initialized with {n_ranks} ranks")
        print(f"[Distributed] {solver.partition.n_local_cells} local cells, "
              f"{solver.partition.n_halo} halo cells")
        print("[Distributed] 传统模式：每个 rank 都加载了完整网格（内存非最优；大网格请用 --fully-distributed）")

    # 中间 checkpoint 保存回调（2026-09-02 补齐——此前只在 solve()
    # 返回之后保存一次最终 checkpoint，跑到一半崩溃/被杀会丢失全部
    # 进度，且没有任何"从分布式 checkpoint 继续跑"的机制，与单机
    # 路径 `_checkpoint_cb` 同一个设计，见 `DistributedFRSolver.
    # solve`/`solve_commands.py::resume` 的 `--n-ranks`/`--multi-gpu`
    # 分支说明）。
    from autoflowcfd.core.mpi.distributed_checkpoint import (
        distributed_save_results,
    )

    _distributed_checkpoint_cb = distributed_periodic_checkpoint_callback(
        checkpoint_interval, output_dir, input_file, turbulence_model, surface_mesh=surface_mesh)

    # 执行分布式求解
    try:
        result = solver.solve(
            max_iter=max_iter, dt=STEADY_DT, tol=tol,
            checkpoint_callback=_distributed_checkpoint_cb,
            phase_max_iter=phase_max_iter, residual_drop_threshold=residual_drop_threshold,
        )
        print(f"\n✅ Distributed Simulation Finished: Iterations={result.iterations}, "
              f"Residual={result.final_residual:.6e}")

        # 保存结果（分布式版本：root 收集全局数据后保存）
        distributed_save_results(solver, output_dir)
        solver.save_checkpoint_distributed(
            output_dir, result.iterations, input_file, solver.current_order, turbulence_model,
            target_order=solver.order, surface_mesh=surface_mesh,
        )
        report_distributed_aerodynamic_coefficients(
            solver, distributed_reference_area(solver, volume_data, reference_area))

    except Exception as e:
        print(f"\n❌ Distributed Simulation Failed: {str(e)}")
        raise click.Abort()
