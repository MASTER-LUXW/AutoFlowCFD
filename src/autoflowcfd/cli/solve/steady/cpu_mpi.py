"""AutoFlowCFD V2.0 - `solve steady` 的CPU MPI 分布式（传统模式 / 完全分布式加载）后端分支。

从 `cli/solve/steady.py` 拆出（2026-09-25，项目「单文件不超 500 行」规范）；参数由
`command.py` 在公共前段（物理常数解析与范围校验）之后逐名传入。
"""

import click

from autoflowcfd.cli.solve.wall_distance import wall_distance_source_if_needed
from autoflowcfd.core.time_integration.base import scheme_from_name
from autoflowcfd.cli.solve.helpers import load_mesh_for_solver


def _run_cpu_mpi(
    *,
    use_eikonal,
    aoa_deg, aos_deg, backend, cfl_max, cfl_min, cfl_start, checkpoint_interval,
    fully_distributed, input_file, max_iter, mu_molecular, n_ranks, order,
    output_dir, p_inf, phase_max_iter, residual_drop_threshold, rho_inf,
    skip_quality_check, surface_mesh, threads, time_scheme, turbulence_intensity,
    turbulence_model, vel_inf, viscosity_ratio,
):
    """`solve steady` 的CPU MPI 分布式（传统模式 / 完全分布式加载）路径。"""
    # 分布式求解器路径。
    #
    # #2（V2.0 专家组盲审第4轮，2026-08-28）：此前这里唯一的路径是
    # "完全分布式网格加载"（只有 root 加载完整网格，通过
    # distributed_mesh_load 分发局部数据）——但 distributed_mesh_load
    # 产出的 local_mesh 缺少 face_connectivity/face_flux_points（见
    # distributed_mesh_loader.py::build_local_mesh_from_data 文档），
    # DistributedFRSolver.__init__ 内部 build_distributed_flat_face
    # 需要真实的全局面几何才能构建分布式面几何，这条路径构造期必然
    # 失败——这是一个更深的架构缺口（真正做到"只有 root 持有完整
    # 网格"需要 root 逐 rank 预构建+分发压缩几何数据，工作量与 GPU
    # 侧 #1 修复相当，未在本次修复范围内）。
    #
    # 改为"传统模式"（与 --multi-gpu 已经在用的模式一致，见
    # solve_steady_command.py 的 --multi-gpu 分支）：每个 rank 独立
    # 加载完整网格，进程内分区——不是内存最优，但 mesh.face_
    # connectivity 是真实、完整的，build_distributed_flat_face 能
    # 正确工作，DistributedMeshAdapter/distributed_compute_*_residual
    # 的 local+halo 压缩索引空间重排（#2 修复）才有意义。
    from autoflowcfd.core.mpi import mpi_available, is_root
    if not mpi_available:
        print("\n❌ MPI not available. Please install mpi4py and run with mpirun.")
        print("   pip install mpi4py")
        print("   mpirun -np {n_ranks} autoflowcfd solve steady ...")
        raise click.Abort()

    from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
    from autoflowcfd.fr.operators import generate_fr_operators

    if fully_distributed:
        # 真正的完全分布式网格加载（2026-09-02 实现，同日续接补齐
        # SST/DDES/IDDES/WMLES/LES——见 distributed_mesh_loader.py::
        # distributed_mesh_load_v2/DistributedFRSolver.from_fully_
        # distributed_package 文档"范围边界"一节）。此前这里的
        # 硬编码 `!= 'NONE'` 拒绝早于 SST 支持接入、且从未跟随后续
        # 批次同步更新，是过时的信息源——真实 bug：既拒绝了后端已经
        # 支持的全部湍流模型，也从未把 `turbulence_model` 传给
        # `distributed_mesh_load_v2`（该函数签名里 `turb_model_name`
        # 参数一直被忽略，恒用默认值 'NONE'）。
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
            use_eikonal=use_eikonal,
            turbulence_intensity=turbulence_intensity, viscosity_ratio=viscosity_ratio,
            cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min,
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
            backend=backend,
            order=order,
            turb_model_name=turbulence_model,
            wall_distance_source=wall_distance_source_if_needed(turbulence_model, volume_data, use_eikonal),
            time_scheme=scheme_from_name(time_scheme),
            n_threads=threads,
            turbulence_intensity=turbulence_intensity,
            viscosity_ratio=viscosity_ratio,
            mu_molecular=mu_molecular,
            rho_inf=rho_inf, vel_inf=vel_inf, p_inf=p_inf,
            aoa_deg=aoa_deg, aos_deg=aos_deg,
            # 见多 GPU 传统模式同一处注释：CFL 边界参数此前在分布式
            # 路径上被静默丢弃。DistributedFRSolver 从 solver_kwargs
            # 读这三个键（见其 _cfl_controller 构造处）。
            cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min,
        )

        # 初始化状态
        print(f"[Distributed] Initialized with {n_ranks} ranks")
        print(f"[Distributed] {solver.partition.n_local_cells} local cells, "
              f"{solver.partition.n_halo} halo cells")
        print(f"[Distributed] Traditional mode: every rank loaded the full mesh "
              f"(not memory-optimal, see #2 fix notes; use --fully-distributed "
              f"for the memory-optimal path)")

    # 中间 checkpoint 保存回调（2026-09-02 补齐——此前只在 solve()
    # 返回之后保存一次最终 checkpoint，跑到一半崩溃/被杀会丢失全部
    # 进度，且没有任何"从分布式 checkpoint 继续跑"的机制，与单机
    # 路径 `_checkpoint_cb` 同一个设计，见 `DistributedFRSolver.
    # solve`/`solve_commands.py::resume` 的 `--n-ranks`/`--multi-gpu`
    # 分支说明）。
    from autoflowcfd.core.mpi.distributed_checkpoint import (
        distributed_save_results,
        distributed_save_checkpoint,
    )

    def _distributed_checkpoint_cb(solver_ref, iteration):
        if iteration % checkpoint_interval != 0:
            return
        try:
            # order 传 solver_ref.current_order（不是固定的目标
            # order）：Order Continuation 接入分布式路径后
            # （2026-09-02），爬坡阶段中途保存的 checkpoint 的
            # `U_sps` 形状对应的是当时的 current_order，不是最终
            # 目标阶数，见 distributed_save_checkpoint 同名参数
            # 文档。target_order 记录真正的目标，供 resume 时继续
            # 爬坡。
            saved_path = distributed_save_checkpoint(
                solver_ref, output_dir, iteration,
                input_file, solver_ref.current_order, turbulence_model, backend,
                target_order=solver_ref.order, surface_mesh=surface_mesh,
            )
            if saved_path and is_root():
                print(f"   [Checkpoint] iter {iteration} saved: {saved_path}")
        except Exception as e:
            if is_root():
                print(f"   [Checkpoint] Warning: save failed at iter {iteration}: {e}")

    # 执行分布式求解
    try:
        result = solver.solve(
            n_steps=max_iter, dt=1e-3, output_interval=checkpoint_interval,
            checkpoint_callback=_distributed_checkpoint_cb,
            phase_max_iter=phase_max_iter, residual_drop_threshold=residual_drop_threshold,
        )
        print(f"\n✅ Distributed Simulation Finished: Iterations={result.iterations}, "
              f"Residual={result.final_residual:.6e}")

        # 保存结果（分布式版本：root 收集全局数据后保存）
        distributed_save_results(solver, output_dir)
        distributed_save_checkpoint(
            solver, output_dir, result.iterations,
            input_file, solver.current_order, turbulence_model, backend,
            target_order=solver.order, surface_mesh=surface_mesh,
        )

    except Exception as e:
        print(f"\n❌ Distributed Simulation Failed: {str(e)}")
        raise click.Abort()
