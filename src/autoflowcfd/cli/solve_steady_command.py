"""`solve steady` 命令 —— 从 solve_steady_commands.py 拆出，控制单文件行数。

见 solve_steady_commands.py 文档说明整体拆分结构。
"""

import click

from autoflowcfd.core import FRSolver
from autoflowcfd.core.time_integration.base import TimeIntegrationScheme
from autoflowcfd.cli.solve_helpers import (
    compute_wall_distance_for_solver,
    load_mesh_for_solver,
    save_results,
    write_checkpoint,
    load_physical_config_if_given,
    resolve_physical_constants,
    resolve_turbulence_model,
)
from autoflowcfd.cli.solve_aero_coefficients import (
    _compute_reference_area_auto,
    _report_aerodynamic_coefficients,
)
from autoflowcfd.cli.solve_commands import solve


@solve.command(name='steady')
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--backend', type=click.Choice(['cpu', 'gpu']), default='cpu', help='计算后端 (CPU/GPU)')
@click.option('--order', type=int, default=2, help='FR 多项式阶数 (P1/P2/P3)')
@click.option('--flux-type', type=click.Choice(['radau', 'gauss']), default='radau',
              help="FR 修正函数族（#14）：'radau'（默认，此前唯一使用过的方案，"
                   "Huynh 记法 g_DG）；'gauss' 是与 Spectral Difference 等价的新方案"
                   "（见 fr/matrix_operators.py 文档）。目前仅单机 CPU 路径支持，"
                   "GPU/多 GPU/MPI 分布式路径传非默认值会报错而不是静默忽略")
@click.option('--turbulence-model', type=click.Choice(['none', 'sst', 'ddes', 'wmles']), default='sst', help='湍流模型')
@click.option('--max-iter', type=int, default=1000, help='最大迭代次数')
@click.option('--output', '-o', 'output_dir', type=click.Path(), default='./results', help='结果输出目录')
@click.option('--checkpoint-interval', type=int, default=100, help='检查点保存间隔')
@click.option('--use-eikonal', is_flag=True, help='使用 Eikonal 方程求解壁面距离（更精确但较慢）')
@click.option('--surface-mesh', '-s', type=click.Path(exists=True), default=None,
              help='原始面网格路径 - input_file 是 .nas 体网格时必填，用于反推边界分组；input_file 是 .pkl 时不需要')
@click.option('--skip-quality-check', is_flag=True, help='跳过求解前的网格质量门检查（不建议，仅用于临时诊断）')
@click.option('--reference-area', type=float, default=None, help='气动系数参考面积 (m^2)，提供时求解结束后打印 Cd/Cl')
@click.option('--threads', '-j', type=int, default=-1, help='CPU 后端 numba 并行 kernel 使用的线程数，默认 -1 = 4（本机真实网格实测扩展性甜点，不是核数）')
@click.option('--n-ranks', '--np', type=int, default=1, help='MPI 并行 rank 数（域分解并行，需配合 mpirun 使用。默认 1 = 单机模式）')
@click.option('--gpu-device', type=int, default=0, help='GPU 设备 ID（默认 0，多 GPU 时每个 rank 自动分配）')
@click.option('--multi-gpu', is_flag=True, help='启用多 GPU + MPI 分布式求解（每个 rank 使用一块 GPU）')
@click.option('--turbulence-intensity', type=float, default=0.01,
              help='来流湍流强度 Tu（默认 0.01=1%%），外部气动 ≤1%%，城市道路 3-5%%。'
                   '也驱动 --turbulence-model ddes 时 BD-02 SEM 入口的目标雷诺应力')
@click.option('--viscosity-ratio', type=float, default=5.0, help='来流粘性比 VR=nu_t/nu（默认 5.0），外部气动推荐 2-10')
@click.option('--sem-num-eddies', type=int, default=200,
              help='--turbulence-model ddes 时 BD-02 合成湍流入口 (SEM) 的涡核数量（默认 200）')
@click.option('--mu-molecular', type=float, default=1.8e-5, help='分子动力粘度 (Pa*s)，默认 1.8e-5（标准状态下空气），非标准工况请覆盖')
@click.option('--rho-inf', type=float, default=1.225, help='自由流密度 (kg/m^3)，默认 1.225（标准海平面空气）')
@click.option('--vel-inf', type=float, default=33.33, help='自由流速度大小 (m/s)，默认 33.33')
@click.option('--p-inf', type=float, default=101325.0, help='自由流静压 (Pa)，默认 101325.0（标准大气压）')
@click.option('--config', 'config_path', type=click.Path(exists=True), default=None,
              help='从 YAML 文件读取物理常量默认值（mu_molecular/rho_inf/vel_inf/p_inf/'
                   'turbulence_intensity/viscosity_ratio）；显式传入的同名 --xxx 选项优先于此文件')
def solve_steady(input_file, backend, order, flux_type, turbulence_model, max_iter, output_dir, checkpoint_interval, use_eikonal, surface_mesh, skip_quality_check, reference_area, threads, n_ranks, gpu_device, multi_gpu, turbulence_intensity, viscosity_ratio, sem_num_eddies, mu_molecular, rho_inf, vel_inf, p_inf, config_path):
    """执行稳态 FR 求解。

    支持高阶精度 (P1-P4) 和多种湍流模型 (SST, DDES, WMLES)。
    输入文件必须是体网格 - .pkl（`grid generate-volume`/`grid
    import-volume` 的输出）或 .nas 体网格（需要配合 --surface-mesh 反推边界
    分组）。求解前会强制检查网格质量门，除非传了 --skip-quality-check。
    """
    print(f"=== Starting Steady FR Simulation ===")

    # 物理常量解析：显式 CLI 选项 > --config YAML > 上面 click 声明的内建默认值。
    # 不能硬编码——mu_molecular/rho_inf/vel_inf/p_inf 等基础物理量必须能被
    # 用户为非标准工况（不同流体/温度/高度）覆盖，见 solve_physical_constants.py 文档。
    _phys_cfg = load_physical_config_if_given(config_path)
    _resolved = resolve_physical_constants(
        click.get_current_context(),
        {
            'turbulence_intensity': turbulence_intensity, 'viscosity_ratio': viscosity_ratio,
            'mu_molecular': mu_molecular, 'rho_inf': rho_inf, 'vel_inf': vel_inf, 'p_inf': p_inf,
            'order': order, 'max_iter': max_iter,
        },
        _phys_cfg,
    )
    turbulence_intensity = _resolved['turbulence_intensity']
    viscosity_ratio = _resolved['viscosity_ratio']
    mu_molecular = _resolved['mu_molecular']
    rho_inf = _resolved['rho_inf']
    vel_inf = _resolved['vel_inf']
    p_inf = _resolved['p_inf']
    order = _resolved['order']
    max_iter = _resolved['max_iter']
    # turbulence_model：CLI 字符串词汇与 SteadyConfig.turbulence 的枚举
    # 命名不完全一致，需要专门的映射，不能靠 resolve_physical_constants
    # 的同名 getattr（见 resolve_turbulence_model 文档）。
    turbulence_model = resolve_turbulence_model(click.get_current_context(), turbulence_model, _phys_cfg)

    # Tu/VR/mu_molecular/rho_inf/vel_inf/p_inf 范围校验：CLI 路径不构造
    # SolverConfig，其 __post_init__ 的校验到不了这里，必须在入口拦截
    # （2026-08-25 代码审查；mu_molecular/rho_inf/vel_inf/p_inf 是本轮新增）。
    if not (0.0 < turbulence_intensity <= 1.0):
        raise click.BadParameter("湍流强度 Tu 必须在 (0, 1] 区间", param_hint="--turbulence-intensity")
    if viscosity_ratio <= 0.0:
        raise click.BadParameter("粘性比 VR 必须 > 0", param_hint="--viscosity-ratio")
    if mu_molecular <= 0.0:
        raise click.BadParameter("分子动力粘度必须 > 0", param_hint="--mu-molecular")
    if rho_inf <= 0.0:
        raise click.BadParameter("自由流密度必须 > 0", param_hint="--rho-inf")
    if vel_inf <= 0.0:
        raise click.BadParameter("自由流速度必须 > 0", param_hint="--vel-inf")
    if p_inf <= 0.0:
        raise click.BadParameter("自由流静压必须 > 0", param_hint="--p-inf")
    # #14：GPU/多GPU/MPI 分布式路径的 ops 构造未接入 flux_type（那些路径
    # 直接调用 generate_fr_operators(order) 不带 flux_point_type，见下方
    # 各分支），非默认值在这些路径上会被静默忽略——宁可显式报错，不静默
    # 退回默认方案（与本项目其余地方"不允许静默降级"的一贯原则一致）。
    if flux_type != 'radau' and (backend == 'gpu' or n_ranks > 1):
        raise click.BadParameter(
            "--flux-type gauss 目前只有单机 CPU 路径支持（GPU/多GPU/MPI 分布式"
            "路径的算子构造尚未接入这个选择）。请去掉 --backend gpu/--multi-gpu/"
            "--n-ranks，或使用默认的 --flux-type radau。",
            param_hint="--flux-type",
        )
    print(f"\nInput Grid : {input_file}")
    print(f"Backend    : {backend} | Order: P{order} | Method: rk3")
    print(f"Turbulence : {turbulence_model} | Max Iter: {max_iter}")
    if n_ranks > 1:
        print(f"MPI Ranks  : {n_ranks} (domain decomposition)")
    if use_eikonal:
        print(f"Wall Dist : Eikonal (graph-Dijkstra approx)\n")
    else:
        print(f"Wall Dist : KD-Tree (Geometric)\n")

    # 1. 网格加载与处理
    if backend == 'gpu' and multi_gpu and n_ranks > 1:
        # 多 GPU + MPI 分布式路径
        from autoflowcfd.core.gpu import gpu_available
        if not gpu_available:
            print("\n❌ CuPy not available. Install with: pip install cupy-cuda12x")
            raise click.Abort()

        from autoflowcfd.core.mpi import mpi_available, is_root
        if not mpi_available:
            print("\n❌ MPI not available. Install mpi4py and run with mpirun.")
            raise click.Abort()

        from autoflowcfd.core.gpu.distributed.gpu_distributed import MultiGPUDistributedSolver
        from autoflowcfd.fr.operators import generate_fr_operators

        mesh, volume_data = load_mesh_for_solver(
            input_file, order, surface_mesh=surface_mesh, skip_quality_check=skip_quality_check
        )
        ops = generate_fr_operators(order)

        solver = MultiGPUDistributedSolver(
            mesh=mesh, ops=ops, n_ranks=n_ranks,
            device_id=gpu_device,
            mu_molecular=mu_molecular,
            rho_inf=rho_inf, vel_inf=vel_inf, p_inf=p_inf,
            # 真实 bug 修复（V2.0 专家组盲审发现）：此前从不传 turb_model，
            # --turbulence-model 无论填什么都被静默丢弃、恒定跑层流，
            # 终端打印的 Turbulence 行却仍显示用户输入的模型名。
            turb_model=turbulence_model.upper(),
        )

        try:
            result = solver.solve(max_iter=max_iter, dt=1e-3, tol=1e-6)
            print(f"\n✅ Multi-GPU Simulation Finished")
            # #4（2026-08-28）：此前这里从不保存结果——分布式 checkpoint
            # save/load 依赖的 self.U_gpu 本地尺寸缺陷（#1）修复之前，
            # 保存也没有意义，见 MultiGPUDistributedSolver.
            # save_checkpoint_distributed 文档。
            saved_path = solver.save_checkpoint_distributed(
                output_dir, max_iter, input_file, order, turbulence_model, backend="gpu",
            )
            if saved_path and is_root():
                print(f"   Checkpoint saved: {saved_path}")
            solver.cleanup()
        except Exception as e:
            print(f"\n❌ Multi-GPU Simulation Failed: {e}")
            raise click.Abort()

    elif backend == 'gpu' and not multi_gpu:
        # 单 GPU 路径
        from autoflowcfd.core.gpu import gpu_available
        if not gpu_available:
            print("\n❌ CuPy not available. Install with: pip install cupy-cuda12x")
            raise click.Abort()

        from autoflowcfd.core.gpu.solver.gpu_solver import GPUFRSolver
        from autoflowcfd.fr.operators import generate_fr_operators

        mesh, volume_data = load_mesh_for_solver(
            input_file, order, surface_mesh=surface_mesh, skip_quality_check=skip_quality_check
        )
        ops = generate_fr_operators(order)

        solver = GPUFRSolver(
            mesh=mesh, ops=ops, order=order,
            device_id=gpu_device,
            turbulence_intensity=turbulence_intensity,
            viscosity_ratio=viscosity_ratio,
            mu_molecular=mu_molecular,
            rho_inf=rho_inf, vel_inf=vel_inf, p_inf=p_inf,
            # 真实 bug 修复（V2.0 专家组盲审发现）：此前从不传 turb_model，
            # --turbulence-model 无论填什么都被静默丢弃、恒定跑层流，
            # 终端打印的 Turbulence 行却仍显示用户输入的模型名。
            turb_model=turbulence_model.upper(),
        )

        try:
            result = solver.solve(max_iter=max_iter, dt=1e-3, tol=1e-6)
            print(f"\n✅ GPU Simulation Finished: Iterations={result['iterations']}")

            # 保存结果
            state_cpu = solver.get_state_cpu()
            import pickle, os
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, 'final_state.pkl'), 'wb') as f:
                pickle.dump(state_cpu, f)
            solver.cleanup()

        except Exception as e:
            print(f"\n❌ GPU Simulation Failed: {e}")
            raise click.Abort()

    elif n_ranks > 1:
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
        from autoflowcfd.core.mpi import mpi_available
        if not mpi_available:
            print("\n❌ MPI not available. Please install mpi4py and run with mpirun.")
            print("   pip install mpi4py")
            print("   mpirun -np {n_ranks} autoflowcfd solve steady ...")
            raise click.Abort()

        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
        from autoflowcfd.fr.operators import generate_fr_operators

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
            time_scheme=TimeIntegrationScheme.SSP_RK3,
            n_threads=threads,
            turbulence_intensity=turbulence_intensity,
            viscosity_ratio=viscosity_ratio,
            mu_molecular=mu_molecular,
            rho_inf=rho_inf, vel_inf=vel_inf, p_inf=p_inf,
        )

        # 初始化状态
        print(f"[Distributed] Initialized with {n_ranks} ranks")
        print(f"[Distributed] {solver.partition.n_local_cells} local cells, "
              f"{solver.partition.n_halo} halo cells")
        print(f"[Distributed] Traditional mode: every rank loaded the full mesh "
              f"(not memory-optimal, see #2 fix notes)")

        # 执行分布式求解
        try:
            solver.solve(n_steps=max_iter, dt=1e-3, output_interval=checkpoint_interval)
            print(f"\n✅ Distributed Simulation Finished")

            # 保存结果（分布式版本：root 收集全局数据后保存）
            from autoflowcfd.core.mpi.distributed_checkpoint import (
                distributed_save_results,
                distributed_save_checkpoint,
            )
            distributed_save_results(solver, output_dir)
            distributed_save_checkpoint(
                solver, output_dir, max_iter,
                input_file, order, turbulence_model, backend,
            )

        except Exception as e:
            print(f"\n❌ Distributed Simulation Failed: {str(e)}")
            raise click.Abort()

    else:
        # 单机求解器路径：所有 rank 加载完整网格
        mesh, volume_data = load_mesh_for_solver(
            input_file, order, surface_mesh=surface_mesh, skip_quality_check=skip_quality_check
        )
        # 单机求解器路径（默认）
        solver = FRSolver(
            mesh=mesh,
            backend=backend,
            order=order,
            turb_model_name=turbulence_model,
            time_scheme=TimeIntegrationScheme.SSP_RK3,
            n_threads=threads,
            turbulence_intensity=turbulence_intensity,
            viscosity_ratio=viscosity_ratio,
            sem_num_eddies=sem_num_eddies,
            mu_molecular=mu_molecular,
            rho_inf=rho_inf, vel_inf=vel_inf, p_inf=p_inf,
            flux_type=flux_type,
        )

        # 2.5. 计算壁面距离场（如果湍流模型需要）
        compute_wall_distance_for_solver(solver, volume_data, use_eikonal=use_eikonal)

        # 传递参考面积到求解器，供迭代中输出气动力系数
        # 如果未指定 --reference-area，尝试从面网格自动计算投影面积
        if reference_area is None:
            auto_ref_area = _compute_reference_area_auto(volume_data)
            if auto_ref_area is not None:
                reference_area = auto_ref_area
        solver._reference_area = reference_area

        # 构建中间 checkpoint 保存回调（每 checkpoint_interval 步保存一次）
        def _checkpoint_cb(solver_ref, iteration):
            if iteration % checkpoint_interval != 0:
                return
            try:
                save_results(solver_ref, output_dir, quiet=True)
                # 必须用 solver_ref.current_order（这一步实际求解用的阶数），
                # 不能用外层闭包捕获的 order（CLI --order，Order Continuation
                # 的最终目标阶数）——真实复现：Order Continuation 还没爬升到
                # 目标阶数时（例如 P0 阶段的中间 checkpoint）两者不相等，用
                # 目标阶数重建 mesh/FRSolver 会得到与 checkpoint 里存的
                # U_sps 形状不匹配的 n_sps，resume 直接报错拒绝恢复。
                write_checkpoint(
                    solver_ref, output_dir, iteration,
                    input_file, solver_ref.current_order, turbulence_model, backend,
                    quiet=True, surface_mesh=surface_mesh, target_order=solver_ref.order,
                )
                print(f"   [Checkpoint] iter {iteration} saved")
            except Exception as e:
                print(f"   [Checkpoint] Warning: save failed at iter {iteration}: {e}")

        # 3. 执行求解
        try:
            result = solver.solve(max_iter=max_iter, dt=1e-3, tol=1e-6,
                                  checkpoint_callback=_checkpoint_cb)
            print(f"\n✅ Simulation Finished: Iterations={result.iterations}, Residual={result.final_residual:.6e}")

            # 4. 保存结果（.pkl 全量状态 + HDF5 checkpoint，后者供 solve resume 使用）
            save_results(solver, output_dir)
            # solver.current_order 而非 order：理由同上方 _checkpoint_cb 里的
            # 说明。正常跑完的情况下 Order Continuation 应该已经爬升到目标
            # 阶数、二者相等，但读活的值而不是假设闭包变量仍然成立更稳妥。
            write_checkpoint(
                solver, output_dir, result.iterations, input_file, solver.current_order,
                turbulence_model, backend, surface_mesh=surface_mesh, target_order=solver.order,
            )

            # 5. 气动系数（提供 --reference-area 时）
            _report_aerodynamic_coefficients(solver, reference_area)

        except Exception as e:
            print(f"\n❌ Simulation Failed: {str(e)}")
            raise click.Abort()
