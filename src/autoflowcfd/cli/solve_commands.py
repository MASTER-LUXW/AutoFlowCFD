"""V2.0 FR 求解器子命令。

本模块提供 V2.0 FR 求解器的 CLI 命令，支持高阶精度、多种时间推进方法和湍流模型。

命令:
    - steady: 运行稳态 FR 仿真
    - transient: 运行瞬态 FR 仿真（专用命令）
    - resume: 从检查点恢复
    - status: 查看求解器状态

示例:
    $ autoflowcfd solve steady model_volume.pkl --backend cpu --order 2 --turbulence-model sst

网格加载/壁面距离场/结果保存三个共享辅助函数在 solve_helpers.py，本文件只保留
steady/transient/resume/status 四个 click 命令本体。
"""

import logging
from pathlib import Path
from typing import Optional

import click

from autoflowcfd.core import FRSolver
from autoflowcfd.core.time_integration import TimeIntegrationScheme
from autoflowcfd.cli.solve_helpers import (
    compute_wall_distance_for_solver,
    load_mesh_for_solver,
    restore_state_from_checkpoint,
    save_results,
    write_checkpoint,
)

logger = logging.getLogger(__name__)


@click.group()
def solve():
    """FR 求解器相关命令 (稳态/瞬态)。"""
    pass


def _report_aerodynamic_coefficients(solver, reference_area: Optional[float]) -> None:
    """求解结束后直接在当前 FRSolver 状态上积分并打印 Cd/Cl。

    不经过 checkpoint/post 命令组的往返（那条路径此前完全打不通，见
    postprocess/fr_coefficients.py 模块文档），直接用求解器仍在内存里的
    mesh+state 计算——这是让 CLI 验收标准"Cd/Cl 非零、符号正确"能够
    兑现的最短路径。reference_area 未提供时跳过（不猜一个可能误导的
    默认值）。
    """
    if reference_area is None or reference_area <= 0:
        print("\n(未提供 --reference-area，跳过气动系数计算；如需 Cd/Cl 请指定参考面积)")
        return
    try:
        from autoflowcfd.postprocess.fr_coefficients import compute_aerodynamic_coefficients_fr

        coeffs = compute_aerodynamic_coefficients_fr(solver, reference_area=reference_area)
        print(f"\n=== Aerodynamic Coefficients (reference_area={reference_area} m^2) ===")
        print(f"   Cd (drag) = {coeffs.Cd:.6f}")
        print(f"   Cl (lift) = {coeffs.Cl:.6f}")
        print(f"   Cs (side) = {coeffs.Cs:.6f}")
    except Exception as e:
        print(f"\n⚠️  Aerodynamic coefficient calculation failed: {e}")


@solve.command(name='steady')
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--backend', type=click.Choice(['cpu', 'gpu']), default='cpu', help='计算后端 (CPU/GPU)')
@click.option('--order', type=int, default=2, help='FR 多项式阶数 (P1/P2/P3)')
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
def solve_steady(input_file, backend, order, turbulence_model, max_iter, output_dir, checkpoint_interval, use_eikonal, surface_mesh, skip_quality_check, reference_area, threads, n_ranks, gpu_device, multi_gpu):
    """
    执行稳态 FR 求解。

    支持高阶精度 (P1-P4) 和多种湍流模型 (SST, DDES, WMLES)。
    输入文件必须是体网格 - .pkl（`grid generate-volume`/`grid
    import-volume` 的输出）或 .nas 体网格（需要配合 --surface-mesh 反推边界
    分组）。求解前会强制检查网格质量门，除非传了 --skip-quality-check。
    """
    print(f"=== Starting Steady FR Simulation ===")
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

        from autoflowcfd.core.mpi import mpi_available
        if not mpi_available:
            print("\n❌ MPI not available. Install mpi4py and run with mpirun.")
            raise click.Abort()

        from autoflowcfd.core.gpu.gpu_distributed import MultiGPUDistributedSolver
        from autoflowcfd.core.operators import FROperators

        mesh, volume_data = load_mesh_for_solver(
            input_file, order, surface_mesh=surface_mesh, skip_quality_check=skip_quality_check
        )
        ops = FROperators(order=order, n_points_1d=mesh.n_points_1d)

        solver = MultiGPUDistributedSolver(
            mesh=mesh, ops=ops, n_ranks=n_ranks,
            device_id=gpu_device,
            mu_molecular=1.8e-5,
        )

        try:
            result = solver.solve(max_iter=max_iter, dt=1e-3, tol=1e-6)
            print(f"\n✅ Multi-GPU Simulation Finished")
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

        from autoflowcfd.core.gpu.gpu_solver import GPUFRSolver
        from autoflowcfd.core.operators import FROperators

        mesh, volume_data = load_mesh_for_solver(
            input_file, order, surface_mesh=surface_mesh, skip_quality_check=skip_quality_check
        )
        ops = FROperators(order=order, n_points_1d=mesh.n_points_1d)

        solver = GPUFRSolver(
            mesh=mesh, ops=ops, order=order,
            device_id=gpu_device,
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
        # 分布式求解器路径：完全分布式网格加载（只有 root 加载完整网格）
        from autoflowcfd.core.mpi import mpi_available
        if not mpi_available:
            print("\n❌ MPI not available. Please install mpi4py and run with mpirun.")
            print("   pip install mpi4py")
            print("   mpirun -np {n_ranks} autoflowcfd solve steady ...")
            raise click.Abort()

        from autoflowcfd.core.mpi.distributed_mesh_loader import distributed_mesh_load
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
        from autoflowcfd.core.operators import FROperators

        # 完全分布式网格加载：root 加载 + 分区 + 分发
        local_mesh, local_fc_data, partition_info = distributed_mesh_load(
            input_file, order, surface_mesh, n_ranks, skip_quality_check
        )

        # 构建算子
        ops = FROperators(order=order, n_points_1d=local_mesh.n_points_1d)

        # 创建分布式求解器（使用局部网格）
        solver = DistributedFRSolver(
            mesh=local_mesh,
            ops=ops,
            face_connectivity_data=local_fc_data,
            partition_info=partition_info,
            n_ranks=n_ranks,
            backend=backend,
            order=order,
            turb_model_name=turbulence_model,
            time_scheme=TimeIntegrationScheme.SSP_RK3,
            n_threads=threads,
        )

        # 初始化状态
        print(f"[Distributed] Initialized with {n_ranks} ranks")
        print(f"[Distributed] {solver.partition.n_local_cells} local cells, "
              f"{solver.partition.n_halo} halo cells")
        print(f"[Distributed] Memory optimized: only root loaded full mesh")

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
        )

        # 2.5. 计算壁面距离场（如果湍流模型需要）
        compute_wall_distance_for_solver(solver, volume_data, use_eikonal=use_eikonal)

        # 3. 执行求解
        try:
            result = solver.solve(max_iter=max_iter, dt=1e-3, tol=1e-6)
            print(f"\n✅ Simulation Finished: Iterations={result.iterations}, Residual={result.final_residual:.6e}")

            # 4. 保存结果（.pkl 全量状态 + HDF5 checkpoint，后者供 solve resume 使用）
            save_results(solver, output_dir)
            write_checkpoint(
                solver, output_dir, result.iterations, input_file, order, turbulence_model, backend
            )

            # 5. 气动系数（提供 --reference-area 时）
            _report_aerodynamic_coefficients(solver, reference_area)

        except Exception as e:
            print(f"\n❌ Simulation Failed: {str(e)}")
            raise click.Abort()


@solve.command(name='transient')
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--backend", "-b", type=click.Choice(["cpu", "gpu"]),
              default="cpu", help="计算后端")
@click.option("--order", "-p", type=click.IntRange(1, 3), default=2,
              help="FR 离散阶数")
@click.option("--time-method", "-t",
              type=click.Choice(["rk3", "imex", "dual-time"]),
              default="rk3", help="时间推进方法")
@click.option("--turbulence-model", "-m",
              type=click.Choice(["sst", "ddes", "wmles", "les"]),
              default="ddes", help="湍流模型")
@click.option("--max-iter", "-n", default=100, help="最大迭代次数")
@click.option("--dt", default=1e-5, help="时间步长 (秒)")
@click.option("--physical-time", default=None, help="总物理时间（秒）")
@click.option("--output", "-o", "output_dir", default="./transient_results", help="输出目录")
@click.option("--use-eikonal", is_flag=True, help='使用 Eikonal 方程求解壁面距离')
@click.option("--surface-mesh", "-s", type=click.Path(exists=True), default=None,
              help='原始面网格路径 - input_file 是 .nas 体网格时必填，用于反推边界分组；input_file 是 .pkl 时不需要')
@click.option("--skip-quality-check", is_flag=True, help='跳过求解前的网格质量门检查（不建议，仅用于临时诊断）')
@click.option('--reference-area', type=float, default=None, help='气动系数参考面积 (m^2)，提供时求解结束后打印 Cd/Cl')
@click.option('--dual-time-inner-iter', type=int, default=20,
              help='--time-method dual-time 时每个物理步的伪时间内迭代次数（此前恒为硬编码3，'
                   '真实测得默认保守CFL策略下通常不足以收敛到物理时间精度，见 TimeIntegrator 文档）')
@click.option('--threads', '-j', type=int, default=-1, help='CPU 后端 numba 并行 kernel 使用的线程数，默认 -1 = 4（本机真实网格实测扩展性甜点，不是核数）')
@click.option('--init-from', 'init_checkpoint', type=click.Path(exists=True), default=None,
              help='从稳态 checkpoint 文件初始化瞬态求解器（典型工作流：先稳态 SST 收敛，'
                   '再从该流场启动 DES/LES 瞬态计算，避免从均匀流场直接启动需要极长的瞬态发展时间）')
def transient(input_file: str, backend: str, order: int, time_method: str,
              turbulence_model: str, max_iter: int, dt: float, physical_time: float,
              output_dir: str, use_eikonal: bool, surface_mesh: Optional[str],
              skip_quality_check: bool, reference_area: Optional[float],
              dual_time_inner_iter: int, threads: int, init_checkpoint: Optional[str]) -> None:
    """运行瞬态 FR 仿真 (DES/LES)。

    Args:
        input_file: 输入体网格文件 - .pkl 或 .nas 体网格（需要配合
            --surface-mesh）- 先用 'grid generate-volume' 或 'grid
            import-volume' 从面网格生成/导入体网格
        backend: 计算后端
        order: FR 阶数
        time_method: 时间推进方法
        turbulence_model: 湍流模型 (推荐 DDES 或 LES)
        max_iter: 最大迭代次数
        dt: 时间步长
        physical_time: 总物理时间（秒）
        output_dir: 输出目录
        use_eikonal: 是否使用 Eikonal 方程
        surface_mesh: 原始面网格路径，input_file 是 .nas 体网格时必填
        skip_quality_check: 跳过求解前的网格质量门检查
        init_checkpoint: 从稳态 checkpoint 初始化（可选）
    """
    print(f"=== Starting Transient FR Simulation (DES/LES) ===")
    print(f"\nInput Grid : {input_file}")
    print(f"Backend    : {backend} | Order: P{order} | Method: {time_method}")
    print(f"Turbulence : {turbulence_model} | dt: {dt:.2e}")
    if init_checkpoint:
        print(f"Init From  : {init_checkpoint}")
    if physical_time:
        max_iter = int(float(physical_time) / dt)
        print(f"Physical Time: {physical_time}s | Iterations: {max_iter}\n")
    else:
        print(f"Iterations : {max_iter}\n")

    # 1. 网格加载与处理（含求解前质量门检查）
    mesh, volume_data = load_mesh_for_solver(
        input_file, order, surface_mesh=surface_mesh, skip_quality_check=skip_quality_check
    )

    # 2. 映射时间推进方法
    time_scheme_map = {
        'rk3': TimeIntegrationScheme.SSP_RK3,
        'imex': TimeIntegrationScheme.IMEX_EULER,
        'dual-time': TimeIntegrationScheme.DUAL_TIME
    }
    time_scheme = time_scheme_map.get(time_method, TimeIntegrationScheme.SSP_RK3)

    # 3. 初始化求解器
    solver = FRSolver(
        mesh=mesh,
        backend=backend,
        order=order,
        turb_model_name=turbulence_model.upper(),
        time_scheme=time_scheme,
        dual_time_inner_iter=dual_time_inner_iter,
        n_threads=threads,
    )

    # 4. 计算壁面距离场（DES/LES/WMLES 必须）
    compute_wall_distance_for_solver(solver, volume_data, use_eikonal=use_eikonal)

    # 4.5. 从 checkpoint 初始化（可选：以稳态结果为初场启动瞬态计算）
    if init_checkpoint:
        from types import SimpleNamespace
        from autoflowcfd.core.checkpoint import CheckpointManager

        print(f"\n🔄 从 checkpoint 加载稳态结果作为瞬态初场...")
        solution, history, ckpt_iter, ckpt_meta = CheckpointManager(
            config=SimpleNamespace(), output_dir="."
        ).load(init_checkpoint)

        restore_state_from_checkpoint(init_checkpoint, solver, ckpt_meta)
        print(f"   源 checkpoint 迭代数: {ckpt_iter}\n")

    # 5. 执行瞬态求解
    try:
        # 瞬态求解通常不需要 tol，而是跑满指定的时间步
        result = solver.solve(max_iter=max_iter, dt=dt, tol=0.0)
        print(f"\n✅ Transient Simulation Finished: Steps={result.iterations}, Final Residual={result.final_residual:.6e}")

        # 6. 保存结果（.pkl 全量状态 + HDF5 checkpoint，后者供 solve resume 使用）
        save_results(solver, output_dir)
        write_checkpoint(
            solver, output_dir, result.iterations, input_file, order, turbulence_model, backend,
            history={"iterations": [result.iterations]},
        )

        # 7. 气动系数（提供 --reference-area 时）
        _report_aerodynamic_coefficients(solver, reference_area)

    except Exception as e:
        print(f"\n❌ Transient Simulation Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise click.Abort()


@solve.command()
@click.argument("checkpoint_file", type=click.Path(exists=True))
@click.option("--max-iter", "-n", default=500, help="额外迭代次数")
@click.option("--backend", "-b", type=click.Choice(["cpu", "gpu"]),
              default=None, help="后端覆盖")
@click.option("--surface-mesh", "-s", type=click.Path(exists=True), default=None,
              help="原始面网格路径——checkpoint 里记录的 input_file 若是 .nas 体网格则必填")
@click.option('--reference-area', type=float, default=None, help='气动系数参考面积 (m^2)')
@click.option('--threads', '-j', type=int, default=-1, help='CPU 后端 numba 并行 kernel 使用的线程数，默认 -1 = 4（本机真实网格实测扩展性甜点，不是核数）')
def resume(checkpoint_file: str, max_iter: int, backend: Optional[str],
           surface_mesh: Optional[str], reference_area: Optional[float], threads: int) -> None:
    """从检查点真正恢复并继续求解（不是只打印元信息）。

    重建流程：checkpoint 的 metadata 记录了重建 FRSolver 所需的全部
    构造参数（input_file/order/turbulence_model/backend/自由来流条件，
    见 solve_helpers.write_checkpoint 文档），用它们重新走一遍
    load_mesh_for_solver + FRSolver(...) 构造出一个全新求解器，再用
    checkpoint 里完整保存的 (n_cells,n_sps,n_vars) 状态（metadata['fields']
    ['U_sps']，不是拍扁过的单元中心近似）整体替换掉初始化时生成的均匀
    流场，然后调用 solver.solve() 继续迭代——此前这里只是把 checkpoint
    读出来打几行日志，从不重建求解器也不继续迭代，是伪装成可用命令的
    stub（V2.0 二次评审 CL-01 发现）。

    Args:
        checkpoint_file: checkpoint 文件路径（solve steady/transient 产出）
        max_iter: 从当前迭代数继续跑的额外迭代次数
        backend: 后端覆盖，None 时沿用 checkpoint 记录的原始后端
        surface_mesh: checkpoint 记录的 input_file 若是 .nas 体网格，
            需要提供原始面网格来反推边界分组（与 solve steady 的同名
            参数语义一致）
        reference_area: 气动系数参考面积
    """
    from types import SimpleNamespace
    from autoflowcfd.core.checkpoint import CheckpointManager

    logger.info(f"Resuming simulation from checkpoint: {checkpoint_file}")

    solution, history, iteration, metadata = CheckpointManager(
        config=SimpleNamespace(), output_dir="."
    ).load(checkpoint_file)

    fields = metadata.get("fields", {})
    if "U_sps" not in fields:
        raise click.ClickException(
            f"Checkpoint '{checkpoint_file}' 缺少 'U_sps' 字段（完整的 (n_cells,n_sps,n_vars) "
            f"求解器状态）——这不是本版本 solve_helpers.write_checkpoint 写出的 checkpoint，"
            f"无法精确恢复（只有拍扁过的单元中心近似不足以重建高阶解）。"
        )

    input_file = metadata.get("input_file")
    order = int(metadata.get("order", 2))
    turbulence_model = metadata.get("turbulence_model", "sst")
    target_backend = backend or metadata.get("backend", "cpu")

    if not input_file:
        raise click.ClickException("Checkpoint metadata 缺少 'input_file'，无法重新加载网格。")

    logger.info(f"Rebuilding solver from checkpoint: input={input_file}, order=P{order}, "
                f"turbulence={turbulence_model}, backend={target_backend}, resuming at iter={iteration}")

    mesh, volume_data = load_mesh_for_solver(input_file, order, surface_mesh=surface_mesh)

    solver = FRSolver(
        mesh=mesh,
        backend=target_backend,
        order=order,
        turb_model_name=turbulence_model,
        rho_inf=metadata.get("rho_inf", 1.225),
        vel_inf=metadata.get("vel_inf", 30.0),
        p_inf=metadata.get("p_inf", 101325.0),
        n_threads=threads,
    )
    compute_wall_distance_for_solver(solver, volume_data)

    U_restored = fields["U_sps"]
    if U_restored.shape != solver.state.U.shape:
        raise click.ClickException(
            f"Checkpoint 状态形状 {U_restored.shape} 与重建求解器的状态形状 "
            f"{solver.state.U.shape} 不匹配（网格或阶数可能已变化），拒绝恢复。"
        )
    solver.state.U = U_restored
    solver.state._update_primitives()

    logger.info(f"State restored from checkpoint, continuing for {max_iter} more iterations...")
    result = solver.solve(max_iter=max_iter, dt=1e-3, tol=1e-6)
    print(f"\n✅ Resumed simulation finished: total_iterations~={iteration + result.iterations}, "
          f"Residual={result.final_residual:.6e}")

    output_dir = str(Path(checkpoint_file).parent.parent)
    save_results(solver, output_dir)
    write_checkpoint(solver, output_dir, iteration + result.iterations, input_file, order,
                      turbulence_model, target_backend)
    _report_aerodynamic_coefficients(solver, reference_area)


@solve.command()
@click.option("--backend", "-b", is_flag=True, help="列出可用后端")
def status(backend: bool) -> None:
    """查看求解器状态。

    Args:
        backend: 列出可用后端
    """
    if backend:
        from autoflowcfd.core.backend import get_available_backends
        backends = get_available_backends()
        logger.info(f"Available backends: {backends}")
    else:
        logger.info("V2.0 FR Solver Status: Ready")
        logger.info("Supported features:")
        logger.info("  - Orders: P1, P2, P3")
        logger.info("  - Time methods: RK3, IMEX, Dual-Time")
        logger.info("  - Turbulence models: SST, DDES, WMLES, LES")
        logger.info("  - Order continuation: P0 → P2/P3 smooth transition")
