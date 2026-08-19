"""`solve steady` 命令 —— 从 solve_steady_commands.py 拆出，控制单文件行数。

见 solve_steady_commands.py 文档说明整体拆分结构。
"""

import logging

import click

from autoflowcfd.core import FRSolver
from autoflowcfd.core.time_integration.base import TimeIntegrationScheme
from autoflowcfd.cli.solve_helpers import (
    compute_wall_distance_for_solver,
    load_mesh_for_solver,
    save_results,
    write_checkpoint,
)
from autoflowcfd.cli.solve_aero_coefficients import (
    _compute_reference_area_auto,
    _report_aerodynamic_coefficients,
)
from autoflowcfd.cli.solve_commands import solve

logger = logging.getLogger(__name__)


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
    """执行稳态 FR 求解。

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
        from autoflowcfd.fr.operators import generate_fr_operators

        mesh, volume_data = load_mesh_for_solver(
            input_file, order, surface_mesh=surface_mesh, skip_quality_check=skip_quality_check
        )
        ops = generate_fr_operators(order)

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
        from autoflowcfd.fr.operators import generate_fr_operators

        mesh, volume_data = load_mesh_for_solver(
            input_file, order, surface_mesh=surface_mesh, skip_quality_check=skip_quality_check
        )
        ops = generate_fr_operators(order)

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
        from autoflowcfd.fr.operators import generate_fr_operators

        # 完全分布式网格加载：root 加载 + 分区 + 分发
        local_mesh, local_fc_data, partition_info = distributed_mesh_load(
            input_file, order, surface_mesh, n_ranks, skip_quality_check
        )

        # 构建算子
        ops = generate_fr_operators(order)

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
                write_checkpoint(
                    solver_ref, output_dir, iteration,
                    input_file, order, turbulence_model, backend,
                    quiet=True
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
            write_checkpoint(
                solver, output_dir, result.iterations, input_file, order, turbulence_model, backend
            )

            # 5. 气动系数（提供 --reference-area 时）
            _report_aerodynamic_coefficients(solver, reference_area)

        except Exception as e:
            print(f"\n❌ Simulation Failed: {str(e)}")
            raise click.Abort()
