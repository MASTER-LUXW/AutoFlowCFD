"""稳态/瞬态求解 CLI 命令 —— 从 solve_commands.py 拆出，控制单文件行数。

包含 solve steady 和 solve transient 两个 click 子命令，以及共用的
_report_aerodynamic_coefficients 气动系数积分打印辅助函数。
"""

import logging
from typing import Optional

import click
import numpy as np

from autoflowcfd.core import FRSolver
from autoflowcfd.core.time_integration.base import TimeIntegrationScheme
from autoflowcfd.cli.solve_helpers import (
    compute_wall_distance_for_solver,
    load_mesh_for_solver,
    restore_state_from_checkpoint,
    save_results,
    write_checkpoint,
)

logger = logging.getLogger(__name__)


def _compute_reference_area_auto(volume_data) -> Optional[float]:
    """从面网格自动计算参考面积（X 方向正投影面积）。

    当用户未指定 --reference-area 时调用，从保存的原始面网格数据计算
    车身迎风面的投影面积，作为气动力系数的参考面积。

    Args:
        volume_data: VolumeMeshData 对象，应包含 surface_mesh 属性

    Returns:
        参考面积 (m^2)，计算失败返回 None
    """
    surface_mesh = getattr(volume_data, 'surface_mesh', None)
    if surface_mesh is None:
        logger.debug("Auto reference area: surface_mesh is None")
        return None

    try:
        surface_nodes = surface_mesh.get('nodes')
        surface_faces = surface_mesh.get('faces')
        surface_boundaries = surface_mesh.get('boundaries')

        if surface_nodes is None or surface_faces is None or surface_boundaries is None:
            logger.debug(f"Auto reference area: missing data - nodes={surface_nodes is not None}, faces={surface_faces is not None}, boundaries={surface_boundaries is not None}")
            return None

        # 获取面网格边界名称
        all_boundary_names = list(surface_boundaries.boundary_names)
        logger.debug(f"Auto reference area: surface mesh boundaries = {all_boundary_names}")

        # 查找车身边界面（BODY/CAR/WALL，排除 INLET/OUTLET/SYMMETRY 等）
        body_boundary_names = [
            name for name in all_boundary_names
            if ('BODY' in name.upper() or 'CAR' in name.upper() or 'WALL' in name.upper())
            and 'INLET' not in name.upper() and 'OUTLET' not in name.upper() and 'SYMMETRY' not in name.upper()
        ]

        if not body_boundary_names:
            logger.debug(f"Auto reference area: no body boundary found in {all_boundary_names}")
            return None

        # 收集车身面索引
        body_face_indices = []
        for boundary_name in body_boundary_names:
            face_indices = surface_boundaries.get_cell_indices(boundary_name)
            body_face_indices.extend(face_indices)

        if len(body_face_indices) == 0:
            return None

        body_face_indices = np.array(body_face_indices, dtype=np.int64)

        # 取车身面的节点坐标
        v0 = surface_nodes[surface_faces[body_face_indices, 0]]
        v1 = surface_nodes[surface_faces[body_face_indices, 1]]
        v2 = surface_nodes[surface_faces[body_face_indices, 2]]

        # 计算面法向量和面积
        e1 = v1 - v0
        e2 = v2 - v0
        normals = np.cross(e1, e2)
        areas = 0.5 * np.linalg.norm(normals, axis=1)

        # 归一化法向量
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        unit_normals = normals / norms

        # 计算 X 方向投影面积（迎风面：法向 n_x < 0）
        x_component = unit_normals[:, 0]
        upstream_mask = x_component < 0
        projected_areas = -x_component[upstream_mask] * areas[upstream_mask]
        ref_area = np.sum(projected_areas)

        if ref_area <= 0 or not np.isfinite(ref_area):
            # 兆底：用绝对投影除以 2（适用于对称车身）
            projected_areas_all = np.abs(x_component) * areas
            ref_area = np.sum(projected_areas_all) / 2.0

        if ref_area > 0 and np.isfinite(ref_area):
            logger.info(f"Auto-computed reference area (frontal projected area): {ref_area:.6f} m^2")
            return float(ref_area)

        return None

    except Exception as e:
        logger.warning(f"Failed to auto-compute reference area: {e}")
        return None


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


# 延迟导入 solve 命令组（避免循环导入：本模块注册命令到 solve 组，
# solve_commands.py 导入本模块触发注册）
def _get_solve_group():
    from autoflowcfd.cli.solve_commands import solve
    return solve


@click.pass_context
def _register_steady(ctx):
    """注册 steady 子命令到 solve 组。"""
    pass


# 导入 solve 命令组
from autoflowcfd.cli.solve_commands import solve  # noqa: E402


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
                save_results(solver_ref, output_dir)
                write_checkpoint(
                    solver_ref, output_dir, iteration,
                    input_file, order, turbulence_model, backend
                )
                print(f"   [Checkpoint] Saved at iteration {iteration} to {output_dir}")
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
        from autoflowcfd.core.utils.checkpoint import CheckpointManager

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
