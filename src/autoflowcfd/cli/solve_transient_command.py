"""`solve transient` 命令 (DES/LES) —— 从 solve_steady_commands.py 拆出，控制单文件行数。

见 solve_steady_commands.py 文档说明整体拆分结构。
"""

from typing import Optional

import click

from autoflowcfd.core import FRSolver
from autoflowcfd.core.time_integration.base import TimeIntegrationScheme
from autoflowcfd.cli.solve_helpers import (
    compute_wall_distance_for_solver,
    load_mesh_for_solver,
    restore_state_from_checkpoint,
    save_results,
    write_checkpoint,
)
from autoflowcfd.cli.solve_aero_coefficients import _report_aerodynamic_coefficients
from autoflowcfd.cli.solve_commands import solve


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
