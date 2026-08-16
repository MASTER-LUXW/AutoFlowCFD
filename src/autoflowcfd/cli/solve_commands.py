"""Solver subcommands (V2.0 Pure FR).

本模块提供 V2.0 FR 求解器的 CLI 命令，支持高阶精度、多种时间推进方法和湍流模型。

Commands:
    - steady: 运行稳态 FR 仿真
    - transient: 运行瞬态 FR 仿真（专用命令）
    - resume: 从检查点恢复
    - status: 查看求解器状态

Example:
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
def solve_steady(input_file, backend, order, turbulence_model, max_iter, output_dir, checkpoint_interval, use_eikonal, surface_mesh, skip_quality_check, reference_area, threads):
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
    if use_eikonal:
        print(f"Wall Dist : Eikonal (graph-Dijkstra approx)\n")
    else:
        print(f"Wall Dist : KD-Tree (Geometric)\n")

    # 1. 网格加载与处理（含求解前质量门检查）
    mesh, volume_data = load_mesh_for_solver(
        input_file, order, surface_mesh=surface_mesh, skip_quality_check=skip_quality_check
    )

    # 2. 初始化求解器
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
def transient(input_file: str, backend: str, order: int, time_method: str,
              turbulence_model: str, max_iter: int, dt: float, physical_time: float,
              output_dir: str, use_eikonal: bool, surface_mesh: Optional[str],
              skip_quality_check: bool, reference_area: Optional[float],
              dual_time_inner_iter: int, threads: int) -> None:
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
    """
    print(f"=== Starting Transient FR Simulation (DES/LES) ===")
    print(f"\nInput Grid : {input_file}")
    print(f"Backend    : {backend} | Order: P{order} | Method: {time_method}")
    print(f"Turbulence : {turbulence_model} | dt: {dt:.2e}")
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
