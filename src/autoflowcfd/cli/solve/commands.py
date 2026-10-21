"""V2.0 FR 求解器子命令。

本模块提供 V2.0 FR 求解器的 CLI 命令，支持高阶精度、多种时间推进方法和湍流模型。

命令:
    - steady: 运行稳态 FR 仿真
    - transient: 运行瞬态 FR 仿真（专用命令）
    - resume: 从检查点恢复
    - status: 查看求解器状态

示例:
    $ autoflowcfd solve steady model_volume.pkl --backend cpu --order 2 --turbulence-model sst

steady/transient 命令本体在同目录的 `steady.py`/`transient.py`，本文件只保留
命令组定义 + resume/status。
"""

from typing import Optional

import click
from loguru import logger

from autoflowcfd.cli.solve.option_help import PHASE_MAX_ITER_HELP, RESIDUAL_DROP_THRESHOLD_HELP, THREADS_HELP
from autoflowcfd.cli.solve.solver_factory import validate_backend_options
from autoflowcfd.cli.solve.distributed_checkpoint_io import distributed_periodic_checkpoint_callback
from autoflowcfd.config.solver_config import SteadyConfig, field_default
from autoflowcfd.core.utils.checkpoint_time import run_tolerance
from autoflowcfd.cli.solve.checkpoint_io import continue_single_node, rebuild_solver_from_checkpoint

# 真实 bug（已修复，2026-08-21）：此前这里 `import logging` +
# `logging.getLogger(__name__)` 用的是标准库 logging，不是本项目全局
# 统一用的 loguru（见 cli/main.py 顶层 `cli()` group 回调对 loguru 的
# sink 配置——标准库 logging 完全不受它影响）。项目从未对标准库
# logging 做过 basicConfig/加 handler，root logger 默认没有任何
# handler、默认级别 WARNING，`logger.info(...)` 因此被静默吞掉，不会
# 出现在 stdout 也不会出现在 stderr——真实复现：`solve status`（不带
# --backend）唯一的输出就是 5 行 logger.info，此前这条命令跑完退出码
# 0、终端上什么都不打印，是一个看起来"能跑但什么也不做"的伪装可用
# 命令；`solve resume` 同样两行进度日志被吞掉（其余关键结果走的是
# print()，未受影响，问题比 status 轻但同样是真实 bug）。改成和本
# 代码库其余所有文件一致的 `from loguru import logger`，走同一个已在
# cli/main.py 里配置好、路由到 stderr 的 sink。


@click.group()
def solve():
    """FR 求解器相关命令 (稳态/瞬态)。"""
    pass


# 导入子命令模块，触发 @solve.command() 注册
from autoflowcfd.cli.solve import steady as _steady_cmd  # noqa: F401,E402
from autoflowcfd.cli.solve import transient as _transient_cmd  # noqa: F401,E402
from autoflowcfd.cli.solve.aero_coefficients import _report_aerodynamic_coefficients  # noqa: E402


@solve.command()
@click.argument("checkpoint_file", type=click.Path(exists=True))
@click.option("--max-iter", "-n", default=500, help="额外迭代次数")
@click.option("--backend", "-b", type=click.Choice(["cpu", "gpu"]),
              default=None, help="后端覆盖")
@click.option("--surface-mesh", "-s", type=click.Path(exists=True), default=None,
              help="原始面网格路径覆盖——原 solve steady/transient 运行若传了它，"
                   "checkpoint 里已经存了一份，此处可不传；两者都缺失且 "
                   "input_file 是 .nas 体网格时才会报错")
@click.option('--reference-area', type=float, default=None, help='气动系数参考面积 (m^2)')
@click.option('--threads', '-j', type=int, default=field_default(SteadyConfig, 'n_threads'), show_default=True,
              help=THREADS_HELP)
@click.option('--skip-quality-check', is_flag=True,
              help='跳过重建时的网格质量门检查（B-11：原求解靠该选项才跑得起来的'
                   '网格，resume 同样需要跳过；不建议，仅用于临时诊断）')
@click.option('--checkpoint-interval', type=int, default=field_default(SteadyConfig, 'checkpoint_interval'),
              help='中间 checkpoint 保存间隔（本次 resume 自己新跑的额外迭代数，'
                   '非绝对迭代数）——与 solve steady 同名参数含义一致')
@click.option('--time-scheme', type=click.Choice(['rk3', 'newton-krylov']), default=None,
              help='在两个稳态格式之间切换；不给时沿用 checkpoint 记录的格式（瞬态 dual-time 的 checkpoint 续算仍是 dual-time：物理时间步长、上一时间层、内迭代次数上限'
                   '随 checkpoint 恢复，跑满 --max-iter 步）。单机与分布式续算都支持')
@click.option('--low-mach-precond/--no-low-mach-precond', 'low_mach_precond', default=None,
              help='低马赫数伪时间预处理开关（只作用于稳态格式 rk3/newton-krylov），默认沿用 checkpoint 记录；'
                   '不改变定常解，只改变到达它的路径')
@click.option('--cfl-start', type=float, default=None,
              help='自适应 CFL 初始值，缺省按续算所用时间格式取（同 solve steady --cfl-start）。CFL 是纯数值加速参数，不影响物理解，每次续算可按上一段的收敛表现重新给。单机与分布'
                   '式续算都生效')
@click.option('--cfl-max', type=float, default=None,
              help='自适应 CFL 上限，缺省按续算所用时间格式取（同 solve steady --cfl-max）。单机与分布式续算都生效')
@click.option('--cfl-min', type=float, default=None,
              help='自适应 CFL 下限，缺省按续算所用时间格式取（同 solve steady --cfl-min）。单机与分布式续算都生效')
@click.option('--phase-max-iter', type=int, default=None,
              help=PHASE_MAX_ITER_HELP)
@click.option('--residual-drop-threshold', type=float, default=field_default(SteadyConfig, 'residual_drop_threshold'),
              help=RESIDUAL_DROP_THRESHOLD_HELP)
@click.option('--n-ranks', type=int, default=1,
              help='MPI rank 总数（>1 时重建为分布式求解器：CPU MPI 传统模式，或配合 --multi-gpu / --fully-distributed）。必须与 mpirun -np 的进程'
                   '数一致')
@click.option('--multi-gpu', is_flag=True,
              help='重建为多GPU分布式求解器（需要 --n-ranks>1）')
@click.option('--fully-distributed', is_flag=True,
              help='走"完全分布式加载"重建（只有 root rank 加载完整网格，需要 --n-ranks>1，'
                   '与 --multi-gpu 互斥）')
@click.option('--gpu-device', type=int, default=None,
              help='GPU 设备号（单 GPU 续算默认 0；--multi-gpu 时按 rank 分配）')
@click.option('--dt', type=float, default=None,
              help='时间步长覆盖。不给时沿用 checkpoint 记录的值（dual-time 瞬态的物理时间步长必须与原算例一致；checkpoint 没有记录时须显式给出）')
@click.option('--tol', type=float, default=field_default(SteadyConfig, 'convergence_tol'), show_default=True,
              help='相对收敛容差（与 solve steady --tol 同义）')
@click.option('--output', '-o', 'output_dir', type=click.Path(), default=None,
              help='续算的输出目录（checkpoint 与结果）；不给时为 checkpoint 所在算例目录')
@click.option('--reset-turbulence', is_flag=True,
              help='checkpoint 的湍流场有超过 10%% 的自由度贴近上界（湍流场曾失控、未恢复），或缺少湍流场 / 涡粘时，'
                   '把湍流场重置为来流值再续算（平均流保留）。不给时这两种 checkpoint 拒绝续算并报告')
def resume(checkpoint_file: str, max_iter: int, backend: Optional[str],
           surface_mesh: Optional[str], reference_area: Optional[float], threads: int,
           skip_quality_check: bool, checkpoint_interval: int,
           time_scheme: Optional[str], low_mach_precond: Optional[bool],
           cfl_start: Optional[float], cfl_max: Optional[float], cfl_min: Optional[float],
           phase_max_iter: Optional[int], residual_drop_threshold: float,
           n_ranks: int, multi_gpu: bool, fully_distributed: bool,
           gpu_device: Optional[int], dt: Optional[float], tol: float, output_dir: Optional[str],
           reset_turbulence: bool) -> None:
    """从 checkpoint 续算。

    按 checkpoint 记录的网格（输出目录的 volume_mesh.pkl）、阶数、湍流模型与全部物理参数重建求解器，恢复解点状态、
    湍流场、收敛历史与时间状态后继续迭代 `--max-iter` 步。CFL、步数、容差等数值参数可以重新给出；`--time-scheme`
    只在两个稳态格式之间切换。

    \b
    Examples:
        $ autoflowcfd solve resume results/checkpoints/checkpoint_iter_000500.h5 --max-iter 1000
        $ mpirun -np 8 autoflowcfd solve resume results/checkpoints/checkpoint_iter_000500.h5 --n-ranks 8
    \f

    Args:
        checkpoint_file: checkpoint 文件路径（solve steady/transient 产出）
        max_iter: 从当前迭代数继续跑的额外迭代次数
        backend: 后端覆盖，None 时沿用 checkpoint 记录的原始后端
        surface_mesh: 面网格路径覆盖。None 时回退到 checkpoint metadata
            里存的 surface_mesh（原 solve steady/transient 运行传了的话）；
            input_file 是 .nas 体网格、且两者都缺失时才会报错
        reference_area: 气动系数参考面积
        checkpoint_interval: 中间 checkpoint 保存间隔（额外迭代数）
        cfl_start, cfl_max, cfl_min: 自适应 CFL 的初始值/上限/下限（纯数值加速参数，不影响
            物理解，不从 checkpoint 恢复——每次 resume 由本次命令行重新指定，
            方便根据上一段收敛表现调整）。单机与分布式续算都生效。
        phase_max_iter: Order Continuation 非最终阶段最大步数上限，None 时取
            max_iter // len(orders) 作为这个数字的默认值——不论默认还是
            显式值，目标阶数都吃掉剩余全部步数，见同名 CLI 选项帮助文本
        residual_drop_threshold: Order Continuation 单阶段提前升阶所需的残差下降倍数
        dt: 时间步长覆盖（None 时沿用 checkpoint 记录的值，见 core/utils/checkpoint_time.py::resume_time_step）
        tol: 相对收敛容差
        low_mach_precond: 低马赫预处理覆盖（None 时沿用 checkpoint 记录）
        output_dir: 输出目录，None 时为 checkpoint 所在算例目录
        reset_turbulence: 湍流场被上界大面积截断或 checkpoint 缺湍流场时重置为来流值（不给时拒绝续算，
            `core/utils/order_continuation/turbulence_reset.py`）
    """
    logger.info(f"Resuming simulation from checkpoint: {checkpoint_file}")

    # 后端组合校验（与 `solve steady/transient` 同一规则）
    validate_backend_options(backend, n_ranks, multi_gpu)

    if n_ranks > 1:
        _resume_distributed(
            checkpoint_file, max_iter, n_ranks, multi_gpu, fully_distributed,
            gpu_device, surface_mesh, threads, skip_quality_check,
            checkpoint_interval, phase_max_iter, residual_drop_threshold,
            cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min, reset_turbulence=reset_turbulence,
            time_scheme=time_scheme, dt=dt, tol=tol, reference_area=reference_area,
            low_mach_precond=low_mach_precond, output_dir=output_dir,
        )
        return

    # 单机（CPU / 单 GPU，按 checkpoint 记录的后端或 --backend 覆盖重建，见 solver_factory.py）
    solver, iteration, metadata = rebuild_solver_from_checkpoint(
        checkpoint_file, backend=backend, surface_mesh=surface_mesh, threads=threads,
        reference_area=reference_area,
        skip_quality_check=skip_quality_check,
        cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min,
        time_scheme=time_scheme,
        gpu_device=0 if gpu_device is None else gpu_device,
        low_mach_precond=low_mach_precond,
    )
    solver._reset_turbulence_on_resume = bool(reset_turbulence)
    logger.info(f"State restored from checkpoint (iter={iteration}), "
                f"continuing for {max_iter} more iterations...")
    # 求解 + 中间 checkpoint + 收尾写出：与 Python API 共用（checkpoint_io/resume_run.py）
    result = continue_single_node(
        solver, iteration, metadata, checkpoint_file, max_iter=max_iter, dt=_resume_dt(dt, metadata, solver),
        tol=tol, checkpoint_interval=checkpoint_interval, phase_max_iter=phase_max_iter,
        residual_drop_threshold=residual_drop_threshold, output_dir=output_dir)
    print(f"\n✅ Resumed simulation finished: total_iterations~={iteration + result.iterations}, "
          f"Residual={result.final_residual:.6e}")
    # solver._reference_area 而非 CLI 参数：重建时两者都是 None 会自动估算并存到这个属性上
    _report_aerodynamic_coefficients(solver.host_view(), getattr(solver, "_reference_area", None))


def _resume_dt(dt_option: Optional[float], metadata: dict, solver) -> float:
    """续算时间步长（`core/utils/checkpoint_time.py::resume_time_step`），无法确定时转成命令行错误。"""
    from autoflowcfd.core.utils.checkpoint_time import resume_time_step

    try:
        return resume_time_step(dt_option, metadata, solver)
    except ValueError as e:
        raise click.ClickException(str(e)) from e


def _resume_distributed(
    checkpoint_file: str, max_iter: int, n_ranks: int, multi_gpu: bool,
    fully_distributed: bool, gpu_device: Optional[int],
    surface_mesh: Optional[str], threads: int, skip_quality_check: bool,
    checkpoint_interval: int,
    phase_max_iter: Optional[int] = None,
    residual_drop_threshold: float = 100.0,
    cfl_start: Optional[float] = None,
    cfl_max: Optional[float] = None,
    cfl_min: Optional[float] = None,
    time_scheme: Optional[str] = None,
    dt: Optional[float] = None,
    tol: float = field_default(SteadyConfig, "convergence_tol"),
    reference_area: Optional[float] = None,
    reset_turbulence: bool = False,
    low_mach_precond: Optional[bool] = None,
    output_dir: Optional[str] = None,
) -> None:
    """`resume` 的分布式分支：CPU MPI 传统模式 / 完全分布式加载 / 多 GPU 共用同一个重建入口
    `rebuild_distributed_solver_from_checkpoint`，续算（中间 checkpoint 的文件名按续算起点的绝对迭代数偏移）后
    与 `solve steady/transient` 同一个收尾（`distributed_build.py::finish_distributed`：final_state.pkl、结束时
    的 checkpoint、气动系数）。Order Continuation 的 `--phase-max-iter` / `--residual-drop-threshold` 照常生效。
    """
    from autoflowcfd.core.mpi import is_root
    from autoflowcfd.cli.solve.distributed_checkpoint_io import (
        rebuild_distributed_solver_from_checkpoint,
    )

    solver, iteration, metadata = rebuild_distributed_solver_from_checkpoint(
        checkpoint_file, n_ranks=n_ranks, multi_gpu=multi_gpu,
        fully_distributed=fully_distributed, gpu_device=gpu_device,
        surface_mesh=surface_mesh, threads=threads,
        cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min,
        skip_quality_check=skip_quality_check, time_scheme=time_scheme,
        reference_area=reference_area, low_mach_precond=low_mach_precond,
    )
    solver._reset_turbulence_on_resume = bool(reset_turbulence)
    input_file = metadata["input_file"]
    turbulence_model = metadata["turbulence_model"]
    resolved_surface_mesh = metadata.get("surface_mesh")
    from autoflowcfd.cli.solve.checkpoint_io.resume_run import resume_output_dir

    output_dir = output_dir or resume_output_dir(checkpoint_file)
    from autoflowcfd.core.probes import attach_probes_from_checkpoint

    attach_probes_from_checkpoint(solver, metadata, output_dir)       # 集体操作

    if is_root():
        logger.info(
            f"[Distributed resume] State restored from checkpoint (iter={iteration}), "
            f"continuing for {max_iter} more iterations..."
        )
    resume_dt = _resume_dt(dt, metadata, solver)
    from autoflowcfd.cli.solve.distributed_build import finish_distributed

    result = solver.solve(
        max_iter=max_iter, dt=resume_dt, tol=run_tolerance(tol, solver),
        checkpoint_callback=distributed_periodic_checkpoint_callback(
            checkpoint_interval, output_dir, input_file, turbulence_model, surface_mesh=resolved_surface_mesh,
            iteration_offset=iteration),
        phase_max_iter=phase_max_iter, residual_drop_threshold=residual_drop_threshold)
    total_iterations = iteration + result.iterations
    if is_root():
        print(f"\n✅ 续算结束：累计 {total_iterations} 步，最终残差 {result.final_residual:.6e}，状态 {solver.solve_status}")
    finish_distributed(solver, output_dir, total_iterations, input_file, turbulence_model, resolved_surface_mesh,
                       solver._report_reference_area)


@solve.command()
@click.option("--backend", "-b", is_flag=True, help="列出可用后端")
def status(backend: bool) -> None:
    """查看求解器状态：可用后端，或 `solve steady` / `solve transient` 支持的阶数、时间格式与湍流模型。

    \f
    能力清单直接读两个命令自己的选项定义（唯一来源）。

    Args:
        backend: 列出可用后端
    """
    if backend:
        from autoflowcfd.core.backend import get_available_backends
        click.echo(f"Available backends: {get_available_backends()}")
        return

    def option(command, name):
        return next(p for p in command.params if p.name == name)

    for label, time_option in (("steady", "time_scheme"), ("transient", "time_method")):
        command = solve.commands[label]
        order = option(command, "order").type
        click.echo(f"solve {label}:")
        click.echo(f"  阶数        P{order.min} ~ P{order.max}（>= 1 时从 P0 逐阶爬升，Order Continuation）")
        click.echo(f"  时间格式    {', '.join(option(command, time_option).type.choices)}")
        click.echo(f"  湍流模型    {', '.join(option(command, 'turbulence_model').type.choices)}")
