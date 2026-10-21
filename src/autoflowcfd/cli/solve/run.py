"""AutoFlowCFD V2.0 - 一次求解的运行参数（`SolveRun`）与按后端执行（`run_solve`）：`solve steady` 与 `solve transient`
共用的唯一实现。

命令只负责解析选项、把 `.nas` 体网格转存成米制 `.pkl`、校验物理量范围，然后构造 `SolveRun` 交给 `run_solve`：
单机（CPU `FRSolver` / 单 GPU `GPUFRSolver`，`solver_factory.py`）或分布式（四种组合，`distributed_build.py`）构造
求解器 -> 可选的 `--init-from` 初场 -> 参考面积 -> 探针 -> 求解（中间 checkpoint）-> 结果落盘与气动系数。

此前稳态三个后端分支、瞬态单机与三个分布式分支各写一遍这条流程（参数逐个转发约 90 行样板，构造、保存、报告
逐段重复），失败时 `print` 一行后 `click.Abort()`，异常栈丢失。
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import click
from loguru import logger

if TYPE_CHECKING:
    from autoflowcfd.core.time_integration.base import TimeIntegrationScheme


@dataclass(frozen=True)
class SolveRun:
    """一次求解的全部运行参数（命令行选项经 `--config` 覆盖后的值）。

    `physics` 为决定物理算例的参数，键同 `checkpoint_io/rebuild.py::PHYSICS_KEYS`（写进 checkpoint、续算原样恢复）。
    """
    input_file: str
    surface_mesh: Optional[str]
    skip_quality_check: bool
    output_dir: str
    checkpoint_interval: int
    backend: str
    n_ranks: int
    multi_gpu: bool
    fully_distributed: bool
    gpu_device: Optional[int]
    threads: int
    order: int
    turbulence_model: str
    time_scheme: "TimeIntegrationScheme"
    time_accurate: bool
    dual_time_inner_iter: int
    cfl_start: Optional[float]
    cfl_max: Optional[float]
    cfl_min: Optional[float]
    low_mach_precond: bool
    physics: dict
    sem_num_eddies: Optional[int]
    reference_area: Optional[float]
    init_checkpoint: Optional[str]
    probes: Tuple
    max_iter: int
    dt: float
    tol: float
    phase_max_iter: Optional[int]
    residual_drop_threshold: float

    @property
    def distributed(self) -> bool:
        return self.n_ranks > 1


def physics_from_options(rho_inf, vel_inf, p_inf, aoa_deg, aos_deg, mu_molecular, turbulence_intensity,
                         viscosity_ratio, artificial_viscosity_enabled, artificial_viscosity_alpha,
                         wall_treatment) -> dict:
    """命令行选项 -> 物理参数字典（键同 `PHYSICS_KEYS`）。"""
    return dict(rho_inf=float(rho_inf), vel_inf=float(vel_inf), p_inf=float(p_inf), aoa_deg=float(aoa_deg),
                aos_deg=float(aos_deg), mu_molecular=float(mu_molecular),
                turbulence_intensity=float(turbulence_intensity), viscosity_ratio=float(viscosity_ratio),
                artificial_viscosity_enabled=bool(artificial_viscosity_enabled),
                artificial_viscosity_alpha=float(artificial_viscosity_alpha), wall_treatment=str(wall_treatment))


def solve_run_from_config(config, *, input_file: str, aoa_deg: float, aos_deg: float,
                          reference_area: Optional[float], skip_quality_check: bool, time_accurate: bool,
                          low_mach_precond: bool, max_iter: int, dt: float, tol: float,
                          init_checkpoint: Optional[str] = None, probes=()) -> SolveRun:
    """配置对象（`SteadyConfig` / `TransientConfig`，Python API 的参数）-> `SolveRun`：单机运行。"""
    from autoflowcfd.config.turbulence_names import turbulence_solver_name
    from autoflowcfd.core.fr_solver.boundary.constants import _SEM_DEFAULT_NUM_EDDIES
    from autoflowcfd.core.time_integration.base import DEFAULT_DUAL_TIME_STEPS

    return SolveRun(
        input_file=input_file, surface_mesh=None, skip_quality_check=skip_quality_check,
        output_dir=config.output_dir, checkpoint_interval=config.checkpoint_interval,
        backend=config.backend.value, n_ranks=1, multi_gpu=False, fully_distributed=False,
        gpu_device=config.gpu_device, threads=config.n_threads, order=config.order,
        turbulence_model=turbulence_solver_name(config.turbulence), time_scheme=config.time_scheme,
        time_accurate=time_accurate, dual_time_inner_iter=DEFAULT_DUAL_TIME_STEPS,
        cfl_start=config.cfl_init, cfl_max=config.cfl_max, cfl_min=config.cfl_min, low_mach_precond=low_mach_precond,
        physics=physics_from_options(config.rho_inf, config.vel_inf, config.p_inf, aoa_deg, aos_deg,
                                     config.mu_molecular, config.turbulence_intensity, config.viscosity_ratio,
                                     config.artificial_viscosity_enabled, config.artificial_viscosity_alpha,
                                     config.wall_treatment),
        sem_num_eddies=_SEM_DEFAULT_NUM_EDDIES, reference_area=reference_area, init_checkpoint=init_checkpoint,
        probes=tuple(probes or ()), max_iter=max_iter, dt=dt, tol=tol, phase_max_iter=config.phase_max_iter,
        residual_drop_threshold=config.residual_drop_threshold)


def single_node_solver_kwargs(run: SolveRun) -> dict:
    """单机求解器（`FRSolver` / `GPUFRSolver`）的构造参数（CLI 与 Python API 共用）。"""
    return dict(order=run.order, turb_model_name=run.turbulence_model, time_scheme=run.time_scheme,
                dual_time_inner_iter=run.dual_time_inner_iter, n_threads=run.threads, cfl_start=run.cfl_start,
                cfl_max=run.cfl_max, cfl_min=run.cfl_min, low_mach_precond=run.low_mach_precond,
                time_accurate=run.time_accurate, sem_num_eddies=run.sem_num_eddies, **run.physics)


def _build(run: SolveRun):
    """构造求解器并接上初场；返回 `(solver, volume_data)`。"""
    if run.distributed:
        from autoflowcfd.cli.solve.checkpoint_io import start_distributed_from_checkpoint
        from autoflowcfd.cli.solve.distributed_build import build_distributed_solver

        solver, volume_data = build_distributed_solver(
            multi_gpu=run.multi_gpu, fully_distributed=run.fully_distributed, input_file=run.input_file,
            order=run.order, surface_mesh=run.surface_mesh, skip_quality_check=run.skip_quality_check,
            n_ranks=run.n_ranks, gpu_device=run.gpu_device, threads=run.threads,
            turbulence_model=run.turbulence_model, time_scheme=run.time_scheme, physics=run.physics,
            cfl_start=run.cfl_start, cfl_max=run.cfl_max, cfl_min=run.cfl_min,
            low_mach_precond=run.low_mach_precond, time_accurate=run.time_accurate,
            dual_time_inner_iter=run.dual_time_inner_iter, sem_num_eddies=run.sem_num_eddies)
        if run.init_checkpoint:
            start_distributed_from_checkpoint(solver, run.init_checkpoint)
        return solver, volume_data

    from autoflowcfd.cli.solve.checkpoint_io import start_single_node_from_checkpoint
    from autoflowcfd.cli.solve.mesh_loader import load_mesh_for_solver
    from autoflowcfd.cli.solve.solver_factory import build_single_node_solver

    mesh, volume_data = load_mesh_for_solver(run.input_file, run.order, surface_mesh=run.surface_mesh,
                                             skip_quality_check=run.skip_quality_check)
    solver = build_single_node_solver(run.backend, mesh, volume_data, gpu_device=run.gpu_device,
                                      **single_node_solver_kwargs(run))
    if run.init_checkpoint:
        start_single_node_from_checkpoint(solver, run.init_checkpoint)
    return solver, volume_data


def _print_header(run: SolveRun) -> None:
    """运行参数摘要（只在 root 打印一次；此前分布式时每个 rank 各打印一遍）。"""
    from autoflowcfd.core.mpi import is_root
    from autoflowcfd.core.time_integration.base import TimeIntegrationScheme

    if not is_root():
        return
    backend = "gpu (multi-GPU)" if run.multi_gpu else run.backend
    if run.distributed:
        backend += f" | MPI ranks: {run.n_ranks}" + (" (fully distributed)" if run.fully_distributed else "")
    transient = run.time_accurate or run.time_scheme == TimeIntegrationScheme.DUAL_TIME
    print(f"\nInput Grid : {run.input_file}")
    print(f"Backend    : {backend} | Order: P{run.order} | Time scheme: {run.time_scheme.value}"
          + (" (time-accurate)" if run.time_accurate else ""))
    print(f"Turbulence : {run.turbulence_model} | Steps: {run.max_iter}"
          + (f" | dt: {run.dt:.2e} s" if transient else ""))
    if run.init_checkpoint:
        print(f"Init From  : {run.init_checkpoint}")
    if run.probes:
        print(f"Probes     : {len(run.probes)}")
    print()


def run_solve(run: SolveRun):
    """按后端执行一次求解并落盘（分布式时全部 rank 调用）；返回 `SolverResult`。

    Raises:
        click.ClickException: 求解或落盘失败（异常栈记进日志）
    """
    from autoflowcfd.cli.solve.aero_coefficients import (
        _report_aerodynamic_coefficients, distributed_reference_area, resolve_reference_area,
    )
    from autoflowcfd.cli.solve.checkpoint_io import periodic_checkpoint_callback, write_single_node_outputs
    from autoflowcfd.cli.solve.distributed_build import finish_distributed
    from autoflowcfd.cli.solve.distributed_checkpoint_io import distributed_periodic_checkpoint_callback
    from autoflowcfd.core.mpi import is_root
    from autoflowcfd.core.probes import attach_probes

    _print_header(run)
    solver, volume_data = _build(run)
    if run.distributed:
        reference_area = distributed_reference_area(solver, volume_data, run.reference_area)
        callback = distributed_periodic_checkpoint_callback(run.checkpoint_interval, run.output_dir, run.input_file,
                                                            run.turbulence_model, surface_mesh=run.surface_mesh)
    else:
        reference_area = resolve_reference_area(solver, volume_data, run.reference_area)
        callback = periodic_checkpoint_callback(run.checkpoint_interval, run.output_dir, run.input_file,
                                                run.turbulence_model, run.backend, surface_mesh=run.surface_mesh)
    attach_probes(solver, run.probes, run.output_dir)

    try:
        result = solver.solve(max_iter=run.max_iter, dt=run.dt, tol=run.tol, checkpoint_callback=callback,
                              phase_max_iter=run.phase_max_iter, residual_drop_threshold=run.residual_drop_threshold)
        if is_root():
            print(f"\n✅ 求解结束：{result.iterations} 步，最终残差 {result.final_residual:.6e}，"
                  f"状态 {solver.solve_status}")
        if run.distributed:
            finish_distributed(solver, run.output_dir, result.iterations, run.input_file, run.turbulence_model,
                               run.surface_mesh, reference_area)
        else:
            write_single_node_outputs(solver, run.output_dir, result.iterations, run.input_file,
                                      run.turbulence_model, run.backend, surface_mesh=run.surface_mesh)
            _report_aerodynamic_coefficients(solver.host_view(), reference_area)
    except Exception as e:
        logger.opt(exception=True).error(f"求解失败：{e}")
        raise click.ClickException(f"求解失败：{e}") from e
    return result
