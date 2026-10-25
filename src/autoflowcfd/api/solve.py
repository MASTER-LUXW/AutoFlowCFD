"""AutoFlowCFDAPI 的求解入口：稳态、瞬态、从 checkpoint 续算。

与 CLI 走同一套构造：求解前质量门 + 高阶网格（`grid/validation/solve_gate.py`）、求解器工厂
（`cli/solve/solver_factory.py::build_single_node_solver`）、checkpoint 重建与续算的时间步长/容差
（`cli/solve/checkpoint_io/rebuild.py`、`core/utils/checkpoint_time.py`）。参数就是配置类的字段
（`SteadyConfig` / `TransientConfig`，默认值与校验都在那里，见 docs/CONFIGURATION_GUIDE.md）。

2026-10-19 以前：`run_steady`/`run_transient` 各自逐行解析约 40 行"显式参数 > config > 写死默认值"，写死的默认值
是第三份来源（1.225/33.33/101325/1000 步/"sst"/order 2/1e2），参数名与配置字段不一致（turbulence_model/tol/
threads/physical_time/time_method），未知参数透传进求解器构造函数；求解前不跑质量门；续算把 dt=1e-3、tol=1e-6
写死并总是传给 solve()，续算瞬态时静默改掉 checkpoint 里的物理时间步。
"""

import dataclasses
from typing import Any, Optional

from loguru import logger

from autoflowcfd.config.solver_config import SteadyConfig, TransientConfig, field_default
from autoflowcfd.grid.schema import VolumeMeshData


def _resolved_config(config_cls, config, fields: dict):
    """`config`（缺省为配置类默认值）上覆盖 `fields`；重新构造，经配置类同一处校验。"""
    if config is not None and not isinstance(config, config_cls):
        raise TypeError(f"config 必须是 {config_cls.__name__}，收到 {type(config).__name__}")
    names = {f.name for f in dataclasses.fields(config_cls) if f.init}
    unknown = sorted(set(fields) - names)
    if unknown:
        raise TypeError(f"{config_cls.__name__} 没有字段 {unknown}；可用字段：{sorted(names)}")
    return dataclasses.replace(config if config is not None else config_cls(), **fields)


class _APISolveMixin:
    """求解入口：稳态、瞬态、从 checkpoint 续算"""

    # 最近一次求解用的体网格（`AutoFlowCFDAPI.__init__` 置 None）
    volume_mesh: Optional[VolumeMeshData]

    def _solve_and_write(self, solver, run):
        """求解并按 `run.output_dir` / `checkpoint_interval` 落盘（中间 checkpoint + 收尾结果），与 CLI 同一组写出函数。

        2026-10-19 以前 API 从不写结果：配置里的 `output_dir`、`checkpoint_interval` 被静默忽略，长程计算中断即丢失。
        """
        from autoflowcfd.cli.solve.checkpoint_io import periodic_checkpoint_callback, write_single_node_outputs

        names = (run.turbulence_model, run.backend)
        result = solver.solve(
            max_iter=run.max_iter, dt=run.dt, tol=run.tol,
            checkpoint_callback=periodic_checkpoint_callback(run.checkpoint_interval, run.output_dir, run.input_file,
                                                             *names),
            phase_max_iter=run.phase_max_iter, residual_drop_threshold=run.residual_drop_threshold)
        write_single_node_outputs(solver, run.output_dir, result.iterations, run.input_file, *names)
        return result

    def _build_solver(self, volume_mesh: VolumeMeshData, run):
        """质量门 + 高阶网格 + 求解器（CPU / 单 GPU）+ 初场 + 参考面积 + 探针 / 逐步气动力历史，构造参数与 CLI
        单机分支同一份（`cli/solve/run.py::single_node_solver_kwargs`）。"""
        from autoflowcfd.cli.solve.aero_coefficients import resolve_reference_area
        from autoflowcfd.cli.solve.run import single_node_solver_kwargs
        from autoflowcfd.cli.solve.solver_factory import build_single_node_solver
        from autoflowcfd.core.probes import attach_force_history, attach_probes
        from autoflowcfd.grid.validation.solve_gate import high_order_mesh_for_solver

        mesh = high_order_mesh_for_solver(volume_mesh, run.order, skip_quality_check=run.skip_quality_check)
        solver = build_single_node_solver(run.backend, mesh, volume_mesh, gpu_device=run.gpu_device,
                                          **single_node_solver_kwargs(run))
        if run.init_checkpoint:
            from autoflowcfd.cli.solve.checkpoint_io import start_single_node_from_checkpoint

            start_single_node_from_checkpoint(solver, run.init_checkpoint)
        # 参考面积（未给时沿来流方向自动估算）：每步日志的 Cd/Cl/Cs 读它，与 CLI 同一个函数
        resolve_reference_area(solver, volume_mesh, run.reference_area)
        attach_probes(solver, run.probes, run.output_dir)
        if run.force_history:
            attach_force_history(solver, solver._reference_area, run.output_dir)
        self.volume_mesh = volume_mesh
        self.solver = solver
        return solver

    def _run(self, volume_mesh: VolumeMeshData, config, **run_fields):
        """配置 -> 运行参数（`cli/solve/run.py::solve_run_from_config`）-> 构造 -> 求解与落盘。体网格先存成输出
        目录下的 volume_mesh.pkl（checkpoint 记录它，`solve resume` 与后处理用它）。"""
        from autoflowcfd.cli.solve.mesh_loader import save_volume_mesh_cache
        from autoflowcfd.cli.solve.run import solve_run_from_config

        run = solve_run_from_config(config, input_file=save_volume_mesh_cache(volume_mesh, config.output_dir),
                                    **run_fields)
        return self._solve_and_write(self._build_solver(volume_mesh, run), run)

    def run_steady(self, volume_mesh: VolumeMeshData, config: Optional[SteadyConfig] = None, *,
                   aoa_deg: float = 0.0, aos_deg: float = 0.0, reference_area: Optional[float] = None,
                   skip_quality_check: bool = False, **fields) -> Any:
        """稳态求解。

        Args:
            volume_mesh: 体网格（`generate_volume_mesh` 的输出；面网格不能直接求解）
            config: 稳态配置，缺省为 `SteadyConfig()` 的默认值
            aoa_deg, aos_deg: 攻角/侧滑角（度），约定同 CLI `--aoa/--aos`
            reference_area: 气动系数参考面积 (m^2)，None 时沿来流方向的迎风投影自动估算（同 CLI `--reference-area`）
            skip_quality_check: 跳过求解前质量门（同 CLI `--skip-quality-check`，仅用于临时诊断）
            **fields: 覆盖 `config` 的字段（名字同 YAML，如 order=3、turbulence="sa"、time_scheme="newton-krylov"）

        Returns:
            SolverResult（求解器同时存于 `self.solver`，供 calculate_coefficients/export_vtk 使用）。结果与中间
            checkpoint 写在 `output_dir`（体网格存为其中的 volume_mesh.pkl，checkpoint 可用 `solve resume` 续算）

        Raises:
            MeshQualityGateError: 体网格未通过求解前质量门
        """
        from autoflowcfd.core.time_integration.base import STEADY_DT

        config = _resolved_config(SteadyConfig, config, fields)
        logger.info(f"Starting steady-state FR simulation: P{config.order}, {config.turbulence.value}, "
                    f"{config.time_scheme.value}")
        result = self._run(volume_mesh, config, aoa_deg=aoa_deg, aos_deg=aos_deg, reference_area=reference_area,
                           skip_quality_check=skip_quality_check, time_accurate=False,
                           low_mach_precond=config.low_mach_precond, max_iter=config.max_iter, dt=STEADY_DT,
                           tol=config.convergence_tol)
        logger.info(f"Simulation complete: {result.iterations} iterations, converged={result.converged}")
        return result

    def run_transient(self, volume_mesh: VolumeMeshData, config: Optional[TransientConfig] = None, *,
                      aoa_deg: float = 0.0, aos_deg: float = 0.0, reference_area: Optional[float] = None,
                      skip_quality_check: bool = False, probes=None, force_history: bool = False,
                      **fields) -> Any:
        """瞬态求解（双时间步或显式时间精确推进），参数同 `run_steady`；配置类为 `TransientConfig`。

        步数为 `config.n_steps`（给了 total_time 时按 total_time/dt 换算，否则 max_iter）；给了
        `init_from_checkpoint` 时以该 checkpoint 的场为初场（阶数可低于 order，精确延拓）。瞬态不以残差为停止
        条件，跑满全部物理步。`probes`：探针位置 `(k, 3)`（米，同 CLI `--probe`），每个物理步采样写进
        `output_dir/probes.csv`（`core/probes`）。`force_history`：每个物理步把总气动力系数（压力 + 粘性）及其压力
        部分写进 `output_dir/forces.csv`（同 CLI `--force-history`，`core/probes/forces.py`；参考面积未给时沿来流方向
        自动估算）。
        """
        from autoflowcfd.core.time_integration.base import TimeIntegrationScheme

        config = _resolved_config(TransientConfig, config, fields)
        logger.info(f"Starting transient FR simulation: P{config.order}, {config.turbulence.value}, "
                    f"{config.time_scheme.value}, {config.n_steps} steps x dt={config.dt}")
        result = self._run(
            volume_mesh, config, aoa_deg=aoa_deg, aos_deg=aos_deg, reference_area=reference_area,
            skip_quality_check=skip_quality_check,
            # 瞬态的显式格式做时间精确推进（全场同一物理步长的子步）；双时间步本身时间精确
            time_accurate=config.time_scheme != TimeIntegrationScheme.DUAL_TIME,
            # 低马赫伪时间预处理的请求值：dual-time 不用它，显式时间精确推进强制关闭
            low_mach_precond=True, max_iter=config.n_steps, dt=config.dt, tol=0.0,
            init_checkpoint=config.init_from_checkpoint, probes=probes, force_history=force_history)
        logger.info(f"Transient simulation complete: {result.iterations} steps")
        return result

    def resume_simulation(
        self,
        checkpoint_file: str,
        max_iter: int = 0,
        *,
        dt: Optional[float] = None,
        tol: float = field_default(SteadyConfig, "convergence_tol"),
        checkpoint_interval: int = field_default(SteadyConfig, "checkpoint_interval"),
        backend: Optional[str] = None,
        surface_mesh: Optional[str] = None,
        threads: int = field_default(SteadyConfig, "n_threads"),
        gpu_device: Optional[int] = None,
        skip_quality_check: bool = False,
        cfl_start: Optional[float] = None,
        cfl_max: Optional[float] = None,
        cfl_min: Optional[float] = None,
        time_scheme: Optional[str] = None,
        low_mach_precond: Optional[bool] = None,
        phase_max_iter: Optional[int] = None,
        residual_drop_threshold: float = field_default(SteadyConfig, "residual_drop_threshold"),
        force_history: bool = False,
    ) -> Any:
        """从 checkpoint 重建求解器并（可选）继续求解，与 CLI `solve resume` 单机分支同一套重建、续算与落盘
        （结果与中间 checkpoint 写在 checkpoint 所在算例目录，`checkpoint_io/resume_run.py`）。

        Args:
            checkpoint_file: checkpoint 文件
            max_iter: 继续迭代的步数；0 只恢复状态（求解器与体网格存于 `self.solver`/`self.volume_mesh`，可直接导出），
                返回未收敛的结果
            dt: 时间步长覆盖；None 沿用 checkpoint 记录（瞬态为物理时间步长，`checkpoint_time.resume_time_step`）
            tol: 相对收敛容差（时间精确的续算跑满全部步，`run_tolerance`）
            force_history: 从本次续算起写逐步气动力历史（checkpoint 已记录时自动接着写，同 CLI `--force-history`）
            其余: 同 CLI `solve resume` 同名选项（time_scheme 只在稳态格式之间切换；low_mach_precond 为 None 时沿用
                checkpoint 记录）

        Returns:
            SolverResult
        """
        from pathlib import Path

        from autoflowcfd.cli.solve.checkpoint_io import continue_single_node, rebuild_from_checkpoint
        from autoflowcfd.core.utils.checkpoint_time import resume_time_step

        if not Path(checkpoint_file).exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
        logger.info(f"Resuming simulation from checkpoint: {checkpoint_file}")
        solver, iteration, metadata, volume_mesh = rebuild_from_checkpoint(
            checkpoint_file, backend=backend, surface_mesh=surface_mesh, threads=threads,
            skip_quality_check=skip_quality_check, cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min,
            time_scheme=time_scheme, gpu_device=0 if gpu_device is None else gpu_device,
            low_mach_precond=low_mach_precond,
        )
        self.solver = solver
        self.volume_mesh = volume_mesh
        logger.info(f"State restored from checkpoint (iter={iteration})")
        if max_iter <= 0:
            from autoflowcfd.core.fr_solver.state import SolverResult

            return SolverResult(converged=False, iterations=iteration, final_residual=float("nan"))
        return continue_single_node(
            solver, iteration, metadata, checkpoint_file, max_iter=max_iter, dt=resume_time_step(dt, metadata, solver),
            tol=tol, checkpoint_interval=checkpoint_interval, phase_max_iter=phase_max_iter,
            residual_drop_threshold=residual_drop_threshold, force_history=force_history)
