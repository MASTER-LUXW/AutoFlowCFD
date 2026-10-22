"""`post transient-mean` / `transient-rms` / `transient-psd`：瞬态历史统计。

mean / rms 依次读取算例目录下全部 checkpoint（按物理时间取样，`--warmup-time` 之前的不计），累积节点分辨率的
时均与脉动均方根（`postprocess/transient_stats.py`）；mean 另可统计气动系数的时均与标准差。psd 读求解器每个物理步
写出的探针时间序列（`core/probes`）。三个命令用普通 `@click.command()` 定义，由 `commands.py` 注册到 `post`。
"""

import csv
from pathlib import Path
from typing import Optional

import click
import numpy as np
from loguru import logger

from autoflowcfd.cli.post.helpers import (
    _export_point_fields_vtk,
    _list_checkpoints,
    _load_grid_data,
    _locate_grid_file,
    _physical_time,
)
from autoflowcfd.postprocess.solution_vector import SolutionVector


def _accumulate_history(case: str, grid: Optional[str], warmup_time: float, on_sample=None):
    """逐个 checkpoint 累积时均 / RMS（mean 与 rms 共用）；`on_sample(stats, ckpt_file, metadata)` 在每个计入统计
    的样本上调用一次（mean 的气动系数统计）。返回 `(grid_data, stats)`。

    Raises:
        click.ClickException: 算例目录没有 checkpoint，或全部样本都早于 warmup_time
    """
    from autoflowcfd.core.utils.checkpoint import CheckpointManager
    from autoflowcfd.postprocess import TransientStatistics

    case_path = Path(case)
    grid_data = _load_grid_data(_locate_grid_file(case_path, grid))
    ckpt_files = _list_checkpoints(case_path)
    if not ckpt_files:
        raise click.ClickException(f"No checkpoints found under {case_path / 'checkpoints'}")

    manager = CheckpointManager()
    stats = TransientStatistics(grid_data)
    n_excluded = 0
    for ckpt_file in ckpt_files:
        solution_data, _history, _iteration, metadata = manager.load(ckpt_file)
        time = _physical_time(metadata, ckpt_file)
        if time < warmup_time:
            n_excluded += 1
            continue
        stats.accumulate(SolutionVector(solution_data), time=time)
        if on_sample is not None:
            on_sample(stats, ckpt_file, metadata)
    if n_excluded == len(ckpt_files):
        raise click.ClickException(f"全部 {n_excluded} 个 checkpoint 都早于 --warmup-time={warmup_time}s")
    if n_excluded:
        logger.info(f"Excluded {n_excluded} checkpoint(s) with time < warmup_time={warmup_time}s")
    return grid_data, stats


def _vtk_path(output: str) -> Path:
    output_path = Path(output)
    return output_path if output_path.suffix else output_path.with_suffix('.vtk')


@click.command(name="transient-mean")
@click.option("--case", "-c", required=True, type=click.Path(exists=True),
              help="算例目录")
@click.option("--grid", "-g", type=click.Path(exists=True),
              help="体网格文件（缺省取算例目录的 volume_mesh.pkl 或 checkpoint 记录的求解输入）")
@click.option("--output", "-o", type=click.Path(), default="mean_flow.vtk",
              help="输出文件")
@click.option("--warmup-time", type=float, default=0.0,
              help="排除物理时间早于此值的 checkpoint（起动瞬态：尾流 / 边界层尚未发展完全的阶段），不计入统计。"
                   "默认 0.0（不排除）")
@click.option("--coefficients/--no-coefficients", default=True, show_default=True,
              help="同时统计气动系数（Cd/Cl/Cs）的时均与标准差：用第一个计入统计的 checkpoint 重建一次求解器，"
                   "之后每个 checkpoint 只替换状态、积分物理无滑移固壁上的力。没有物体（无物理无滑移固壁）的算例用 "
                   "--no-coefficients")
@click.option("--reference-area", type=float, default=None,
              help="气动系数参考面积 (m^2)；不给时沿来流方向由物体面的迎风投影自动估算（同 solve 命令）")
@click.option("--skip-quality-check", is_flag=True,
              help="重建求解器时跳过网格质量门（原求解靠该选项才跑得起来的网格，统计气动系数时同样需要）")
def transient_mean(case: str, grid: Optional[str], output: str, warmup_time: float, coefficients: bool,
                   reference_area: Optional[float], skip_quality_check: bool) -> None:
    """时间平均流场（节点分辨率，导出为 VTK）与气动系数时均。

    \b
    Examples:
        $ autoflowcfd post transient-mean --case transient_results/ --warmup-time 0.05
    """
    logger.info(f"Computing time-averaged flow for case: {case}")

    try:
        coeff = {}

        def on_sample(stats, ckpt_file, metadata):
            from autoflowcfd.cli.solve.checkpoint_io import (
                rebuild_solver_from_checkpoint, restore_solver_state_from_fields,
            )
            from autoflowcfd.postprocess.fr_coefficients import compute_aerodynamic_coefficients_fr

            if "solver" not in coeff:
                # 只重建一次（真实网格上一次重建要 1~2 分钟），之后的 checkpoint 只替换状态数组；
                # 后处理只读状态，一律在 CPU 上重建（与 post coefficients 同一约定）
                coeff["solver"], _it, _meta = rebuild_solver_from_checkpoint(
                    str(ckpt_file), backend="cpu", reference_area=reference_area,
                    skip_quality_check=skip_quality_check)
            else:
                restore_solver_state_from_fields(coeff["solver"], metadata["fields"], metadata)
            solver = coeff["solver"]
            c = compute_aerodynamic_coefficients_fr(solver, reference_area=solver._reference_area)
            stats.accumulate_coefficients(c.Cd, c.Cl, c.Cs)

        grid_data, stats = _accumulate_history(case, grid, warmup_time, on_sample=on_sample if coefficients else None)
        result = stats.compute_statistics()

        vector_fields, scalar_fields = {}, {}
        if all(k in result.mean_fields for k in ('velocity_u', 'velocity_v', 'velocity_w')):
            vector_fields['MeanVelocity'] = np.column_stack([
                result.mean_fields['velocity_u'], result.mean_fields['velocity_v'], result.mean_fields['velocity_w']])
        if 'pressure' in result.mean_fields:
            scalar_fields['MeanPressure'] = result.mean_fields['pressure']
        output_path = _vtk_path(output)
        _export_point_fields_vtk(output_path, grid_data, vector_fields, scalar_fields)

        click.echo(f"✓ Time-averaged flow field exported: {output_path} "
                   f"({result.num_samples} samples, {result.sampling_time:.4f}s)")
        if result.mean_coefficients:
            click.echo(
                f"✓ Mean aerodynamic coefficients ({stats.n_coeff_samples} samples): "
                f"Cd={result.mean_coefficients['Cd']:.6f} (std={result.std_coefficients['Cd']:.6f}), "
                f"Cl={result.mean_coefficients['Cl']:.6f} (std={result.std_coefficients['Cl']:.6f}), "
                f"Cs={result.mean_coefficients['Cs']:.6f} (std={result.std_coefficients['Cs']:.6f})")

    except click.ClickException:
        raise
    except Exception as e:
        logger.error(f"Transient mean calculation failed: {e}")
        raise click.ClickException(str(e))


@click.command(name="transient-rms")
@click.option("--case", "-c", required=True, type=click.Path(exists=True),
              help="算例目录")
@click.option("--grid", "-g", type=click.Path(exists=True),
              help="体网格文件（缺省取算例目录的 volume_mesh.pkl 或 checkpoint 记录的求解输入）")
@click.option("--output", "-o", type=click.Path(), default="rms.vtk",
              help="输出文件")
@click.option("--warmup-time", type=float, default=0.0,
              help="排除物理时间早于此值的 checkpoint（起动瞬态），不计入统计。默认 0.0（不排除）")
def transient_rms(case: str, grid: Optional[str], output: str, warmup_time: float) -> None:
    """速度与压力脉动的均方根（节点分辨率，导出为 VTK）。

    \b
    Examples:
        $ autoflowcfd post transient-rms --case transient_results/ --warmup-time 0.05
    """
    logger.info(f"Computing RMS fluctuations for case: {case}")

    try:
        grid_data, stats = _accumulate_history(case, grid, warmup_time)
        result = stats.compute_statistics()

        vector_fields, scalar_fields = {}, {}
        if all(k in result.rms_fields for k in ('velocity_u_rms', 'velocity_v_rms', 'velocity_w_rms')):
            vector_fields['VelocityRMS'] = np.column_stack([
                result.rms_fields['velocity_u_rms'], result.rms_fields['velocity_v_rms'],
                result.rms_fields['velocity_w_rms']])
        if 'pressure_rms' in result.rms_fields:
            scalar_fields['PressureRMS'] = result.rms_fields['pressure_rms']
        output_path = _vtk_path(output)
        _export_point_fields_vtk(output_path, grid_data, vector_fields, scalar_fields)

        click.echo(f"✓ RMS fluctuation field exported: {output_path} "
                   f"({result.num_samples} samples, {result.sampling_time:.4f}s)")

    except click.ClickException:
        raise
    except Exception as e:
        logger.error(f"RMS calculation failed: {e}")
        raise click.ClickException(str(e))


@click.command(name="transient-psd")
@click.option("--case", "-c", required=True, type=click.Path(exists=True),
              help="算例目录：求解时用 `solve transient --probe X Y Z` 写出的 probes.csv / probes.json 所在目录")
@click.option("--output", "-o", type=click.Path(), default="psd.csv",
              help="输出文件（频率与各探针压力的单边功率谱密度 Pa^2/Hz）")
@click.option("--warmup-time", type=float, default=0.0,
              help="排除物理时间早于此值的样本（起动瞬态混进频谱会抬高低频）。默认 0.0（不排除）")
@click.option("--nperseg", type=int, default=None,
              help="Welch 法每段样本数；不给时样本少于 64 个取单段，否则取约 8 段（50% 重叠）的段长")
def transient_psd(case: str, output: str, warmup_time: float, nperseg: Optional[int]) -> None:
    """探针压力的功率谱密度（Welch 法，`postprocess/pressure_psd.py`）。

    样本是求解器每个物理步在探针位置求的 FR 解多项式值（`core/probes`），采样间隔即物理时间步长。

    \b
    Examples:
        $ autoflowcfd solve transient mesh.pkl --dt 1e-4 --probe 1.5 0.0 0.5 -o transient_results/
        $ autoflowcfd post transient-psd --case transient_results/ --warmup-time 0.05
    """
    logger.info(f"Performing PSD analysis for case: {case}")

    try:
        from autoflowcfd.core.probes import read_probe_series
        from autoflowcfd.postprocess import PressurePSD

        points, times, values = read_probe_series(case)
        keep = times >= warmup_time
        if not np.any(keep):
            raise click.ClickException(f"全部 {len(times)} 个样本都早于 --warmup-time={warmup_time}s")
        psd_analyzer = PressurePSD(monitor_points=[tuple(p) for p in points])
        for t, pressures in zip(times[keep], values["p"][keep]):
            psd_analyzer.add_sample(time=float(t), pressures=pressures.tolist())

        columns = [psd_analyzer.compute_psd(i, nperseg=nperseg) for i in range(len(points))]
        freqs = columns[0][0]
        output_path = Path(output)
        if not output_path.suffix:
            output_path = output_path.with_suffix('.csv')
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frequency_hz'] + [f'psd_probe{i}' for i in range(len(points))])
            for row_idx, freq in enumerate(freqs):
                writer.writerow([freq] + [col[1][row_idx] for col in columns])

        click.echo(f"✓ PSD written: {output_path} ({len(points)} probe(s), {int(keep.sum())} samples, "
                   f"excluded {int((~keep).sum())} before t={warmup_time}s)")
        for i, p in enumerate(points):
            dom_freq, _ = psd_analyzer.find_dominant_frequency(i, min_freq=freqs[1], max_freq=freqs[-1],
                                                               nperseg=nperseg)
            click.echo(f"  Probe {i} {tuple(p)}: dominant f={dom_freq:.2f} Hz")

    except click.ClickException:
        raise
    except Exception as e:
        logger.error(f"PSD analysis failed: {e}")
        raise click.ClickException(str(e))
