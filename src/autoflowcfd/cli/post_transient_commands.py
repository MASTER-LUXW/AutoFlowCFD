"""`post transient-mean` / `transient-rms` / `transient-psd` 命令
(从 post_commands.py 拆分)。

从 post_commands.py 拆出来（该文件原有 974 行，超过 400 行硬性拆分
阈值）：这三个命令都是围绕"瞬态历史统计"这一主题（依次遍历案例目录下
全部 checkpoint、累积统计量），合计约 260 行，与文件里其余偏"单个
checkpoint"的命令（coefficients/export-vtk/report/convergence）自成
一组，是清晰的拆分边界。用普通 `@click.command()`（而不是
`@post.command()`）定义——因为定义时这里还拿不到 `post` 这个 group
对象——由 post_commands.py 在模块加载末尾 `post.add_command(...)`
注册，与 cli/grid_commands.py 给 generate-volume/import-volume 注册到
`grid` 完全是同一套机制。纯代码搬移，不改变任何行为。
"""

import csv
from pathlib import Path
from typing import List, Optional

import click
import numpy as np
from loguru import logger

from .post_helpers import (
    _cell_centroids,
    _export_point_fields_vtk,
    _list_checkpoints,
    _load_grid_data,
    _locate_grid_file,
    _to_solution_vector,
)


@click.command(name="transient-mean")
@click.option("--case", "-c", required=True, type=click.Path(exists=True),
              help="Case directory")
@click.option("--grid", "-g", type=click.Path(exists=True),
              help="Grid file path (if not in case directory)")
@click.option("--output", "-o", type=click.Path(), default="mean_flow.vtk",
              help="Output file")
def transient_mean(case: str, grid: Optional[str], output: str) -> None:
    """Calculate time-averaged flow field.

    Compute mean flow statistics from transient simulation data by
    accumulating every saved checkpoint in the case directory's
    checkpoints/ folder (see TransientStatistics.accumulate) and
    exporting the resulting node-resolution mean fields to VTK.

    Args:
        case: Case directory
        grid: Grid file path (auto-detected from case dir if omitted)
        output: Output file

    Examples:
        $ autoflowcfd post transient-mean --case transient_results/
    """
    logger.info(f"Computing time-averaged flow for case: {case}")

    try:
        from autoflowcfd.postprocess import TransientStatistics

        case_path = Path(case)
        grid_file = _locate_grid_file(case_path, grid)
        grid_data = _load_grid_data(grid_file)

        ckpt_files = _list_checkpoints(case_path)
        if not ckpt_files:
            raise click.ClickException(f"No checkpoints found under {case_path / 'checkpoints'}")

        from autoflowcfd.core.utils.checkpoint import CheckpointManager
        ckpt_manager = CheckpointManager(str(ckpt_files[0].parent))
        stats = TransientStatistics(grid_data, window_size=len(ckpt_files))

        for ckpt_file in ckpt_files:
            solution_data, _history, iteration, metadata = ckpt_manager.load(ckpt_file)
            solution = _to_solution_vector(solution_data)
            time = float(metadata.get('current_time', iteration))
            stats.accumulate(solution, time=time)

        result = stats.compute_statistics()

        vector_fields = {}
        scalar_fields = {}
        if all(k in result.mean_fields for k in ('velocity_u', 'velocity_v', 'velocity_w')):
            vector_fields['MeanVelocity'] = np.column_stack([
                result.mean_fields['velocity_u'],
                result.mean_fields['velocity_v'],
                result.mean_fields['velocity_w'],
            ])
        if 'pressure' in result.mean_fields:
            scalar_fields['MeanPressure'] = result.mean_fields['pressure']

        output_path = Path(output)
        if not output_path.suffix:
            output_path = output_path.with_suffix('.vtk')
        _export_point_fields_vtk(output_path, grid_data, vector_fields, scalar_fields)

        click.echo(
            f"✓ Time-averaged flow field exported: {output_path} "
            f"({result.num_samples} samples, {result.sampling_time:.4f}s)"
        )

    except click.ClickException:
        raise
    except Exception as e:
        logger.error(f"Transient mean calculation failed: {e}")
        raise click.ClickException(str(e))


@click.command(name="transient-rms")
@click.option("--case", "-c", required=True, type=click.Path(exists=True),
              help="案例目录")
@click.option("--grid", "-g", type=click.Path(exists=True),
              help="网格文件路径（如果不在案例目录中）")
@click.option("--output", "-o", type=click.Path(), default="rms.vtk",
              help="输出文件")
def transient_rms(case: str, grid: Optional[str], output: str) -> None:
    """计算 RMS 脉动。

    从瞬态数据中计算流场脉动的均方根 (RMS)，使用与
    transient-mean 相同的检查点累积过程（参见 TransientStatistics.accumulate/compute_statistics）。

    Args:
        case: 案例目录
        grid: 网格文件路径（如果省略则从案例目录自动检测）
        output: 输出文件

    Examples:
        $ autoflowcfd post transient-rms --case transient_results/
    """
    logger.info(f"Computing RMS fluctuations for case: {case}")

    try:
        from autoflowcfd.postprocess import TransientStatistics

        case_path = Path(case)
        grid_file = _locate_grid_file(case_path, grid)
        grid_data = _load_grid_data(grid_file)

        ckpt_files = _list_checkpoints(case_path)
        if not ckpt_files:
            raise click.ClickException(f"No checkpoints found under {case_path / 'checkpoints'}")

        from autoflowcfd.core.utils.checkpoint import CheckpointManager
        ckpt_manager = CheckpointManager(str(ckpt_files[0].parent))
        stats = TransientStatistics(grid_data, window_size=len(ckpt_files))

        for ckpt_file in ckpt_files:
            solution_data, _history, iteration, metadata = ckpt_manager.load(ckpt_file)
            solution = _to_solution_vector(solution_data)
            time = float(metadata.get('current_time', iteration))
            stats.accumulate(solution, time=time)

        result = stats.compute_statistics()

        vector_fields = {}
        scalar_fields = {}
        if all(k in result.rms_fields for k in ('velocity_u_rms', 'velocity_v_rms', 'velocity_w_rms')):
            vector_fields['VelocityRMS'] = np.column_stack([
                result.rms_fields['velocity_u_rms'],
                result.rms_fields['velocity_v_rms'],
                result.rms_fields['velocity_w_rms'],
            ])
        if 'pressure_rms' in result.rms_fields:
            scalar_fields['PressureRMS'] = result.rms_fields['pressure_rms']

        output_path = Path(output)
        if not output_path.suffix:
            output_path = output_path.with_suffix('.vtk')
        _export_point_fields_vtk(output_path, grid_data, vector_fields, scalar_fields)

        click.echo(
            f"✓ RMS fluctuation field exported: {output_path} "
            f"({result.num_samples} samples, {result.sampling_time:.4f}s)"
        )

    except click.ClickException:
        raise
    except Exception as e:
        logger.error(f"RMS calculation failed: {e}")
        raise click.ClickException(str(e))


@click.command(name="transient-psd")
@click.option("--case", "-c", required=True, type=click.Path(exists=True),
              help="案例目录")
@click.option("--grid", "-g", type=click.Path(exists=True),
              help="网格文件路径（如果不在案例目录中）")
@click.option("--output", "-o", type=click.Path(), default="psd.csv",
              help="输出文件")
@click.option("--probe-location", nargs=3, type=float, multiple=True,
              help="探针位置 (x y z)")
def transient_psd(case: str, grid: Optional[str], output: str, probe_location: tuple) -> None:
    """执行频谱分析 (PSD)。

    计算在给定探针位置处压力脉动的功率谱密度，数据采样自案例
    目录中的每个检查点（位于离每个探针最近的单元中心的压力 - 参见 PressurePSD）。

    Args:
        case: 案例目录
        grid: 网格文件路径（如果省略则从案例目录自动检测）
        output: 输出文件
        probe_location: 探针位置坐标

    Examples:
        $ autoflowcfd post transient-psd --case transient_results/ \
          --probe-location 1.5 0.0 0.5
    """
    logger.info(f"Performing PSD analysis for case: {case}")

    if not probe_location:
        raise click.ClickException(
            "At least one --probe-location x y z is required for PSD analysis."
        )

    try:
        from autoflowcfd.postprocess import PressurePSD

        case_path = Path(case)
        grid_file = _locate_grid_file(case_path, grid)
        grid_data = _load_grid_data(grid_file)

        ckpt_files = _list_checkpoints(case_path)
        if len(ckpt_files) < 8:
            raise click.ClickException(
                f"PSD analysis needs >= 8 samples, found {len(ckpt_files)} "
                f"checkpoints under {case_path / 'checkpoints'}."
            )

        centroids = _cell_centroids(grid_data)
        probes = np.asarray(probe_location, dtype=np.float64)
        probe_cell_idx = [int(np.argmin(np.linalg.norm(centroids - p, axis=1))) for p in probes]

        from autoflowcfd.core.utils.checkpoint import CheckpointManager
        ckpt_manager = CheckpointManager(str(ckpt_files[0].parent))

        times: List[float] = []
        pressures_per_ckpt: List[List[float]] = []
        for ckpt_file in ckpt_files:
            solution_data, _history, iteration, metadata = ckpt_manager.load(ckpt_file)
            solution = _to_solution_vector(solution_data)
            p_field = solution.get_pressure()
            times.append(float(metadata.get('current_time', iteration)))
            pressures_per_ckpt.append([float(p_field[idx]) for idx in probe_cell_idx])

        # PSD via FFT assumes uniform sampling - use the median spacing
        # between saved checkpoints (they are normally saved at a fixed
        # iteration/time interval) and warn if the actual spacing varies
        # a lot, rather than silently feeding a non-uniform series into rfft.
        dts = np.diff(times)
        dt = float(np.median(dts)) if len(dts) else 1.0
        if len(dts) and np.std(dts) > 0.1 * abs(dt):
            logger.warning(
                f"Checkpoint sampling interval is not uniform (std={np.std(dts):.3e}s, "
                f"median={dt:.3e}s) - PSD assumes uniform dt, so results may be "
                f"inaccurate. Save checkpoints at a fixed interval for a clean spectrum."
            )

        psd_analyzer = PressurePSD(monitor_points=list(probe_location), dt=max(dt, 1e-12))
        for t, pressures in zip(times, pressures_per_ckpt):
            psd_analyzer.add_sample(time=t, pressures=pressures)

        output_path = Path(output)
        if not output_path.suffix:
            output_path = output_path.with_suffix('.csv')

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['frequency_hz']
            header.extend(f'psd_probe{i}' for i in range(len(probes)))
            writer.writerow(header)

            freqs = None
            psd_columns = []
            for i in range(len(probes)):
                freqs, psd_values = psd_analyzer.compute_psd(i)
                psd_columns.append(psd_values)
            for row_idx, freq in enumerate(freqs):
                writer.writerow([freq] + [col[row_idx] for col in psd_columns])

        click.echo(f"✓ PSD written: {output_path} ({len(probes)} probe(s), {len(ckpt_files)} samples)")
        for i in range(len(probes)):
            dom_freq, dom_psd = psd_analyzer.find_dominant_frequency(i)
            click.echo(f"  Probe {i} {tuple(probe_location[i])}: dominant f={dom_freq:.2f} Hz")

    except click.ClickException:
        raise
    except Exception as e:
        logger.error(f"PSD analysis failed: {e}")
        raise click.ClickException(str(e))
