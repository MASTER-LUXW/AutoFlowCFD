"""后处理子命令。

本模块提供仿真结果后处理相关的 CLI 命令。

命令:
    - coefficients: 计算气动系数
    - export-vtk: 导出 VTK 场数据
    - report: 生成仿真报告
    - convergence: 绘制收敛曲线
    - transient-mean: 瞬态平均流场分析
    - transient-rms: 瞬态 RMS 脉动分析
    - transient-psd: 瞬态频谱分析

示例:
    $ autoflowcfd post coefficients --case results/
    $ autoflowcfd post export-vtk --case results/ --output output.vtk
"""

import csv
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import numpy as np
from loguru import logger


# ----------------------------------------------------------------------
# 案例加载辅助函数：coefficients/export-vtk/report/convergence/
# transient-mean/transient-rms/transient-psd 都需要从案例目录里找到网格
# 文件和 checkpoint 文件——这段逻辑原来只写死在 export_vtk 一个命令里，
# 现在提出来给其它命令共用，避免重复实现同一套"自动探测网格/checkpoint"
# 逻辑。
# ----------------------------------------------------------------------

def _locate_grid_file(case_path: Path, grid: Optional[str]) -> Path:
    """定位网格文件（支持 .pkl 体网格缓存和 .nas 面网格）。"""
    if grid:
        grid_file = Path(grid)
        logger.info(f"Using specified grid file: {grid_file}")
        return grid_file

    # 优先级：volume_mesh.pkl（已保存的体网格）> grid/*.nas（面网格）> *.nas（面网格）
    grid_candidates = [
        case_path / "volume_mesh.pkl",
        case_path / "grid" / "*.nas",
        case_path / "*.nas",
    ]

    for pattern in grid_candidates:
        if pattern.exists():
            logger.info(f"Auto-detected grid file: {pattern}")
            return pattern
        if '*' in str(pattern):
            matches = list(pattern.parent.glob(pattern.name))
            if matches:
                logger.info(f"Auto-detected grid file: {matches[0]}")
                return matches[0]

    raise FileNotFoundError(
        f"Grid file not found in case directory: {case_path}\n"
        f"Please specify grid file with --grid option.\n"
        f"Expected locations:\n"
        f"  - {case_path}/volume_mesh.pkl (saved volume mesh)\n"
        f"  - {case_path}/grid/*.nas (surface mesh)\n"
        f"  - {case_path}/*.nas (surface mesh)"
    )


def _load_grid_data(grid_file: Path):
    """加载网格数据（.pkl 直接反序列化；.nas 解析并重新生成体网格）。"""
    logger.info("Loading grid data...")
    if grid_file.suffix.lower() == '.pkl':
        logger.info(f"Loading volume mesh from PKL: {grid_file}")
        try:
            with open(grid_file, 'rb') as f:
                grid_data = pickle.load(f)
            logger.success(f"✓ Volume mesh loaded: {grid_data.node_count} nodes, "
                         f"{grid_data.cell_count} cells")
        except Exception as e:
            raise ValueError(f"Failed to load volume mesh from {grid_file}: {e}")
    else:
        from autoflowcfd.grid import NASParser

        logger.warning(f"⚠ Parsing surface mesh file: {grid_file}")
        logger.warning("  This will RE-GENERATE the volume mesh!")
        logger.warning("  For best results, use volume_mesh.pkl if available.")

        parser = NASParser(str(grid_file))
        grid_data = parser.parse(generate_volume_mesh=True)
        logger.info(f"✓ Grid generated: {grid_data.node_count} nodes, "
                   f"{grid_data.cell_count} cells")
    return grid_data


def _locate_checkpoint(case_path: Path, checkpoint: Optional[str]) -> Path:
    """定位单个 checkpoint 文件（默认取最新的一个）。"""
    if checkpoint:
        ckpt_file = Path(checkpoint)
        logger.info(f"Using specified checkpoint: {ckpt_file}")
        return ckpt_file

    ckpt_dir = case_path / "checkpoints"
    latest_link = ckpt_dir / "latest"

    if latest_link.exists() and latest_link.is_symlink():
        ckpt_file = latest_link.resolve()
        logger.info(f"Auto-detected latest checkpoint: {ckpt_file}")
        return ckpt_file

    ckpt_files = _list_checkpoints(case_path)
    if ckpt_files:
        ckpt_file = ckpt_files[-1]
        logger.info(f"Auto-detected checkpoint: {ckpt_file}")
        return ckpt_file

    raise FileNotFoundError(
        f"No checkpoint files found in: {ckpt_dir}\n"
        f"Please specify checkpoint with --checkpoint option."
    )


def _list_checkpoints(case_path: Path) -> List[Path]:
    """列出案例目录下全部 checkpoint 文件，按迭代数排序（供
    transient-mean/transient-rms/transient-psd 遍历整条瞬态历史使用）。"""
    ckpt_dir = case_path / "checkpoints"
    if not ckpt_dir.exists():
        return []
    return sorted(
        ckpt_dir.glob("checkpoint_iter_*.h5"),
        key=lambda p: int(p.stem.split('_')[-1]),
    )


def _to_solution_vector(solution_data):
    """把 checkpoint 里读出的 numpy 数组包装成 SolutionVector。"""
    from autoflowcfd.core.backend.base import SolutionVector

    if isinstance(solution_data, np.ndarray):
        n_cells = solution_data.shape[0]
        n_variables = solution_data.shape[1] if len(solution_data.shape) > 1 else 5
        return SolutionVector(data=solution_data, n_cells=n_cells, n_variables=n_variables)
    return solution_data


def _load_case(case: str, grid: Optional[str] = None, checkpoint: Optional[str] = None) -> Tuple:
    """加载网格 + 单个 checkpoint 的解，供 coefficients/export-vtk 使用。

    Returns:
        (grid_data, solution, history, iteration, metadata)
    """
    from autoflowcfd.core.checkpoint import CheckpointManager

    case_path = Path(case)
    grid_file = _locate_grid_file(case_path, grid)
    grid_data = _load_grid_data(grid_file)

    ckpt_file = _locate_checkpoint(case_path, checkpoint)
    ckpt_manager = CheckpointManager(str(ckpt_file.parent))
    solution_data, history, iteration, metadata = ckpt_manager.load(ckpt_file, target_backend=None)
    logger.info(f"✓ Solution loaded from iteration {iteration}")

    solution = _to_solution_vector(solution_data)

    if grid_data.cell_count != solution.n_cells:
        raise ValueError(
            f"Grid-solution mismatch!\n"
            f"  Grid has {grid_data.cell_count} cells\n"
            f"  Solution expects {solution.n_cells} cells\n"
            f"  Please use the SAME grid file that was used in the original simulation."
        )

    return grid_data, solution, history, iteration, metadata


def _load_history_only(case: str, checkpoint: Optional[str] = None) -> Tuple[dict, int, dict]:
    """只加载一个 checkpoint 的收敛历史/元数据（report/convergence 用，不需要网格）。"""
    from autoflowcfd.core.checkpoint import CheckpointManager

    case_path = Path(case)
    ckpt_file = _locate_checkpoint(case_path, checkpoint)
    ckpt_manager = CheckpointManager(str(ckpt_file.parent))
    _solution, history, iteration, metadata = ckpt_manager.load(ckpt_file, target_backend=None)
    return history, iteration, metadata


def _replay_history(history: dict):
    """把 checkpoint 里的收敛历史（每方程残差 + 系数的并行数组）重放进
    一个新的 ConvergenceAnalyzer，供 report/convergence 复用
    ConvergenceAnalyzer/SimulationReport 已有的分析/导出逻辑，而不是
    重新实现一遍。"""
    from autoflowcfd.postprocess import AerodynamicCoefficients, ConvergenceAnalyzer

    analyzer = ConvergenceAnalyzer()
    iterations = history.get('iterations', [])
    residuals_by_eq = history.get('residuals', {})
    coeffs_by_name = history.get('coefficients', {})
    cfl_history = history.get('cfl_history', [])

    for idx, it in enumerate(iterations):
        residuals = {eq: values[idx] for eq, values in residuals_by_eq.items() if idx < len(values)}
        cfl = cfl_history[idx] if idx < len(cfl_history) else 0.0
        coefficients = None
        if 'Cd' in coeffs_by_name and idx < len(coeffs_by_name['Cd']):
            coefficients = AerodynamicCoefficients(
                Cd=coeffs_by_name.get('Cd', [0.0] * (idx + 1))[idx],
                Cl=coeffs_by_name.get('Cl', [0.0] * (idx + 1))[idx],
            )
        analyzer.add_iteration(iteration=it, residuals=residuals, cfl=cfl, coefficients=coefficients)

    return analyzer


def _cell_centroids(grid_data) -> np.ndarray:
    """计算逐单元中心点坐标，形状 (n_cells, 3)。

    与 core/solver_steady_setup.py、core/transient_solver_loop.py 里
    的中心点计算完全一致：三棱柱单元（若存在）占据全局单元索引空间的
    前段，四面体在后。
    """
    nodes_array = np.column_stack([grid_data.nodes.x, grid_data.nodes.y, grid_data.nodes.z])
    tet_connectivity = grid_data.cells.connectivity.astype(np.int64)
    tet_centroids = nodes_array[tet_connectivity].mean(axis=1)
    prism_cells_obj = getattr(grid_data, 'prism_cells', None)
    if prism_cells_obj is not None:
        prism_connectivity = prism_cells_obj.connectivity.astype(np.int64)
        prism_centroids = nodes_array[prism_connectivity].mean(axis=1)
        return np.vstack([prism_centroids, tet_centroids])
    return tet_centroids


def _export_point_fields_vtk(
    output_path: Path,
    grid_data,
    vector_fields: Dict[str, np.ndarray],
    scalar_fields: Dict[str, np.ndarray],
    binary: bool = False,
) -> None:
    """把已经是节点分辨率的场（TransientStatistics 算出的 mean/RMS 场）
    写成只含 POINT_DATA 的 legacy VTK 文件。

    复用 VTKExporter 里已经验证过的网格写入逻辑（_write_points/
    _write_cells，处理三棱柱+四面体混合网格）和标量/矢量场写入逻辑
    （_write_scalar/_write_vector），而不是重新实现一遍 VTK 格式细节——
    只是这里的场数据来源（已经在节点分辨率上）和 VTKExporter.export()
    的主路径（从单元中心的求解器数据出发，逐单元/逐节点各写一份）不同，
    所以没有直接复用 export()/export_boundaries()。
    """
    from autoflowcfd.postprocess import VTKExporter

    exporter = VTKExporter(grid_data, solution=None)
    n_points = grid_data.node_count

    mode = 'wb' if binary else 'w'
    with open(output_path, mode) as f:
        exporter._wl(f, "# vtk DataFile Version 3.0\n", binary)
        exporter._wl(f, f"AutoFlowCFD Export - {output_path.name}\n", binary)
        exporter._wl(f, ("BINARY\n" if binary else "ASCII\n"), binary)
        exporter._wl(f, "\n", binary)
        exporter._wl(f, "DATASET UNSTRUCTURED_GRID\n", binary)
        exporter._wl(f, "\n", binary)

        exporter._write_points(f, binary)
        exporter._write_cells(f, binary)

        exporter._wl(f, f"POINT_DATA {n_points}\n", binary)
        for name, values in vector_fields.items():
            exporter._write_vector(f, name, values, binary)
        for name, values in scalar_fields.items():
            exporter._write_scalar(f, name, values, binary)


@click.group()
def post() -> None:
    """后处理命令。

    分析和可视化仿真结果。

    Examples:
        # Calculate coefficients
        $ autoflowcfd post coefficients --case results/

        # Export to VTK
        $ autoflowcfd post export-vtk --case results/
    """
    pass


@post.command()
@click.option("--case", "-c", required=True, type=click.Path(exists=True),
              help="Case directory or result file")
@click.option("--grid", "-g", type=click.Path(exists=True),
              help="Grid file path (if not in case directory)")
@click.option("--checkpoint", type=click.Path(exists=True),
              help="Checkpoint file path (defaults to latest)")
@click.option("--reference-area", type=float, default=2.2,
              help="Reference area (m²)")
@click.option("--reference-length", type=float, default=4.5,
              help="Reference length (m)")
@click.option("--density", type=float, default=1.225,
              help="Air density (kg/m³)")
@click.option("--velocity", type=float, default=30.0,
              help="Free-stream velocity (m/s)")
@click.option("--output", "-o", type=click.Path(), default="coefficients.json",
              help="Output file")
@click.option("--json", "-j", "json_output", is_flag=True, help="Output as JSON")
def coefficients(
    case: str,
    grid: Optional[str],
    checkpoint: Optional[str],
    reference_area: float,
    reference_length: float,
    density: float,
    velocity: float,
    output: str,
    json_output: bool
) -> None:
    """Calculate aerodynamic coefficients.

    Compute drag coefficient (Cd), lift coefficient (Cl), and other
    aerodynamic coefficients from simulation results.

    Args:
        case: Case directory or result file
        grid: Grid file path (auto-detected from case dir if omitted)
        checkpoint: Checkpoint file path (defaults to latest)
        reference_area: Reference area
        reference_length: Reference length
        density: Air density
        velocity: Free-stream velocity
        output: Output file path
        json_output: Output as JSON

    Examples:
        # Basic calculation
        $ autoflowcfd post coefficients --case results/

        # Custom reference values
        $ autoflowcfd post coefficients --case results/ \
          --reference-area 2.5 --velocity 35.0
    """
    logger.info(f"Calculating aerodynamic coefficients for case: {case}")

    try:
        from autoflowcfd.postprocess import CoefficientCalculator

        grid_data, solution, history, iteration, metadata = _load_case(case, grid, checkpoint)

        calc = CoefficientCalculator(
            grid_data, solution,
            reference_area=reference_area, reference_length=reference_length,
            density=density, velocity=velocity,
        )
        coeffs = calc.calculate()
        result = coeffs.to_dict()

        output_path = Path(output)
        with open(output_path, 'w') as f:
            json.dump({'iteration': iteration, **result}, f, indent=2)
        logger.success(f"Coefficients written to: {output_path}")

        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\nAerodynamic Coefficients (iteration {iteration})")
            click.echo(f"{'='*40}")
            click.echo(f"Cd (Drag):     {result['Cd']:.4f}")
            click.echo(f"Cl (Lift):     {result['Cl']:.4f}")
            click.echo(f"Cm (Pitch):    {result['Cm']:.4f}")
            click.echo(f"Cs (Side):     {result['Cs']:.4f}")

    except Exception as e:
        logger.error(f"Coefficient calculation failed: {e}")
        raise click.ClickException(f"Failed to calculate coefficients: {e}")


@post.command()
@click.option("--case", "-c", required=True, type=click.Path(exists=True),
              help="Case directory")
@click.option("--output", "-o", type=click.Path(), default="output.vtk",
              help="Output VTK file")
@click.option("--variables", multiple=True,
              help="Variables to export (pressure, velocity, etc.)")
@click.option("--time-step", type=int, help="Specific time step (for transient)")
@click.option("--grid", "-g", type=click.Path(exists=True),
              help="Grid file path (if not in case directory)")
@click.option("--checkpoint", type=click.Path(exists=True),
              help="Checkpoint file path (defaults to latest)")
@click.option("--binary/--ascii", "binary", default=None,
              help="Write binary payloads instead of ASCII text (much smaller/"
                   "faster for real mesh sizes). Default: ASCII for .vtk, "
                   "binary+compressed for .vtu.")
@click.option("--boundaries-only", is_flag=True, default=False,
              help="Export only the named boundary patches (WALL/INLET/OUTLET/"
                   "...), tagged with BoundaryID/BoundaryTypeID + a name "
                   "legend, instead of the full volume mesh - lets you filter/"
                   "color by named zone in ParaView (Fluent/OpenFOAM-style "
                   "patch workflow). Requires a volume mesh (VolumeMeshData).")
def export_vtk(
    case: str,
    output: str,
    variables: tuple,
    time_step: int,
    grid: Optional[str],
    checkpoint: Optional[str],
    binary: Optional[bool],
    boundaries_only: bool,
) -> None:
    """Export field data to VTK format.

    Export simulation results to VTK format for visualization in
    ParaView or other VTK-compatible viewers.

    Args:
        case: Case directory containing simulation results
        output: Output VTK file path
        variables: Variables to export (velocity, pressure, k, omega, nut)
        time_step: Specific time step for transient simulations
        grid: Path to volume mesh file (.nas)
        checkpoint: Path to checkpoint file (.h5)

    Examples:
        # Basic export (auto-detects grid and checkpoint from case dir)
        $ autoflowcfd post export-vtk --case results/steady/

        # Specify grid and checkpoint explicitly
        $ autoflowcfd post export-vtk \
          --case results/ \
          --grid results/grid/sedan.nas \
          --checkpoint results/checkpoints/checkpoint_0500.h5 \
          --output flow_field.vtk

        # Export specific variables
        $ autoflowcfd post export-vtk \
          --case results/ \
          --variables velocity pressure \
          --output vel_pres.vtk

        # Transient: export specific time step
        $ autoflowcfd post export-vtk \
          --case results/transient/ \
          --time-step 100 \
          --output step_100.vtk

    Required Data:
        1. Volume mesh file (.nas) - provides grid geometry
        2. Checkpoint file (.h5) - provides solution vector (velocity, pressure, etc.)

    Note:
        If --grid and --checkpoint are not specified, the command will attempt
        to auto-detect them from the case directory structure.
    """
    logger.info(f"Exporting VTK data from case: {case}")

    try:
        from autoflowcfd.postprocess import VTKExporter

        grid_data, solution, history, iteration, metadata = _load_case(case, grid, checkpoint)

        # Step 5: Prepare variables list
        if not variables:
            var_list = ['velocity', 'pressure']
            logger.info(f"No variables specified, using defaults: {var_list}")
        else:
            var_list = list(variables)
            logger.info(f"Exporting variables: {var_list}")

        # Validate variable names
        valid_vars = {'velocity', 'pressure', 'k', 'omega', 'nut'}
        invalid_vars = set(var_list) - valid_vars
        if invalid_vars:
            raise ValueError(
                f"Invalid variables: {invalid_vars}\n"
                f"Valid options: {valid_vars}"
            )

        # Step 6: Create VTK exporter and export
        # mu_t (exact solver eddy viscosity), if the checkpoint has it -
        # see CheckpointManager.save's extra_fields / VTKExporter's mu_t
        # param. Absent for checkpoints written before this was added, in
        # which case 'nut' falls back to a logged-as-approximate estimate.
        mu_t = metadata.get('fields', {}).get('mu_t')
        logger.info("Creating VTK exporter...")
        exporter = VTKExporter(
            grid_data=grid_data,
            solution=solution,
            mu_t=mu_t,
        )

        # Determine output format based on extension
        output_path = Path(output)
        if output_path.suffix == '.vtu':
            fmt = 'xml'
        elif output_path.suffix == '.vtk' or not output_path.suffix:
            fmt = 'legacy'
            if not output_path.suffix:
                output_path = output_path.with_suffix('.vtk')
        else:
            raise ValueError(f"Unsupported file format: {output_path.suffix}")

        logger.info(f"Exporting to: {output_path} (format: {fmt}, boundaries_only: {boundaries_only})")
        if boundaries_only:
            vtk_path = exporter.export_boundaries(
                output_path=str(output_path),
                fields=var_list,
                format=fmt,
                binary=binary,
            )
        else:
            vtk_path = exporter.export(
                output_path=str(output_path),
                fields=var_list,
                format=fmt,
                binary=binary,
            )

        # Success message
        click.echo("\n" + "="*70)
        click.echo("✅ VTK Export Successful")
        click.echo("="*70)
        click.echo(f"Output file:     {vtk_path}")
        click.echo(f"Format:          {fmt.upper()}")
        click.echo(f"Variables:       {', '.join(var_list)}")
        click.echo(f"Iteration:       {iteration}")
        click.echo(f"Grid cells:      {grid_data.cell_count:,}")
        click.echo("="*70)
        click.echo("\n💡 Next steps:")
        click.echo("  1. Open ParaView")
        click.echo(f"  2. File → Open → {vtk_path}")
        click.echo("  3. Click Apply to load data")
        click.echo("  4. Select coloring variable (Velocity/Pressure)")
        click.echo("="*70)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise click.ClickException(str(e))

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise click.ClickException(str(e))

    except Exception as e:
        logger.error(f"VTK export failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise click.ClickException(f"VTK export failed: {e}")


@post.command()
@click.option("--case", "-c", required=True, type=click.Path(exists=True),
              help="Case directory")
@click.option("--checkpoint", type=click.Path(exists=True),
              help="Checkpoint file path (defaults to latest)")
@click.option("--output", "-o", type=click.Path(), default="report.json",
              help="Output report file")
@click.option("--format", "-f", type=click.Choice(["markdown", "html", "pdf", "json"]),
              default="json", help="Report format")
def report(case: str, checkpoint: Optional[str], output: str, format: str) -> None:
    """Generate simulation report.

    Create a comprehensive report including convergence history and
    aerodynamic coefficients, built from the checkpoint's saved
    convergence history (residuals/coefficients/CFL per iteration).

    Args:
        case: Case directory
        checkpoint: Checkpoint to report on (defaults to latest)
        output: Output report file
        format: Report format

    Examples:
        # JSON report (the only format actually implemented - see Note)
        $ autoflowcfd post report --case results/ --format json

    Note:
        Only 'json' is currently implemented (SimulationReport writes
        JSON). Requesting markdown/html/pdf falls back to JSON with a
        warning rather than silently producing an empty/fake file in a
        format nothing actually generates.
    """
    logger.info(f"Generating report for case: {case}")

    try:
        from autoflowcfd.postprocess import SimulationReport

        history, iteration, metadata = _load_history_only(case, checkpoint)
        analyzer = _replay_history(history)

        if format != "json":
            logger.warning(
                f"--format {format} was requested, but report generation only "
                "implements JSON output - writing JSON instead of silently "
                "producing an empty/fake file in an unsupported format."
            )

        # The checkpoint only stores a hash of the original solver
        # configuration (see CheckpointManager._compute_config_hash), not
        # the configuration itself - report honestly with what is actually
        # available instead of fabricating a full config dict.
        config_summary = {
            'config_hash': metadata.get('config_hash'),
            'original_backend': metadata.get('original_backend'),
            'timestamp': metadata.get('timestamp'),
        }

        output_path = Path(output)
        if output_path.suffix not in ('.json',):
            output_path = output_path.with_suffix('.json')

        sim_report = SimulationReport(config_summary, analyzer)
        path = sim_report.generate(str(output_path), metadata={'source_iteration': iteration})

        click.echo(f"✓ Report generated: {path}")

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise click.ClickException(f"Report generation failed: {e}")


@post.command()
@click.option("--case", "-c", required=True, type=click.Path(exists=True),
              help="Case directory")
@click.option("--checkpoint", type=click.Path(exists=True),
              help="Checkpoint file path (defaults to latest)")
@click.option("--output", "-o", type=click.Path(), default="convergence.png",
              help="Output plot file")
@click.option("--variables", multiple=True, default=["residual"],
              help="Variables to plot")
def convergence(case: str, checkpoint: Optional[str], output: str, variables: tuple) -> None:
    """Plot convergence history.

    Visualize residual convergence history and other monitoring variables.

    Args:
        case: Case directory
        checkpoint: Checkpoint to plot from (defaults to latest)
        output: Output plot file
        variables: Variables to plot ('residual' and/or 'cfl' and/or 'coefficients')

    Examples:
        # Plot residuals
        $ autoflowcfd post convergence --case results/

        # Save to file
        $ autoflowcfd post convergence --case results/ -o conv.png
    """
    logger.info(f"Plotting convergence for case: {case}")

    try:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            raise click.ClickException(
                "Convergence plotting requires matplotlib, which is not "
                "installed. Install it with: pip install matplotlib"
            )

        history, iteration, metadata = _load_history_only(case, checkpoint)
        iterations = history.get('iterations', [])
        if not iterations:
            raise click.ClickException(
                f"Checkpoint has no convergence history to plot (iteration {iteration})."
            )

        want_residual = not variables or 'residual' in variables
        want_cfl = 'cfl' in variables
        want_coeffs = 'coefficients' in variables or not variables

        n_panels = sum([want_residual, want_cfl, want_coeffs and 'coefficients' in history])
        n_panels = max(n_panels, 1)
        fig, axes = plt.subplots(n_panels, 1, figsize=(8, 3.2 * n_panels), sharex=True)
        if n_panels == 1:
            axes = [axes]
        panel = 0

        if want_residual and history.get('residuals'):
            ax = axes[panel]; panel += 1
            for eq_name, values in history['residuals'].items():
                ax.semilogy(iterations[:len(values)], np.maximum(values, 1e-16), label=eq_name)
            ax.set_ylabel("Residual")
            ax.legend()
            ax.grid(True, which='both', alpha=0.3)

        if want_cfl and history.get('cfl_history'):
            ax = axes[panel]; panel += 1
            ax.plot(iterations[:len(history['cfl_history'])], history['cfl_history'])
            ax.set_ylabel("CFL")
            ax.grid(True, alpha=0.3)

        if want_coeffs and history.get('coefficients'):
            ax = axes[panel]; panel += 1
            for name, values in history['coefficients'].items():
                ax.plot(iterations[:len(values)], values, label=name)
            ax.set_ylabel("Coefficient")
            ax.legend()
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Iteration")
        fig.tight_layout()

        output_path = Path(output)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        click.echo(f"✓ Convergence plot saved: {output_path}")

    except click.ClickException:
        raise
    except Exception as e:
        logger.error(f"Convergence plotting failed: {e}")
        raise click.ClickException(f"Convergence plotting failed: {e}")


@post.command()
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

        from autoflowcfd.core.checkpoint import CheckpointManager
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


@post.command()
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

        from autoflowcfd.core.checkpoint import CheckpointManager
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


@post.command()
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

        from autoflowcfd.core.checkpoint import CheckpointManager
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
