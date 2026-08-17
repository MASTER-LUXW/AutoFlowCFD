"""`post export-vtk` 命令 (从 post_commands.py 拆分)。

从 post_commands.py 拆出来（该文件原有 974 行，超过 400 行硬性拆分
阈值）：export-vtk 单个命令本身就有约 170 行（含较长的 docstring），
是文件里最重的单个命令，独立成一个文件最清晰。用普通
`@click.command()`（而不是 `@post.command()`）定义——因为定义时这里
还拿不到 `post` 这个 group 对象——由 post_commands.py 在模块加载末尾
`post.add_command(...)` 注册，与 cli/main.py 给顶层命令组注册到
`cli`、cli/grid_commands.py 给 generate-volume/import-volume 注册到
`grid` 完全是同一套机制。纯代码搬移，不改变任何行为。
"""

from pathlib import Path
from typing import Optional

import click
from loguru import logger

from .post_helpers import _load_case


@click.command(name="export-vtk")
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
        valid_vars = {'velocity', 'pressure', 'k', 'omega', 'nut', 'q_criterion'}
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
