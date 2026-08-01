"""Postprocessing subcommands.

This module provides CLI commands for post-processing simulation results.

Commands:
    - coefficients: Calculate aerodynamic coefficients
    - export-vtk: Export VTK field data
    - report: Generate simulation report
    - convergence: Plot convergence curves
    - transient-mean: Transient mean flow analysis
    - transient-rms: Transient RMS fluctuation analysis
    - transient-psd: Transient spectral analysis

Example:
    $ autoflowcfd post coefficients --case results/
    $ autoflowcfd post export-vtk --case results/ --output output.vtk
"""

import click
import json
import numpy as np
from pathlib import Path
from typing import Optional
from loguru import logger


@click.group()
def post() -> None:
    """Postprocessing commands.
    
    Analyze and visualize simulation results.
    
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
        # TODO: Implement coefficient calculation
        # This requires loading solution data and integrating forces
        
        result = {
            "Cd": 0.0,  # Drag coefficient
            "Cl": 0.0,  # Lift coefficient
            "Cm": 0.0,  # Pitching moment coefficient
            "Cs": 0.0,  # Side force coefficient
        }
        
        logger.warning("Coefficient calculation is under development")
        
        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\nAerodynamic Coefficients")
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
        from pathlib import Path
        import pickle
        from autoflowcfd.grid import NASParser
        from autoflowcfd.core.checkpoint import CheckpointManager
        from autoflowcfd.postprocess import VTKExporter
        
        case_path = Path(case)
        
        # Step 1: Locate grid file (support both .nas and .pkl formats)
        if grid:
            grid_file = Path(grid)
            logger.info(f"Using specified grid file: {grid_file}")
        else:
            # Auto-detect grid file from case directory
            # Priority: volume_mesh.pkl > *.nas in grid/ > *.nas in root
            grid_candidates = [
                case_path / "volume_mesh.pkl",      # Saved volume mesh (preferred)
                case_path / "grid" / "*.nas",       # Surface mesh in grid/
                case_path / "*.nas",                 # Surface mesh in root
            ]
            
            grid_file = None
            for pattern in grid_candidates:
                if pattern.exists():
                    grid_file = pattern
                    break
                elif '*' in str(pattern):
                    matches = list(pattern.parent.glob(pattern.name))
                    if matches:
                        grid_file = matches[0]
                        break
            
            if grid_file is None:
                raise FileNotFoundError(
                    f"Grid file not found in case directory: {case}\n"
                    f"Please specify grid file with --grid option.\n"
                    f"Expected locations:\n"
                    f"  - {case_path}/volume_mesh.pkl (saved volume mesh)\n"
                    f"  - {case_path}/grid/*.nas (surface mesh)\n"
                    f"  - {case_path}/*.nas (surface mesh)"
                )
            
            logger.info(f"Auto-detected grid file: {grid_file}")
        
        # Step 2: Load grid data (handle both .pkl and .nas formats)
        logger.info("Loading grid data...")
        if grid_file.suffix.lower() == '.pkl':
            # Load saved volume mesh from pickle
            logger.info(f"Loading volume mesh from PKL: {grid_file}")
            try:
                with open(grid_file, 'rb') as f:
                    grid_data = pickle.load(f)
                logger.success(f"✓ Volume mesh loaded: {grid_data.node_count} nodes, "
                             f"{grid_data.cell_count} cells")
            except Exception as e:
                raise ValueError(f"Failed to load volume mesh from {grid_file}: {e}")
        else:
            # Parse surface mesh and generate volume mesh
            logger.warning(f"⚠ Parsing surface mesh file: {grid_file}")
            logger.warning("  This will RE-GENERATE the volume mesh!")
            logger.warning("  For best results, use volume_mesh.pkl if available.")
            
            parser = NASParser(str(grid_file))
            grid_data = parser.parse(generate_volume_mesh=True)
            logger.info(f"✓ Grid generated: {grid_data.node_count} nodes, "
                       f"{grid_data.cell_count} cells")
        
        # Step 3: Locate checkpoint file
        if checkpoint:
            ckpt_file = Path(checkpoint)
            logger.info(f"Using specified checkpoint: {ckpt_file}")
        else:
            # Auto-detect latest checkpoint
            ckpt_dir = case_path / "checkpoints"
            latest_link = ckpt_dir / "latest"
            
            if latest_link.exists() and latest_link.is_symlink():
                ckpt_file = latest_link.resolve()
                logger.info(f"Auto-detected latest checkpoint: {ckpt_file}")
            else:
                # Find most recent checkpoint by modification time
                ckpt_files = sorted(ckpt_dir.glob("checkpoint_*.h5"))
                if ckpt_files:
                    ckpt_file = ckpt_files[-1]  # Last one (highest iteration)
                    logger.info(f"Auto-detected checkpoint: {ckpt_file}")
                else:
                    raise FileNotFoundError(
                        f"No checkpoint files found in: {ckpt_dir}\n"
                        f"Please specify checkpoint with --checkpoint option."
                    )
        
        # Step 4: Load solution from checkpoint
        logger.info("Loading solution from checkpoint...")
        ckpt_manager = CheckpointManager(str(ckpt_file.parent))
        solution_data, history, iteration, metadata = ckpt_manager.load(
            ckpt_file,
            target_backend=None  # Auto-detect
        )
        logger.info(f"✓ Solution loaded from iteration {iteration}")
        logger.info(f"  Solution shape: {solution_data.shape}")
        
        # Convert numpy array to SolutionVector if needed
        from autoflowcfd.core.backend.base import SolutionVector
        
        if isinstance(solution_data, np.ndarray):
            logger.info("Converting numpy array to SolutionVector...")
            n_cells = solution_data.shape[0]
            n_variables = solution_data.shape[1] if len(solution_data.shape) > 1 else 5
            
            solution = SolutionVector(
                data=solution_data,
                n_cells=n_cells,
                n_variables=n_variables
            )
            logger.info(f"✓ SolutionVector created: {n_cells} cells, {n_variables} variables")
        else:
            solution = solution_data
        
        # Validate grid-solution compatibility
        if grid_data.metadata.cell_count != solution.n_cells:
            raise ValueError(
                f"Grid-solution mismatch!\n"
                f"  Grid has {grid_data.metadata.cell_count} cells\n"
                f"  Solution expects {solution.n_cells} cells\n"
                f"  Please use the SAME grid file that was used in the original simulation."
            )
        
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
        click.echo(f"Grid cells:      {grid_data.metadata.cell_count:,}")
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
@click.option("--output", "-o", type=click.Path(), default="report.md",
              help="Output report file")
@click.option("--format", "-f", type=click.Choice(["markdown", "html", "pdf"]),
              default="markdown", help="Report format")
def report(case: str, output: str, format: str) -> None:
    """Generate simulation report.
    
    Create a comprehensive report including grid info, solver settings,
    convergence history, and aerodynamic coefficients.
    
    Args:
        case: Case directory
        output: Output report file
        format: Report format
    
    Examples:
        # Markdown report
        $ autoflowcfd post report --case results/ --format markdown
        
        # HTML report
        $ autoflowcfd post report --case results/ --format html
    """
    logger.info(f"Generating report for case: {case}")
    
    try:
        # TODO: Implement report generation
        logger.warning("Report generation is under development")
        click.echo("⚠ Report generation feature coming soon")
    
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise click.ClickException(f"Report generation failed: {e}")


@post.command()
@click.option("--case", "-c", required=True, type=click.Path(exists=True),
              help="Case directory")
@click.option("--output", "-o", type=click.Path(), default="convergence.png",
              help="Output plot file")
@click.option("--variables", multiple=True, default=["residual"],
              help="Variables to plot")
def convergence(case: str, output: str, variables: tuple) -> None:
    """Plot convergence history.
    
    Visualize residual convergence history and other monitoring variables.
    
    Args:
        case: Case directory
        output: Output plot file
        variables: Variables to plot
    
    Examples:
        # Plot residuals
        $ autoflowcfd post convergence --case results/
        
        # Save to file
        $ autoflowcfd post convergence --case results/ -o conv.png
    """
    logger.info(f"Plotting convergence for case: {case}")
    
    try:
        # TODO: Implement convergence plotting
        logger.warning("Convergence plotting is under development")
        click.echo("⚠ Convergence plotting feature coming soon")
    
    except Exception as e:
        logger.error(f"Convergence plotting failed: {e}")
        raise click.ClickException(f"Convergence plotting failed: {e}")


@post.command()
@click.option("--case", "-c", required=True, type=click.Path(exists=True),
              help="Case directory")
@click.option("--output", "-o", type=click.Path(), default="mean_flow.vtk",
              help="Output file")
def transient_mean(case: str, output: str) -> None:
    """Calculate time-averaged flow field.
    
    Compute mean flow statistics from transient simulation data.
    
    Args:
        case: Case directory
        output: Output file
    
    Examples:
        $ autoflowcfd post transient-mean --case transient_results/
    """
    logger.info(f"Computing time-averaged flow for case: {case}")
    
    try:
        # TODO: Implement transient mean calculation
        logger.warning("Transient mean analysis is under development")
        click.echo("⚠ Transient mean analysis coming soon")
    
    except Exception as e:
        logger.error(f"Transient mean calculation failed: {e}")
        raise click.ClickException(str(e))


@post.command()
@click.option("--case", "-c", required=True, type=click.Path(exists=True),
              help="Case directory")
@click.option("--output", "-o", type=click.Path(), default="rms.vtk",
              help="Output file")
def transient_rms(case: str, output: str) -> None:
    """Calculate RMS fluctuations.
    
    Compute root-mean-square (RMS) of flow fluctuations from transient data.
    
    Args:
        case: Case directory
        output: Output file
    
    Examples:
        $ autoflowcfd post transient-rms --case transient_results/
    """
    logger.info(f"Computing RMS fluctuations for case: {case}")
    
    try:
        # TODO: Implement RMS calculation
        logger.warning("RMS analysis is under development")
        click.echo("⚠ RMS analysis coming soon")
    
    except Exception as e:
        logger.error(f"RMS calculation failed: {e}")
        raise click.ClickException(str(e))


@post.command()
@click.option("--case", "-c", required=True, type=click.Path(exists=True),
              help="Case directory")
@click.option("--output", "-o", type=click.Path(), default="psd.csv",
              help="Output file")
@click.option("--probe-location", nargs=3, type=float, multiple=True,
              help="Probe location (x y z)")
def transient_psd(case: str, output: str, probe_location: tuple) -> None:
    """Perform spectral analysis (PSD).
    
    Calculate power spectral density of pressure or velocity fluctuations.
    
    Args:
        case: Case directory
        output: Output file
        probe_location: Probe location coordinates
    
    Examples:
        $ autoflowcfd post transient-psd --case transient_results/ \
          --probe-location 1.5 0.0 0.5
    """
    logger.info(f"Performing PSD analysis for case: {case}")
    
    try:
        # TODO: Implement PSD analysis
        logger.warning("PSD analysis is under development")
        click.echo("⚠ PSD analysis coming soon")
    
    except Exception as e:
        logger.error(f"PSD analysis failed: {e}")
        raise click.ClickException(str(e))
