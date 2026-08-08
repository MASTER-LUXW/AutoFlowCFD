"""Solver subcommands.

This module provides CLI commands for running CFD simulations.

Commands:
    - run: Run steady-state RANS simulation
    - transient: Run transient LES/DES simulation
    - resume: Resume from checkpoint
    - status: Check solver status

Example:
    $ autoflowcfd solve run model.nas --backend gpu --order 3
    $ autoflowcfd solve transient model.nas --mode des --physical-time 0.3
"""

import click
import json
from typing import Optional
from pathlib import Path
from loguru import logger


@click.group()
def solve() -> None:
    """Solver commands.
    
    Run steady-state or transient CFD simulations.
    
    Examples:
        # Steady-state RANS
        $ autoflowcfd solve run sedan.nas
        
        # Transient DES
        $ autoflowcfd solve transient sedan.nas --physical-time 0.3
    """
    pass


@solve.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--backend", "-b", type=click.Choice(["cpu", "gpu", "auto"]),
              default="auto", help="Compute backend")
@click.option("--order", "-p", type=click.IntRange(1, 3), default=3,
              help="FR discretization order (1/2/3)")
@click.option("--turbulence", "-t", type=click.Choice(["sst_kw", "sa"]),
              default="sst_kw", help="Turbulence model")
@click.option("--max-iter", "-n", default=500, help="Maximum iterations")
@click.option("--cfl-init", type=float, default=0.05, help="Initial CFL number (recommended: 0.05-0.1 for complex grids)")
@click.option("--cfl-max", type=float, default=50.0, help="Maximum CFL number")
@click.option("--convergence-tol", type=float, default=1e-6,
              help="Convergence tolerance")
@click.option("--config", "-c", type=click.Path(exists=True),
              help="YAML config file path")
@click.option("--output", "-o", type=click.Path(), default="results/",
              help="Output directory")
@click.option("--checkpoint-interval", type=int, default=100,
              help="Checkpoint save interval (steps)")
@click.option("--threads", type=int, default=-1,
              help="CPU thread count (-1 for auto)")
@click.option("--gpu-device", type=int, default=0,
              help="GPU device ID")
@click.option("--growth-rate", type=float, default=None,
              help="Boundary-layer geometric growth rate (overrides config)")
@click.option("--bl-layers", type=int, default=None,
              help="How many layers count as the fine boundary-layer stage before "
                   "switching to the (fixed-rate) transition stage; unset defaults "
                   "to 8 (overrides config)")
@click.option("--min-cell-size", type=float, default=None,
              help="Minimum cell size in meters (overrides config)")
@click.option("--max-cell-size", type=float, default=None,
              help="Max core-region cell size in meters, graded outward from the BL's "
                   "near-wall size (overrides config); unset means no cap")
@click.option("--surface-mesh", "-s", type=click.Path(exists=True), default=None,
              help="Original surface .nas file INPUT_FILE was generated from - passing "
                   "this treats INPUT_FILE as an EXTERNALLY-generated volume mesh (e.g. "
                   "ANSA's own volume export: GRID + CTETRA + CPENTA) instead of a "
                   "surface mesh to regenerate from or a cached .pkl. Runs the same "
                   "parse -> geometric boundary matching -> quality check -> best-effort "
                   "Stage A repair flow as 'autoflowcfd grid import-volume', inline, "
                   "without needing a separate .pkl round-trip first.")
@click.option("--wall-functions", is_flag=True, default=False,
              help="Enable Menter scalable/automatic wall treatment (log-law based) "
                   "on WALL/GROUND faces, instead of resolving all the way to the "
                   "wall - lets a coarser near-wall mesh (y+ up to ~100+, not just "
                   "y+~1) still give physically meaningful skin friction and "
                   "near-wall turbulence (overrides config)")
@click.option("--skip-quality-check", is_flag=True, default=False,
              help="Skip the pre-solve volume mesh quality gate (MeshQualityValidator) "
                   "and solve regardless of a failing report - e.g. negative-volume "
                   "or extreme volume-ratio cells, which are diagnostic of degenerate "
                   "(sliver) tetrahedra that reliably seed a divergence once solved. "
                   "By default a failing report aborts before any solve iterations "
                   "run, instead of burning compute on a doomed case.")
@click.option("--json", "-j", "json_output", is_flag=True, help="Output as JSON")
def run(
    input_file: str,
    backend: str,
    order: int,
    turbulence: str,
    max_iter: int,
    cfl_init: float,
    cfl_max: float,
    convergence_tol: float,
    config: str,
    output: str,
    checkpoint_interval: int,
    threads: int,
    gpu_device: int,
    growth_rate: Optional[float],
    bl_layers: Optional[int],
    min_cell_size: Optional[float],
    max_cell_size: Optional[float],
    surface_mesh: Optional[str],
    wall_functions: bool,
    skip_quality_check: bool,
    json_output: bool
) -> None:
    """Run steady-state RANS simulation.
    
    Solve steady-state Reynolds-Averaged Navier-Stokes equations
    using Flux Reconstruction method.
    
    Args:
        input_file: Path to a surface .nas grid file (volume mesh is
            generated fresh), a cached volume_mesh.pkl (from a prior
            `solve run`/`transient`, or from `grid import-volume`'s own
            external-mesh import - loaded as-is, not regenerated), or -
            when --surface-mesh is also given - an externally-generated
            volume-mesh .nas file
        surface_mesh: Original surface .nas INPUT_FILE was generated from;
            only meaningful (and required) when INPUT_FILE is an external
            volume mesh, not a surface .nas or cached .pkl
        backend: Compute backend (cpu/gpu/auto)
        order: FR discretization order
        turbulence: Turbulence model
        max_iter: Maximum iteration count
        cfl_init: Initial CFL number
        cfl_max: Maximum CFL number
        convergence_tol: Convergence tolerance
        config: YAML config file
        output: Output directory
        checkpoint_interval: Checkpoint interval
        threads: CPU thread count
        gpu_device: GPU device ID
        skip_quality_check: Skip the pre-solve mesh quality gate
        json_output: Output as JSON

    Examples:
        # Basic steady RANS
        $ autoflowcfd solve run sedan.nas
        
        # GPU with 3rd order
        $ autoflowcfd solve run sedan.nas --backend gpu --order 3
        
        # With config file
        $ autoflowcfd solve run sedan.nas -c simulation.yaml

        # Directly from an externally-generated volume mesh (e.g. ANSA)
        $ autoflowcfd solve run car_volume.nas -s car_surface.nas
    """
    from autoflowcfd.config import ConfigLoader, SteadyConfig, BackendType, TurbulenceModel
    from autoflowcfd.grid import NASParser
    from autoflowcfd.core import FRSolver
    
    logger.info(f"Starting steady-state simulation")
    logger.info(f"Grid file: {input_file}")
    
    try:
        # Load configuration
        if config:
            logger.info(f"Loading config from {config}")
            loader = ConfigLoader()
            steady_config = loader.load(config)

            # Only apply a CLI flag if the user actually typed it - every
            # one of these options carries a click default (e.g. max_iter
            # defaults to 500), so unconditionally applying them would
            # silently clobber whatever the YAML config specifies even
            # when the user never asked to override it. This previously
            # caused two different bugs at once: backend/order/turbulence
            # were force-applied from their CLI defaults regardless of the
            # config file's values, while every other flag (max-iter,
            # cfl-init, cfl-max, convergence-tol, output, checkpoint-
            # interval, threads, gpu-device) was silently ignored even when
            # explicitly passed alongside --config.
            ctx = click.get_current_context()

            def _explicit(name: str) -> bool:
                return ctx.get_parameter_source(name) == click.core.ParameterSource.COMMANDLINE

            if _explicit('backend'):
                steady_config.backend = BackendType(backend)
            if _explicit('order'):
                steady_config.order = order
            if _explicit('turbulence'):
                steady_config.turbulence = TurbulenceModel(turbulence)
            if _explicit('max_iter'):
                steady_config.max_iter = max_iter
            if _explicit('cfl_init'):
                steady_config.cfl_init = cfl_init
            if _explicit('cfl_max'):
                steady_config.cfl_max = cfl_max
            if _explicit('convergence_tol'):
                steady_config.convergence_tol = convergence_tol
            if _explicit('output'):
                steady_config.output_dir = output
            if _explicit('checkpoint_interval'):
                steady_config.checkpoint_interval = checkpoint_interval
            if _explicit('threads'):
                steady_config.n_threads = threads if threads > 0 else -1
            if _explicit('gpu_device'):
                steady_config.gpu_device = gpu_device
            if _explicit('wall_functions'):
                steady_config.use_wall_functions = wall_functions
        else:
            # Create config from CLI options
            steady_config = SteadyConfig(
                backend=BackendType(backend),
                order=order,
                turbulence=TurbulenceModel(turbulence),
                max_iter=max_iter,
                cfl_init=cfl_init,
                cfl_max=cfl_max,
                convergence_tol=convergence_tol,
                output_dir=output,
                checkpoint_interval=checkpoint_interval,
                n_threads=threads if threads > 0 else -1,
                gpu_device=gpu_device,
                use_wall_functions=wall_functions,
            )

        # --growth-rate/--bl-layers/--min-cell-size are CLI-only
        # overrides: when passed, they win over whatever steady_config
        # carries (defaults, or values loaded from --config yaml).
        if growth_rate is not None:
            steady_config.growth_rate = growth_rate
        if bl_layers is not None:
            steady_config.bl_layers = bl_layers
        if min_cell_size is not None:
            steady_config.min_cell_size = min_cell_size
        if max_cell_size is not None:
            steady_config.max_cell_size = max_cell_size

        logger.info(f"Configuration: backend={steady_config.backend}, "
                   f"order={steady_config.order}, turbulence={steady_config.turbulence}")

        # Load grid: a saved volume_mesh.pkl (from a prior `solve run`/
        # `transient`, OR from `grid import-volume`'s own external-mesh
        # import) is loaded as-is, same convention `transient`/`resume`
        # already use - no re-validation, since import-volume's own
        # quality report (printed when that command ran) already told the
        # user whether it passed; --surface-mesh means input_file is an
        # externally-generated volume mesh, imported inline (same flow as
        # `grid import-volume`, just without the separate .pkl round
        # trip); otherwise input_file is a surface .nas, parsed and
        # tetrahedralized fresh using this config's own BL parameters.
        input_path = Path(input_file)
        if input_path.suffix.lower() == '.pkl':
            logger.info(f"Loading saved volume mesh: {input_file}")
            import pickle
            try:
                with open(input_path, 'rb') as f:
                    grid_data = pickle.load(f)
                logger.success(
                    f"Volume mesh loaded: {grid_data.node_count} nodes, "
                    f"{grid_data.cell_count} cells"
                )
            except Exception as e:
                raise ValueError(f"Failed to load volume mesh from {input_file}: {e}")
            if skip_quality_check:
                logger.debug("--skip-quality-check has no effect when input_file is a cached volume_mesh.pkl")
        elif surface_mesh is not None:
            from autoflowcfd.grid.mesh_gen.mesh_external_import import import_external_volume_mesh
            logger.info(f"Importing external volume mesh: {input_file}")
            grid_data, quality_report = import_external_volume_mesh(
                input_file, surface_mesh, repair=True, check_overlap=True,
            )
            logger.info(f"External volume mesh loaded: {grid_data.node_count} nodes, "
                       f"{grid_data.cell_count} cells")
            if not quality_report.passed and not skip_quality_check:
                raise click.ClickException(
                    "External volume mesh quality check failed (see report above) - "
                    "solving would very likely diverge. Pass --skip-quality-check to "
                    "solve anyway, or address the implicated cells (e.g. re-mesh the "
                    "sliver regions in the original tool) and re-import."
                )
        else:
            logger.info("Parsing grid file...")
            parser = NASParser(input_file)

            logger.info(
                f"Using BL parameters: growth_rate={steady_config.growth_rate}, "
                f"min_cell_size={steady_config.min_cell_size}m"
            )

            # Enable volume mesh generation by default for accurate CFD.
            # Defaults are conservative BL parameters chosen to avoid
            # self-intersection on sharp features (e.g. Ahmed Body's tight
            # underbody gaps): few layers, small initial cell size, low growth
            # rate -- see SteadyConfig field docs for the full rationale.
            grid_data = parser.parse(
                generate_volume_mesh=True,
                volume_mesh_params={
                    'growth_rate': steady_config.growth_rate,
                    'bl_layers': steady_config.bl_layers,
                    'min_cell_size': steady_config.min_cell_size,
                    'target_cells': steady_config.target_cells,
                    'max_cell_size': steady_config.max_cell_size,
                }
            )

            logger.info(f"Grid loaded: {grid_data.node_count} nodes, "
                       f"{grid_data.cell_count} cells")

            # Pre-solve mesh quality gate. Degenerate (sliver) tetrahedra -
            # near-zero volume relative to the mesh's typical cell size - are
            # numerically ill-conditioned for gradient reconstruction and flux
            # computation, and reliably seed a local blow-up that spreads
            # through the domain over iterations (root-caused this way for a
            # real case: BL extrusion at a body's sharp convex edges/corners
            # produced sliver cells whose positions matched the eventual
            # divergence's pressure/velocity hotspots almost exactly). Catching
            # this before solve() burns any iterations is much cheaper than
            # discovering it from a diverged run's checkpoint history.
            if not skip_quality_check:
                from autoflowcfd.grid import MeshQualityValidator
                logger.info("Validating volume mesh quality before solving...")
                quality_report = MeshQualityValidator().validate_volume_mesh(grid_data)
                if quality_report.passed:
                    logger.info(f"\n{quality_report.summary()}")
                else:
                    logger.error(f"\n{quality_report.summary()}")
                    raise click.ClickException(
                        "Volume mesh quality check failed (see report above) - solving "
                        "would very likely diverge. Common causes: sharp convex edges/"
                        "corners on the body (BL extrusion degrades there; consider a "
                        "small chamfer/fillet in the source geometry), or an overly "
                        "aggressive --growth-rate/--min-cell-size for this geometry's "
                        "feature sizes. Pass --skip-quality-check to solve anyway."
                    )

        # Save volume mesh for future resume operations
        import pickle
        output_dir = Path(steady_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        volume_mesh_path = output_dir / "volume_mesh.pkl"
        
        try:
            with open(volume_mesh_path, 'wb') as f:
                pickle.dump(grid_data, f)
            logger.success(f"Volume mesh saved to: {volume_mesh_path}")
            logger.info("This file can be used for resume operations with --grid option")
        except Exception as e:
            logger.warning(f"Failed to save volume mesh: {e}")
            logger.warning("Resume will require re-generating the volume mesh")
        
        # Create solver
        logger.info("Initializing solver...")
        solver = FRSolver(grid_data, steady_config)
        
        # Run simulation. Log the config's effective max_iter (not the raw
        # CLI variable) since --config may have set a different value that
        # the CLI flag didn't explicitly override.
        logger.info(f"Starting simulation (max_iter={steady_config.max_iter})...")
        result = solver.solve()
        
        # Output results
        result_dict = {
            "command": "solve.run",
            "status": "success" if result.converged else "not_converged",
            "iterations": result.iterations,
            "final_residual": result.final_residual,
            "converged": result.converged,
            "output_dir": output,
        }
        
        if json_output:
            click.echo(json.dumps(result_dict, indent=2))
        else:
            click.echo(f"\n{'='*60}")
            click.echo(f"Simulation Complete")
            click.echo(f"{'='*60}")
            click.echo(f"Status: {'CONVERGED' if result.converged else 'NOT CONVERGED'}")
            click.echo(f"Iterations: {result.iterations}")
            click.echo(f"Final Residual: {result.final_residual:.6e}")
            click.echo(f"Output Directory: {output}")
            click.echo(f"{'='*60}")
        
        # Exit code based on convergence
        if not result.converged:
            raise SystemExit(1)
    
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        if json_output:
            error_result = {
                "command": "solve.run",
                "status": "error",
                "error": str(e)
            }
            click.echo(json.dumps(error_result, indent=2))
        raise click.ClickException(f"Simulation failed: {e}")


@solve.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--backend", "-b", type=click.Choice(["cpu", "gpu", "auto"]),
              default="auto", help="Compute backend")
@click.option("--order", "-p", type=click.IntRange(1, 3), default=3,
              help="FR discretization order")
@click.option("--mode", "-m", type=click.Choice(["des", "ddes", "les"]),
              default="des", help="Turbulence mode")
@click.option("--time-integration", type=click.Choice(["backward_euler", "rk2", "rk3", "ab3"]),
              default="backward_euler", help="Time integration scheme")
@click.option("--physical-time", type=float, required=True,
              help="Total physical time (seconds)")
@click.option("--dt", type=float, default=1e-4, help="Time step size")
@click.option("--init-from", type=click.Path(exists=True),
              help="Initialize from steady-state checkpoint")
@click.option("--config", "-c", type=click.Path(exists=True),
              help="YAML config file")
@click.option("--output", "-o", type=click.Path(), default="transient_results/",
              help="Output directory")
@click.option("--sample-interval", type=int, default=10,
              help="Sampling interval for statistics")
@click.option("--checkpoint-interval", type=int, default=None,
              help="Checkpoint save interval (steps; overrides config)")
@click.option("--threads", type=int, default=None,
              help="CPU thread count (-1 for auto; overrides config)")
@click.option("--gpu-device", type=int, default=None,
              help="GPU device ID (overrides config)")
@click.option("--growth-rate", type=float, default=None,
              help="Boundary-layer geometric growth rate (overrides config; ignored "
                   "when input_file is a cached volume_mesh.pkl)")
@click.option("--bl-layers", type=int, default=None,
              help="How many layers count as the fine boundary-layer stage before "
                   "switching to the (fixed-rate) transition stage; unset defaults "
                   "to 8 (overrides config; ignored when input_file is a cached "
                   "volume_mesh.pkl)")
@click.option("--min-cell-size", type=float, default=None,
              help="Minimum cell size in meters (overrides config; ignored when "
                   "input_file is a cached volume_mesh.pkl)")
@click.option("--max-cell-size", type=float, default=None,
              help="Max core-region cell size in meters, graded outward from the BL's "
                   "near-wall size (overrides config); unset means no cap. Ignored "
                   "when input_file is a cached volume_mesh.pkl")
@click.option("--surface-mesh", "-s", type=click.Path(exists=True), default=None,
              help="Original surface .nas file INPUT_FILE was generated from - passing "
                   "this treats INPUT_FILE as an EXTERNALLY-generated volume mesh "
                   "(see `solve run --help` for the same option's full rationale). "
                   "Ignored when input_file is a cached volume_mesh.pkl.")
@click.option("--wall-functions", is_flag=True, default=False,
              help="Enable Menter scalable/automatic wall treatment (log-law based) "
                   "on WALL/GROUND faces, instead of resolving all the way to the "
                   "wall (overrides config; see `solve run --help` for the same "
                   "option's rationale)")
@click.option("--skip-quality-check", is_flag=True, default=False,
              help="Skip the pre-solve volume mesh quality gate (see `solve run --help` "
                   "for the same option's rationale). No effect when input_file is a "
                   "cached volume_mesh.pkl.")
@click.option("--json", "-j", "json_output", is_flag=True, help="Output as JSON")
def transient(
    input_file: str,
    backend: str,
    order: int,
    mode: str,
    time_integration: str,
    physical_time: float,
    dt: float,
    init_from: str,
    config: str,
    output: str,
    sample_interval: int,
    checkpoint_interval: Optional[int],
    threads: Optional[int],
    gpu_device: Optional[int],
    growth_rate: Optional[float],
    bl_layers: Optional[int],
    min_cell_size: Optional[float],
    max_cell_size: Optional[float],
    surface_mesh: Optional[str],
    wall_functions: bool,
    skip_quality_check: bool,
    json_output: bool
) -> None:
    """Run transient LES/DES simulation.

    Solve unsteady flow using Large Eddy Simulation or Detached
    Eddy Simulation.

    Args:
        input_file: Path to a .nas surface grid file, OR a saved
            volume_mesh.pkl (from a prior `solve run`/`solve transient`).
            Passing the .pkl is strongly recommended whenever --init-from
            points at a checkpoint produced on a specific volume mesh:
            re-parsing the .nas here regenerates the volume mesh from
            scratch using THIS command's own BL parameters, which reliably
            produces a different cell count than the
            checkpoint's solution array and fails to load.
        backend: Compute backend
        order: FR order
        mode: Turbulence mode (des/ddes/les)
        time_integration: Time integration scheme
        physical_time: Total physical time
        dt: Time step size
        init_from: Initialize from checkpoint
        config: YAML config file
        output: Output directory
        sample_interval: Sampling interval
        checkpoint_interval: Checkpoint save interval (steps)
        threads: CPU thread count
        gpu_device: GPU device ID
        growth_rate, bl_layers, min_cell_size, max_cell_size:
            Volume mesh generation parameters (same meaning as `solve run`);
            ignored when input_file is a cached volume_mesh.pkl
        wall_functions: Enable Menter scalable/automatic wall treatment
        skip_quality_check: Skip the pre-solve volume mesh quality gate
        json_output: Output as JSON

    Examples:
        # Basic DES simulation
        $ autoflowcfd solve transient sedan.nas --physical-time 0.3

        # DDES with RK2
        $ autoflowcfd solve transient sedan.nas --mode ddes \
          --time-integration rk2 --physical-time 0.5

        # Initialize from steady solution - reuse the EXACT volume mesh the
        # checkpoint was solved on (see `solve run`'s "Volume mesh saved
        # to" log line for its path) instead of re-generating from the
        # surface .nas, so cell counts are guaranteed to match
        $ autoflowcfd solve transient steady_results/volume_mesh.pkl \
          --physical-time 0.3 --init-from steady_results/checkpoint.h5
    """
    from autoflowcfd.config import ConfigLoader, TransientConfig, BackendType, TurbulenceModel, TimeIntegrationScheme
    from autoflowcfd.grid import NASParser
    from autoflowcfd.core import TransientSolver

    logger.info(f"Starting transient simulation")
    logger.info(f"Physical time: {physical_time}s, dt: {dt}s")

    try:
        # Map mode to turbulence model
        turbulence_map = {
            'des': TurbulenceModel.DES,
            'ddes': TurbulenceModel.DDES,
            'les': TurbulenceModel.LES,
        }

        # Load or create configuration
        if config:
            loader = ConfigLoader()
            transient_config = loader.load(config)

            # Only apply a CLI flag if the user actually typed it - see the
            # matching comment in `run()` above for why: every one of these
            # options carries a click default, so applying them
            # unconditionally would silently clobber the YAML config's
            # values. Previously --config to `transient` ignored EVERY CLI
            # flag (backend/order/mode/time-integration/dt/output/
            # sample-interval all silently dropped) - worse than `run`,
            # which at least applied backend/order/turbulence.
            ctx = click.get_current_context()

            def _explicit(name: str) -> bool:
                return ctx.get_parameter_source(name) == click.core.ParameterSource.COMMANDLINE

            if _explicit('backend'):
                transient_config.backend = BackendType(backend)
            if _explicit('order'):
                transient_config.order = order
            if _explicit('mode'):
                transient_config.turbulence = turbulence_map[mode]
            if _explicit('time_integration'):
                transient_config.time_scheme = TimeIntegrationScheme(time_integration)
            if _explicit('dt'):
                transient_config.dt = dt
            if _explicit('physical_time'):
                transient_config.total_time = physical_time
            if _explicit('output'):
                transient_config.output_dir = output
            if _explicit('sample_interval'):
                transient_config.sample_interval = sample_interval
            if _explicit('checkpoint_interval'):
                transient_config.checkpoint_interval = checkpoint_interval
            if _explicit('threads'):
                transient_config.n_threads = threads if threads > 0 else -1
            if _explicit('gpu_device'):
                transient_config.gpu_device = gpu_device
            if _explicit('wall_functions'):
                transient_config.use_wall_functions = wall_functions
        else:
            transient_config = TransientConfig(
                backend=BackendType(backend),
                order=order,
                turbulence=turbulence_map[mode],
                time_scheme=TimeIntegrationScheme(time_integration),
                dt=dt,
                total_time=physical_time,
                output_dir=output,
                sample_interval=sample_interval,
                n_threads=threads if (threads is not None and threads > 0) else -1,
                gpu_device=gpu_device if gpu_device is not None else 0,
                checkpoint_interval=checkpoint_interval if checkpoint_interval is not None else 100,
                use_wall_functions=wall_functions,
            )

        # --growth-rate/--bl-layers/--min-cell-size/--max-cell-size
        # are CLI-only overrides (same convention as `run()` above): when
        # passed, they win over whatever transient_config carries (defaults,
        # or values loaded from --config yaml). No effect when input_file
        # below turns out to be a cached volume_mesh.pkl - that mesh is
        # loaded as-is, not regenerated from these parameters.
        if growth_rate is not None:
            transient_config.growth_rate = growth_rate
        if bl_layers is not None:
            transient_config.bl_layers = bl_layers
        if min_cell_size is not None:
            transient_config.min_cell_size = min_cell_size
        if max_cell_size is not None:
            transient_config.max_cell_size = max_cell_size

        # Calculate total steps from the EFFECTIVE config values (not the
        # raw CLI variables), which may differ if --config set its own
        # dt/total_time and the CLI didn't explicitly override them.
        total_steps = int(transient_config.total_time / transient_config.dt)
        logger.info(f"Total time steps: {total_steps}")

        # Load grid: a saved volume_mesh.pkl is loaded as-is (same
        # convention as `solve resume`'s --grid); a .nas surface file is
        # parsed and tetrahedralized fresh using THIS config's own BL
        # parameters (which default differently from `solve run`'s, and
        # aren't exposed as CLI flags here) - see the pkl-vs-nas rationale
        # in this command's docstring. Re-generating is what silently
        # produced a different cell count than an --init-from checkpoint's
        # solution array, which then fails at load_checkpoint() below with
        # a confusing-looking mismatch error far from its actual cause.
        input_path = Path(input_file)
        if input_path.suffix.lower() == '.pkl':
            logger.info(f"Loading saved volume mesh: {input_file}")
            import pickle
            try:
                with open(input_path, 'rb') as f:
                    grid_data = pickle.load(f)
                logger.success(
                    f"Volume mesh loaded: {grid_data.node_count} nodes, "
                    f"{grid_data.cell_count} cells"
                )
            except Exception as e:
                raise ValueError(f"Failed to load volume mesh from {input_file}: {e}")

            # No quality gate here: this mesh already passed (or had
            # --skip-quality-check explicitly accepted) whatever solve
            # generated it in the first place - re-validating an
            # already-accepted mesh on every subsequent transient run adds
            # cost without new information. Matches `solve resume`'s
            # identical treatment of a cached volume_mesh.pkl.
            if skip_quality_check:
                logger.debug("--skip-quality-check has no effect when input_file is a cached volume_mesh.pkl")
        elif surface_mesh is not None:
            from autoflowcfd.grid.mesh_gen.mesh_external_import import import_external_volume_mesh
            logger.info(f"Importing external volume mesh: {input_file}")
            grid_data, quality_report = import_external_volume_mesh(
                input_file, surface_mesh, repair=True, check_overlap=True,
            )
            logger.info(f"External volume mesh loaded: {grid_data.node_count} nodes, "
                       f"{grid_data.cell_count} cells")
            if not quality_report.passed and not skip_quality_check:
                raise click.ClickException(
                    "External volume mesh quality check failed (see report above) - "
                    "solving would very likely diverge. Pass --skip-quality-check to "
                    "solve anyway, or address the implicated cells (e.g. re-mesh the "
                    "sliver regions in the original tool) and re-import."
                )
        else:
            logger.info("Parsing grid file...")
            parser = NASParser(input_file)
            grid_data = parser.parse(
                generate_volume_mesh=True,
                volume_mesh_params={
                    'growth_rate': transient_config.growth_rate,
                    'bl_layers': transient_config.bl_layers,
                    'min_cell_size': transient_config.min_cell_size,
                    'target_cells': transient_config.target_cells,
                    'max_cell_size': transient_config.max_cell_size,
                }
            )
            logger.info(f"Grid loaded: {grid_data.node_count} nodes, {grid_data.cell_count} cells")

            # Pre-solve mesh quality gate - see `run`'s (solve run) identical
            # check for the full rationale.
            if not skip_quality_check:
                from autoflowcfd.grid import MeshQualityValidator
                logger.info("Validating volume mesh quality before solving...")
                quality_report = MeshQualityValidator().validate_volume_mesh(grid_data)
                if quality_report.passed:
                    logger.info(f"\n{quality_report.summary()}")
                else:
                    logger.error(f"\n{quality_report.summary()}")
                    raise click.ClickException(
                        "Volume mesh quality check failed (see report above) - solving "
                        "would very likely diverge. Common causes: sharp convex edges/"
                        "corners on the body (BL extrusion degrades there; consider a "
                        "small chamfer/fillet in the source geometry), or an overly "
                        "aggressive --growth-rate/--min-cell-size for this geometry's "
                        "feature sizes. Pass --skip-quality-check to solve anyway."
                    )

        # Create solver
        logger.info("Initializing transient solver...")
        solver = TransientSolver(grid_data, transient_config)
        
        # Initialize from checkpoint if specified
        if init_from:
            logger.info(f"Initializing from checkpoint: {init_from}")
            solver.load_checkpoint(init_from)
        
        # Run simulation
        logger.info(f"Starting transient simulation ({total_steps} steps)...")
        result = solver.solve()
        
        # Output results (field names match TransientResult - see transient_result.py).
        result_dict = {
            "command": "solve.transient",
            "status": "success",
            "physical_time": result.total_time,
            "time_steps": result.n_steps,
            "output_dir": output,
        }

        if json_output:
            click.echo(json.dumps(result_dict, indent=2))
        else:
            click.echo(f"\n{'='*60}")
            click.echo(f"Transient Simulation Complete")
            click.echo(f"{'='*60}")
            click.echo(f"Physical Time: {result.total_time:.6f}s")
            click.echo(f"Time Steps: {result.n_steps}")
            click.echo(f"Output Directory: {output}")
            click.echo(f"{'='*60}")

    except Exception as e:
        logger.error(f"Transient simulation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        if json_output:
            error_result = {
                "command": "solve.transient",
                "status": "error",
                "error": str(e)
            }
            click.echo(json.dumps(error_result, indent=2))
        raise click.ClickException(f"Transient simulation failed: {e}")


@solve.command()
@click.argument("checkpoint_file", type=click.Path(exists=True))
@click.option("--grid", "-g", "grid_file", type=click.Path(exists=True), default=None,
              help="Grid file path (required for resume)")
@click.option("--config", "-c", "config_file", type=click.Path(exists=True), default=None,
              help="Configuration file path")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Output directory (overrides config)")
@click.option("--max-iter", "-n", type=int, default=None,
              help="Total iterations to run (not additional)")
@click.option("--backend", "-b", type=click.Choice(["cpu", "gpu"]), default=None,
              help="Backend to use (overrides checkpoint backend)")
@click.option("--skip-quality-check", is_flag=True, default=False,
              help="Skip the mesh quality gate when --grid points at a surface .nas "
                   "that gets re-tetrahedralized (see `solve run --help` for the "
                   "rationale). No effect when --grid is a cached volume_mesh.pkl - "
                   "that mesh already solved successfully up to this checkpoint.")
@click.option("--json", "-j", "json_output", is_flag=True, help="Output as JSON")
def resume(
    checkpoint_file: str,
    grid_file: Optional[str],
    config_file: Optional[str],
    output: Optional[str],
    max_iter: Optional[int],
    backend: Optional[str],
    skip_quality_check: bool,
    json_output: bool
) -> None:
    """Resume simulation from checkpoint.
    
    Continue a previously interrupted simulation from a checkpoint file.
    
    Args:
        checkpoint_file: Path to checkpoint file (.h5)
        grid_file: Grid file path (required)
        config_file: Configuration file path (optional)
        output: Output directory (overrides config)
        max_iter: Total iterations to run (not additional)
        backend: Backend override ("cpu" or "gpu")
        skip_quality_check: Skip the mesh quality gate (only relevant when
            --grid is a surface .nas that gets re-tetrahedralized)
        json_output: Output as JSON

    Examples:
        # 使用网格文件恢复
        $ autoflowcfd solve resume checkpoint.h5 --grid mesh.nas
        
        # 使用配置文件并增加迭代次数
        $ autoflowcfd solve resume checkpoint.h5 --config config.yaml --max-iter 2000
        
        # 切换到 GPU 后端
        $ autoflowcfd solve resume checkpoint.h5 --grid mesh.nas --backend gpu
    """
    logger.info(f"正在从检查点恢复: {checkpoint_file}")
    
    try:
        import h5py
        from pathlib import Path
        from ..core.checkpoint import CheckpointManager
        from ..config.loader import load_config
        from ..grid.nas_io.parser_core import NASParser
        from ..core.solver_steady import FRSolver
        
        # 步骤 1：加载检查点元数据
        logger.info("\n[1/5] 正在加载检查点元数据...")
        with h5py.File(checkpoint_file, 'r') as f:
            iteration = int(f['metadata'].attrs['iteration'])
            original_backend = f['metadata'].attrs['backend']
            config_hash = f['metadata'].attrs['config_hash']
            
            # 如果需要，将字节解码为字符串
            if isinstance(original_backend, bytes):
                original_backend = original_backend.decode('utf-8')
        
        logger.info(f"✓ 检查点已加载:")
        logger.info(f"  - 上次迭代: {iteration}")
        logger.info(f"  - 原始后端: {original_backend}")
        logger.info(f"  - 配置哈希: {config_hash[:16]}...")
        
        # 步骤 2：确定目标后端
        target_backend = backend if backend else original_backend
        if backend and backend != original_backend:
            logger.warning(f"⚠ 后端覆盖: {original_backend} → {target_backend}")
        
        # 步骤 3：加载网格数据（必需）
        if not grid_file:
            raise ValueError(
                "恢复操作需要网格文件。请使用 --grid 选项指定。\n"
                "重要提示：对于体网格恢复，您必须提供与原始仿真中使用的相同"
                "体网格文件，而不是表面 NAS 文件。"
            )
        
        logger.info(f"\n[2/5] 正在加载网格数据...")
        
        # 检查 grid_file 是保存的体网格 (pkl) 还是表面网格 (nas)
        from pathlib import Path
        grid_path = Path(grid_file)
        
        if grid_path.suffix.lower() == '.pkl':
            # 加载保存的体网格
            logger.info(f"正在加载保存的体网格: {grid_file}")
            import pickle
            try:
                with open(grid_file, 'rb') as f:
                    grid_data = pickle.load(f)
                logger.success(f"✓ 体网格已加载: {grid_data.node_count} 节点, {grid_data.cell_count} 单元")
            except Exception as e:
                raise ValueError(f"从 {grid_file} 加载体网格失败: {e}")
        else:
            # 解析表面网格并生成体网格（不推荐用于恢复）
            logger.warning(f"⚠ 正在解析表面网格文件: {grid_file}")
            logger.warning("  这将重新生成体网格，可能与原始网格不同！")
            logger.warning("  为了准确恢复，请改用保存的 volume_mesh.pkl 文件。")
            
            parser = NASParser(grid_file)
            grid_data = parser.parse(generate_volume_mesh=True)
            logger.info(f"✓ 网格已生成: {grid_data.node_count} 节点, {grid_data.cell_count} 单元")

            if not skip_quality_check:
                from ..grid import MeshQualityValidator
                logger.info("在恢复前验证体网格质量...")
                quality_report = MeshQualityValidator().validate_volume_mesh(grid_data)
                if quality_report.passed:
                    logger.info(f"\n{quality_report.summary()}")
                else:
                    logger.error(f"\n{quality_report.summary()}")
                    raise click.ClickException(
                        "体网格质量检查失败（见上方报告）- 这个"
                        "新生成的网格与检查点实际求解的网格不同（见上方关于使用 "
                        "volume_mesh.pkl 的警告），恢复时可能会发散。"
                        "如果仍要恢复，请传递 --skip-quality-check。"
                    )
        
        # 步骤 4：加载或创建配置
        logger.info(f"\n[3/5] 正在加载配置...")
        if config_file:
            logger.info(f"  从以下位置加载: {config_file}")
            config = load_config(config_file)
        else:
            from ..config.solver_config import SteadyConfig
            logger.warning("  未提供配置文件，使用默认值")
            config = SteadyConfig()
        
        # 如果指定了输出目录则覆盖
        if output:
            config.output_dir = output
            logger.info(f"  输出目录: {output}")
        
        # 如果指定了 max_iter 则覆盖
        if max_iter:
            config.max_iter = max_iter
            logger.info(f"  最大迭代次数: {max_iter}")
        
        # 设置后端
        if target_backend:
            from ..config.solver_config import BackendType
            config.backend = BackendType(target_backend.lower())
            logger.info(f"  后端: {target_backend}")
        
        # 步骤 5：创建求解器并加载检查点
        logger.info(f"\n[4/5] 正在创建求解器并加载检查点...")
        solver = FRSolver(grid_data, config)
        
        solution, history, loaded_iteration, metadata = solver.checkpoint_manager.load(
            checkpoint_file,
            target_backend=target_backend
        )
        
        # 验证网格大小与检查点解的形状匹配
        expected_cells = solution.shape[0]
        if grid_data.cell_count != expected_cells:
            raise ValueError(
                f"网格单元数不匹配！\n"
                f"  检查点解期望 {expected_cells} 个单元\n"
                f"  当前网格有 {grid_data.cell_count} 个单元\n"
                f"  请提供与原始仿真中使用的相同体网格文件。"
            )
        
        logger.info(f"✓ 检查点已恢复:")
        logger.info(f"  - 迭代: {loaded_iteration}")
        logger.info(f"  - 解形状: {solution.shape}")
        logger.info(f"  - 历史条目: {len(history.get('iterations', []))}")
        logger.info(f"✓ 网格已验证: {grid_data.cell_count} 单元与检查点匹配")
        
        # 设置初始解
        solver.solution = solution
        
        # 恢复收敛历史
        if history:
            solver.convergence_history = history
            logger.info(f"  - 收敛历史已恢复")
        
        # 步骤 6：继续求解
        logger.info(
            f"\n[5/5] 正在从迭代 {loaded_iteration} 恢复到迭代 {config.max_iter}..."
        )
        logger.info("="*60)

        result = solver.solve(max_iter=config.max_iter, start_iteration=loaded_iteration)

        # 输出结果（字段名与 SteadyResult 匹配 - 参见 solver_steady.py）。
        final_cd = result.cd_history[-1] if result.cd_history else 0.0
        final_cl = result.cl_history[-1] if result.cl_history else 0.0
        logger.info("\n" + "="*60)
        logger.info("✓ 仿真成功完成！")
        logger.info("="*60)
        logger.info(f"最终迭代: {result.iterations}")
        logger.info(f"最终残差: {result.final_residual:.6e}")
        logger.info(f"最终 Cd: {final_cd:.6f}")
        logger.info(f"最终 Cl: {final_cl:.6f}")
        logger.info(f"输出目录: {config.output_dir}")
        logger.info("="*60)

        if json_output:
            result_dict = {
                "command": "solve.resume",
                "status": "success",
                "final_iteration": result.iterations,
                "final_residual": float(result.final_residual),
                "final_Cd": float(final_cd),
                "final_Cl": float(final_cl),
                "output_dir": str(config.output_dir),
            }
            click.echo(json.dumps(result_dict, indent=2))
    
    except ValueError as e:
        logger.error(f"配置错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        error_result = {
            "command": "solve.resume",
            "status": "error",
            "error": str(e)
        }
        if json_output:
            click.echo(json.dumps(error_result, indent=2))
        raise click.ClickException(str(e))
    
    except ImportError as e:
        logger.error(f"缺少依赖: {e}")
        error_result = {
            "command": "solve.resume",
            "status": "error",
            "error": f"缺少依赖: {str(e)}"
        }
        if json_output:
            click.echo(json.dumps(error_result, indent=2))
        raise click.ClickException(f"恢复失败: {e}")
    
    except Exception as e:
        logger.error(f"恢复失败: {e}")
        import traceback
        logger.error(traceback.format_exc())

        if json_output:
            error_result = {
                "command": "solve.resume",
                "status": "error",
                "error": str(e)
            }
            click.echo(json.dumps(error_result, indent=2))
        raise click.ClickException(f"恢复失败: {e}")


@solve.command()
@click.argument("case_dir", type=click.Path(exists=True))
@click.option("--json", "-j", "json_output", is_flag=True, help="以 JSON 格式输出")
def status(case_dir: str, json_output: bool) -> None:
    """检查求解器状态。
    
    显示正在运行或已完成的仿真的当前状态。
    
    Args:
        case_dir: 案例目录路径
        json_output: 以 JSON 格式输出
    
    Examples:
        # 检查状态
        $ autoflowcfd solve status results/
    """
    logger.info(f"正在检查案例状态: {case_dir}")
    
    try:
        # TODO: 实现状态检查
        logger.warning("状态检查功能正在开发中")
        
        result_dict = {
            "command": "solve.status",
            "status": "pending",
            "message": "状态检查尚未完全实现",
        }
        
        if json_output:
            click.echo(json.dumps(result_dict, indent=2))
        else:
            click.echo("⚠ 状态检查功能将在下次更新中提供")
    
    except Exception as e:
        logger.error(f"状态检查失败: {e}")
        if json_output:
            error_result = {
                "command": "solve.status",
                "status": "error",
                "error": str(e)
            }
            click.echo(json.dumps(error_result, indent=2))
        raise click.ClickException(f"状态检查失败: {e}")
