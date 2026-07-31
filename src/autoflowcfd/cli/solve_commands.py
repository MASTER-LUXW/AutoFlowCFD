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
@click.option("--max-layers", type=int, default=None,
              help="Maximum boundary layer layers (overrides config)")
@click.option("--min-cell-size", type=float, default=None,
              help="Minimum cell size in meters (overrides config)")
@click.option("--max-cell-size", type=float, default=None,
              help="Max core-region cell size in meters, graded outward from the BL's "
                   "near-wall size (overrides config); unset means no cap")
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
    max_layers: Optional[int],
    min_cell_size: Optional[float],
    max_cell_size: Optional[float],
    json_output: bool
) -> None:
    """Run steady-state RANS simulation.
    
    Solve steady-state Reynolds-Averaged Navier-Stokes equations
    using Flux Reconstruction method.
    
    Args:
        input_file: Path to .nas grid file
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
        json_output: Output as JSON
    
    Examples:
        # Basic steady RANS
        $ autoflowcfd solve run sedan.nas
        
        # GPU with 3rd order
        $ autoflowcfd solve run sedan.nas --backend gpu --order 3
        
        # With config file
        $ autoflowcfd solve run sedan.nas -c simulation.yaml
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
            )

        # --max-layers/--min-cell-size are CLI-only overrides: when passed,
        # they win over whatever steady_config carries (defaults, or values
        # loaded from --config yaml).
        if max_layers is not None:
            steady_config.max_layers = max_layers
        if min_cell_size is not None:
            steady_config.min_cell_size = min_cell_size
        if max_cell_size is not None:
            steady_config.max_cell_size = max_cell_size

        logger.info(f"Configuration: backend={steady_config.backend}, "
                   f"order={steady_config.order}, turbulence={steady_config.turbulence}")

        # Parse grid and generate volume mesh
        logger.info("Parsing grid file...")
        parser = NASParser(input_file)

        logger.info(
            f"Using BL parameters: growth_rate={steady_config.growth_rate}, "
            f"max_layers={steady_config.max_layers}, "
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
                'max_layers': steady_config.max_layers,
                'min_cell_size': steady_config.min_cell_size,
                'target_cells': steady_config.target_cells,
                'max_cell_size': steady_config.max_cell_size,
            }
        )

        logger.info(f"Grid loaded: {grid_data.node_count} nodes, "
                   f"{grid_data.cell_count} cells")
        
        # Save volume mesh for future resume operations
        import pickle
        from pathlib import Path
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
    json_output: bool
) -> None:
    """Run transient LES/DES simulation.
    
    Solve unsteady flow using Large Eddy Simulation or Detached
    Eddy Simulation.
    
    Args:
        input_file: Path to .nas grid file
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
        json_output: Output as JSON
    
    Examples:
        # Basic DES simulation
        $ autoflowcfd solve transient sedan.nas --physical-time 0.3
        
        # DDES with RK2
        $ autoflowcfd solve transient sedan.nas --mode ddes \
          --time-integration rk2 --physical-time 0.5
        
        # Initialize from steady solution
        $ autoflowcfd solve transient sedan.nas --physical-time 0.3 \
          --init-from steady_results/checkpoint.h5
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
            )

        # Calculate total steps from the EFFECTIVE config values (not the
        # raw CLI variables), which may differ if --config set its own
        # dt/total_time and the CLI didn't explicitly override them.
        total_steps = int(transient_config.total_time / transient_config.dt)
        logger.info(f"Total time steps: {total_steps}")

        # Parse grid and generate volume mesh (same conservative BL defaults
        # as `solve run` - see SteadyConfig field docs for the rationale).
        logger.info("Parsing grid file...")
        parser = NASParser(input_file)
        grid_data = parser.parse(
            generate_volume_mesh=True,
            volume_mesh_params={
                'growth_rate': transient_config.growth_rate,
                'max_layers': transient_config.max_layers,
                'min_cell_size': transient_config.min_cell_size,
                'target_cells': transient_config.target_cells,
                'max_cell_size': transient_config.max_cell_size,
            }
        )
        logger.info(f"Grid loaded: {grid_data.node_count} nodes, {grid_data.cell_count} cells")

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
@click.option("--json", "-j", "json_output", is_flag=True, help="Output as JSON")
def resume(
    checkpoint_file: str,
    grid_file: Optional[str],
    config_file: Optional[str],
    output: Optional[str],
    max_iter: Optional[int],
    backend: Optional[str],
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
        json_output: Output as JSON
    
    Examples:
        # Resume with grid file
        $ autoflowcfd solve resume checkpoint.h5 --grid mesh.nas
        
        # With config file and more iterations
        $ autoflowcfd solve resume checkpoint.h5 --config config.yaml --max-iter 2000
        
        # Switch to GPU backend
        $ autoflowcfd solve resume checkpoint.h5 --grid mesh.nas --backend gpu
    """
    logger.info(f"Resuming from checkpoint: {checkpoint_file}")
    
    try:
        import h5py
        from pathlib import Path
        from ..core.checkpoint import CheckpointManager
        from ..config.loader import load_config
        from ..grid.parser_core import NASParser
        from ..core.solver_steady import FRSolver
        
        # Step 1: Load checkpoint metadata
        logger.info("\n[1/5] Loading checkpoint metadata...")
        with h5py.File(checkpoint_file, 'r') as f:
            iteration = int(f['metadata'].attrs['iteration'])
            original_backend = f['metadata'].attrs['backend']
            config_hash = f['metadata'].attrs['config_hash']
            
            # Decode bytes to string if necessary
            if isinstance(original_backend, bytes):
                original_backend = original_backend.decode('utf-8')
        
        logger.info(f"✓ Checkpoint loaded:")
        logger.info(f"  - Last iteration: {iteration}")
        logger.info(f"  - Original backend: {original_backend}")
        logger.info(f"  - Config hash: {config_hash[:16]}...")
        
        # Step 2: Determine target backend
        target_backend = backend if backend else original_backend
        if backend and backend != original_backend:
            logger.warning(f"⚠ Backend override: {original_backend} → {target_backend}")
        
        # Step 3: Load grid data (required)
        if not grid_file:
            raise ValueError(
                "Grid file is required for resume operation. "
                "Please specify with --grid option.\n"
                "IMPORTANT: For volume mesh resume, you must provide the SAME "
                "volume mesh file used in the original simulation, NOT the "
                "surface NAS file."
            )
        
        logger.info(f"\n[2/5] Loading grid data...")
        
        # Check if grid_file is a saved volume mesh (pkl) or surface mesh (nas)
        from pathlib import Path
        grid_path = Path(grid_file)
        
        if grid_path.suffix.lower() == '.pkl':
            # Load saved volume mesh
            logger.info(f"Loading saved volume mesh: {grid_file}")
            import pickle
            try:
                with open(grid_file, 'rb') as f:
                    grid_data = pickle.load(f)
                logger.success(f"✓ Volume mesh loaded: {grid_data.node_count} nodes, {grid_data.cell_count} cells")
            except Exception as e:
                raise ValueError(f"Failed to load volume mesh from {grid_file}: {e}")
        else:
            # Parse surface mesh and generate volume mesh (NOT recommended for resume)
            logger.warning(f"⚠ Parsing surface mesh file: {grid_file}")
            logger.warning("  This will RE-GENERATE the volume mesh, which may differ from the original!")
            logger.warning("  For accurate resume, use the saved volume_mesh.pkl file instead.")
            
            parser = NASParser(grid_file)
            grid_data = parser.parse(generate_volume_mesh=True)
            logger.info(f"✓ Grid generated: {grid_data.node_count} nodes, {grid_data.cell_count} cells")
        
        # Step 4: Load or create configuration
        logger.info(f"\n[3/5] Loading configuration...")
        if config_file:
            logger.info(f"  Loading from: {config_file}")
            config = load_config(config_file)
        else:
            from ..config.solver_config import SteadyConfig
            logger.warning("  No config file provided, using defaults")
            config = SteadyConfig()
        
        # Override output directory if specified
        if output:
            config.output_dir = output
            logger.info(f"  Output directory: {output}")
        
        # Override max_iter if specified
        if max_iter:
            config.max_iter = max_iter
            logger.info(f"  Max iterations: {max_iter}")
        
        # Set backend
        if target_backend:
            from ..config.solver_config import BackendType
            config.backend = BackendType(target_backend.lower())
            logger.info(f"  Backend: {target_backend}")
        
        # Step 5: Create solver and load checkpoint
        logger.info(f"\n[4/5] Creating solver and loading checkpoint...")
        solver = FRSolver(grid_data, config)
        
        solution, history, loaded_iteration, metadata = solver.checkpoint_manager.load(
            checkpoint_file,
            target_backend=target_backend
        )
        
        # Validate grid size matches checkpoint solution shape
        expected_cells = solution.shape[0]
        if grid_data.cell_count != expected_cells:
            raise ValueError(
                f"Grid cell count mismatch!\n"
                f"  Checkpoint solution expects {expected_cells} cells\n"
                f"  Current grid has {grid_data.cell_count} cells\n"
                f"  Please provide the SAME volume mesh file used in the original simulation."
            )
        
        logger.info(f"✓ Checkpoint restored:")
        logger.info(f"  - Iteration: {loaded_iteration}")
        logger.info(f"  - Solution shape: {solution.shape}")
        logger.info(f"  - History entries: {len(history.get('iterations', []))}")
        logger.info(f"✓ Grid validated: {grid_data.cell_count} cells matches checkpoint")
        
        # Set initial solution
        solver.solution = solution
        
        # Restore convergence history
        if history:
            solver.convergence_history = history
            logger.info(f"  - Convergence history restored")
        
        # Step 6: Continue solving
        logger.info(
            f"\n[5/5] Resuming simulation from iteration {loaded_iteration} "
            f"to iteration {config.max_iter}..."
        )
        logger.info("="*60)

        result = solver.solve(max_iter=config.max_iter, start_iteration=loaded_iteration)

        # Output results (field names match SteadyResult - see solver_steady.py).
        final_cd = result.cd_history[-1] if result.cd_history else 0.0
        final_cl = result.cl_history[-1] if result.cl_history else 0.0
        logger.info("\n" + "="*60)
        logger.info("✓ Simulation completed successfully!")
        logger.info("="*60)
        logger.info(f"Final iteration: {result.iterations}")
        logger.info(f"Final residual: {result.final_residual:.6e}")
        logger.info(f"Final Cd: {final_cd:.6f}")
        logger.info(f"Final Cl: {final_cl:.6f}")
        logger.info(f"Output directory: {config.output_dir}")
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
        logger.error(f"Configuration error: {e}")
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
        logger.error(f"Missing dependency: {e}")
        error_result = {
            "command": "solve.resume",
            "status": "error",
            "error": f"Missing dependency: {str(e)}"
        }
        if json_output:
            click.echo(json.dumps(error_result, indent=2))
        raise click.ClickException(f"Resume failed: {e}")
    
    except Exception as e:
        logger.error(f"Resume failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

        if json_output:
            error_result = {
                "command": "solve.resume",
                "status": "error",
                "error": str(e)
            }
            click.echo(json.dumps(error_result, indent=2))
        raise click.ClickException(f"Resume failed: {e}")


@solve.command()
@click.argument("case_dir", type=click.Path(exists=True))
@click.option("--json", "-j", "json_output", is_flag=True, help="Output as JSON")
def status(case_dir: str, json_output: bool) -> None:
    """Check solver status.
    
    Display current status of a running or completed simulation.
    
    Args:
        case_dir: Case directory path
        json_output: Output as JSON
    
    Examples:
        # Check status
        $ autoflowcfd solve status results/
    """
    logger.info(f"Checking status of case: {case_dir}")
    
    try:
        # TODO: Implement status checking
        logger.warning("Status check functionality is under development")
        
        result_dict = {
            "command": "solve.status",
            "status": "pending",
            "message": "Status check not fully implemented",
        }
        
        if json_output:
            click.echo(json.dumps(result_dict, indent=2))
        else:
            click.echo("⚠ Status check feature coming in next update")
    
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        if json_output:
            error_result = {
                "command": "solve.status",
                "status": "error",
                "error": str(e)
            }
            click.echo(json.dumps(error_result, indent=2))
        raise click.ClickException(f"Status check failed: {e}")
