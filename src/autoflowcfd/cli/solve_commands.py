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
            
            # Override with CLI options
            steady_config.backend = BackendType(backend)
            steady_config.order = order
            steady_config.turbulence = TurbulenceModel(turbulence)
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
            }
        )
        
        logger.info(f"Grid loaded: {grid_data.node_count} nodes, "
                   f"{grid_data.cell_count} cells")
        
        # Create solver
        logger.info("Initializing solver...")
        solver = FRSolver(grid_data, steady_config)
        
        # Run simulation
        logger.info(f"Starting simulation (max_iter={max_iter})...")
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
    from autoflowcfd.config import TransientConfig, BackendType, TurbulenceModel, TimeIntegrationScheme
    from autoflowcfd.grid import NASParser
    from autoflowcfd.core import TransientSolver
    
    logger.info(f"Starting transient simulation")
    logger.info(f"Physical time: {physical_time}s, dt: {dt}s")
    
    try:
        # Calculate total steps
        total_steps = int(physical_time / dt)
        logger.info(f"Total time steps: {total_steps}")
        
        # Load or create configuration
        if config:
            loader = ConfigLoader()
            transient_config = loader.load(config)
        else:
            # Map mode to turbulence model
            turbulence_map = {
                'des': TurbulenceModel.DES,
                'ddes': TurbulenceModel.DDES,
                'les': TurbulenceModel.LES,
            }
            
            transient_config = TransientConfig(
                backend=BackendType(backend),
                order=order,
                turbulence=turbulence_map[mode],
                time_integration=TimeIntegrationScheme(time_integration),
                dt=dt,
                total_time=physical_time,
                output_dir=output,
                sample_interval=sample_interval,
            )
        
        # Parse grid
        logger.info("Parsing grid file...")
        parser = NASParser(input_file)
        grid_data = parser.parse()
        
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
        
        # Output results
        result_dict = {
            "command": "solve.transient",
            "status": "success",
            "physical_time": result.physical_time,
            "time_steps": result.time_steps,
            "output_dir": output,
        }
        
        if json_output:
            click.echo(json.dumps(result_dict, indent=2))
        else:
            click.echo(f"\n{'='*60}")
            click.echo(f"Transient Simulation Complete")
            click.echo(f"{'='*60}")
            click.echo(f"Physical Time: {result.physical_time:.6f}s")
            click.echo(f"Time Steps: {result.time_steps}")
            click.echo(f"Output Directory: {output}")
            click.echo(f"{'='*60}")
    
    except Exception as e:
        logger.error(f"Transient simulation failed: {e}")
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
@click.option("--output", "-o", type=click.Path(), default="results/",
              help="Output directory")
@click.option("--max-iter", "-n", default=5000,
              help="Additional iterations")
@click.option("--json", "-j", "json_output", is_flag=True, help="Output as JSON")
def resume(
    checkpoint_file: str,
    output: str,
    max_iter: int,
    json_output: bool
) -> None:
    """Resume simulation from checkpoint.
    
    Continue a previously interrupted simulation from a checkpoint file.
    
    Args:
        checkpoint_file: Path to checkpoint file (.h5)
        output: Output directory
        max_iter: Additional iterations to run
        json_output: Output as JSON
    
    Examples:
        # Resume from checkpoint
        $ autoflowcfd solve resume results/checkpoint_1000.h5
        
        # With more iterations
        $ autoflowcfd solve resume checkpoint.h5 --max-iter 2000
    """
    logger.info(f"Resuming from checkpoint: {checkpoint_file}")
    
    try:
        # TODO: Implement checkpoint loading and resumption
        logger.warning("Checkpoint resume functionality is under development")
        
        result_dict = {
            "command": "solve.resume",
            "status": "pending",
            "message": "Checkpoint resume not fully implemented",
        }
        
        if json_output:
            click.echo(json.dumps(result_dict, indent=2))
        else:
            click.echo("⚠ Checkpoint resume feature coming in next update")
    
    except Exception as e:
        logger.error(f"Resume failed: {e}")
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
