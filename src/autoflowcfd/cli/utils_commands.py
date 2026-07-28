"""Utility subcommands.

This module provides CLI utility commands for AutoFlowCFD.

Commands:
    - version: Display version information
    - doctor: Environment diagnostics
    - benchmark: Performance benchmarking

Example:
    $ autoflowcfd utils version
    $ autoflowcfd utils doctor
    $ autoflowcfd utils benchmark --grid model.nas --backend cpu
"""

import click
import json
import sys
from pathlib import Path
from loguru import logger


@click.group()
def utils() -> None:
    """Utility commands.
    
    System utilities and diagnostic tools.
    
    Examples:
        # Check version
        $ autoflowcfd utils version
        
        # Run diagnostics
        $ autoflowcfd utils doctor
        
        # Performance benchmark
        $ autoflowcfd utils benchmark --grid model.nas
    """
    pass


@utils.command()
@click.option("--json", "-j", "json_output", is_flag=True, help="Output as JSON")
def version(json_output: bool) -> None:
    """Display version information.
    
    Show AutoFlowCFD version and build information.
    
    Args:
        json_output: Output as JSON
    
    Examples:
        # Basic version
        $ autoflowcfd utils version
        
        # Detailed JSON output
        $ autoflowcfd utils version --json
    """
    from autoflowcfd import __version__
    
    try:
        import platform
        import sys
        
        version_info = {
            "autoflowcfd": __version__,
            "python": sys.version,
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
        }
        
        if json_output:
            click.echo(json.dumps(version_info, indent=2))
        else:
            click.echo(f"\nAutoFlowCFD v{__version__}")
            click.echo(f"{'='*40}")
            click.echo(f"Python:   {sys.version.split()[0]}")
            click.echo(f"Platform: {platform.system()} {platform.release()}")
            click.echo(f"Machine:  {platform.machine()}")
            click.echo(f"{'='*40}")
    
    except Exception as e:
        logger.error(f"Failed to get version info: {e}")
        raise click.ClickException(str(e))


@utils.command()
@click.option("--json", "-j", "json_output", is_flag=True, help="Output as JSON")
def doctor(json_output: bool) -> None:
    """Run environment diagnostics.
    
    Check system environment for potential issues and missing dependencies.
    
    Args:
        json_output: Output as JSON
    
    Examples:
        # Run diagnostics
        $ autoflowcfd utils doctor
        
        # JSON output
        $ autoflowcfd utils doctor --json
    """
    logger.info("Running environment diagnostics...")
    
    try:
        issues = []
        warnings = []
        info = {}
        
        # Check Python version
        python_version = sys.version_info
        info['python_version'] = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
        
        if python_version < (3, 8):
            issues.append(f"Python version {python_version} is too old. Requires Python 3.8+")
        
        # Check required packages
        required_packages = {
            'numpy': 'NumPy',
            'click': 'Click',
            'loguru': 'Loguru',
            'pyyaml': 'PyYAML',
        }
        
        installed_packages = {}
        for pkg_name, pkg_display in required_packages.items():
            try:
                if pkg_name == 'pyyaml':
                    import yaml
                    installed_packages[pkg_display] = yaml.__version__
                else:
                    pkg = __import__(pkg_name)
                    installed_packages[pkg_display] = getattr(pkg, '__version__', 'unknown')
            except ImportError:
                issues.append(f"Missing required package: {pkg_display}")
        
        info['installed_packages'] = installed_packages
        
        # Check optional packages
        optional_packages = {
            'cupy': 'CuPy (GPU support)',
            'numba': 'Numba (CPU acceleration)',
            'h5py': 'h5py (HDF5 support)',
        }
        
        optional_installed = {}
        for pkg_name, pkg_display in optional_packages.items():
            try:
                pkg = __import__(pkg_name)
                optional_installed[pkg_display] = getattr(pkg, '__version__', 'available')
            except ImportError:
                warnings.append(f"Optional package not installed: {pkg_display}")
        
        info['optional_packages'] = optional_installed
        
        # Check GPU availability
        gpu_available = False
        try:
            import cupy as cp
            # Try to create a simple array on GPU
            test_array = cp.array([1, 2, 3])
            gpu_available = True
            info['gpu_status'] = 'available'
        except Exception:
            info['gpu_status'] = 'not available'
            warnings.append("GPU (CUDA) not available. Install CuPy and CUDA toolkit for GPU acceleration.")
        
        # Check CPU cores
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        info['cpu_cores'] = cpu_count
        
        # Compile results
        status = "healthy" if not issues else "unhealthy"
        
        result = {
            "command": "utils.doctor",
            "status": status,
            "issues": issues,
            "warnings": warnings,
            "info": info,
        }
        
        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\nEnvironment Diagnostics")
            click.echo(f"{'='*60}")
            
            if issues:
                click.echo(f"Status: ❌ UNHEALTHY")
                click.echo(f"\nIssues ({len(issues)}):")
                for issue in issues:
                    click.echo(f"  ❌ {issue}")
            else:
                click.echo(f"Status: ✅ HEALTHY")
            
            if warnings:
                click.echo(f"\nWarnings ({len(warnings)}):")
                for warning in warnings:
                    click.echo(f"  ⚠️  {warning}")
            
            click.echo(f"\nSystem Information:")
            click.echo(f"  Python:      {info.get('python_version', 'unknown')}")
            click.echo(f"  CPU Cores:   {info.get('cpu_cores', 'unknown')}")
            click.echo(f"  GPU Status:  {info.get('gpu_status', 'unknown')}")
            
            if info.get('installed_packages'):
                click.echo(f"\nInstalled Packages:")
                for pkg, ver in info['installed_packages'].items():
                    click.echo(f"  ✓ {pkg:<20} {ver}")
            
            if info.get('optional_packages'):
                click.echo(f"\nOptional Packages:")
                for pkg, ver in info['optional_packages'].items():
                    click.echo(f"  {'✓' if ver != 'not installed' else '✗'} {pkg:<20} {ver}")
            
            click.echo(f"{'='*60}")
            
            if not issues and not warnings:
                click.echo("\n✅ Everything looks good!")
            elif not issues:
                click.echo(f"\n⚠️  {len(warnings)} warning(s) found. System is functional but could be improved.")
    
    except Exception as e:
        logger.error(f"Diagnostics failed: {e}")
        raise click.ClickException(f"Diagnostics failed: {e}")


@utils.command()
@click.argument("grid_file", type=click.Path(exists=True))
@click.option("--backend", "-b", type=click.Choice(["cpu", "gpu"]),
              default="cpu", help="Backend to benchmark")
@click.option("--iterations", "-n", type=int, default=100,
              help="Number of test iterations")
@click.option("--order", "-p", type=click.IntRange(1, 3), default=2,
              help="FR order")
@click.option("--json", "-j", "json_output", is_flag=True, help="Output as JSON")
def benchmark(
    grid_file: str,
    backend: str,
    iterations: int,
    order: int,
    json_output: bool
) -> None:
    """Run performance benchmark.
    
    Measure computation speed and memory usage for specified grid and backend.
    
    Args:
        grid_file: Path to .nas grid file
        backend: Backend to benchmark (cpu/gpu)
        iterations: Number of iterations
        order: FR order
        json_output: Output as JSON
    
    Examples:
        # CPU benchmark
        $ autoflowcfd utils benchmark sedan.nas --backend cpu
        
        # GPU benchmark
        $ autoflowcfd utils benchmark sedan.nas --backend gpu -n 200
    """
    logger.info(f"Running benchmark: grid={grid_file}, backend={backend}")
    
    try:
        # TODO: Implement actual benchmarking
        # This requires loading grid, setting up solver, and running iterations
        
        logger.warning("Benchmark functionality is under development")
        
        result = {
            "command": "utils.benchmark",
            "status": "pending",
            "message": "Benchmark feature coming soon",
            "grid_file": grid_file,
            "backend": backend,
            "iterations": iterations,
        }
        
        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo("\nPerformance Benchmark")
            click.echo(f"{'='*60}")
            click.echo(f"Grid File:  {grid_file}")
            click.echo(f"Backend:    {backend.upper()}")
            click.echo(f"Iterations: {iterations}")
            click.echo(f"\n⚠️  Benchmark feature under development")
            click.echo(f"{'='*60}")
    
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise click.ClickException(f"Benchmark failed: {e}")
