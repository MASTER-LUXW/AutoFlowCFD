"""Command-line interface for AutoFlowCFD.

This module provides the main CLI entry point using Click framework.
It supports subcommands for grid processing, solving, postprocessing,
configuration management, and utilities.

Example:
    $ autoflowcfd --version
    AutoFlowCFD v0.1.0
    
    $ autoflowcfd grid parse --help
    Usage: autoflowcfd grid parse [OPTIONS] INPUT_FILE
    
    Parse ANSA .nas grid file.
    
    $ autoflowcfd solve steady --help
    Usage: autoflowcfd solve steady [OPTIONS] INPUT_FILE

    Run steady-state RANS simulation.
"""

import sys
import os

# 修复Windows控制台中文乱码问题
if sys.platform == 'win32':
    # 设置标准输出编码为UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
    # 设置环境变量
    os.environ['PYTHONIOENCODING'] = 'utf-8'

import click
from loguru import logger

from .. import __version__
from .grid_commands import grid
from .solve_commands import solve
from .post_commands import post
from .config_commands import config
from .utils_commands import utils

# Log messages and click.echo() calls throughout this CLI use Unicode symbols
# (checkmarks, °, ³, ...). On Windows, stdout/stderr default to the active
# console codepage (e.g. GBK/936 on Chinese locales) rather than UTF-8, so
# those characters raise UnicodeEncodeError and abort the command. Force
# UTF-8 with a safe fallback so output never crashes the CLI regardless of
# the host console's codepage.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


@click.group()
@click.version_option(version=__version__, prog_name="AutoFlowCFD")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def cli(verbose: bool) -> None:
    """AutoFlowCFD - High-performance CFD for automotive aerodynamics.
    
    AutoFlowCFD is an open-source Computational Fluid Dynamics software
    specialized for automotive external flow field simulation.
    
    Features:
        - FR (Flux Reconstruction) high-order discretization
        - SST k-ω, DES, DDES turbulence models
        - CPU (Numba) and GPU (CUDA) backends
        - ANSA .nas grid file support
        - Comprehensive post-processing tools
    
    Command Groups:
        grid     Grid processing (parse, validate, info, convert, generate-volume, import-volume)
        solve    Solver commands (steady, transient, resume, status)
        post     Post-processing (coefficients, export-vtk, report, etc.)
        config   Configuration management (init, show, validate)
        utils    Utilities (version, doctor, benchmark)

    Examples:
        # Parse grid file
        $ autoflowcfd grid parse sedan.nas

        # Generate a volume mesh from the surface mesh (required before solving)
        $ autoflowcfd grid generate-volume sedan.nas -o sedan_volume.nas

        # Run steady-state simulation (input must be a .pkl volume mesh)
        $ autoflowcfd solve steady sedan_volume.pkl --backend gpu --order 3

        # Run transient DES simulation
        $ autoflowcfd solve transient sedan_volume.pkl --physical-time 0.3
        
        # Calculate aerodynamic coefficients
        $ autoflowcfd post coefficients --case results/
        
        # Generate config template
        $ autoflowcfd config init --template steady
        
        # Check environment
        $ autoflowcfd utils doctor
    
    For more information on a specific command, use:
        $ autoflowcfd <command> --help
        $ autoflowcfd <command> <subcommand> --help
    """
    if verbose:
        logger.remove()
        logger.add(
            lambda msg: click.echo(msg),
            level="DEBUG",
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        )
    else:
        logger.remove()
        logger.add(
            lambda msg: click.echo(msg),
            level="INFO",
            format="<level>{message}</level>",
        )


# Register subcommand groups
cli.add_command(grid)
cli.add_command(solve)
cli.add_command(post)
cli.add_command(config)
cli.add_command(utils)


if __name__ == "__main__":
    cli()
