"""Grid processing subcommands.

This module provides CLI commands for grid file parsing, validation,
and information display.

Commands:
    - parse: Parse .nas grid files
    - validate: Validate grid quality
    - info: Display grid statistics
    - convert: Convert grid formats (v1.0)

Example:
    $ autoflowcfd grid parse model.nas
    $ autoflowcfd grid validate model.nas --report report.json
    $ autoflowcfd grid info model.nas --json
"""

import click
import json
from pathlib import Path
from loguru import logger


@click.group()
def grid() -> None:
    """Grid processing commands.
    
    Parse, validate, and analyze ANSA .nas grid files.
    
    Examples:
        # Parse grid file
        $ autoflowcfd grid parse sedan.nas
        
        # Validate grid quality
        $ autoflowcfd grid validate sedan.nas
        
        # Show grid info
        $ autoflowcfd grid info sedan.nas
    """
    pass


@grid.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default="grid_info.json",
              help="Output JSON file path")
@click.option("--encoding", default="UTF-8", help="File encoding")
@click.option("--streaming", is_flag=True, help="Enable streaming parse for large files")
@click.option("--skip-validation", is_flag=True, help="Skip grid quality validation")
@click.option("--json", "-j", "json_output", is_flag=True, help="Output as JSON")
def parse(
    input_file: str,
    output: str,
    encoding: str,
    streaming: bool,
    skip_validation: bool,
    json_output: bool
) -> None:
    """Parse ANSA .nas grid file.
    
    Extract nodes, cells, and boundary information from NAS format.
    
    Args:
        input_file: Path to .nas grid file
        output: Output JSON file path
        encoding: File encoding
        streaming: Enable streaming mode for large files
        skip_validation: Skip quality validation
        json_output: Output result as JSON
    
    Examples:
        # Basic parsing
        $ autoflowcfd grid parse sedan.nas
        
        # With custom output
        $ autoflowcfd grid parse sedan.nas -o output/grid.json
        
        # Streaming mode for large files
        $ autoflowcfd grid parse large.nas --streaming
    """
    from autoflowcfd.grid import NASParser
    
    logger.info(f"Parsing grid file: {input_file}")
    
    try:
        # Parse grid
        parser = NASParser(input_file, encoding=encoding)
        
        if streaming:
            logger.info("Using streaming mode for large file")
            # TODO: Implement streaming parsing
            grid_data = parser.parse_streaming()
        else:
            grid_data = parser.parse()
        
        # Get grid statistics
        result = {
            "node_count": grid_data.node_count,
            "cell_count": grid_data.cell_count,
            "boundary_groups": {},
        }
        
        # Get boundary information
        if hasattr(grid_data, 'boundaries'):
            for name in grid_data.boundaries.boundary_names:
                nodes = grid_data.boundaries.get_node_indices(name)
                result["boundary_groups"][name] = len(nodes)
        
        # Quality report (if not skipped)
        if not skip_validation:
            from autoflowcfd.grid import GridValidator
            validator = GridValidator(grid_data)
            quality_report = validator.validate()
            result["quality_report"] = quality_report
        
        # Output
        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            # Save to file
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"Grid info saved to {output_path}")
            click.echo(f"✓ Parsed {result['node_count']} nodes, {result['cell_count']} cells")
            click.echo(f"✓ Boundaries: {len(result['boundary_groups'])} groups")
            click.echo(f"✓ Output saved to {output}")
    
    except Exception as e:
        logger.error(f"Failed to parse grid: {e}")
        if json_output:
            error_result = {
                "command": "grid.parse",
                "status": "error",
                "error": str(e)
            }
            click.echo(json.dumps(error_result, indent=2))
        raise click.ClickException(f"Grid parsing failed: {e}")


@grid.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--report", "-r", type=click.Path(), default="quality_report.json",
              help="Quality report output file")
@click.option("--threshold-aspect-ratio", type=float, default=1000.0,
              help="Aspect ratio threshold")
@click.option("--threshold-area", type=float, default=1e-12,
              help="Minimum cell area threshold (m²)")
@click.option("--fix-duplicates", is_flag=True, help="Auto-merge duplicate nodes")
@click.option("--fix-normals", is_flag=True, help="Auto-fix normal directions")
@click.option("--json", "-j", "json_output", is_flag=True, help="Output as JSON")
def validate(
    input_file: str,
    report: str,
    threshold_aspect_ratio: float,
    threshold_area: float,
    fix_duplicates: bool,
    fix_normals: bool,
    json_output: bool
) -> None:
    """Validate grid quality and compatibility.
    
    Check mesh quality metrics including aspect ratio, skewness,
    and Jacobian determinant.
    
    Args:
        input_file: Path to .nas grid file
        report: Quality report output file
        threshold_aspect_ratio: Aspect ratio threshold
        threshold_area: Minimum cell area threshold
        fix_duplicates: Auto-merge duplicate nodes
        fix_normals: Auto-fix normal directions
        json_output: Output result as JSON
    
    Examples:
        # Basic validation
        $ autoflowcfd grid validate sedan.nas
        
        # Custom thresholds
        $ autoflowcfd grid validate sedan.nas --threshold-aspect-ratio 500.0
        
        # Auto-fix issues
        $ autoflowcfd grid validate sedan.nas --fix-duplicates --fix-normals
    """
    from autoflowcfd.grid import NASParser, GridValidator
    
    logger.info(f"Validating grid: {input_file}")
    
    try:
        # Parse grid
        parser = NASParser(input_file)
        grid_data = parser.parse()
        
        # Validate
        validator = GridValidator(
            grid_data,
            threshold_aspect_ratio=threshold_aspect_ratio,
            threshold_area=threshold_area
        )
        
        if fix_duplicates or fix_normals:
            logger.info("Applying automatic fixes...")
            validator.fix_issues(fix_duplicates=fix_duplicates, fix_normals=fix_normals)
        
        quality_report = validator.validate()
        
        # Determine status
        error_count = quality_report.get('error_count', 0)
        warning_count = quality_report.get('warning_count', 0)
        
        if error_count > 0:
            status = "error"
            exit_code = 2
        elif warning_count > 10:
            status = "warning"
            exit_code = 1
        else:
            status = "success"
            exit_code = 0
        
        result = {
            "command": "grid.validate",
            "status": status,
            "result": quality_report,
        }
        
        # Output
        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            # Save report
            report_path = Path(report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(quality_report, f, indent=2)
            
            logger.info(f"Quality report saved to {report_path}")
            
            # Print summary
            click.echo(f"\nGrid Quality Report")
            click.echo(f"{'='*50}")
            click.echo(f"Status: {status.upper()}")
            click.echo(f"Errors: {error_count}")
            click.echo(f"Warnings: {warning_count}")
            
            if quality_report.get('recommendations'):
                click.echo(f"\nRecommendations:")
                for rec in quality_report['recommendations']:
                    click.echo(f"  • {rec}")
            
            click.echo(f"\n✓ Report saved to {report}")
        
        if exit_code != 0:
            raise SystemExit(exit_code)
    
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        if json_output:
            error_result = {
                "command": "grid.validate",
                "status": "error",
                "error": str(e)
            }
            click.echo(json.dumps(error_result, indent=2))
        raise click.ClickException(f"Grid validation failed: {e}")


@grid.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--json", "-j", "json_output", is_flag=True, help="Output as JSON")
def info(input_file: str, json_output: bool) -> None:
    """Display grid statistics.
    
    Quick view of grid information without generating files.
    
    Args:
        input_file: Path to .nas grid file
        json_output: Output result as JSON
    
    Examples:
        # Quick view
        $ autoflowcfd grid info sedan.nas
        
        # JSON output
        $ autoflowcfd grid info sedan.nas --json
    """
    from autoflowcfd.grid import NASParser
    
    logger.info(f"Getting grid info: {input_file}")
    
    try:
        # Parse grid
        parser = NASParser(input_file)
        grid_data = parser.parse()
        
        # Get statistics
        node_count = grid_data.node_count
        cell_count = grid_data.cell_count
        
        result = {
            "file": input_file,
            "node_count": node_count,
            "cell_count": cell_count,
            "boundary_groups": {},
        }
        
        # Boundary info
        if hasattr(grid_data, 'boundaries'):
            for name in grid_data.boundaries.boundary_names:
                nodes = grid_data.boundaries.get_node_indices(name)
                result["boundary_groups"][name] = len(nodes)
        
        # Estimate memory usage (rough estimate)
        # ~44 bytes per cell + ~24 bytes per node
        estimated_memory_mb = (cell_count * 44 + node_count * 24) / (1024 * 1024)
        result["estimated_memory_mb"] = round(estimated_memory_mb, 2)
        
        # Output
        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            # Pretty print
            filename = Path(input_file).name
            click.echo(f"\nGrid Information: {filename}")
            click.echo(f"{'='*50}")
            click.echo(f"Nodes:          {node_count:,}")
            click.echo(f"Cells:          {cell_count:,}")
            click.echo(f"Boundaries:     {len(result['boundary_groups'])} groups")
            
            for name, count in result['boundary_groups'].items():
                click.echo(f"  - {name:<15} {count:,} cells")
            
            click.echo(f"Memory Usage:   ~{estimated_memory_mb:.1f} MB (estimated)")
    
    except Exception as e:
        logger.error(f"Failed to get grid info: {e}")
        if json_output:
            error_result = {
                "command": "grid.info",
                "status": "error",
                "error": str(e)
            }
            click.echo(json.dumps(error_result, indent=2))
        raise click.ClickException(f"Failed to get grid info: {e}")


@grid.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--format", "-f", type=click.Choice(["vtk", "cgns", "stl"]),
              required=True, help="Output format")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
def convert(input_file: str, format: str, output: str) -> None:
    """Convert grid to different format.
    
    Convert .nas grid to VTK, CGNS, or STL format.
    
    Args:
        input_file: Path to .nas grid file
        format: Output format (vtk/cgns/stl)
        output: Output file path
    
    Examples:
        # Convert to VTK
        $ autoflowcfd grid convert sedan.nas -f vtk -o sedan.vtk
        
        # Convert to STL
        $ autoflowcfd grid convert sedan.nas -f stl -o sedan.stl
    
    Note:
        This feature is planned for v1.0 release.
    """
    logger.warning("Grid conversion is planned for v1.0 release")
    click.echo({"status": "pending", "message": "Grid conversion not yet implemented"})
    # TODO: Implement grid conversion in v1.0
