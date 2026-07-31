"""Grid processing subcommands.

This module provides CLI commands for grid file parsing, validation,
and information display.

Commands:
    - parse: Parse .nas grid files
    - validate: Validate grid quality
    - info: Display grid statistics
    - generate-volume: Generate + export a volume mesh from a surface .nas
    - convert: Convert grid formats (v1.0)

Example:
    $ autoflowcfd grid parse model.nas
    $ autoflowcfd grid validate model.nas --report report.json
    $ autoflowcfd grid info model.nas --json
    $ autoflowcfd grid generate-volume model.nas -o model_volume.nas
"""

import click
import json
from pathlib import Path
from typing import Optional
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
            logger.info(
                "NASParser already parses node/cell cards line-by-line; "
                "--streaming has no additional effect"
            )
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

        # Validate (GridValidator only checks aspect ratio / skewness /
        # Jacobian; there is no per-cell area threshold or auto-fix support)
        validator = GridValidator(grid_data)
        validator.thresholds['aspect_ratio_max'] = threshold_aspect_ratio

        if threshold_area != 1e-12:
            logger.warning(
                "--threshold-area is not supported by GridValidator and will be ignored"
            )
        if fix_duplicates or fix_normals:
            logger.warning(
                "--fix-duplicates/--fix-normals are not implemented; "
                "no automatic fixes were applied"
            )

        quality_report = validator.validate()

        # Determine status
        passed = quality_report['passed']
        status = "success" if passed else "error"
        exit_code = 0 if passed else 2

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
            click.echo(quality_report['summary'])
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


@grid.command(name="generate-volume")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), required=True,
              help="Output volume mesh .nas file path")
@click.option("--growth-rate", type=float, default=1.2, help="Boundary layer growth rate")
@click.option("--max-layers", type=int, default=12, help="Maximum extrusion layers")
@click.option("--min-cell-size", type=float, default=0.01, help="Minimum cell size (m)")
@click.option("--target-cells", type=int, default=400000, help="Target total volume cell count")
@click.option("--max-cell-size", type=float, default=None,
              help="Max core-region cell size (m), graded outward from the BL's near-wall "
                   "size; unset means the core fill has no size cap beyond tetgen's own "
                   "shape-quality bounds")
@click.option("--skip-quality-report", is_flag=True, help="Skip volume mesh quality report")
@click.option("--json", "-j", "json_output", is_flag=True, help="Output as JSON")
def generate_volume(
    input_file: str,
    output: str,
    growth_rate: float,
    max_layers: int,
    min_cell_size: float,
    target_cells: int,
    max_cell_size: Optional[float],
    skip_quality_report: bool,
    json_output: bool
) -> None:
    """Generate a volume mesh from a surface .nas file and export it.

    Runs the full grid pipeline: parse surface mesh -> validate surface
    quality -> generate hybrid volume mesh (BL extrusion + Cartesian
    background) -> validate volume mesh quality -> export to Nastran .nas.

    Args:
        input_file: Path to surface .nas grid file
        output: Output volume mesh .nas file path
        growth_rate: Boundary layer growth rate
        max_layers: Maximum extrusion layers
        min_cell_size: Minimum cell size (m)
        target_cells: Target total volume cell count
        skip_quality_report: Skip volume mesh quality report
        json_output: Output result as JSON

    Examples:
        # Basic volume mesh generation
        $ autoflowcfd grid generate-volume sedan.nas -o sedan_volume.nas

        # Coarser mesh for a quick check
        $ autoflowcfd grid generate-volume sedan.nas -o sedan_volume.nas --target-cells 100000
    """
    from autoflowcfd.grid import (
        NASParser, GridValidator, MeshQualityValidator, export_volume_mesh_to_nas
    )

    logger.info(f"Generating volume mesh: {input_file}")

    try:
        parser = NASParser(input_file)

        logger.info("Step 1/4: Parsing surface mesh...")
        surface_grid = parser.parse()

        logger.info("Step 2/4: Validating surface mesh quality...")
        surface_report = GridValidator(surface_grid).validate()
        if not surface_report['passed']:
            logger.warning(
                "Surface mesh quality validation failed; "
                "continuing with volume mesh generation anyway"
            )

        logger.info("Step 3/4: Generating volume mesh (BL extrusion + background)...")
        volume_mesh = parser.parse(
            generate_volume_mesh=True,
            volume_mesh_params={
                'growth_rate': growth_rate,
                'max_layers': max_layers,
                'min_cell_size': min_cell_size,
                'target_cells': target_cells,
                'max_cell_size': max_cell_size,
            }
        )

        quality_report = None
        if not skip_quality_report:
            logger.info("Validating volume mesh quality...")
            quality_report = MeshQualityValidator().validate_volume_mesh(volume_mesh)

        logger.info("Step 4/4: Exporting volume mesh to NAS...")
        output_path = export_volume_mesh_to_nas(volume_mesh, output)

        boundary_names = list(volume_mesh.boundaries.groups.keys())
        result = {
            "command": "grid.generate-volume",
            "status": "success",
            "surface_quality_passed": surface_report['passed'],
            "node_count": volume_mesh.node_count,
            "cell_count": volume_mesh.cell_count,
            "total_volume_m3": volume_mesh.total_volume,
            "boundary_groups": boundary_names,
            "volume_quality_passed": quality_report.passed if quality_report else None,
            "output_file": output_path,
        }

        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\nVolume Mesh Generated: {Path(input_file).name}")
            click.echo("=" * 50)
            click.echo(f"Nodes: {volume_mesh.node_count:,}")
            click.echo(f"Cells: {volume_mesh.cell_count:,}")
            click.echo(f"Total volume: {volume_mesh.total_volume:.6e} m^3")
            click.echo(f"Boundary groups: {', '.join(boundary_names)}")
            if quality_report is not None:
                click.echo(quality_report.summary())
            click.echo(f"\n✓ Exported to: {output_path}")

    except Exception as e:
        logger.error(f"Volume mesh generation failed: {e}")
        if json_output:
            error_result = {
                "command": "grid.generate-volume",
                "status": "error",
                "error": str(e)
            }
            click.echo(json.dumps(error_result, indent=2))
        raise click.ClickException(f"Volume mesh generation failed: {e}")


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
