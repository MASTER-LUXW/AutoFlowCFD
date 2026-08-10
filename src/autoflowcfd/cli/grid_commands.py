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
        quality_passed = True
        if not skip_validation:
            from autoflowcfd.grid import GridValidator
            validator = GridValidator(grid_data)
            quality_report = validator.validate()
            result["quality_report"] = quality_report
            quality_passed = quality_report['passed']

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
            if not quality_passed:
                click.echo("✗ Quality validation failed - see quality_report for details")

        # Exit non-zero when the embedded quality_report failed, matching
        # `grid validate`'s identical-shaped report (which does gate on
        # this). Previously `parse` always exited 0 regardless of
        # quality_report['passed'], so a caller relying on the exit code
        # alone (rather than digging into the JSON) had no signal that
        # quality failed.
        if not quality_passed:
            raise SystemExit(2)

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
@click.option("--threshold-aspect-ratio", type=float, default=100.0,
              help="Aspect ratio threshold (matches GridValidator's own "
              "default - see validator.py - so an unmodified `grid validate` "
              "run agrees with `grid parse`/`generate-volume`'s quality gate "
              "on the same mesh instead of being 10x more permissive)")
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
@click.option("--output", "-o", required=True, help="Output volume mesh .nas file path")
@click.option("--growth-rate", default=1.2, show_default=True, help="Boundary layer growth rate")
@click.option("--min-cell-size", default=0.001, show_default=True, help="Minimum cell size (m)")
@click.option("--target-cells", default=500000, show_default=True, help="Target total volume cell count")
@click.option("--max-cell-size", default=None, type=float, help="Maximum cell size (m)")
@click.option("--bl-layers", default=None, type=int, help="Number of BL layers")
@click.option("--skip-quality-report", is_flag=True, help="Skip quality report computation")
@click.option("--json-output", is_flag=True, help="Output result as JSON")
@click.option("--bl-only", is_flag=True, help="Generate and export only the BL prism layer mesh")
@click.option(
    "--core-only", is_flag=True,
    help="Export the mesh right after core-region tetgen fill (core tets "
    "alone, not spliced with BL) - skips all later steps",
)
def generate_volume(
    input_file: str,
    output: str,
    growth_rate: float,
    min_cell_size: float,
    target_cells: int,
    max_cell_size: Optional[float],
    bl_layers: Optional[int],
    skip_quality_report: bool,
    json_output: bool,
    bl_only: bool,
    core_only: bool,
) -> None:
    """Generate a volume mesh from a surface .nas file and export it.

    Runs the full grid pipeline: parse surface mesh -> validate surface
    quality -> generate hybrid volume mesh (BL extrusion + Cartesian
    background) -> validate volume mesh quality -> export to Nastran .nas.
    The volume mesh is always exported, whether or not it passes the
    quality report - a case that genuinely can't converge (see
    mesh_repair.py's documented limits) should still produce a mesh file to
    inspect or hand off, not nothing at all. `autoflowcfd solve run` is the
    actual enforcement point that blocks solving a mesh that fails this
    check.

    Args:
        input_file: Path to surface .nas grid file
        output: Output volume mesh .nas file path
        growth_rate: Boundary layer growth rate
        min_cell_size: Minimum cell size (m)
        target_cells: Target total volume cell count
        bl_layers: How many layers the BL stage extrudes before the
            remaining volume is filled directly from the BL's own outer
            surface by tetgen (see mesh_background_merge._build_merged_mesh
            - there is no separate structured "transition" stage anymore,
            ProjectFiles Part13 P49); None defaults to 8
        skip_quality_report: Skip computing/printing the quality report
            (export always happens regardless)
        json_output: Output result as JSON
        bl_only: If set, only generate and export the BL prism layer mesh.
        core_only: If set, export right after core-region tetgen fill (core
            tets alone, not spliced with BL) and stop.
    """
    from autoflowcfd.grid import (
        NASParser, GridValidator, MeshQualityValidator, export_volume_mesh_to_nas
    )

    if bl_only and core_only:
        raise click.ClickException(
            "--bl-only and --core-only are mutually exclusive - each stops "
            "the pipeline at a different stage"
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
        # Reuses surface_grid (already parsed above for the Step 2 quality
        # check) directly - parser.parse(generate_volume_mesh=True) would
        # re-parse the same NAS file from scratch a second time.
        volume_mesh = parser.generate_volume_mesh_from_surface(
            surface_grid,
            volume_mesh_params={
                'growth_rate': growth_rate,
                'min_cell_size': min_cell_size,
                'target_cells': target_cells,
                'max_cell_size': max_cell_size,
                'bl_layers': bl_layers,
                'bl_only': bl_only,
                'core_only': core_only,
                'output': output,
            }
        )

        quality_report = None
        if not skip_quality_report:
            logger.info("Validating volume mesh quality...")
            quality_report = MeshQualityValidator().validate_volume_mesh(volume_mesh)
            # Stage A/B/C (mesh_gen/mesh_repair.py, volume_mesh_generator.py's
            # backoff loop) already ran to completion during generation above -
            # this is purely informational on their outcome, not an export
            # gate: the volume mesh file is always written below regardless
            # of pass/fail, since a case that genuinely can't converge (e.g.
            # a real sharp convex corner - see mesh_repair.py's own
            # documented, measured limits here) would otherwise never
            # produce any output at all to inspect or hand-fix. The
            # solve-time quality gate (cli/solve_commands.py) is the actual
            # enforcement point before any iterations run.
            if quality_report.passed:
                logger.info(f"\n{quality_report.summary()}")
            else:
                logger.error(
                    f"\n{quality_report.summary()}\n"
                    "Volume mesh quality check failed after Stage A/B/C repair - "
                    "exporting anyway (see report above). This mesh would very "
                    "likely diverge if solved as-is; common causes: sharp convex "
                    "edges/corners on the body (BL extrusion degrades there; "
                    "consider a small chamfer/fillet in the source geometry), or "
                    "an overly aggressive --growth-rate/--min-cell-size for this "
                    "geometry's feature sizes. 'autoflowcfd solve run' will still enforce this gate "
                    "before any iterations run, unless --skip-quality-check is "
                    "passed there too."
                )

        logger.info("Step 4/4: Exporting volume mesh to NAS...")
        # scale_factor=1000.0（默认值即为此，这里显式写出便于阅读）：内部网格坐标
        # 始终是米（NASParser 导入时按 mm->m 换算），导出为 mm 与 NASParser
        # 默认导入单位一致，避免往返导入导出时几何体缩小 1000 倍。
        output_path = export_volume_mesh_to_nas(volume_mesh, output, scale_factor=1000.0)

        # 同时保存pickle格式的体网格文件，供求解器直接使用
        import pickle
        pkl_output = Path(output).with_suffix('.pkl')
        with open(pkl_output, 'wb') as f:
            pickle.dump(volume_mesh, f)
        logger.info(f"Volume mesh cache saved: {pkl_output}")

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
            "cache_file": str(pkl_output),
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
            click.echo(f"\n✓ Exported to: {output_path}")
            click.echo(f"✓ Cache saved to: {pkl_output}")

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
@click.option("--json", "-j", "json_output", is_flag=True, help="Output as JSON")
def convert(input_file: str, format: str, output: str, json_output: bool) -> None:
    """Convert grid to different format.

    Convert .nas grid to VTK, CGNS, or STL format.

    Args:
        input_file: Path to .nas grid file
        format: Output format (vtk/cgns/stl)
        output: Output file path
        json_output: Output result as JSON

    Examples:
        # Convert to VTK
        $ autoflowcfd grid convert sedan.nas -f vtk -o sedan.vtk

        # Convert to STL
        $ autoflowcfd grid convert sedan.nas -f stl -o sedan.stl

    Note:
        This feature is planned for v1.0 release. Every other grid/solve
        subcommand supports --json and emits real JSON on both success and
        error paths; this one previously had neither (no --json flag, and
        click.echo({...}) prints Python's dict repr - single-quoted keys,
        not parseable by json.loads()) despite otherwise matching their
        {"command", "status", ...} shape, which would misinform a caller
        that reasonably expects the same contract as every sibling command.
        This command still isn't implemented; it just now fails that way
        loudly and machine-readably instead of silently.
    """
    logger.warning("Grid conversion is planned for v1.0 release")
    result = {
        "command": "grid.convert",
        "status": "not_implemented",
        "message": "Grid conversion not yet implemented",
    }
    if json_output:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"{result['status']}: {result['message']}")
    raise click.ClickException("Grid conversion is not yet implemented (planned for v1.0)")


@grid.command(name="import-volume")
@click.argument("volume_mesh_file", type=click.Path(exists=True))
@click.option("--surface-mesh", "-s", type=click.Path(exists=True), required=True,
              help="Original surface .nas file the volume mesh was generated from "
                   "(supplies boundary-group geometry for inlet/outlet/wall/... "
                   "attribution - the volume mesh file itself typically carries none)")
@click.option("--output", "-o", type=click.Path(), required=True,
              help="Output path for the validated/repaired mesh, as a pickled "
                   "VolumeMeshData (.pkl) ready for 'autoflowcfd solve run'/"
                   "'transient' - NOT a .nas file")
@click.option("--skip-repair", is_flag=True,
              help="Skip Stage A smoothing when the initial quality check fails - "
                   "just report and export the mesh exactly as parsed")
@click.option("--max-repair-passes", type=int, default=5,
              help="Stage A smoothing's own max passes")
@click.option("--skip-overlap-check", is_flag=True,
              help="Skip the physical-overlap check (the most expensive single "
                   "quality check on a large mesh) - use for a quick preliminary look")
@click.option("--json", "-j", "json_output", is_flag=True, help="Output as JSON")
def import_volume(
    volume_mesh_file: str,
    surface_mesh: str,
    output: str,
    skip_repair: bool,
    max_repair_passes: int,
    skip_overlap_check: bool,
    json_output: bool,
) -> None:
    """Import an externally-generated volume mesh (e.g. ANSA's own volume
    export) for quality-checking, best-effort repair, and solving.

    Parses a volume-mesh .nas file (GRID + CTETRA + CPENTA cards) some
    OTHER tool produced, attributes boundary groups (inlet/outlet/wall/...)
    from the companion surface mesh it was generated from by geometric
    (nearest-centroid) matching, runs the same MeshQualityValidator this
    project's own generate-volume uses, and - if the check fails - applies
    Stage A smoothing (quality-gated Laplacian smoothing of skewed/non-
    orthogonal/volume-mismatched cells) as a best-effort repair. The
    result is saved as a pickled VolumeMeshData, the same cache format
    'autoflowcfd solve run'/'transient' already consume directly.

    Args:
        volume_mesh_file: Path to the volume-mesh .nas file
        surface_mesh: Path to the original surface .nas file
        output: Output .pkl path
        skip_repair: Skip Stage A smoothing on a failing quality check
        max_repair_passes: Stage A's own max passes
        skip_overlap_check: Skip the (expensive) physical-overlap check
        json_output: Output result as JSON

    Examples:
        # Import, repair if needed, and prepare for solving
        $ autoflowcfd grid import-volume car_volume.nas -s car_surface.nas -o car_volume.pkl

        # Then solve directly from the cache
        $ autoflowcfd solve run car_volume.pkl
    """
    from autoflowcfd.grid.mesh_gen.mesh_external_import import import_external_volume_mesh

    logger.info(f"Importing external volume mesh: {volume_mesh_file}")

    try:
        volume_mesh, report = import_external_volume_mesh(
            volume_mesh_file, surface_mesh,
            repair=not skip_repair,
            max_repair_passes=max_repair_passes,
            check_overlap=not skip_overlap_check,
        )

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(volume_mesh, f)

        result = {
            "command": "grid.import-volume",
            "status": "success",
            "node_count": volume_mesh.node_count,
            "cell_count": volume_mesh.cell_count,
            "total_volume": volume_mesh.total_volume,
            "boundary_groups": list(volume_mesh.boundaries.groups.keys()),
            "quality_passed": report.passed,
            "output": str(output_path),
        }
        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\nImported: {Path(volume_mesh_file).name}")
            click.echo("=" * 50)
            click.echo(f"Nodes: {volume_mesh.node_count:,}")
            click.echo(f"Cells: {volume_mesh.cell_count:,}")
            click.echo(f"Total volume: {volume_mesh.total_volume:.6e} m^3")
            click.echo(f"Boundary groups: {', '.join(result['boundary_groups'])}")
            click.echo(f"Quality gate: {'PASSED' if report.passed else 'FAILED'}")
            click.echo(f"\n✓ Saved to: {output_path}")
            if not report.passed:
                click.echo(
                    "✗ Quality gate failed (see report above) - "
                    "'solve run'/'transient' will still enforce this before solving, "
                    "unless --skip-quality-check is passed there too"
                )

    except Exception as e:
        logger.error(f"External volume mesh import failed: {e}")
        if json_output:
            error_result = {
                "command": "grid.import-volume",
                "status": "error",
                "error": str(e),
            }
            click.echo(json.dumps(error_result, indent=2))
        raise click.ClickException(f"External volume mesh import failed: {e}")
    # TODO: Implement grid conversion in v1.0
