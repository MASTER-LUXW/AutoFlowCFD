"""网格处理子命令。

本模块提供网格文件解析、验证和信息显示的 CLI 命令。

命令:
    - parse: 解析 .nas 网格文件
    - validate: 验证网格质量
    - info: 显示网格统计信息
    - convert: 转换网格格式（v1.0）
    - generate-volume: 从面网格 .nas 生成体网格
      （已搬至 grid_volume_commands.py，见下）
    - import-volume: 导入外部生成的体网格
      （已搬至 grid_volume_commands.py，见下）

拆分说明（本文件原有 700 行，超过 400 行硬性拆分阈值）：
`generate-volume`/`import-volume` 这两个围绕"体网格"主题的重量级命令
（合计约 300 行）已搬到 grid_volume_commands.py，用普通
`@click.command()` 定义后在本文件末尾通过 `grid.add_command(...)`
注册——与 cli/main.py 给 `grid`/`solve`/`post`/... 这几个顶层命令组
注册到 `cli` 的方式完全一致，只是往下多了一层。命令名、选项、帮助
文本、`autoflowcfd grid --help` 里的可见效果都与拆分前完全一致。

Example:
    $ autoflowcfd grid parse model.nas
    $ autoflowcfd grid validate model.nas --report report.json
    $ autoflowcfd grid info model.nas --json
    $ autoflowcfd grid generate-volume model.nas -o model_volume.nas
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


# generate-volume / import-volume 两个体网格重量级命令已搬到
# grid_volume_commands.py（见本文件顶部拆分说明），这里用与
# cli/main.py 给 grid/solve/post/... 注册到 cli 完全一致的
# add_command 机制接回来，注册后 CLI 可见效果与拆分前完全一致。
from .grid_volume_commands import generate_volume, import_volume  # noqa: E402

grid.add_command(generate_volume)
grid.add_command(import_volume)
