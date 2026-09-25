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
    """网格处理命令。

    解析、校验、分析 ANSA .nas 网格文件。

    Examples:
        # 解析网格文件
        $ autoflowcfd grid parse sedan.nas

        # 校验网格质量
        $ autoflowcfd grid validate sedan.nas

        # 显示网格信息
        $ autoflowcfd grid info sedan.nas
    """
    pass


@grid.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default="grid_info.json",
              help="输出 JSON 文件路径")
@click.option("--encoding", default="UTF-8", help="文件编码")
@click.option("--streaming", is_flag=True, help="大文件启用流式解析")
@click.option("--skip-validation", is_flag=True, help="跳过网格质量校验")
@click.option("--json", "-j", "json_output", is_flag=True, help="以 JSON 格式输出")
def parse(
    input_file: str,
    output: str,
    encoding: str,
    streaming: bool,
    skip_validation: bool,
    json_output: bool
) -> None:
    """解析 ANSA .nas 网格文件。

    从 NAS 格式提取节点、单元和边界信息。

    Args:
        input_file: .nas 网格文件路径
        output: 输出 JSON 文件路径
        encoding: 文件编码
        streaming: 大文件启用流式模式
        skip_validation: 跳过质量校验
        json_output: 以 JSON 格式输出结果

    Examples:
        # 基本解析
        $ autoflowcfd grid parse sedan.nas

        # 自定义输出路径
        $ autoflowcfd grid parse sedan.nas -o output/grid.json

        # 大文件用流式模式
        $ autoflowcfd grid parse large.nas --streaming
    """
    from autoflowcfd.grid import NASParser

    logger.info(f"Parsing grid file: {input_file}")

    try:
        # 解析网格
        parser = NASParser(input_file, encoding=encoding)

        if streaming:
            logger.info(
                "NASParser already parses node/cell cards line-by-line; "
                "--streaming has no additional effect"
            )
        grid_data = parser.parse()

        # 获取网格统计信息
        result = {
            "node_count": grid_data.node_count,
            "cell_count": grid_data.cell_count,
            "boundary_groups": {},
        }

        # 获取边界信息
        if hasattr(grid_data, 'boundaries'):
            for name in grid_data.boundaries.boundary_names:
                # 这里统计的是该边界组的**单元**数（BoundaryMap.groups 按
                # 契约存单元索引）。原变量名 `nodes` 是错的，且用的是已删除
                # 的错名别名 get_node_indices，见 grid_boundaries.py 那处说明。
                cells = grid_data.boundaries.get_cell_indices(name)
                result["boundary_groups"][name] = len(cells)

        # 质量报告（若未跳过）
        quality_passed = True
        if not skip_validation:
            from autoflowcfd.grid import GridValidator
            validator = GridValidator(grid_data)
            quality_report = validator.validate()
            result["quality_report"] = quality_report
            quality_passed = quality_report['passed']

        # 输出
        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            # 保存到文件
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

        # 内嵌的 quality_report 未通过时以非零码退出，与 `grid validate`
        # 同形状的报告保持一致（那边确实会拿它当门槛）。此前 `parse`
        # 无论 quality_report['passed'] 是什么恒退出 0，只看退出码
        # （不深挖 JSON 内容）的调用方完全得不到质量未过关的信号。
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
              help="质量报告输出文件")
@click.option("--threshold-aspect-ratio", type=float, default=100.0,
              help="长宽比阈值（与 GridValidator 自身默认值一致——见 "
              "validator.py——保证未修改参数的 `grid validate` 在同一"
              "网格上与 `grid parse`/`generate-volume` 的质量门判据一致，"
              "而不是宽松 10 倍）")
@click.option("--threshold-area", type=float, default=1e-12,
              help="最小单元面积阈值 (m²)")
@click.option("--fix-duplicates", is_flag=True, help="自动合并重复节点")
@click.option("--fix-normals", is_flag=True, help="自动修正法向")
@click.option("--json", "-j", "json_output", is_flag=True, help="以 JSON 格式输出")
def validate(
    input_file: str,
    report: str,
    threshold_aspect_ratio: float,
    threshold_area: float,
    fix_duplicates: bool,
    fix_normals: bool,
    json_output: bool
) -> None:
    """校验网格质量与兼容性。

    检查长宽比、扭曲度、Jacobian 行列式等网格质量指标。

    Args:
        input_file: .nas 网格文件路径
        report: 质量报告输出文件
        threshold_aspect_ratio: 长宽比阈值
        threshold_area: 最小单元面积阈值
        fix_duplicates: 自动合并重复节点
        fix_normals: 自动修正法向
        json_output: 以 JSON 格式输出结果

    Examples:
        # 基本校验
        $ autoflowcfd grid validate sedan.nas

        # 自定义阈值
        $ autoflowcfd grid validate sedan.nas --threshold-aspect-ratio 500.0

        # 自动修复问题
        $ autoflowcfd grid validate sedan.nas --fix-duplicates --fix-normals
    """
    from autoflowcfd.grid import NASParser, GridValidator

    logger.info(f"Validating grid: {input_file}")

    try:
        # 解析网格
        parser = NASParser(input_file)
        grid_data = parser.parse()

        # 校验（GridValidator 只检查长宽比/扭曲度/Jacobian；不支持
        # 逐单元面积阈值或自动修复）
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

        # 确定状态
        passed = quality_report['passed']
        status = "success" if passed else "error"
        exit_code = 0 if passed else 2

        result = {
            "command": "grid.validate",
            "status": status,
            "result": quality_report,
        }

        # 输出
        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            # 保存报告
            report_path = Path(report)
            report_path.parent.mkdir(parents=True, exist_ok=True)

            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(quality_report, f, indent=2)

            logger.info(f"Quality report saved to {report_path}")

            # 打印摘要
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
@click.option("--json", "-j", "json_output", is_flag=True, help="以 JSON 格式输出")
def info(input_file: str, json_output: bool) -> None:
    """显示网格统计信息。

    不生成任何文件，快速查看网格信息。

    Args:
        input_file: .nas 网格文件路径
        json_output: 以 JSON 格式输出结果

    Examples:
        # 快速查看
        $ autoflowcfd grid info sedan.nas

        # JSON 输出
        $ autoflowcfd grid info sedan.nas --json
    """
    from autoflowcfd.grid import NASParser

    logger.info(f"Getting grid info: {input_file}")

    try:
        # 解析网格
        parser = NASParser(input_file)
        grid_data = parser.parse()

        # 获取统计信息
        node_count = grid_data.node_count
        cell_count = grid_data.cell_count

        result = {
            "file": input_file,
            "node_count": node_count,
            "cell_count": cell_count,
            "boundary_groups": {},
        }

        # 边界信息
        if hasattr(grid_data, 'boundaries'):
            for name in grid_data.boundaries.boundary_names:
                # 这里统计的是该边界组的**单元**数（BoundaryMap.groups 按
                # 契约存单元索引）。原变量名 `nodes` 是错的，且用的是已删除
                # 的错名别名 get_node_indices，见 grid_boundaries.py 那处说明。
                cells = grid_data.boundaries.get_cell_indices(name)
                result["boundary_groups"][name] = len(cells)

        # 估算内存占用（粗略估计）
        # 每单元约 44 字节 + 每节点约 24 字节
        estimated_memory_mb = (cell_count * 44 + node_count * 24) / (1024 * 1024)
        result["estimated_memory_mb"] = round(estimated_memory_mb, 2)

        # 输出
        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            # 美化打印
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
              required=True, help="输出格式")
@click.option("--output", "-o", type=click.Path(), help="输出文件路径")
@click.option("--json", "-j", "json_output", is_flag=True, help="以 JSON 格式输出")
def convert(input_file: str, format: str, output: str, json_output: bool) -> None:
    """转换网格为其他格式。

    把 .nas 网格转换为 VTK、CGNS 或 STL 格式。

    Args:
        input_file: .nas 网格文件路径
        format: 输出格式（vtk/cgns/stl）
        output: 输出文件路径
        json_output: 以 JSON 格式输出结果

    Examples:
        # 转换为 VTK
        $ autoflowcfd grid convert sedan.nas -f vtk -o sedan.vtk

        # 转换为 STL
        $ autoflowcfd grid convert sedan.nas -f stl -o sedan.stl

    Note:
        这个功能计划在 v1.0 发布。其余全部 grid/solve 子命令都支持
        --json，成功/失败两条路径都产出真正的 JSON；这个命令此前两者
        都没有（没有 --json 选项，且 `click.echo({...})` 打印的是
        Python dict 的 repr——单引号键名，`json.loads()` 解析不了），
        尽管形状上其余部分与其他命令的 `{"command", "status", ...}`
        约定一致，会误导调用方以为它遵循同一套契约。这个命令本身仍未
        实现；现在只是让它以响亮、机器可读的方式失败，而不是静默。
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
from autoflowcfd.cli.grid.volume_commands import generate_volume, import_volume  # noqa: E402

grid.add_command(generate_volume)
grid.add_command(import_volume)
