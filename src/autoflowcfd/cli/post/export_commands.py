"""`post export-vtk` 命令 (从 post_commands.py 拆分)。

从 post_commands.py 拆出来（该文件原有 974 行，超过 400 行硬性拆分
阈值）：export-vtk 单个命令本身就有约 170 行（含较长的 docstring），
是文件里最重的单个命令，独立成一个文件最清晰。用普通
`@click.command()`（而不是 `@post.command()`）定义——因为定义时这里
还拿不到 `post` 这个 group 对象——由 post_commands.py 在模块加载末尾
`post.add_command(...)` 注册，与 cli/main.py 给顶层命令组注册到
`cli`、cli/grid/commands.py 给 generate-volume/import-volume 注册到
`grid` 完全是同一套机制。纯代码搬移，不改变任何行为。
"""

from pathlib import Path
from typing import Optional

import click
from loguru import logger

from autoflowcfd.cli.post.helpers import _load_case


@click.command(name="export-vtk")
@click.option("--case", "-c", required=True, type=click.Path(exists=True),
              help="算例目录")
@click.option("--output", "-o", type=click.Path(), default="output.vtk",
              help="输出 VTK 文件路径")
@click.option("--variables", multiple=True,
              help="要导出的变量（pressure、velocity 等）")
@click.option("--time-step", type=int, help="指定时间步（瞬态算例用）")
@click.option("--grid", "-g", type=click.Path(exists=True),
              help="网格文件路径（若不在算例目录里则需显式指定）")
@click.option("--checkpoint", type=click.Path(exists=True),
              help="checkpoint 文件路径（默认取最新一个）")
@click.option("--binary/--ascii", "binary", default=None,
              help="写二进制而非 ASCII 文本（真实网格规模下体积小得多、速度也"
                   "快得多）。默认：.vtk 用 ASCII，.vtu 用二进制+压缩。")
@click.option("--boundaries-only", is_flag=True, default=False,
              help="只导出命名边界面片（WALL/INLET/OUTLET/...），带 "
                   "BoundaryID/BoundaryTypeID 标记 + 名称图例，而不是完整体"
                   "网格——可以在 ParaView 里按命名分区筛选/着色（类似 "
                   "Fluent/OpenFOAM 的 patch 工作流）。需要体网格数据"
                   "（VolumeMeshData）。")
def export_vtk(
    case: str,
    output: str,
    variables: tuple,
    time_step: int,
    grid: Optional[str],
    checkpoint: Optional[str],
    binary: Optional[bool],
    boundaries_only: bool,
) -> None:
    """把场数据导出为 VTK 格式。

    把仿真结果导出为 VTK 格式，供 ParaView 或其他兼容 VTK 的查看器可视化。

    Args:
        case: 存有仿真结果的算例目录
        output: 输出 VTK 文件路径
        variables: 要导出的变量（velocity、pressure、k、omega、nut）
        time_step: 瞬态仿真的指定时间步
        grid: 体网格文件路径（.nas）
        checkpoint: checkpoint 文件路径（.h5）

    Examples:
        # 基本导出（自动从算例目录探测网格和 checkpoint）
        $ autoflowcfd post export-vtk --case results/steady/

        # 显式指定网格和 checkpoint
        $ autoflowcfd post export-vtk \
          --case results/ \
          --grid results/grid/sedan.nas \
          --checkpoint results/checkpoints/checkpoint_0500.h5 \
          --output flow_field.vtk

        # 导出指定变量
        $ autoflowcfd post export-vtk \
          --case results/ \
          --variables velocity pressure \
          --output vel_pres.vtk

        # 瞬态：导出指定时间步
        $ autoflowcfd post export-vtk \
          --case results/transient/ \
          --time-step 100 \
          --output step_100.vtk

    所需数据：
        1. 体网格文件（.nas）——提供网格几何
        2. checkpoint 文件（.h5）——提供解向量（velocity、pressure 等）

    Note:
        若未指定 --grid 和 --checkpoint，命令会尝试从算例目录结构里自动
        探测。
    """
    logger.info(f"Exporting VTK data from case: {case}")

    try:
        from autoflowcfd.postprocess import VTKExporter

        grid_data, solution, history, iteration, metadata = _load_case(case, grid, checkpoint)

        # 准备变量列表
        if not variables:
            var_list = ['velocity', 'pressure']
            logger.info(f"No variables specified, using defaults: {var_list}")
        else:
            var_list = list(variables)
            logger.info(f"Exporting variables: {var_list}")

        # 校验变量名
        valid_vars = {'velocity', 'pressure', 'k', 'omega', 'nut', 'q_criterion'}
        invalid_vars = set(var_list) - valid_vars
        if invalid_vars:
            raise ValueError(
                f"Invalid variables: {invalid_vars}\n"
                f"Valid options: {valid_vars}"
            )

        # 创建 VTK 导出器并导出。
        # mu_t（求解器算出的精确涡粘），如果 checkpoint 里有的话——见
        # CheckpointManager.save 的 extra_fields / VTKExporter 的 mu_t
        # 参数。在这个字段加入之前写的 checkpoint 没有它，此时 'nut'
        # 会退回一个标记为近似估计的值。
        mu_t = metadata.get('fields', {}).get('mu_t')
        logger.info("Creating VTK exporter...")
        exporter = VTKExporter(
            grid_data=grid_data,
            solution=solution,
            mu_t=mu_t,
        )

        # 根据扩展名确定输出格式
        output_path = Path(output)
        if output_path.suffix == '.vtu':
            fmt = 'xml'
        elif output_path.suffix == '.vtk' or not output_path.suffix:
            fmt = 'legacy'
            if not output_path.suffix:
                output_path = output_path.with_suffix('.vtk')
        else:
            raise ValueError(f"Unsupported file format: {output_path.suffix}")

        logger.info(f"Exporting to: {output_path} (format: {fmt}, boundaries_only: {boundaries_only})")
        if boundaries_only:
            vtk_path = exporter.export_boundaries(
                output_path=str(output_path),
                fields=var_list,
                format=fmt,
                binary=binary,
            )
        else:
            vtk_path = exporter.export(
                output_path=str(output_path),
                fields=var_list,
                format=fmt,
                binary=binary,
            )

        # 成功提示
        click.echo("\n" + "="*70)
        click.echo("✅ VTK Export Successful")
        click.echo("="*70)
        click.echo(f"Output file:     {vtk_path}")
        click.echo(f"Format:          {fmt.upper()}")
        click.echo(f"Variables:       {', '.join(var_list)}")
        click.echo(f"Iteration:       {iteration}")
        click.echo(f"Grid cells:      {grid_data.cell_count:,}")
        click.echo("="*70)
        click.echo("\n💡 Next steps:")
        click.echo("  1. Open ParaView")
        click.echo(f"  2. File → Open → {vtk_path}")
        click.echo("  3. Click Apply to load data")
        click.echo("  4. Select coloring variable (Velocity/Pressure)")
        click.echo("="*70)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise click.ClickException(str(e))

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise click.ClickException(str(e))

    except Exception as e:
        logger.error(f"VTK export failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise click.ClickException(f"VTK export failed: {e}")
