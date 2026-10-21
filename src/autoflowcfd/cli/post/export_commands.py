"""`post export-vtk`：把 checkpoint 的场导出为 VTK（体网格或命名边界面片；`--high-order` 导出高阶 Lagrange 单元）。

用普通 `@click.command()` 定义，由 `commands.py` 注册到 `post`。
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
@click.option("--grid", "-g", type=click.Path(exists=True),
              help="网格文件路径（若不在算例目录里则需显式指定）")
@click.option("--checkpoint", type=click.Path(exists=True),
              help="checkpoint 文件路径（默认取最新一个）")
@click.option("--surface-mesh", "-s", type=click.Path(exists=True), default=None,
              help="原始面网格路径——导出 q_criterion 需要从 checkpoint 重建求解器，checkpoint 记录的"
                   "求解输入是 .nas 体网格且未记录面网格时必填（与 'post coefficients' 同义）")
@click.option("--skip-quality-check", is_flag=True,
              help="重建求解器时跳过网格质量门（仅当原求解也跳过时使用）")
@click.option("--binary/--ascii", "binary", default=None,
              help="写二进制而非 ASCII 文本（真实网格规模下体积小得多、速度也"
                   "快得多）。默认：.vtk 用 ASCII，.vtu 用二进制+压缩。")
@click.option("--boundaries-only", is_flag=True, default=False,
              help="只导出命名边界面片（WALL/INLET/OUTLET/...），带 "
                   "BoundaryID/BoundaryTypeID 标记 + 名称图例，而不是完整体"
                   "网格——可以在 ParaView 里按命名分区筛选/着色（类似 "
                   "Fluent/OpenFOAM 的 patch 工作流）。需要体网格数据"
                   "（VolumeMeshData）。")
@click.option("--high-order", is_flag=True, default=False,
              help="按 FR 解的分段多项式导出 VTK 高阶 Lagrange 单元（.vtu；单元内部显示真实分布，而不是单元平均）。"
                   "从 checkpoint 重建求解器；湍流量取解点场，另可导出 density")
def export_vtk(
    case: str,
    output: str,
    variables: tuple,
    grid: Optional[str],
    checkpoint: Optional[str],
    surface_mesh: Optional[str],
    skip_quality_check: bool,
    binary: Optional[bool],
    boundaries_only: bool,
    high_order: bool,
) -> None:
    """把 checkpoint 的场导出为 VTK（ParaView 等查看器可视化）。

    低阶导出取单元积分平均（同时写单元值与按体积加权的节点值），--high-order 按 FR 解多项式导出高阶 Lagrange
    单元，--boundaries-only 只写命名边界面片。格式按扩展名：.vtk legacy、.vtu XML。

    \b
    Examples:
        $ autoflowcfd post export-vtk --case results -o flow.vtu
        $ autoflowcfd post export-vtk --case results -o flow.vtu --variables velocity --variables q_criterion
        $ autoflowcfd post export-vtk --case results -o flow_ho.vtu --high-order
        $ autoflowcfd post export-vtk --case results -o walls.vtu --boundaries-only
        $ autoflowcfd post export-vtk --case results --checkpoint results/checkpoints/checkpoint_iter_000100.h5 -o t100.vtu
    \f

    Args:
        case: 存有仿真结果的算例目录
        output: 输出 VTK 文件路径
        variables: 要导出的变量（velocity、pressure、k、omega、nu_tilde、nut、turbulence、q_criterion）
        grid: 体网格文件路径（.pkl 或 .nas）
        checkpoint: checkpoint 文件路径（.h5）；瞬态算例要导出某一时刻就指定那一时刻的 checkpoint

    Note:
        若未指定 --grid 和 --checkpoint，命令会尝试从算例目录结构里自动
        探测。
    """
    logger.info(f"Exporting VTK data from case: {case}")

    try:
        from autoflowcfd.postprocess import VTKExporter

        # 准备变量列表
        if not variables:
            var_list = ['velocity', 'pressure']
            logger.info(f"No variables specified, using defaults: {var_list}")
        else:
            var_list = list(variables)
            logger.info(f"Exporting variables: {var_list}")

        if high_order:
            _export_high_order(case, checkpoint, surface_mesh, skip_quality_check, output, var_list, binary,
                               boundaries_only)
            return

        grid_data, solution, history, iteration, metadata = _load_case(case, grid, checkpoint)

        # 校验变量名
        from autoflowcfd.postprocess.vtk_export_fields import VALID_FIELDS
        invalid_vars = set(var_list) - VALID_FIELDS
        if invalid_vars:
            raise ValueError(f"无效的变量 {sorted(invalid_vars)}，可用：{sorted(VALID_FIELDS)}")

        q_field = None
        if 'q_criterion' in var_list:
            # Q 准则需要解点上的速度梯度（与湍流模型同一实现），从 checkpoint 精确重建求解器状态
            from autoflowcfd.cli.post.helpers import _locate_checkpoint
            from autoflowcfd.cli.solve.checkpoint_io import rebuild_solver_from_checkpoint
            from autoflowcfd.postprocess.q_criterion import q_criterion_cell_field

            solver, _it, _meta = rebuild_solver_from_checkpoint(
                str(_locate_checkpoint(Path(case), checkpoint)), backend="cpu", surface_mesh=surface_mesh,
                skip_quality_check=skip_quality_check)
            q_field = q_criterion_cell_field(solver)

        # 创建 VTK 导出器并导出。湍流量取 checkpoint 里湍流模型输运场与涡粘的单元平均
        # （`core/turbulence/output.py` 的 `turb_cell_*` 字段）。
        from autoflowcfd.core.turbulence.output import turbulence_fields_from_checkpoint

        logger.info("Creating VTK exporter...")
        exporter = VTKExporter(
            grid_data=grid_data,
            solution=solution,
            turbulence=turbulence_fields_from_checkpoint(metadata.get('fields', {})),
            q_criterion=q_field,
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


def _export_high_order(case, checkpoint, surface_mesh, skip_quality_check, output, var_list, binary, boundaries_only):
    """`--high-order`：从 checkpoint 精确重建求解器状态，按解点多项式导出（`postprocess/vtk_export_highorder`）。"""
    from autoflowcfd.cli.post.helpers import _locate_checkpoint
    from autoflowcfd.cli.solve.checkpoint_io import rebuild_solver_from_checkpoint
    from autoflowcfd.postprocess.vtk_export_highorder import export_solver_highorder_vtk

    if boundaries_only:
        raise ValueError("--high-order 导出体单元，不能与 --boundaries-only 同用")
    output_path = Path(output)
    if output_path.suffix not in ("", ".vtu"):
        raise ValueError(f"--high-order 只写 .vtu（VTK 高阶 Lagrange 单元），收到 {output_path.suffix}")
    solver, _it, _meta = rebuild_solver_from_checkpoint(
        str(_locate_checkpoint(Path(case), checkpoint)), backend="cpu", surface_mesh=surface_mesh,
        skip_quality_check=skip_quality_check)
    path = export_solver_highorder_vtk(solver, output_path.with_suffix(".vtu"), fields=var_list,
                                       binary=True if binary is None else binary)
    click.echo(f"\n✅ 高阶 VTK 导出完成：{path}")

