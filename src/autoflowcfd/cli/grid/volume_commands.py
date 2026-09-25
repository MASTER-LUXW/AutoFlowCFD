"""体网格生成/导入子命令。

从 grid_commands.py 中拆分出来（该文件超过 400 行硬性拆分阈值）：
`generate-volume`（从面网格生成体网格）和 `import-volume`（导入外部
已生成的体网格，例如 ANSA 自己的体网格导出）这两个命令都是围绕"体
网格"这一主题的重量级命令（合计约 300 行），与 grid_commands.py 里
剩下的 parse/validate/info/convert（轻量、面网格层面的操作）自成
一组，是清晰的拆分边界。

这里的两个命令用普通的 `@click.command()`（而不是 `@grid.command()`）
定义——因为定义时这里还拿不到 `grid` 这个 group 对象——由
grid_commands.py 在模块加载末尾 `grid.add_command(...)` 注册，注册后
在 `autoflowcfd grid --help` 里的可见效果、命令名、选项、帮助文本与
拆分前完全一致（都用了显式 `name=` 参数，不依赖函数名推导）。
"""

import click
import json
from pathlib import Path
from typing import Optional
from loguru import logger


@click.command(name="generate-volume")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", required=True, help="输出体网格 .nas 文件路径")
@click.option("--growth-rate", default=1.2, show_default=True, help="边界层增长率")
@click.option("--min-cell-size", default=0.001, show_default=True, help="最小单元尺寸 (m)")
@click.option("--target-cells", default=500000, show_default=True, help="目标体网格单元总数")
@click.option("--max-cell-size", default=None, type=float, help="最大单元尺寸 (m)")
@click.option("--bl-layers", default=None, type=int, help="边界层层数")
@click.option("--skip-quality-report", is_flag=True, help="跳过质量报告计算")
@click.option("--json-output", is_flag=True, help="以 JSON 格式输出结果")
@click.option("--bl-only", is_flag=True, help="只生成并导出边界层棱柱层网格")
@click.option(
    "--core-only", is_flag=True,
    help="在核心区 tetgen 填充完成后立即导出网格（只有核心区四面体，未与"
    "边界层拼接）——跳过后续全部步骤",
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
    """从面网格 .nas 文件生成体网格并导出。

    运行完整的网格流水线：解析面网格 -> 校验面网格质量 -> 生成混合体网格
    （边界层挤出 + 笛卡尔背景网格）-> 校验体网格质量 -> 导出为 Nastran
    .nas。无论质量报告是否通过，体网格都会被导出——一个真正无法收敛的
    case（见 mesh_repair.py 里记录的已知极限）也应该产出一个可供检查或
    转交的网格文件，而不是什么都不产出。真正拦截"质量不过关的网格不能
    求解"的地方是 `autoflowcfd solve steady`。

    Args:
        input_file: 面网格 .nas 文件路径
        output: 输出体网格 .nas 文件路径
        growth_rate: 边界层增长率
        min_cell_size: 最小单元尺寸 (m)
        target_cells: 目标体网格单元总数
        bl_layers: 边界层阶段挤出多少层之后，剩余体积直接由边界层自身
            外表面交给 tetgen 填充（见
            mesh_background_merge._build_merged_mesh——现在已经没有单独
            的结构化"过渡层"阶段了，见 ProjectFiles Part13 P49）；
            None 时默认为 8
        skip_quality_report: 跳过质量报告的计算/打印（导出本身始终照常
            进行）
        json_output: 以 JSON 格式输出结果
        bl_only: 若设置，只生成并导出边界层棱柱层网格。
        core_only: 若设置，在核心区 tetgen 填充完成后立即导出（只有核心区
            四面体，未与边界层拼接）并停止。
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
        # 直接复用 surface_grid（上面 Step 2 质量检查时已经解析过）——
        # parser.parse(generate_volume_mesh=True) 会把同一个 NAS 文件
        # 从头再解析一遍。
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
            # Stage A/B（mesh_gen/mesh_repair.py）在上面生成体网格的过程中
            # 已经跑完了——这里纯粹是对其结果的信息展示，不是导出门槛：
            # 无论通过与否，下面都会照常写出体网格文件，因为一个真正无法
            # 收敛的 case（例如真实的尖锐凸角——见 mesh_repair.py 自己
            # 记录的、实测出的极限）否则就完全产不出任何可供检查/手动
            # 修复的结果。这里以前还有一个 Stage C（全局 min_cell_size
            # 退让 + 完整重新生成）——按用户要求已移除：实测下来是个不
            # 可靠的净收益（3 组受控 cube_demo 对比里有 2 组反而比原始
            # 参数更差），同时还会让导出的网格静默偏离用户实际要求的
            # min_cell_size。真正的求解期质量门（cli/solve/commands.py）
            # 才是任何迭代开始前真正拦截的地方。
            if quality_report.passed:
                logger.info(f"\n{quality_report.summary()}")
            else:
                logger.error(
                    f"\n{quality_report.summary()}\n"
                    "Volume mesh quality check failed after Stage A/B repair - "
                    "exporting anyway (see report above). This mesh would very "
                    "likely diverge if solved as-is; common causes: sharp convex "
                    "edges/corners on the body (BL extrusion degrades there; "
                    "consider a small chamfer/fillet in the source geometry), or "
                    "an overly aggressive --growth-rate/--min-cell-size for this "
                    "geometry's feature sizes. 'autoflowcfd solve steady' will still enforce this gate "
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


@click.command(name="import-volume")
@click.argument("volume_mesh_file", type=click.Path(exists=True))
@click.option("--surface-mesh", "-s", type=click.Path(exists=True), required=True,
              help="生成该体网格所用的原始面网格 .nas 文件（提供 inlet/"
                   "outlet/wall/... 边界分组归属所需的几何信息——体网格"
                   "文件本身通常不携带这些信息）")
@click.option("--output", "-o", type=click.Path(), required=True,
              help="校验/修复后网格的输出路径，格式为 pickle 化的 "
                   "VolumeMeshData (.pkl)，可直接供 'autoflowcfd solve "
                   "steady'/'transient' 使用——不是 .nas 文件")
@click.option("--skip-repair", is_flag=True,
              help="初始质量检查未通过时跳过 Stage A 平滑——按解析结果原样"
                   "报告并导出网格")
@click.option("--max-repair-passes", type=int, default=5,
              help="Stage A 平滑自身的最大迭代次数")
@click.option("--skip-overlap-check", is_flag=True,
              help="跳过物理重叠检查（大网格上单项开销最大的质量检查）——"
                   "用于快速预览")
@click.option("--json", "-j", "json_output", is_flag=True, help="以 JSON 格式输出")
def import_volume(
    volume_mesh_file: str,
    surface_mesh: str,
    output: str,
    skip_repair: bool,
    max_repair_passes: int,
    skip_overlap_check: bool,
    json_output: bool,
) -> None:
    """导入外部生成的体网格（例如 ANSA 自己导出的体网格），做质量检查、
    尽力修复、并准备求解。

    解析某个其他工具产出的体网格 .nas 文件（GRID + CTETRA + CPENTA
    卡片），通过几何（最近形心）匹配，从生成该体网格所用的配套面网格
    反推边界分组（inlet/outlet/wall/...），运行本项目自己
    generate-volume 用的同一套 MeshQualityValidator——若检查未通过，
    应用 Stage A 平滑（对畸变/非正交/体积失配单元做质量门控的
    Laplacian 平滑）作为尽力修复。结果保存为 pickle 化的
    VolumeMeshData，与 'autoflowcfd solve steady'/'transient' 已经直接
    消费的缓存格式相同。

    Args:
        volume_mesh_file: 体网格 .nas 文件路径
        surface_mesh: 原始面网格 .nas 文件路径
        output: 输出 .pkl 路径
        skip_repair: 质量检查未通过时跳过 Stage A 平滑
        max_repair_passes: Stage A 自身的最大迭代次数
        skip_overlap_check: 跳过（开销较大的）物理重叠检查
        json_output: 以 JSON 格式输出结果

    Examples:
        # 导入、按需修复、并准备求解
        $ autoflowcfd grid import-volume car_volume.nas -s car_surface.nas -o car_volume.pkl

        # 然后直接从缓存求解
        $ autoflowcfd solve steady car_volume.pkl
    """
    from autoflowcfd.grid.mesh_gen.utils.mesh_external_import import import_external_volume_mesh

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
                    "'solve steady'/'transient' will still enforce this before solving, "
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
