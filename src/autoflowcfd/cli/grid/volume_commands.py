"""体网格子命令：`generate-volume`（从面网格生成体网格）与 `import-volume`（导入外部已生成的体网格，例如 ANSA
导出的体网格）。

两个命令用普通的 `@click.command()`（显式 `name=`）定义，由 `commands.py` 在模块末尾 `grid.add_command(...)`
注册到 `grid` 组（定义时这里还拿不到 `grid` 组对象）。
"""

import click
import json
from pathlib import Path
from autoflowcfd.core.freestream_defaults import MU_MOLECULAR, RHO_INF
from autoflowcfd.grid.mesh_gen.tetgen.volume_mesh_generator import DEFAULT_GROWTH_RATE, DEFAULT_MIN_CELL_SIZE
from autoflowcfd.cli.grid.options import bc_type_option, units_option
from typing import Optional
from loguru import logger




def _resolve_first_layer_height(first_layer_height, target_yplus, ref_velocity, ref_length, rho, mu):
    """`--first-layer-height` 或 `--target-yplus`（二选一）-> 第一层高度（`first_layer.resolve_first_layer_height`，
    与 Python API 共用）；参数组合不对时转成命令行用法错误。"""
    from autoflowcfd.grid.mesh_gen.extrusion.first_layer import resolve_first_layer_height

    try:
        height = resolve_first_layer_height(first_layer_height, target_yplus, ref_velocity, ref_length, rho, mu)
    except ValueError as e:
        raise click.UsageError(f"{e}（--first-layer-height / --target-yplus / --ref-velocity / --ref-length）")
    if target_yplus is not None:
        logger.info(f"目标 y+={target_yplus:g}（U={ref_velocity:g} m/s, L={ref_length:g} m）-> 第一层高度 {height:.4g} m")
    return height


@click.command(name="generate-volume")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", required=True, help="输出体网格 .nas 文件路径")
@click.option("--growth-rate", default=DEFAULT_GROWTH_RATE, show_default=True, help="边界层增长率")
@click.option("--min-cell-size", default=DEFAULT_MIN_CELL_SIZE, show_default=True, help="最小单元尺寸 (m)")
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
@bc_type_option
@units_option
@click.option("--first-layer-height", type=float, default=None,
              help="边界层第一层高度 (m)；不给时取 min(--min-cell-size, 0.2% 域尺寸)。与 --target-yplus 二选一")
@click.option("--target-yplus", type=float, default=None,
              help="目标第一层 y+，按湍流平板关联式推出第一层高度（需 --ref-velocity 与 --ref-length）")
@click.option("--ref-velocity", type=float, default=None, help="y+ 估计的参考速度 (m/s)")
@click.option("--ref-length", type=float, default=None, help="y+ 估计的参考长度 (m)，例如车长/弦长")
@click.option("--rho-inf", type=float, default=RHO_INF, show_default=True, help="y+ 估计的密度 (kg/m^3)")
@click.option("--mu-molecular", type=float, default=MU_MOLECULAR, show_default=True, help="y+ 估计的动力粘度 (Pa*s)")
def generate_volume(
    input_file: str,
    output: str,
    growth_rate: float,
    min_cell_size: float,
    max_cell_size: Optional[float],
    bl_layers: Optional[int],
    skip_quality_report: bool,
    json_output: bool,
    bl_only: bool,
    core_only: bool,
    bc_type: tuple = (),
    first_layer_height: Optional[float] = None,
    target_yplus: Optional[float] = None,
    ref_velocity: Optional[float] = None,
    ref_length: Optional[float] = None,
    rho_inf: float = RHO_INF,
    mu_molecular: float = MU_MOLECULAR,
    units: str = "mm",
) -> None:
    """从面网格 .nas 生成体网格并导出为 .nas。

    解析面网格 -> 校验面网格质量 -> 边界层棱柱挤出 + tetgen 核心区四面体 -> 体网格质量报告 -> 导出（与输入同一
    坐标单位）。质量报告不通过时照常导出（便于检查或手工修改），求解命令在迭代前用质量门拦截。

    \b
    Examples:
        $ autoflowcfd grid generate-volume car.nas -o car_volume.nas --bl-layers 6
        $ autoflowcfd grid generate-volume car.nas -o car_volume.nas --target-yplus 1 --ref-velocity 40 --ref-length 4.5
    \f

    Args:
        input_file: 面网格 .nas 文件路径
        output: 输出体网格 .nas 文件路径
        growth_rate: 边界层增长率
        min_cell_size: 最小单元尺寸 (m)
        bl_layers: 边界层挤出层数，之后剩余体积由边界层外表面交给 tetgen 填充
            （`mesh_background_merge._build_merged_mesh`）；None 时取生成器缺省（8）
        skip_quality_report: 跳过质量报告的计算/打印（导出本身始终照常
            进行）
        json_output: 以 JSON 格式输出结果
        bl_only: 若设置，只生成并导出边界层棱柱层网格。
        core_only: 若设置，在核心区 tetgen 填充完成后立即导出（只有核心区
            四面体，未与边界层拼接）并停止。
        bc_type: `--bc-type 组名=类型`（可重复），见 `grid/nas_io/boundary_types.py`
        first_layer_height / target_yplus / ref_velocity / ref_length / rho_inf / mu_molecular: 边界层第一层
            高度，直接给出或由目标 y+ 推出（`grid/mesh_gen/extrusion/first_layer.py`）
        units: 输入面网格的坐标单位（mm / m / auto）；导出的体网格 .nas 用同一单位（auto 时为判定出的单位），
            内部与 .pkl 一律为米
    """
    from autoflowcfd.grid import (
        NASParser, GridValidator, MeshQualityValidator, export_volume_mesh_to_nas
    )
    from autoflowcfd.grid.mesh_gen.background.mesh_background_merge_utils import PartialMeshReady
    from autoflowcfd.grid.nas_io.units import SCALE_TO_METERS

    if bl_only and core_only:
        raise click.ClickException(
            "--bl-only and --core-only are mutually exclusive - each stops "
            "the pipeline at a different stage"
        )

    logger.info(f"Generating volume mesh: {input_file}")

    try:
        from autoflowcfd.grid.nas_io.boundary_types import parse_bc_type_options
        parser = NASParser(input_file, units=units, bc_types=parse_bc_type_options(bc_type))

        logger.info("Step 1/4: Parsing surface mesh...")
        surface_grid = parser.parse()
        export_scale = 1.0 / SCALE_TO_METERS[parser.resolved_units]

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
                'max_cell_size': max_cell_size,
                'bl_layers': bl_layers,
                'bl_only': bl_only,
                'core_only': core_only,
                'first_layer_height': _resolve_first_layer_height(
                    first_layer_height, target_yplus, ref_velocity, ref_length, rho_inf, mu_molecular),
            }
        )

        quality_report = None
        if not skip_quality_report:
            logger.info("Validating volume mesh quality...")
            quality_report = MeshQualityValidator().validate_volume_mesh(volume_mesh)
            # 修复 Stage A/B（grid/mesh_gen/repair/mesh_repair.py）在上面生成体网格的过程中已经跑完——这里只展示
            # 结果，不是导出门槛：无论通过与否都照常写出体网格，否则真正修不好的 case（例如尖锐凸角，见
            # mesh_repair.py 记录的实测极限）就产不出任何可供检查 / 手动修复的结果。拦截在求解前的质量门
            # （grid/validation/solve_gate.py）。
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
        # 导出单位与输入面网格相同（内部坐标为米）：用同一个 --units 读回时几何不变
        output_path = export_volume_mesh_to_nas(volume_mesh, output, scale_factor=export_scale)

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

    except PartialMeshReady as done:
        partial_path = export_volume_mesh_to_nas(done.mesh, output, scale_factor=export_scale)
        if json_output:
            click.echo(json.dumps({"command": "grid.generate-volume", "status": "partial",
                                   "stage": done.label, "output_file": partial_path}, indent=2))
        else:
            click.echo(f"\n✓ {done.label} mesh exported to: {partial_path}")
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
@bc_type_option
@units_option
def import_volume(
    volume_mesh_file: str,
    surface_mesh: str,
    output: str,
    skip_repair: bool,
    max_repair_passes: int,
    skip_overlap_check: bool,
    json_output: bool,
    bc_type: tuple = (),
    units: str = "mm",
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

    \b
    Examples:
        # 导入、按需修复、并准备求解
        $ autoflowcfd grid import-volume car_volume.nas -s car_surface.nas -o car_volume.pkl

    \b
        # 然后直接从缓存求解
        $ autoflowcfd solve steady car_volume.pkl
    
    \f

    Args:
        volume_mesh_file: 体网格 .nas 文件路径
        surface_mesh: 原始面网格 .nas 文件路径
        output: 输出 .pkl 路径
        skip_repair: 质量检查未通过时跳过 Stage A 平滑
        max_repair_passes: Stage A 自身的最大迭代次数
        skip_overlap_check: 跳过（开销较大的）物理重叠检查
        json_output: 以 JSON 格式输出结果
        bc_type: `--bc-type 组名=类型`（可重复），见 `grid/nas_io/boundary_types.py`
        units: 体网格与面网格的坐标单位（mm / m / auto，两者同一单位）
    """
    from autoflowcfd.grid.mesh_gen.utils.mesh_external_import import import_external_volume_mesh
    from autoflowcfd.grid.nas_io.boundary_types import parse_bc_type_options

    logger.info(f"Importing external volume mesh: {volume_mesh_file}")

    try:
        volume_mesh, report = import_external_volume_mesh(
            volume_mesh_file, surface_mesh,
            repair=not skip_repair,
            max_repair_passes=max_repair_passes,
            check_overlap=not skip_overlap_check,
            units=units,
            bc_types=parse_bc_type_options(bc_type),
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
