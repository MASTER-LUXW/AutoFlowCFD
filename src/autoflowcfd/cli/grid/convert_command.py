"""`autoflowcfd grid convert`：.nas 网格（面网格或体网格）转换为 VTK / STL / CGNS。"""

import json
from pathlib import Path
from typing import Optional

import click
from loguru import logger

from autoflowcfd.cli.grid.commands import grid


@grid.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--format", "-f", "fmt", type=click.Choice(["vtk", "cgns", "stl"]), required=True, help="输出格式")
@click.option("--output", "-o", type=click.Path(dir_okay=False),
              help="输出文件路径（默认：与输入同目录同名，扩展名 .vtu/.cgns/.stl）")
@click.option("--units", type=click.Choice(["mm", "m", "auto"]), default="mm", show_default=True,
              help="输入文件的坐标单位；输出一律为米")
@click.option("--json", "-j", "json_output", is_flag=True, help="以 JSON 格式输出")
def convert(input_file: str, fmt: str, output: Optional[str], units: str, json_output: bool) -> None:
    """转换网格为 VTK、CGNS 或 STL。

    \b
    输入（自动识别）：
      - 面网格 .nas（CTRIA3 + PSHELL 分组）
      - 体网格 .nas（CTETRA/CPENTA；文件里带 CTRIA3 边界面时按这些面的 PSHELL 分组给外部面定组）
    输出（单位米）：
      - vtk ：非结构网格（.vtu XML / .vtk 旧版，按扩展名）。体网格写体单元，面网格写三角形与组号
      - stl ：ASCII STL，每个边界组一个 solid（体网格为外部面）
      - cgns：CGNS/HDF5。体网格：PENTA_6/TETRA_4 体单元、每组一个 TRI_3 面单元分区、ZoneBC（每组一个
              Family_t + FamilySpecified 的 BC_t）；面网格：每组一个 TRI_3 分区

    \b
    Examples:
      autoflowcfd grid convert sedan.nas -f stl -o sedan.stl
      autoflowcfd grid convert volume.nas -f cgns
    """
    from autoflowcfd.grid.conversion import EXTENSIONS, WRITERS, load_convertible_mesh

    out = Path(output) if output else Path(input_file).with_suffix(EXTENSIONS[fmt])
    try:
        mesh = load_convertible_mesh(input_file, units=units)
        WRITERS[fmt](mesh, str(out))
    except Exception as e:
        logger.error(f"Grid conversion failed: {e}")
        if json_output:
            click.echo(json.dumps({"command": "grid.convert", "status": "error", "error": str(e)}, indent=2))
        raise click.ClickException(f"Grid conversion failed: {e}")

    result = {
        "command": "grid.convert",
        "status": "success",
        "input": str(input_file),
        "output": str(out),
        "format": fmt,
        "kind": "volume" if mesh.is_volume else "surface",
        "nodes": int(len(mesh.nodes)),
        "tetrahedra": int(len(mesh.tets)),
        "prisms": int(len(mesh.prisms)),
        "boundary_triangles": int(len(mesh.surface_tris)),
        "boundary_groups": {name: int((mesh.surface_group == gi).sum()) for gi, name in enumerate(mesh.group_names)},
    }
    if json_output:
        click.echo(json.dumps(result, indent=2))
        return
    click.echo(f"Converted {result['kind']} mesh -> {fmt}: {out}")
    click.echo(f"  nodes={result['nodes']:,} tets={result['tetrahedra']:,} prisms={result['prisms']:,} "
               f"boundary triangles={result['boundary_triangles']:,}")
    for name, count in result["boundary_groups"].items():
        click.echo(f"  - {name}: {count:,}")
