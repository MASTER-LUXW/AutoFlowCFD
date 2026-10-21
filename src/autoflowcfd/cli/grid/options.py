"""grid 子命令共用的选项（唯一定义）。"""

import click

from autoflowcfd.grid.nas_io.units import UNITS_CHOICES


def units_option(func):
    """--units：输入 .nas 文件的坐标单位（内部一律换算为米；'auto' 按原始包围盒大小判定）。"""
    return click.option("--units", type=click.Choice(list(UNITS_CHOICES)), default="mm", show_default=True,
                        help="输入 .nas 文件的坐标单位；内部一律换算为米，'auto' 按原始包围盒大小判定")(func)


def bc_type_option(func):
    """--bc-type 组名=类型（可重复）：覆盖按组名识别的边界条件类型（`grid/nas_io/boundary_types.py`）。"""
    from autoflowcfd.grid.nas_io.boundary_types import SOLVER_BC_TYPES

    return click.option("--bc-type", "bc_type", multiple=True,
                        help="边界条件类型，组名=类型（可重复），覆盖按组名识别的结果；名字识别不了或有歧义的组必须给出。"
                             f"类型取 {'/'.join(SOLVER_BC_TYPES)}")(func)
