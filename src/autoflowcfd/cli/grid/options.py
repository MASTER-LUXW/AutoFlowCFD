"""grid 子命令共用的选项（唯一定义）。"""

import click

from autoflowcfd.grid.nas_io.units import UNITS_CHOICES


def units_option(func):
    """--units：输入 .nas 文件的坐标单位（内部一律换算为米；'auto' 按原始包围盒大小判定）。"""
    return click.option("--units", type=click.Choice(list(UNITS_CHOICES)), default="mm", show_default=True,
                        help="输入 .nas 文件的坐标单位；内部一律换算为米，'auto' 按原始包围盒大小判定")(func)
