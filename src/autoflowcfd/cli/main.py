"""AutoFlowCFD 命令行接口。

本模块基于 Click 框架提供主 CLI 入口，支持网格处理、求解、后处理、
配置管理和实用工具等子命令。

示例:
    $ autoflowcfd --version
    AutoFlowCFD v0.2.0

    $ autoflowcfd grid parse --help
    Usage: autoflowcfd grid parse [OPTIONS] INPUT_FILE

    解析 ANSA .nas 网格文件。

    $ autoflowcfd solve steady --help
    Usage: autoflowcfd solve steady [OPTIONS] INPUT_FILE

    运行稳态 RANS 仿真。
"""

import sys
import os

# 本 CLI 的日志输出和 click.echo() 调用中使用 Unicode 符号
# （勾选标记、°、³ 等）。Windows 下 stdout/stderr 默认使用活动控制台
# 代码页（如中文环境为 GBK/936），而非 UTF-8，这些字符会触发
# UnicodeEncodeError 并中断命令执行。强制使用 UTF-8 并设置安全回退，
# 确保输出不会因宿主控制台代码页不同而导致 CLI 崩溃。
#
# 行缓冲（2026-10-08）：输出被重定向到文件时 Python 默认整块缓冲，长程计算的日志会滞后几十分钟
# （cube_demo 稳态实测：日志停在第 53 步时 checkpoint 已写到第 150 步），进程异常退出时最后一段直接丢失。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
# 子进程（mpirun 拉起的各 rank 等）同样用 UTF-8
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 以下导入放在标准流重配置之后（导入过程中的输出也要走 UTF-8）
import click  # noqa: E402
from loguru import logger  # noqa: E402

from .. import __version__  # noqa: E402
from autoflowcfd.cli.grid.commands import grid  # noqa: E402
from autoflowcfd.cli.solve.commands import solve  # noqa: E402
from autoflowcfd.cli.post.commands import post  # noqa: E402
from .config_commands import config  # noqa: E402
from .utils_commands import utils  # noqa: E402


@click.group()
@click.version_option(version=__version__, prog_name="AutoFlowCFD")
@click.option("--verbose", "-v", is_flag=True, help="启用详细输出")
def cli(verbose: bool) -> None:
    """AutoFlowCFD - 汽车外流场高阶 CFD 求解器（通量重构 FR，P0~P3）。

    \b
    命令组:
        grid     网格：解析、校验、信息、转换、生成体网格、导入体网格
        solve    求解：稳态、瞬态、续算、状态
        post     后处理：气动系数、VTK 导出、收敛曲线、报告、瞬态统计与压力谱
        config   配置文件：生成模板、查看、校验
        utils    工具：环境检查、性能基准

    \b
    典型流程（.nas 坐标缺省按毫米，米制文件加 --units m）:
        $ autoflowcfd grid generate-volume car.nas -o car_volume.nas
        $ autoflowcfd solve steady car_volume.nas --surface-mesh car.nas -o results
        $ autoflowcfd post coefficients --case results --reference-area 2.2
        $ autoflowcfd post export-vtk --case results -o flow.vtu

    各命令的选项：autoflowcfd <命令组> <命令> --help
    """
    # 日志一律走 stderr（err=True），不写 stdout：本 CLI 的多个子命令支持
    # `--json`/`-j` 输出机器可解析的 JSON 到 stdout，供下游 Agent 工具化
    # 调用（见项目功能点 CL-01）——真实 bug（已修复，2026-08-21）：此前
    # 这里用 `click.echo(msg)`（默认写 stdout），INFO 级日志（"正在验证
    # 配置: ..." 等，默认级别就会打印）与 --json 的 JSON payload 混在
    # 同一个 stdout 流里，`autoflowcfd config validate x.yaml --json`
    # 这类调用的输出根本不是合法 JSON，管道给 `jq`/`json.loads` 直接解析
    # 失败——不是某个子命令的孤立问题，是这里的全局 logger sink 配置
    # 影响所有子命令。终端交互式使用不受影响（stdout/stderr 都会显示在
    # 同一个终端里），只有把 stdout 单独重定向/管道消费（脚本化调用的
    # 标准做法）时才看得出区别，这正是这个 bug 会被忽略的原因。
    if verbose:
        logger.remove()
        logger.add(
            lambda msg: click.echo(msg, err=True),
            level="DEBUG",
            format=("<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
                    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"),
        )
    else:
        logger.remove()
        logger.add(
            lambda msg: click.echo(msg, err=True),
            level="INFO",
            format="<level>{message}</level>",
        )


# 注册子命令组
cli.add_command(grid)
cli.add_command(solve)
cli.add_command(post)
cli.add_command(config)
cli.add_command(utils)


if __name__ == "__main__":
    cli()
