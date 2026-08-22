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

# 修复Windows控制台中文乱码问题
if sys.platform == 'win32':
    # 设置标准输出编码为UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
    # 设置环境变量
    os.environ['PYTHONIOENCODING'] = 'utf-8'

import click
from loguru import logger

from .. import __version__
from .grid_commands import grid
from .solve_commands import solve
from .post_commands import post
from .config_commands import config
from .utils_commands import utils

# 本 CLI 的日志输出和 click.echo() 调用中使用 Unicode 符号
#（勾选标记、°、³ 等）。Windows 下 stdout/stderr 默认使用活动控制台
#代码页（如中文环境为 GBK/936），而非 UTF-8，这些字符会触发
# UnicodeEncodeError 并中断命令执行。强制使用 UTF-8 并设置安全回退，
# 确保输出不会因宿主控制台代码页不同而导致 CLI 崩溃。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


@click.group()
@click.version_option(version=__version__, prog_name="AutoFlowCFD")
@click.option("--verbose", "-v", is_flag=True, help="启用详细输出")
def cli(verbose: bool) -> None:
    """AutoFlowCFD - 高性能汽车外流场 CFD 求解器。
    
    AutoFlowCFD 是专注于汽车外流场仿真的开源计算流体力学软件。
    
    主要特性:
        - FR（通量重构）高阶离散方法
        - SST k-ω、DDES、WMLES、LES 漫流模型体系
        - CPU (Numba) 和 GPU (CUDA) 计算后端
        - ANSA .nas 网格文件支持
        - 完善的后处理工具
    
    命令组:
        grid     网格处理（解析、验证、信息、转换、生成体网格、导入体网格）
        solve    求解器命令（稳态、瞬态、恢复、状态）
        post     后处理（气动系数、VTK导出、报告等）
        config   配置管理（初始化、查看、验证）
        utils    实用工具（版本、环境检查、性能测试）

    示例:
        # 解析网格文件
        $ autoflowcfd grid parse sedan.nas

        # 从面网格生成体网格（求解前必需步骤）
        $ autoflowcfd grid generate-volume sedan.nas -o sedan_volume.nas

        # 运行稳态仿真（输入必须为 .pkl 体网格）
        $ autoflowcfd solve steady sedan_volume.pkl --backend gpu --order 3

        # 运行瞬态 DES 仿真
        $ autoflowcfd solve transient sedan_volume.pkl --physical-time 0.3
        
        # 计算气动系数
        $ autoflowcfd post coefficients --case results/
        
        # 生成配置模板
        $ autoflowcfd config init --template steady
        
        # 检查环境
        $ autoflowcfd utils doctor
    
    使用以下命令查看特定命令的帮助信息:
        $ autoflowcfd <command> --help
        $ autoflowcfd <command> <subcommand> --help
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
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        )
    else:
        logger.remove()
        logger.add(
            lambda msg: click.echo(msg, err=True),
            level="INFO",
            format="<level>{message}</level>",
        )


# Register subcommand groups
cli.add_command(grid)
cli.add_command(solve)
cli.add_command(post)
cli.add_command(config)
cli.add_command(utils)


if __name__ == "__main__":
    cli()
