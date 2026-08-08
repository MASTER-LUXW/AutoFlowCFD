"""实用工具子命令。

本模块提供 AutoFlowCFD 的 CLI 实用工具命令。

Commands:
    - version: 显示版本信息
    - doctor: 环境诊断
    - benchmark: 性能基准测试

Example:
    $ autoflowcfd utils version
    $ autoflowcfd utils doctor
    $ autoflowcfd utils benchmark --grid model.nas --backend cpu
"""

import click
import json
import sys
from pathlib import Path
from loguru import logger


@click.group()
def utils() -> None:
    """实用工具命令。
    
    系统实用工具和诊断工具。
    
    Examples:
        # 检查版本
        $ autoflowcfd utils version
        
        # 运行诊断
        $ autoflowcfd utils doctor
        
        # 性能基准测试
        $ autoflowcfd utils benchmark --grid model.nas
    """
    pass


@utils.command()
@click.option("--json", "-j", "json_output", is_flag=True, help="以 JSON 格式输出")
def version(json_output: bool) -> None:
    """显示版本信息。
    
    显示 AutoFlowCFD 版本和构建信息。
    
    Args:
        json_output: 以 JSON 格式输出
    
    Examples:
        # 基本版本
        $ autoflowcfd utils version
        
        # 详细 JSON 输出
        $ autoflowcfd utils version --json
    """
    from autoflowcfd import __version__
    
    try:
        import platform
        import sys
        
        version_info = {
            "autoflowcfd": __version__,
            "python": sys.version,
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
        }
        
        if json_output:
            click.echo(json.dumps(version_info, indent=2))
        else:
            click.echo(f"\nAutoFlowCFD v{__version__}")
            click.echo(f"{'='*40}")
            click.echo(f"Python:   {sys.version.split()[0]}")
            click.echo(f"Platform: {platform.system()} {platform.release()}")
            click.echo(f"Machine:  {platform.machine()}")
            click.echo(f"{'='*40}")
    
    except Exception as e:
        logger.error(f"获取版本信息失败: {e}")
        raise click.ClickException(str(e))


@utils.command()
@click.option("--json", "-j", "json_output", is_flag=True, help="以 JSON 格式输出")
def doctor(json_output: bool) -> None:
    """运行环境诊断。

    检查系统环境中是否存在潜在问题和缺失的依赖项。

    Args:
        json_output: 以 JSON 格式输出

    Examples:
        # 运行诊断
        $ autoflowcfd utils doctor

        # JSON 输出
        $ autoflowcfd utils doctor --json
    """
    if not json_output:
        logger.info("正在运行环境诊断...")
    
    try:
        issues = []
        warnings = []
        info = {}
        
        # 检查 Python 版本
        python_version = sys.version_info
        info['python_version'] = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
        
        if python_version < (3, 8):
            issues.append(f"Python 版本 {python_version} 太旧。需要 Python 3.8+")
        
        # 检查必需的包
        required_packages = {
            'numpy': 'NumPy',
            'click': 'Click',
            'loguru': 'Loguru',
            'pyyaml': 'PyYAML',
        }
        
        installed_packages = {}
        for pkg_name, pkg_display in required_packages.items():
            try:
                if pkg_name == 'pyyaml':
                    import yaml
                    installed_packages[pkg_display] = yaml.__version__
                else:
                    pkg = __import__(pkg_name)
                    installed_packages[pkg_display] = getattr(pkg, '__version__', 'unknown')
            except ImportError:
                issues.append(f"缺少必需的包: {pkg_display}")
        
        info['installed_packages'] = installed_packages
        
        # 检查可选包
        optional_packages = {
            'cupy': 'CuPy (GPU 支持)',
            'numba': 'Numba (CPU 加速)',
            'h5py': 'h5py (HDF5 支持)',
        }
        
        optional_installed = {}
        for pkg_name, pkg_display in optional_packages.items():
            try:
                pkg = __import__(pkg_name)
                optional_installed[pkg_display] = getattr(pkg, '__version__', 'available')
            except ImportError:
                warnings.append(f"未安装可选包: {pkg_display}")
        
        info['optional_packages'] = optional_installed
        
        # 检查 GPU 可用性
        gpu_available = False
        try:
            import cupy as cp
            # 尝试在 GPU 上创建一个简单数组
            test_array = cp.array([1, 2, 3])
            gpu_available = True
            info['gpu_status'] = 'available'
        except Exception:
            info['gpu_status'] = 'not available'
            warnings.append("GPU (CUDA) 不可用。安装 CuPy 和 CUDA 工具包以获得 GPU 加速。")
        
        # 检查 CPU 核心数
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        info['cpu_cores'] = cpu_count
        
        # 汇总结果
        status = "healthy" if not issues else "unhealthy"
        
        result = {
            "command": "utils.doctor",
            "status": status,
            "issues": issues,
            "warnings": warnings,
            "info": info,
        }
        
        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\n环境诊断")
            click.echo(f"{'='*60}")
            
            if issues:
                click.echo(f"状态: ❌ 不健康")
                click.echo(f"\n问题 ({len(issues)}):")
                for issue in issues:
                    click.echo(f"  ❌ {issue}")
            else:
                click.echo(f"状态: ✅ 健康")
            
            if warnings:
                click.echo(f"\n警告 ({len(warnings)}):")
                for warning in warnings:
                    click.echo(f"  ⚠️  {warning}")
            
            click.echo(f"\n系统信息:")
            click.echo(f"  Python:      {info.get('python_version', 'unknown')}")
            click.echo(f"  CPU 核心数:   {info.get('cpu_cores', 'unknown')}")
            click.echo(f"  GPU 状态:  {info.get('gpu_status', 'unknown')}")
            
            if info.get('installed_packages'):
                click.echo(f"\n已安装的包:")
                for pkg, ver in info['installed_packages'].items():
                    click.echo(f"  ✓ {pkg:<20} {ver}")
            
            if info.get('optional_packages'):
                click.echo(f"\n可选包:")
                for pkg, ver in info['optional_packages'].items():
                    click.echo(f"  {'✓' if ver != 'not installed' else '✗'} {pkg:<20} {ver}")
            
            click.echo(f"{'='*60}")
            
            if not issues and not warnings:
                click.echo("\n✅ 一切看起来都很好！")
            elif not issues:
                click.echo(f"\n⚠️  发现 {len(warnings)} 个警告。系统功能正常但可以改进。")
    
    except Exception as e:
        logger.error(f"诊断失败: {e}")
        raise click.ClickException(f"诊断失败: {e}")


@utils.command()
@click.argument("grid_file", type=click.Path(exists=True))
@click.option("--backend", "-b", type=click.Choice(["cpu", "gpu"]),
              default="cpu", help="要基准测试的后端")
@click.option("--iterations", "-n", type=int, default=100,
              help="测试迭代次数")
@click.option("--order", "-p", type=click.IntRange(1, 3), default=2,
              help="FR 阶数")
@click.option("--json", "-j", "json_output", is_flag=True, help="以 JSON 格式输出")
def benchmark(
    grid_file: str,
    backend: str,
    iterations: int,
    order: int,
    json_output: bool
) -> None:
    """运行性能基准测试。
    
    测量指定网格和后端的计算速度和内存使用情况。
    
    Args:
        grid_file: .nas 网格文件路径
        backend: 要基准测试的后端 (cpu/gpu)
        iterations: 迭代次数
        order: FR 阶数
        json_output: 以 JSON 格式输出
    
    Examples:
        # CPU 基准测试
        $ autoflowcfd utils benchmark sedan.nas --backend cpu
        
        # GPU 基准测试
        $ autoflowcfd utils benchmark sedan.nas --backend gpu -n 200
    """
    logger.info(f"运行基准测试: grid={grid_file}, backend={backend}")
    
    try:
        # TODO: 实现实际的基准测试
        # 这需要加载网格、设置求解器并运行迭代
        
        logger.warning("基准测试功能正在开发中")
        
        result = {
            "command": "utils.benchmark",
            "status": "pending",
            "message": "基准测试功能即将推出",
            "grid_file": grid_file,
            "backend": backend,
            "iterations": iterations,
        }
        
        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo("\n性能基准测试")
            click.echo(f"{'='*60}")
            click.echo(f"网格文件:  {grid_file}")
            click.echo(f"后端:    {backend.upper()}")
            click.echo(f"迭代次数: {iterations}")
            click.echo(f"\n⚠️  基准测试功能正在开发中")
            click.echo(f"{'='*60}")
    
    except Exception as e:
        logger.error(f"基准测试失败: {e}")
        raise click.ClickException(f"基准测试失败: {e}")
