"""实用工具子命令。

本模块提供 AutoFlowCFD 的 CLI 实用工具命令。

命令:
    - version: 显示版本信息
    - doctor: 环境诊断
    - benchmark: 性能基准测试

示例:
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
        import time as _time
        import numpy as np

        # 加载网格：真实 bug（已修复，2026-08-21）——此前这里直接
        # `HighOrderMesh(grid_data, order=order)`，grid_data 是
        # NASParser.parse() 返回的*面*网格 GridData（三角面片），既不是
        # HighOrderMesh.__init__ 接受的参数（它只有 order 一个参数，
        # grid_data 位置传参会顶替掉 order，"got multiple values for
        # argument 'order'" 就是这么来的——真实复现），也不是
        # load_from_volume_mesh 需要的 VolumeMeshData（体网格，四面体/
        # 棱柱），这条命令此前对任何输入都会立即报错崩溃，从未真正跑通
        # 过一次基准测试。改成与 `grid generate-volume` 完全相同的管线
        # （parser.generate_volume_mesh_from_surface，见
        # cli/grid_volume_commands.py::generate_volume 文档）先从面网格
        # 生成体网格，再用 HighOrderMesh(order=order).load_from_volume_mesh
        # 加载——这是本项目里唯一真正构造出可用 HighOrderMesh 的路径。
        t_start = _time.perf_counter()
        from autoflowcfd.grid.nas_io.parser import NASParser
        parser = NASParser(grid_file)
        surface_grid = parser.parse()
        volume_mesh = parser.generate_volume_mesh_from_surface(
            surface_grid,
            volume_mesh_params={
                'growth_rate': 1.2,
                'min_cell_size': 0.001,
                'target_cells': 500000,
                'max_cell_size': None,
                'bl_layers': None,
                'bl_only': False,
                'core_only': False,
                'output': None,
            },
        )
        t_load = _time.perf_counter() - t_start

        # 构建高阶网格
        from autoflowcfd.grid.high_order.high_order_mesh import HighOrderMesh
        mesh = HighOrderMesh(order=order)
        mesh.load_from_volume_mesh(volume_mesh)
        t_mesh = _time.perf_counter() - t_start - t_load

        # 初始化自由来流解向量
        from autoflowcfd.core.fr_residual.inviscid import compute_inviscid_residual_fr
        from autoflowcfd.fr.operators import generate_fr_operators
        ops = generate_fr_operators(order)
        n_cells = mesh.n_cells
        n_sps = mesh.n_sps_per_cell
        n_vars = 5
        rho_inf, u_inf, p_inf = 1.225, 30.0, 101325.0
        E_inf = p_inf / 0.4 + 0.5 * rho_inf * u_inf**2
        U_init = np.zeros((n_cells, n_sps, n_vars))
        U_init[:, :, 0] = rho_inf
        U_init[:, :, 1] = rho_inf * u_inf
        U_init[:, :, 4] = E_inf

        # 预热（触发 Numba JIT 编译）——第一次真实求值失败说明基准测试
        # 本身就跑不通，不能吞掉继续假装成功，让它正常抛出。
        _ = compute_inviscid_residual_fr(U_init, mesh, ops)

        # 正式基准测试
        t_bench_start = _time.perf_counter()
        for _ in range(iterations):
            _ = compute_inviscid_residual_fr(U_init, mesh, ops)
        t_bench = _time.perf_counter() - t_bench_start

        # 内存使用
        try:
            import psutil
            mem_mb = psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            mem_mb = -1.0

        rate_per_min = iterations / (t_bench / 60.0) if t_bench > 0 else 0.0

        result = {
            "command": "utils.benchmark",
            "status": "success",
            "grid_file": grid_file,
            "backend": backend,
            "order": order,
            "iterations": iterations,
            "n_cells": int(n_cells),
            "n_sps": int(n_sps),
            "grid_load_time_s": round(t_load, 3),
            "mesh_build_time_s": round(t_mesh, 3),
            "benchmark_time_s": round(t_bench, 3),
            "rate_iter_per_min": round(rate_per_min, 1),
            "memory_mb": round(mem_mb, 1),
        }

        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo("\n性能基准测试结果")
            click.echo(f"{'='*60}")
            click.echo(f"网格文件:    {grid_file}")
            click.echo(f"后端:        {backend.upper()}")
            click.echo(f"多项式阶数:  P{order}")
            click.echo(f"单元数:      {n_cells}")
            click.echo(f"每单元SPs:   {n_sps}")
            click.echo(f"迭代次数:    {iterations}")
            click.echo(f"网格加载:    {t_load:.3f} s")
            click.echo(f"网格构建:    {t_mesh:.3f} s")
            click.echo(f"基准测试:    {t_bench:.3f} s")
            click.echo(f"计算速率:    {rate_per_min:.1f} iter/min")
            if mem_mb > 0:
                click.echo(f"内存占用:    {mem_mb:.1f} MB")
            click.echo(f"{'='*60}")
    
    except Exception as e:
        logger.error(f"基准测试失败: {e}")
        raise click.ClickException(f"基准测试失败: {e}")
