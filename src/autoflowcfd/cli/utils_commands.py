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
        try:
            import cupy as cp
            # 尝试在 GPU 上创建一个简单数组（失败即视为不可用）
            cp.array([1, 2, 3])
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
            click.echo("\n环境诊断")
            click.echo(f"{'='*60}")
            
            if issues:
                click.echo("状态: ❌ 不健康")
                click.echo(f"\n问题 ({len(issues)}):")
                for issue in issues:
                    click.echo(f"  ❌ {issue}")
            else:
                click.echo("状态: ✅ 健康")
            
            if warnings:
                click.echo(f"\n警告 ({len(warnings)}):")
                for warning in warnings:
                    click.echo(f"  ⚠️  {warning}")
            
            click.echo("\n系统信息:")
            click.echo(f"  Python:      {info.get('python_version', 'unknown')}")
            click.echo(f"  CPU 核心数:   {info.get('cpu_cores', 'unknown')}")
            click.echo(f"  GPU 状态:  {info.get('gpu_status', 'unknown')}")
            
            if info.get('installed_packages'):
                click.echo("\n已安装的包:")
                for pkg, ver in info['installed_packages'].items():
                    click.echo(f"  ✓ {pkg:<20} {ver}")
            
            if info.get('optional_packages'):
                click.echo("\n可选包:")
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
@click.option("--surface-mesh", "-s", type=click.Path(exists=True), default=None,
              help="grid_file 是 .nas 体网格时必填：原始面网格（反推边界分组），与 solve steady 相同")
@click.option("--skip-quality-check", is_flag=True, help="跳过网格质量门（与 solve steady 相同）")
@click.option("--backend", "-b", type=click.Choice(["cpu", "gpu"]),
              default="cpu", help="要基准测试的后端")
@click.option("--iterations", "-n", type=int, default=100,
              help="测试迭代次数")
@click.option("--order", "-p", type=click.IntRange(1, 3), default=2,
              help="FR 阶数")
@click.option("--json", "-j", "json_output", is_flag=True, help="以 JSON 格式输出")
def benchmark(
    grid_file: str,
    surface_mesh: str,
    skip_quality_check: bool,
    backend: str,
    iterations: int,
    order: int,
    json_output: bool
) -> None:
    """单机无粘残差求值的计时与内存基准。

    网格加载（含质量门）、边界条件与求解器构造与 `solve steady` 是同一条路径（`load_mesh_for_solver` +
    `build_single_node_solver`），计时的是求解器自己的无粘残差（真实边界幽灵态、自由来流初场）。

    2026-10-19（I-5）改动：此前输入是面网格、命令内部先生成体网格（tetgen 耗时算进"网格加载"），残差调用不传
    边界幽灵态提供者、由核心函数静默换成零梯度外插——计时的是"边界不可见"的残差，与真实求解不是同一个计算。
    体网格请先用 `grid generate-volume` 生成。

    Args:
        grid_file: 体网格（.pkl，或 GRID+CTETRA/CPENTA 的 .nas 并给出 --surface-mesh）
        surface_mesh: .nas 体网格对应的原始面网格
        skip_quality_check: 跳过网格质量门
        backend: 要基准测试的后端 (cpu/gpu)
        iterations: 迭代次数
        order: FR 阶数
        json_output: 以 JSON 格式输出

    Examples:
        $ autoflowcfd utils benchmark volume_mesh.pkl --backend cpu
        $ autoflowcfd utils benchmark volume.nas -s sedan.nas --backend gpu -n 200
    """
    import time as _time

    from autoflowcfd.cli.solve.mesh_loader import load_mesh_for_solver
    from autoflowcfd.cli.solve.solver_factory import build_single_node_solver

    logger.info(f"运行基准测试: grid={grid_file}, backend={backend}")
    t_start = _time.perf_counter()
    mesh, volume_data = load_mesh_for_solver(grid_file, order, surface_mesh=surface_mesh,
                                             skip_quality_check=skip_quality_check)
    t_load = _time.perf_counter() - t_start
    # GPU 不可用时 build_single_node_solver 显式报错（不退化成 CPU 却仍打着 GPU 的标签）
    # 来流与时间格式取配置层默认值（与 solve steady 不给参数时同一组，`config/solver_config.py`）
    from autoflowcfd.config.solver_config import SteadyConfig

    cfg = SteadyConfig()
    solver = build_single_node_solver(backend, mesh, volume_data, order=order, turb_model_name="NONE",
                                      time_scheme=cfg.time_scheme, rho_inf=cfg.rho_inf, vel_inf=cfg.vel_inf,
                                      p_inf=cfg.p_inf, mu_molecular=cfg.mu_molecular)
    t_setup = _time.perf_counter() - t_start - t_load
    residual = solver.compute_inviscid_residual_gpu if backend == "gpu" else solver.compute_inviscid_residual

    # 预热（numba JIT / 首次 GPU kernel 编译）；第一次求值失败说明基准本身跑不通，照常抛出
    residual()
    t_bench_start = _time.perf_counter()
    for _ in range(iterations):
        residual()
    t_bench = _time.perf_counter() - t_bench_start

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
        "n_cells": int(mesh.n_cells),
        "n_sps": int(mesh.n_sps_per_cell),
        "grid_load_time_s": round(t_load, 3),
        "solver_setup_time_s": round(t_setup, 3),
        "benchmark_time_s": round(t_bench, 3),
        "rate_iter_per_min": round(rate_per_min, 1),
        "memory_mb": round(mem_mb, 1),
    }
    if json_output:
        click.echo(json.dumps(result, indent=2))
        return
    click.echo("\n性能基准测试结果")
    click.echo(f"{'='*60}")
    click.echo(f"网格文件:    {grid_file}")
    click.echo(f"后端:        {backend.upper()}")
    click.echo(f"多项式阶数:  P{order}")
    click.echo(f"单元数:      {mesh.n_cells}")
    click.echo(f"每单元SPs:   {mesh.n_sps_per_cell}")
    click.echo(f"迭代次数:    {iterations}")
    click.echo(f"网格加载:    {t_load:.3f} s")
    click.echo(f"求解器构造:  {t_setup:.3f} s")
    click.echo(f"基准测试:    {t_bench:.3f} s")
    click.echo(f"计算速率:    {rate_per_min:.1f} iter/min")
    if mem_mb > 0:
        click.echo(f"内存占用:    {mem_mb:.1f} MB")
    click.echo(f"{'='*60}")
