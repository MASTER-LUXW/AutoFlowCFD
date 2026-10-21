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

from typing import Any, Dict

import click

from autoflowcfd.cli.solve.steady.command import _TIME_SCHEME_CHOICES, _TURBULENCE_CHOICES
from autoflowcfd.cli.grid.options import bc_type_option
from autoflowcfd.grid.nas_io.units import UNITS_CHOICES
import json
import sys
from loguru import logger


@click.group()
def utils() -> None:
    """实用工具命令。

    系统实用工具和诊断工具。

    \b
    Examples:
        # 检查版本
        $ autoflowcfd utils version

    \b
        # 运行诊断
        $ autoflowcfd utils doctor

    \b
        # 性能基准测试
        $ autoflowcfd utils benchmark --grid model.nas
    """
    pass


@utils.command()
@click.option("--json", "-j", "json_output", is_flag=True, help="以 JSON 格式输出")
def version(json_output: bool) -> None:
    """显示版本信息。

    显示 AutoFlowCFD 版本和构建信息。


    \b
    Examples:
        # 基本版本
        $ autoflowcfd utils version

    \b
        # 详细 JSON 输出
        $ autoflowcfd utils version --json

    \f

    Args:
        json_output: 以 JSON 格式输出
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


def _dependency_report():
    """按已安装包的依赖声明（pyproject 的 [project] dependencies 与 optional-dependencies，经包元数据读取，唯一来源）逐项核对。

    Returns:
        (issues, warnings, versions)：必需依赖缺失 / 版本不符与 Python 版本不符进 issues；可选组（extra）的缺失进
        warnings；versions 为 {分发名: 已装版本或 None}
    """
    import re
    from importlib import metadata

    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet

    try:
        requires_python = metadata.metadata("autoflowcfd")["Requires-Python"] or ""
        declared = metadata.requires("autoflowcfd") or []
    except metadata.PackageNotFoundError:
        return (["autoflowcfd 未以包形式安装（在项目目录执行 pip install -e .），无法按依赖声明检查环境"], [], {})
    issues, warnings, versions = [], [], {}
    python = ".".join(str(v) for v in sys.version_info[:3])
    if python not in SpecifierSet(requires_python):
        issues.append(f"Python {python} 不满足 {requires_python}")
    for text in declared:
        req = Requirement(text)
        match = re.search(r'extra == "([^"]+)"', str(req.marker)) if req.marker else None
        extra = match.group(1) if match else None
        if req.marker is not None and not req.marker.evaluate({"extra": extra or ""}):
            continue                                   # 平台 / Python 版本标记不适用于当前环境
        target = warnings if extra else issues
        label = f"可选组 [{extra}] 的依赖" if extra else "必需依赖"
        try:
            version = metadata.version(req.name)
        except metadata.PackageNotFoundError:
            versions[req.name] = None
            target.append(f"缺少{label} {req.name}{req.specifier}")
            continue
        versions[req.name] = version
        if req.specifier and version not in req.specifier:
            target.append(f"{label} {req.name} {version} 不满足 {req.specifier}")
    return issues, warnings, versions


@utils.command()
@click.option("--json", "-j", "json_output", is_flag=True, help="以 JSON 格式输出")
def doctor(json_output: bool) -> None:
    """运行环境诊断。

    检查系统环境中是否存在潜在问题和缺失的依赖项。

    \b
    Examples:
        # 运行诊断
        $ autoflowcfd utils doctor

    \b
        # JSON 输出
        $ autoflowcfd utils doctor --json

    \f

    Args:
        json_output: 以 JSON 格式输出
    """
    if not json_output:
        logger.info("正在运行环境诊断...")

    try:
        issues = []
        warnings = []
        info: Dict[str, Any] = {}

        info['python_version'] = ".".join(str(v) for v in sys.version_info[:3])
        # 依赖按 pyproject 的声明逐项核对（2026-10-20 以前清单写死：scipy、tetgen、pyvista 没查，numba、h5py 被列为
        # "可选"——求解核心依赖 numba 内核、续算依赖 h5py checkpoint，Python 门槛写的 3.8）
        dep_issues, dep_warnings, versions = _dependency_report()
        issues.extend(dep_issues)
        warnings.extend(dep_warnings)
        info['dependencies'] = versions

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

            if info.get('dependencies'):
                click.echo("\n依赖（按 pyproject 声明）:")
                for pkg, ver in info['dependencies'].items():
                    click.echo(f"  {'✓' if ver else '✗'} {pkg:<20} {ver or '未安装'}")

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
@click.option("--units", type=click.Choice(list(UNITS_CHOICES)), default="mm", show_default=True,
              help="grid_file 是 .nas 体网格时它与 --surface-mesh 的坐标单位")
@click.option("--backend", "-b", type=click.Choice(["cpu", "gpu"]),
              default="cpu", help="要基准测试的后端")
@click.option("--iterations", "-n", type=int, default=100,
              help="测试迭代次数")
@click.option("--order", "-p", type=click.IntRange(0, 3), default=2,
              help="FR 阶数（固定阶数计时，不做阶数爬升）")
@click.option("--time-scheme", "-t", type=click.Choice(_TIME_SCHEME_CHOICES), default="rk3", show_default=True,
              help="时间格式（与 solve steady 同义）")
@click.option("--turbulence-model", "-m", type=click.Choice(_TURBULENCE_CHOICES), default="none", show_default=True,
              help="湍流模型（与 solve steady 同义）")
@bc_type_option
@click.option("--json", "-j", "json_output", is_flag=True, help="以 JSON 格式输出")
def benchmark(
    grid_file: str,
    surface_mesh: str,
    skip_quality_check: bool,
    units: str,
    backend: str,
    iterations: int,
    order: int,
    time_scheme: str,
    turbulence_model: str,
    bc_type: tuple,
    json_output: bool
) -> None:
    """单机完整求解步的计时与内存基准（固定阶数，所选时间格式与湍流模型）。

    网格加载（含质量门）、边界条件与求解器构造与 `solve steady` 是同一条路径（`load_mesh_for_solver` +
    `build_single_node_solver`），计时的是 `solve()` 的完整求解步（粘性、湍流、隐式线性求解都在内，预热一步后
    计时 `iterations` 步）。分项：残差求值的次数与总耗时、隐式格式每步的 GMRES 迭代数与矩阵向量积次数，其余
    （线性求解与预处理、湍流、步长、监控）为总耗时减去残差耗时——JFNK 的矩阵向量积本身就是残差求值，按调用拆分
    不重叠。输入是体网格（`grid generate-volume` 的输出或 .pkl）。

    \b
    Examples:
        $ autoflowcfd utils benchmark volume_mesh.pkl --backend cpu
        $ autoflowcfd utils benchmark volume_mesh.pkl -t newton-krylov -m sa -p 2 -n 20
    \f

    Args:
        grid_file: 体网格（.pkl，或 GRID+CTETRA/CPENTA 的 .nas 并给出 --surface-mesh）
        surface_mesh: .nas 体网格对应的原始面网格
        skip_quality_check: 跳过网格质量门
        units: .nas 体网格与面网格的坐标单位
        backend: 要基准测试的后端 (cpu/gpu)
        iterations: 迭代次数
        order: FR 阶数
        time_scheme: 时间格式（rk3 / newton-krylov）
        turbulence_model: 湍流模型
        json_output: 以 JSON 格式输出
        bc_type: `--bc-type 组名=类型`（可重复）
    """
    import time as _time

    from autoflowcfd.cli.solve.mesh_loader import load_mesh_for_solver
    from autoflowcfd.cli.solve.solver_factory import build_single_node_solver

    logger.info(f"运行基准测试: grid={grid_file}, backend={backend}")
    t_start = _time.perf_counter()
    from autoflowcfd.grid.nas_io.boundary_types import parse_bc_type_options

    mesh, volume_data = load_mesh_for_solver(grid_file, order, surface_mesh=surface_mesh,
                                             skip_quality_check=skip_quality_check, units=units,
                                             bc_types=parse_bc_type_options(bc_type))
    t_load = _time.perf_counter() - t_start
    # GPU 不可用时 build_single_node_solver 显式报错（不退化成 CPU 却仍打着 GPU 的标签）
    # 来流与时间格式取配置层默认值（与 solve steady 不给参数时同一组，`config/solver_config.py`）
    from autoflowcfd.config.solver_config import SteadyConfig

    from autoflowcfd.core.time_integration.base import scheme_from_name

    cfg = SteadyConfig()
    solver = build_single_node_solver(backend, mesh, volume_data, order=order, turb_model_name=turbulence_model,
                                      time_scheme=scheme_from_name(time_scheme), rho_inf=cfg.rho_inf,
                                      vel_inf=cfg.vel_inf, p_inf=cfg.p_inf, mu_molecular=cfg.mu_molecular)
    solver.order_continuation_enabled = False                  # 固定阶数计时
    t_setup = _time.perf_counter() - t_start - t_load

    residual_calls = {"n": 0, "t": 0.0}
    names = ("compute_inviscid_residual_gpu", "compute_viscous_residual_gpu") if backend == "gpu" \
        else ("compute_mean_flow_residual",)
    for name in names:
        _time_calls(solver, name, residual_calls, count=(name == names[0]))
    newton = {"gmres": 0, "matvec": 0}
    step = solver.step

    def counted_step(dt):
        res = step(dt)
        info = getattr(solver, "_newton_last_info", None)
        if info:
            newton["gmres"] += int(info["gmres_iters"])
            newton["matvec"] += int(info["n_matvec"])
        return res

    solver.step = counted_step
    # 预热一步（numba JIT / 首次 GPU kernel 编译）；跑不通说明基准本身不成立，照常抛出
    solver.solve(max_iter=1, tol=0.0)
    residual_calls.update(n=0, t=0.0)
    newton.update(gmres=0, matvec=0)
    t_bench_start = _time.perf_counter()
    solver.solve(max_iter=iterations, tol=0.0)
    t_bench = _time.perf_counter() - t_bench_start

    try:
        import psutil
        mem_mb = psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:                                        # 可选组 [perf]
        mem_mb = None
    per_step = t_bench / iterations
    result = {
        "command": "utils.benchmark",
        "status": "success",
        "grid_file": grid_file,
        "backend": backend,
        "order": order,
        "time_scheme": time_scheme,
        "turbulence_model": turbulence_model,
        "steps": iterations,
        "n_cells": int(mesh.n_cells),
        "n_sps": int(mesh.n_sps_per_cell),
        "grid_load_time_s": round(t_load, 3),
        "solver_setup_time_s": round(t_setup, 3),
        "steps_time_s": round(t_bench, 3),
        "time_per_step_s": round(per_step, 4),
        "residual_evaluations_per_step": round(residual_calls["n"] / iterations, 2),
        "residual_time_per_step_s": round(residual_calls["t"] / iterations, 4),
        "other_time_per_step_s": round(per_step - residual_calls["t"] / iterations, 4),
        "gmres_iterations_per_step": round(newton["gmres"] / iterations, 2),
        "matvec_per_step": round(newton["matvec"] / iterations, 2),
        "memory_mb": None if mem_mb is None else round(mem_mb, 1),
    }
    if json_output:
        click.echo(json.dumps(result, indent=2))
        return
    click.echo("\n性能基准测试结果")
    click.echo(f"{'='*60}")
    click.echo(f"网格文件:    {grid_file}")
    click.echo(f"后端 / 阶数: {backend.upper()} / P{order}   时间格式: {time_scheme}   湍流模型: {turbulence_model}")
    click.echo(f"单元数:      {mesh.n_cells}（每单元 {mesh.n_sps_per_cell} 个解点）")
    click.echo(f"网格加载:    {t_load:.3f} s    求解器构造: {t_setup:.3f} s")
    click.echo(f"求解步:      {iterations} 步 {t_bench:.3f} s，每步 {per_step:.4f} s（{60.0 / per_step:.1f} 步/分钟）")
    click.echo(f"  残差求值:  每步 {result['residual_evaluations_per_step']} 次，{result['residual_time_per_step_s']} s")
    click.echo(f"  其余:      每步 {result['other_time_per_step_s']} s（线性求解与预处理、湍流、步长、监控）")
    if newton["gmres"]:
        click.echo(f"  隐式:      每步 GMRES {result['gmres_iterations_per_step']} 次迭代、"
                   f"{result['matvec_per_step']} 次矩阵向量积")
    if mem_mb is not None:
        click.echo(f"内存占用:    {mem_mb:.1f} MB")
    click.echo(f"{'='*60}")


def _time_calls(obj, name: str, acc: dict, count: bool = True) -> None:
    """在实例上把方法 `name` 换成计时包装：累计耗时到 acc["t"]，`count` 时累计调用次数到 acc["n"]。"""
    import time as _time

    method = getattr(obj, name)

    def timed(*args, **kwargs):
        t0 = _time.perf_counter()
        try:
            return method(*args, **kwargs)
        finally:
            acc["t"] += _time.perf_counter() - t0
            if count:
                acc["n"] += 1

    setattr(obj, name, timed)
