"""后处理子命令。

本模块提供仿真结果后处理相关的 CLI 命令。

命令:
    - coefficients: 计算气动系数
    - export-vtk: 导出 VTK 场数据（搬到 post_export_commands.py，见下）
    - report: 生成仿真报告
    - convergence: 绘制收敛曲线
    - transient-mean: 瞬态平均流场分析（搬到 post_transient_commands.py）
    - transient-rms: 瞬态 RMS 脉动分析（搬到 post_transient_commands.py）
    - transient-psd: 瞬态频谱分析（搬到 post_transient_commands.py）

拆分说明（本文件原有 974 行，超过 400 行硬性拆分阈值——本仓库全部
Python 文件里单文件行数最多的一个）：
1. 案例目录/checkpoint 定位与加载的共用辅助函数（10 个）搬到
   post_helpers.py，镜像 cli/solve_commands.py + cli/solve_helpers.py
   已有的拆分方式。
2. export-vtk（单个命令约 170 行，全文件最重）搬到
   post_export_commands.py。
3. transient-mean/transient-rms/transient-psd 三个围绕"瞬态历史统计"
   主题的命令（合计约 260 行）搬到 post_transient_commands.py。
后两批命令都用普通 `@click.command()` 定义、在本文件末尾通过
`post.add_command(...)` 注册——与 cli/main.py 给顶层命令组注册到 cli、
cli/grid_commands.py 给 generate-volume/import-volume 注册到 grid
完全是同一套机制，注册后 `autoflowcfd post --help` 的可见效果、命令名、
选项、帮助文本都与拆分前完全一致。纯代码搬移，不改变任何行为。

示例:
    $ autoflowcfd post coefficients --case results/
    $ autoflowcfd post export-vtk --case results/ --output output.vtk
"""

import json
from pathlib import Path
from typing import Optional

import click
import numpy as np
from loguru import logger

from .post_helpers import _load_case, _load_history_only, _locate_checkpoint, _replay_history
from .solve_helpers import rebuild_solver_from_checkpoint


@click.group()
def post() -> None:
    """后处理命令。

    分析和可视化仿真结果。

    Examples:
        # Calculate coefficients
        $ autoflowcfd post coefficients --case results/

        # Export to VTK
        $ autoflowcfd post export-vtk --case results/
    """
    pass


@post.command()
@click.option("--case", "-c", required=True, type=click.Path(exists=True),
              help="Case directory containing a checkpoint written by 'solve steady/transient/resume'")
@click.option("--checkpoint", type=click.Path(exists=True),
              help="Checkpoint file path (defaults to latest under --case)")
@click.option("--surface-mesh", "-s", type=click.Path(exists=True), default=None,
              help="原始面网格路径——checkpoint 记录的 input_file 若是 .nas 体网格则必填"
                   "（与 'solve resume --surface-mesh' 语义一致，用于反推边界分组）")
@click.option("--reference-area", type=float, required=True,
              help="Reference area A_ref (m^2), 通常是车辆正面投影面积——没有默认值，"
                   "错的参考面积会给出误导性的 Cd/Cl，宁可强制用户显式指定")
@click.option("--backend", "-b", type=click.Choice(["cpu", "gpu"]), default=None,
              help="后端覆盖，默认沿用 checkpoint 记录的原始后端")
@click.option("--threads", type=int, default=-1, help="CPU 后端 numba 并行线程数")
@click.option("--output", "-o", type=click.Path(), default="coefficients.json",
              help="Output file")
@click.option("--json", "-j", "json_output", is_flag=True, help="Output as JSON")
def coefficients(
    case: str,
    checkpoint: Optional[str],
    surface_mesh: Optional[str],
    reference_area: float,
    backend: Optional[str],
    threads: int,
    output: str,
    json_output: bool
) -> None:
    """Calculate aerodynamic coefficients from a checkpoint (post-hoc, no re-solve).

    在 checkpoint 保存的 FR 原生解（(n_cells,n_sps,n_vars) 多点存储）上重建一个
    完整的 FRSolver（网格+面几何+状态），复用与 `solve steady` 收尾阶段完全相同的
    `postprocess.fr_coefficients.compute_aerodynamic_coefficients_fr`（WALL 边界
    压力+粘性力面积分），而不是 V1 时代假设单元中心 `GridData`/`SolutionVector`
    的 `CoefficientCalculator`（该实现调用的 `grid_data.get_face_data()` 从未
    存在过，Cd/Cl 恒为 0，见 ProjectFiles/V2.0/6_整体专家组二次评审.md 发现23）。

    Examples:
        $ autoflowcfd post coefficients --case results/ --reference-area 2.2
    """
    logger.info(f"Calculating aerodynamic coefficients for case: {case}")

    try:
        from autoflowcfd.postprocess.fr_coefficients import compute_aerodynamic_coefficients_fr

        ckpt_file = str(_locate_checkpoint(Path(case), checkpoint))
        solver, iteration, _metadata = rebuild_solver_from_checkpoint(
            ckpt_file, backend=backend, surface_mesh=surface_mesh, threads=threads,
        )
        coeffs = compute_aerodynamic_coefficients_fr(solver, reference_area=reference_area)
        result = coeffs.to_dict()

        output_path = Path(output)
        with open(output_path, 'w') as f:
            json.dump({'iteration': iteration, **result}, f, indent=2)
        logger.success(f"Coefficients written to: {output_path}")

        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\nAerodynamic Coefficients (iteration {iteration})")
            click.echo(f"{'='*40}")
            click.echo(f"Cd (Drag):     {result['Cd']:.4f}")
            click.echo(f"Cl (Lift):     {result['Cl']:.4f}")
            click.echo(f"Cs (Side):     {result['Cs']:.4f}")

    except Exception as e:
        logger.error(f"Coefficient calculation failed: {e}")
        raise click.ClickException(f"Failed to calculate coefficients: {e}")


@post.command()
@click.option("--case", "-c", required=True, type=click.Path(exists=True),
              help="Case directory")
@click.option("--checkpoint", type=click.Path(exists=True),
              help="Checkpoint file path (defaults to latest)")
@click.option("--output", "-o", type=click.Path(), default="report.json",
              help="Output report file")
@click.option("--format", "-f", type=click.Choice(["markdown", "html", "pdf", "json"]),
              default="json", help="Report format")
def report(case: str, checkpoint: Optional[str], output: str, format: str) -> None:
    """Generate simulation report.

    Create a comprehensive report including convergence history and
    aerodynamic coefficients, built from the checkpoint's saved
    convergence history (residuals/coefficients/CFL per iteration).

    Args:
        case: Case directory
        checkpoint: Checkpoint to report on (defaults to latest)
        output: Output report file
        format: Report format

    Examples:
        # JSON report (the only format actually implemented - see Note)
        $ autoflowcfd post report --case results/ --format json

    Note:
        Only 'json' is currently implemented (SimulationReport writes
        JSON). Requesting markdown/html/pdf falls back to JSON with a
        warning rather than silently producing an empty/fake file in a
        format nothing actually generates.
    """
    logger.info(f"Generating report for case: {case}")

    try:
        from autoflowcfd.postprocess import SimulationReport

        history, iteration, metadata = _load_history_only(case, checkpoint)
        analyzer = _replay_history(history)

        if format != "json":
            logger.warning(
                f"--format {format} was requested, but report generation only "
                "implements JSON output - writing JSON instead of silently "
                "producing an empty/fake file in an unsupported format."
            )

        # The checkpoint only stores a hash of the original solver
        # configuration (see CheckpointManager._compute_config_hash), not
        # the configuration itself - report honestly with what is actually
        # available instead of fabricating a full config dict.
        config_summary = {
            'config_hash': metadata.get('config_hash'),
            'original_backend': metadata.get('original_backend'),
            'timestamp': metadata.get('timestamp'),
        }

        output_path = Path(output)
        if output_path.suffix not in ('.json',):
            output_path = output_path.with_suffix('.json')

        sim_report = SimulationReport(config_summary, analyzer)
        path = sim_report.generate(str(output_path), metadata={'source_iteration': iteration})

        click.echo(f"✓ Report generated: {path}")

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise click.ClickException(f"Report generation failed: {e}")


@post.command()
@click.option("--case", "-c", required=True, type=click.Path(exists=True),
              help="Case directory")
@click.option("--checkpoint", type=click.Path(exists=True),
              help="Checkpoint file path (defaults to latest)")
@click.option("--output", "-o", type=click.Path(), default="convergence.png",
              help="Output plot file")
@click.option("--variables", multiple=True, default=["residual"],
              help="Variables to plot")
def convergence(case: str, checkpoint: Optional[str], output: str, variables: tuple) -> None:
    """Plot convergence history.

    Visualize residual convergence history and other monitoring variables.

    Args:
        case: Case directory
        checkpoint: Checkpoint to plot from (defaults to latest)
        output: Output plot file
        variables: Variables to plot ('residual' and/or 'cfl' and/or 'coefficients')

    Examples:
        # Plot residuals
        $ autoflowcfd post convergence --case results/

        # Save to file
        $ autoflowcfd post convergence --case results/ -o conv.png
    """
    logger.info(f"Plotting convergence for case: {case}")

    try:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            raise click.ClickException(
                "Convergence plotting requires matplotlib, which is not "
                "installed. Install it with: pip install matplotlib"
            )

        history, iteration, metadata = _load_history_only(case, checkpoint)
        iterations = history.get('iterations', [])
        if not iterations:
            raise click.ClickException(
                f"Checkpoint has no convergence history to plot (iteration {iteration})."
            )

        want_residual = not variables or 'residual' in variables
        want_cfl = 'cfl' in variables
        want_coeffs = 'coefficients' in variables or not variables

        n_panels = sum([want_residual, want_cfl, want_coeffs and 'coefficients' in history])
        n_panels = max(n_panels, 1)
        fig, axes = plt.subplots(n_panels, 1, figsize=(8, 3.2 * n_panels), sharex=True)
        if n_panels == 1:
            axes = [axes]
        panel = 0

        if want_residual and history.get('residuals'):
            ax = axes[panel]; panel += 1
            for eq_name, values in history['residuals'].items():
                ax.semilogy(iterations[:len(values)], np.maximum(values, 1e-16), label=eq_name)
            ax.set_ylabel("Residual")
            ax.legend()
            ax.grid(True, which='both', alpha=0.3)

        if want_cfl and history.get('cfl_history'):
            ax = axes[panel]; panel += 1
            ax.plot(iterations[:len(history['cfl_history'])], history['cfl_history'])
            ax.set_ylabel("CFL")
            ax.grid(True, alpha=0.3)

        if want_coeffs and history.get('coefficients'):
            ax = axes[panel]; panel += 1
            for name, values in history['coefficients'].items():
                ax.plot(iterations[:len(values)], values, label=name)
            ax.set_ylabel("Coefficient")
            ax.legend()
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Iteration")
        fig.tight_layout()

        output_path = Path(output)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        click.echo(f"✓ Convergence plot saved: {output_path}")

    except click.ClickException:
        raise
    except Exception as e:
        logger.error(f"Convergence plotting failed: {e}")
        raise click.ClickException(f"Convergence plotting failed: {e}")


# export-vtk / transient-mean / transient-rms / transient-psd 已搬到
# post_export_commands.py / post_transient_commands.py（见本文件顶部
# 拆分说明），这里用与 cli/grid_commands.py 给 generate-volume/
# import-volume 注册到 grid 完全一致的 add_command 机制接回来。
from .post_export_commands import export_vtk
from .post_transient_commands import transient_mean, transient_psd, transient_rms

post.add_command(export_vtk)
post.add_command(transient_mean)
post.add_command(transient_rms)
post.add_command(transient_psd)
