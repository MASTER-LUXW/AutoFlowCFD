"""后处理子命令组 `post`。

命令:
    - coefficients: 从 checkpoint 计算气动系数（重建求解器，按组积分壁面力）
    - export-vtk: 导出 VTK 场数据（`export_commands.py`）
    - report: 生成仿真报告
    - convergence: 绘制收敛曲线
    - transient-mean / transient-rms / transient-psd: 瞬态统计与探针频谱（`transient_commands.py`）

示例:
    $ autoflowcfd post coefficients --case results/ --reference-area 2.2
    $ autoflowcfd post export-vtk --case results/ --output output.vtk
"""

import json
from pathlib import Path
from typing import Optional

import click
import numpy as np
from loguru import logger

from autoflowcfd.cli.post.helpers import _load_history_only, _locate_checkpoint, _replay_history
from autoflowcfd.cli.solve.checkpoint_io import rebuild_solver_from_checkpoint


@click.group()
def post() -> None:
    """后处理命令。

    分析和可视化仿真结果。

    \b
    Examples:
        # 计算气动系数
        $ autoflowcfd post coefficients --case results/

    \b
        # 导出 VTK
        $ autoflowcfd post export-vtk --case results/
    """
    pass


@post.command()
@click.option("--case", "-c", required=True, type=click.Path(exists=True),
              help="算例目录，需包含 'solve steady/transient/resume' 写出的 checkpoint")
@click.option("--checkpoint", type=click.Path(exists=True),
              help="checkpoint 文件路径（默认取 --case 目录下最新一个）")
@click.option("--surface-mesh", "-s", type=click.Path(exists=True), default=None,
              help="原始面网格路径——checkpoint 记录的 input_file 若是 .nas 体网格则必填"
                   "（与 'solve resume --surface-mesh' 语义一致，用于反推边界分组）")
@click.option("--reference-area", type=float, required=True,
              help="参考面积 A_ref (m^2)，通常是车辆正面投影面积——没有默认值，"
                   "错的参考面积会给出误导性的 Cd/Cl，宁可强制用户显式指定")
@click.option("--reference-length", type=float, default=1.0, show_default=True,
              help="力矩参考长度 L_ref (m)，车辆通常取轴距")
@click.option("--moment-center", nargs=3, type=float, default=(0.0, 0.0, 0.0), show_default=True,
              help="力矩参考点 x y z (m)，车辆通常取前后轴中点在地面上的投影")
@click.option("--groups", multiple=True,
              help="参与积分的固壁组名（可重复给出；缺省为全部物理无滑移固壁组，例如要排除地面时只给车身各组）")
@click.option("--threads", type=int, default=-1, help="CPU 后端 numba 并行线程数")
@click.option("--skip-quality-check", is_flag=True,
              help="跳过重建时的网格质量门检查（原求解靠该选项才跑得起来的网格，后处理同样需要跳过；不建议，"
                   "仅用于临时诊断）")
@click.option("--output", "-o", type=click.Path(), default="coefficients.json",
              help="输出文件")
@click.option("--json", "-j", "json_output", is_flag=True, help="以 JSON 格式输出")
def coefficients(
    case: str,
    checkpoint: Optional[str],
    surface_mesh: Optional[str],
    reference_area: float,
    reference_length: float,
    moment_center: tuple,
    groups: tuple,
    threads: int,
    skip_quality_check: bool,
    output: str,
    json_output: bool
) -> None:
    """从 checkpoint 计算气动系数（事后计算，不重新求解）。

    在 checkpoint 保存的 FR 解（全部解点）上重建求解器（网格、面几何、状态），用与 `solve` 收尾同一个函数
    `postprocess.fr_coefficients.compute_aerodynamic_coefficients_fr` 积分壁面压力与粘性力。

    \b
    Examples:
        $ autoflowcfd post coefficients --case results/ --reference-area 2.2
    """
    logger.info(f"Calculating aerodynamic coefficients for case: {case}")

    try:
        from autoflowcfd.postprocess.fr_coefficients import compute_aerodynamic_coefficients_fr

        ckpt_file = str(_locate_checkpoint(Path(case), checkpoint))
        solver, iteration, _metadata = rebuild_solver_from_checkpoint(
            # 后处理只读状态、不迭代，一律在 CPU 上重建（GPU 写出的 checkpoint 状态与 CPU 同一格式）
            ckpt_file, backend="cpu", surface_mesh=surface_mesh, threads=threads,
            skip_quality_check=skip_quality_check,
        )
        coeffs = compute_aerodynamic_coefficients_fr(
            solver, reference_area=reference_area, reference_length=reference_length,
            moment_center=moment_center, groups=list(groups) or None)
        result = coeffs.to_dict()

        output_path = Path(output)
        with open(output_path, 'w') as f:
            json.dump({'iteration': iteration, 'reference_area': reference_area,
                       'reference_length': reference_length, 'moment_center': list(moment_center),
                       'groups': list(groups) or 'all WALL groups', **result}, f, indent=2)
        logger.success(f"Coefficients written to: {output_path}")

        if json_output:
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\nAerodynamic Coefficients (iteration {iteration})")
            click.echo(f"{'='*40}")
            click.echo(f"Cd (Drag):     {result['Cd']:.4f}")
            click.echo(f"Cl (Lift):     {result['Cl']:.4f}")
            click.echo(f"Cs (Side):     {result['Cs']:.4f}")
            click.echo(f"Cm (Pitch):    {result['Cm']:.4f}")
            click.echo(f"Cy (Yaw):      {result['Cy']:.4f}")
            click.echo(f"Cr (Roll):     {result['Cr']:.4f}")

    except Exception as e:
        logger.error(f"Coefficient calculation failed: {e}")
        raise click.ClickException(f"Failed to calculate coefficients: {e}")


@post.command()
@click.option("--case", "-c", required=True, type=click.Path(exists=True),
              help="算例目录")
@click.option("--checkpoint", type=click.Path(exists=True),
              help="checkpoint 文件路径（默认取最新一个）")
@click.option("--output", "-o", type=click.Path(), default="report.json",
              help="输出报告文件")
def report(case: str, checkpoint: Optional[str], output: str) -> None:
    """生成仿真报告（JSON）。

    由 checkpoint 保存的收敛历史（每次迭代的残差 / 系数 / CFL）与求解参数构建，包含收敛判定与气动系数。

    \b
    Examples:
        $ autoflowcfd post report --case results/
    
    \f

    Args:
        case: 算例目录
        checkpoint: 要生成报告的 checkpoint（默认取最新一个）
        output: 输出报告文件
    """
    logger.info(f"Generating report for case: {case}")

    try:
        from autoflowcfd.postprocess import SimulationReport

        history, iteration, metadata = _load_history_only(case, checkpoint)
        analyzer = _replay_history(history)

        # checkpoint 记录的求解参数（写出端逐项写进 metadata）
        config_summary = {
            'backend': metadata.get('backend'),
            'order': metadata.get('order'),
            'target_order': metadata.get('target_order'),
            'turbulence_model': metadata.get('turbulence_model'),
            'time_scheme': metadata.get('time_scheme'),
            'timestamp': metadata.get('timestamp'),
        }

        output_path = Path(output)
        if output_path.suffix not in ('.json',):
            output_path = output_path.with_suffix('.json')

        sim_report = SimulationReport(config_summary, analyzer, solve_status=metadata.get('solve_status'))
        path = sim_report.generate(str(output_path), metadata={'source_iteration': iteration})

        click.echo(f"✓ Report generated: {path}")

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise click.ClickException(f"Report generation failed: {e}")


@post.command()
@click.option("--case", "-c", required=True, type=click.Path(exists=True),
              help="算例目录")
@click.option("--checkpoint", type=click.Path(exists=True),
              help="checkpoint 文件路径（默认取最新一个）")
@click.option("--output", "-o", type=click.Path(), default="convergence.png",
              help="输出图片文件")
@click.option("--variables", multiple=True, type=click.Choice(["residual", "cfl", "coefficients"]),
              default=["residual"], show_default=True,
              help="要绘制的量（可重复给出）：residual 各方程残差、cfl、coefficients 逐步力系数")
def convergence(case: str, checkpoint: Optional[str], output: str, variables: tuple) -> None:
    """绘制收敛历史。

    可视化残差收敛历史及其他监控变量。

    \b
    Examples:
        # 绘制残差
        $ autoflowcfd post convergence --case results/

    \b
        # 保存到文件
        $ autoflowcfd post convergence --case results/ -o conv.png
    
    \f

    Args:
        case: 算例目录
        checkpoint: 要绘制的 checkpoint（默认取最新一个）
        output: 输出图片文件
        variables: 要绘制的变量（'residual' 和/或 'cfl' 和/或 'coefficients'）
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

        want_residual = 'residual' in variables
        want_cfl = 'cfl' in variables
        want_coeffs = 'coefficients' in variables
        # 请求的量 checkpoint 里没有记录时报错，不画一张空图
        source = {'residual': 'residuals', 'cfl': 'cfl_history', 'coefficients': 'coefficients'}
        missing = [v for v in dict.fromkeys(variables) if not history.get(source[v])]
        if missing:
            raise click.ClickException(
                f"checkpoint 的收敛历史没有 {missing}（coefficients 只在求解时设置了参考面积才记录）")

        n_panels = sum([want_residual, want_cfl, want_coeffs])
        fig, axes = plt.subplots(n_panels, 1, figsize=(8, 3.2 * n_panels), sharex=True)
        if n_panels == 1:
            axes = [axes]
        panel = 0

        if want_residual:
            ax = axes[panel]; panel += 1
            for eq_name, values in history['residuals'].items():
                ax.semilogy(iterations[:len(values)], np.maximum(values, 1e-16), label=eq_name)
            ax.set_ylabel("Residual")
            ax.legend()
            ax.grid(True, which='both', alpha=0.3)

        if want_cfl:
            ax = axes[panel]; panel += 1
            ax.plot(iterations[:len(history['cfl_history'])], history['cfl_history'])
            ax.set_ylabel("CFL")
            ax.grid(True, alpha=0.3)

        if want_coeffs:
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


# export-vtk 与瞬态统计命令定义在同目录的 export_commands.py / transient_commands.py
from autoflowcfd.cli.post.export_commands import export_vtk
from autoflowcfd.cli.post.transient_commands import transient_mean, transient_psd, transient_rms

post.add_command(export_vtk)
post.add_command(transient_mean)
post.add_command(transient_rms)
post.add_command(transient_psd)
