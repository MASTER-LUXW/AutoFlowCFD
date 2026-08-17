"""V2.0 FR 求解器子命令。

本模块提供 V2.0 FR 求解器的 CLI 命令，支持高阶精度、多种时间推进方法和湍流模型。

命令:
    - steady: 运行稳态 FR 仿真
    - transient: 运行瞬态 FR 仿真（专用命令）
    - resume: 从检查点恢复
    - status: 查看求解器状态

示例:
    $ autoflowcfd solve steady model_volume.pkl --backend cpu --order 2 --turbulence-model sst

steady/transient 命令本体已拆分到 solve_steady_commands.py，本文件只保留
命令组定义 + resume/status。
"""

import logging
from pathlib import Path
from typing import Optional

import click

from autoflowcfd.core import FRSolver
from autoflowcfd.cli.solve_helpers import (
    compute_wall_distance_for_solver,
    load_mesh_for_solver,
    restore_state_from_checkpoint,
    save_results,
    write_checkpoint,
)

logger = logging.getLogger(__name__)


@click.group()
def solve():
    """FR 求解器相关命令 (稳态/瞬态)。"""
    pass


# 导入子命令模块，触发 @solve.command() 注册
from autoflowcfd.cli import solve_steady_commands  # noqa: F401
from autoflowcfd.cli.solve_steady_commands import _report_aerodynamic_coefficients  # noqa: F401



@solve.command()
@click.argument("checkpoint_file", type=click.Path(exists=True))
@click.option("--max-iter", "-n", default=500, help="额外迭代次数")
@click.option("--backend", "-b", type=click.Choice(["cpu", "gpu"]),
              default=None, help="后端覆盖")
@click.option("--surface-mesh", "-s", type=click.Path(exists=True), default=None,
              help="原始面网格路径——checkpoint 里记录的 input_file 若是 .nas 体网格则必填")
@click.option('--reference-area', type=float, default=None, help='气动系数参考面积 (m^2)')
@click.option('--threads', '-j', type=int, default=-1, help='CPU 后端 numba 并行 kernel 使用的线程数，默认 -1 = 4（本机真实网格实测扩展性甜点，不是核数）')
def resume(checkpoint_file: str, max_iter: int, backend: Optional[str],
           surface_mesh: Optional[str], reference_area: Optional[float], threads: int) -> None:
    """从检查点真正恢复并继续求解（不是只打印元信息）。

    重建流程：checkpoint 的 metadata 记录了重建 FRSolver 所需的全部
    构造参数（input_file/order/turbulence_model/backend/自由来流条件，
    见 solve_helpers.write_checkpoint 文档），用它们重新走一遍
    load_mesh_for_solver + FRSolver(...) 构造出一个全新求解器，再用
    checkpoint 里完整保存的 (n_cells,n_sps,n_vars) 状态（metadata['fields']
    ['U_sps']，不是拍扁过的单元中心近似）整体替换掉初始化时生成的均匀
    流场，然后调用 solver.solve() 继续迭代——此前这里只是把 checkpoint
    读出来打几行日志，从不重建求解器也不继续迭代，是伪装成可用命令的
    stub（V2.0 二次评审 CL-01 发现）。

    Args:
        checkpoint_file: checkpoint 文件路径（solve steady/transient 产出）
        max_iter: 从当前迭代数继续跑的额外迭代次数
        backend: 后端覆盖，None 时沿用 checkpoint 记录的原始后端
        surface_mesh: checkpoint 记录的 input_file 若是 .nas 体网格，
            需要提供原始面网格来反推边界分组（与 solve steady 的同名
            参数语义一致）
        reference_area: 气动系数参考面积
    """
    from types import SimpleNamespace
    from autoflowcfd.core.utils.checkpoint import CheckpointManager

    logger.info(f"Resuming simulation from checkpoint: {checkpoint_file}")

    solution, history, iteration, metadata = CheckpointManager(
        config=SimpleNamespace(), output_dir="."
    ).load(checkpoint_file)

    fields = metadata.get("fields", {})
    if "U_sps" not in fields:
        raise click.ClickException(
            f"Checkpoint '{checkpoint_file}' 缺少 'U_sps' 字段（完整的 (n_cells,n_sps,n_vars) "
            f"求解器状态）——这不是本版本 solve_helpers.write_checkpoint 写出的 checkpoint，"
            f"无法精确恢复（只有拍扁过的单元中心近似不足以重建高阶解）。"
        )

    input_file = metadata.get("input_file")
    order = int(metadata.get("order", 2))
    turbulence_model = metadata.get("turbulence_model", "sst")
    target_backend = backend or metadata.get("backend", "cpu")

    if not input_file:
        raise click.ClickException("Checkpoint metadata 缺少 'input_file'，无法重新加载网格。")

    logger.info(f"Rebuilding solver from checkpoint: input={input_file}, order=P{order}, "
                f"turbulence={turbulence_model}, backend={target_backend}, resuming at iter={iteration}")

    mesh, volume_data = load_mesh_for_solver(input_file, order, surface_mesh=surface_mesh)

    solver = FRSolver(
        mesh=mesh,
        backend=target_backend,
        order=order,
        turb_model_name=turbulence_model,
        rho_inf=metadata.get("rho_inf", 1.225),
        vel_inf=metadata.get("vel_inf", 30.0),
        p_inf=metadata.get("p_inf", 101325.0),
        n_threads=threads,
    )
    compute_wall_distance_for_solver(solver, volume_data)

    U_restored = fields["U_sps"]
    if U_restored.shape != solver.state.U.shape:
        raise click.ClickException(
            f"Checkpoint 状态形状 {U_restored.shape} 与重建求解器的状态形状 "
            f"{solver.state.U.shape} 不匹配（网格或阶数可能已变化），拒绝恢复。"
        )
    solver.state.U = U_restored
    solver.state._update_primitives()

    logger.info(f"State restored from checkpoint, continuing for {max_iter} more iterations...")
    result = solver.solve(max_iter=max_iter, dt=1e-3, tol=1e-6)
    print(f"\n✅ Resumed simulation finished: total_iterations~={iteration + result.iterations}, "
          f"Residual={result.final_residual:.6e}")

    output_dir = str(Path(checkpoint_file).parent.parent)
    save_results(solver, output_dir)
    write_checkpoint(solver, output_dir, iteration + result.iterations, input_file, order,
                      turbulence_model, target_backend)
    _report_aerodynamic_coefficients(solver, reference_area)


@solve.command()
@click.option("--backend", "-b", is_flag=True, help="列出可用后端")
def status(backend: bool) -> None:
    """查看求解器状态。

    Args:
        backend: 列出可用后端
    """
    if backend:
        from autoflowcfd.core.backend import get_available_backends
        backends = get_available_backends()
        logger.info(f"Available backends: {backends}")
    else:
        logger.info("V2.0 FR Solver Status: Ready")
        logger.info("Supported features:")
        logger.info("  - Orders: P1, P2, P3")
        logger.info("  - Time methods: RK3, IMEX, Dual-Time")
        logger.info("  - Turbulence models: SST, DDES, WMLES, LES")
        logger.info("  - Order continuation: P0 → P2/P3 smooth transition")
