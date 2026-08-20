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

from autoflowcfd.cli.solve_helpers import (
    rebuild_solver_from_checkpoint,
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
    logger.info(f"Resuming simulation from checkpoint: {checkpoint_file}")

    solver, iteration, metadata = rebuild_solver_from_checkpoint(
        checkpoint_file, backend=backend, surface_mesh=surface_mesh, threads=threads,
    )
    input_file = metadata["input_file"]
    order = metadata["order"]
    turbulence_model = metadata["turbulence_model"]
    target_backend = metadata["backend"]

    logger.info(f"State restored from checkpoint (iter={iteration}), "
                f"continuing for {max_iter} more iterations...")
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
