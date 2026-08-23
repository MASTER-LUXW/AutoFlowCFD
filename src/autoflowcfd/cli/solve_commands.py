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

from pathlib import Path
from typing import Optional

import click
from loguru import logger

from autoflowcfd.cli.solve_helpers import (
    rebuild_solver_from_checkpoint,
    save_results,
    write_checkpoint,
)

# 真实 bug（已修复，2026-08-21）：此前这里 `import logging` +
# `logging.getLogger(__name__)` 用的是标准库 logging，不是本项目全局
# 统一用的 loguru（见 cli/main.py 顶层 `cli()` group 回调对 loguru 的
# sink 配置——标准库 logging 完全不受它影响）。项目从未对标准库
# logging 做过 basicConfig/加 handler，root logger 默认没有任何
# handler、默认级别 WARNING，`logger.info(...)` 因此被静默吞掉，不会
# 出现在 stdout 也不会出现在 stderr——真实复现：`solve status`（不带
# --backend）唯一的输出就是 5 行 logger.info，此前这条命令跑完退出码
# 0、终端上什么都不打印，是一个看起来"能跑但什么也不做"的伪装可用
# 命令；`solve resume` 同样两行进度日志被吞掉（其余关键结果走的是
# print()，未受影响，问题比 status 轻但同样是真实 bug）。改成和本
# 代码库其余所有文件一致的 `from loguru import logger`，走同一个已在
# cli/main.py 里配置好、路由到 stderr 的 sink。


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
              help="原始面网格路径覆盖——原 solve steady/transient 运行若传了它，"
                   "checkpoint 里已经存了一份，此处可不传；两者都缺失且 "
                   "input_file 是 .nas 体网格时才会报错")
@click.option('--reference-area', type=float, default=None, help='气动系数参考面积 (m^2)')
@click.option('--threads', '-j', type=int, default=-1, help='CPU 后端 numba 并行 kernel 使用的线程数，默认 -1 = 4（本机真实网格实测扩展性甜点，不是核数）')
@click.option('--checkpoint-interval', type=int, default=100,
              help='中间 checkpoint 保存间隔（本次 resume 自己新跑的额外迭代数，'
                   '非绝对迭代数）——与 solve steady 同名参数含义一致')
def resume(checkpoint_file: str, max_iter: int, backend: Optional[str],
           surface_mesh: Optional[str], reference_area: Optional[float], threads: int,
           checkpoint_interval: int) -> None:
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
        surface_mesh: 面网格路径覆盖。None 时回退到 checkpoint metadata
            里存的 surface_mesh（原 solve steady/transient 运行传了的话）；
            input_file 是 .nas 体网格、且两者都缺失时才会报错
        reference_area: 气动系数参考面积
        checkpoint_interval: 中间 checkpoint 保存间隔（额外迭代数）
    """
    logger.info(f"Resuming simulation from checkpoint: {checkpoint_file}")

    solver, iteration, metadata = rebuild_solver_from_checkpoint(
        checkpoint_file, backend=backend, surface_mesh=surface_mesh, threads=threads,
        reference_area=reference_area,
    )
    input_file = metadata["input_file"]
    order = metadata["order"]
    turbulence_model = metadata["turbulence_model"]
    target_backend = metadata["backend"]
    resolved_surface_mesh = metadata.get("surface_mesh")
    output_dir = str(Path(checkpoint_file).parent.parent)

    # 真实 bug（已修复，2026-08-22，用户直接问"多少步存一个ckpt"发现）：
    # 此前 resume() 从不把 checkpoint_callback 传给 solver.solve()，只在
    # 整个 max_iter 全部跑完后才写一次 checkpoint——solve_steady_command.py
    # 有 --checkpoint-interval 支持中途定期保存，resume 却没有，跑一次
    # 长程 resume（真实网格上单步 7s 量级，几千步就是数小时）中途崩溃/
    # 中断会丢光这段时间的全部进度，退回到 resume 之前那个 checkpoint。
    # iteration（resume 起点的绝对迭代数）必须加到 callback 收到的本地
    # 计数上，否则文件名会用小迭代数覆盖掉 resume 之前就存在的同名
    # checkpoint（例如 checkpoint_iter_000500.h5 撞上原始运行 P0 阶段
    # 已经存过的那个）。
    def _checkpoint_cb(solver_ref, local_iteration):
        if local_iteration % checkpoint_interval != 0:
            return
        absolute_iteration = iteration + local_iteration
        try:
            save_results(solver_ref, output_dir, quiet=True)
            write_checkpoint(
                solver_ref, output_dir, absolute_iteration, input_file,
                solver_ref.current_order, turbulence_model, target_backend,
                quiet=True, surface_mesh=resolved_surface_mesh, target_order=solver_ref.order,
            )
            print(f"   [Checkpoint] iter {absolute_iteration} saved")
        except Exception as e:
            print(f"   [Checkpoint] Warning: save failed at iter {absolute_iteration}: {e}")

    logger.info(f"State restored from checkpoint (iter={iteration}), "
                f"continuing for {max_iter} more iterations...")
    result = solver.solve(max_iter=max_iter, dt=1e-3, tol=1e-6,
                           checkpoint_callback=_checkpoint_cb)
    print(f"\n✅ Resumed simulation finished: total_iterations~={iteration + result.iterations}, "
          f"Residual={result.final_residual:.6e}")

    save_results(solver, output_dir)
    # solver.current_order 而非上面读出的 order：resume 期间 solver.solve()
    # 自己也可能触发 Order Continuation 爬升（例如这次 resume 正好从 P0
    # 跑过 P0->P1 转换），order 是 solve() 调用*之前*从 metadata 读出的
    # 快照，跑完可能已经过期——同一类 bug，见 solve_steady_command.py 里
    # write_checkpoint 调用点的说明。
    write_checkpoint(solver, output_dir, iteration + result.iterations, input_file,
                      solver.current_order, turbulence_model, target_backend,
                      surface_mesh=resolved_surface_mesh, target_order=solver.order)
    # solver._reference_area 而非上面的 CLI 参数 reference_area：
    # rebuild_solver_from_checkpoint 在两者都是 None 时会尝试自动估算并
    # 存到这个属性上（见该函数 reference_area 参数文档）；用回原始 CLI
    # 参数会导致明明整个 resume 过程都在用自动估算出的参考面积算 Cd/Cl，
    # 这里却因为用户没显式传 --reference-area 而误报"未提供，跳过"。
    _report_aerodynamic_coefficients(solver, getattr(solver, "_reference_area", None))


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
