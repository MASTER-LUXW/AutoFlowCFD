"""V2.0 FR 求解器子命令。

本模块提供 V2.0 FR 求解器的 CLI 命令，支持高阶精度、多种时间推进方法和湍流模型。

命令:
    - steady: 运行稳态 FR 仿真
    - transient: 运行瞬态 FR 仿真（专用命令）
    - resume: 从检查点恢复
    - status: 查看求解器状态

示例:
    $ autoflowcfd solve steady model_volume.pkl --backend cpu --order 2 --turbulence-model sst

steady/transient 命令本体在同目录的 `steady.py`/`transient.py`，本文件只保留
命令组定义 + resume/status。
"""

from pathlib import Path
from typing import Optional

import click
from loguru import logger

from autoflowcfd.cli.solve.helpers import (
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


# 导入子命令模块，触发 @solve.command() 注册（2026-09-25 起直接导入，删掉了
# 只做转发的 solve_steady_commands.py 中间层）
from autoflowcfd.cli.solve import steady as _steady_cmd  # noqa: F401,E402
from autoflowcfd.cli.solve import transient as _transient_cmd  # noqa: F401,E402
from autoflowcfd.cli.solve.aero_coefficients import _report_aerodynamic_coefficients  # noqa: E402



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
@click.option('--skip-quality-check', is_flag=True,
              help='跳过重建时的网格质量门检查（B-11：原求解靠该选项才跑得起来的'
                   '网格，resume 同样需要跳过；不建议，仅用于临时诊断）')
@click.option('--checkpoint-interval', type=int, default=100,
              help='中间 checkpoint 保存间隔（本次 resume 自己新跑的额外迭代数，'
                   '非绝对迭代数）——与 solve steady 同名参数含义一致')
@click.option('--cfl-start', type=float, default=0.03,
              help='自适应 CFL 初始值（默认 0.03，2026-09-17 从 0.1 下调）。'
                   'CFL 是纯数值加速参数，不影响物理解，每次 resume 可根据上一段'
                   '收敛表现重新调。下调依据见 `solve steady --cfl-max` 的帮助：2026-09-17 按直接谱测量 + 两张真实网格的失效点重定，线性极限约 0.117、实测失效点 plate 0.30 / 平板边界层 0.10，默认值留 1.7 倍以上裕度。')
@click.option('--cfl-max', type=float, default=0.06,
              help='自适应 CFL 上限（默认 0.06，2026-09-17 从 0.5 下调）。'
                   '仅单机 CPU 路径（非 --n-ranks>1/--multi-gpu）支持。原文案建议的'
                   '"稳定收敛可试 0.8"已删除——0.8 比实测线性极限高近 7 倍，从来'
                   '不是可达值。下调依据见 `solve steady --cfl-max` 的帮助：2026-09-17 按直接谱测量 + 两张真实网格的失效点重定，线性极限约 0.117、实测失效点 plate 0.30 / 平板边界层 0.10，默认值留 1.7 倍以上裕度。')
@click.option('--cfl-min', type=float, default=0.01,
              help='自适应 CFL 下限（默认 0.01）。**真实缺口修复（2026-09-17）**：'
                   '`solve steady` 早在 2026-09-15 就有这个选项（控制器默认下限 0.05 '
                   '高于真 P1 在 79 万单元 cube_demo 上实测稳定的 CFL 0.03，也高于 '
                   'plate_demo 实测的稳定边界，等于一个已验证可用的工作点通过 CLI 根本'
                   '到不了），但 `solve resume` 从未跟上——续算会静默退回控制器默认 '
                   '0.05，把原本固定 CFL 0.03 的稳定运行抬到发散。默认值与 `solve '
                   'steady` 对齐为 0.01。仅单机 CPU 路径生效。')
@click.option('--phase-max-iter', type=int, default=None,
              help='Order Continuation（目标阶数>=2 时触发）非最终阶段各自的最大迭代'
                   '步数上限。默认(不传)时取本次续算新增的额外迭代数按剩余阶段数机械'
                   '均分的结果作为这一个数字的默认值；不论默认还是显式传值，目标阶数'
                   '永远吃掉这次续算剩余的全部步数，不会被稀释——见'
                   'core/utils/order_continuation.py 文档。CPU/单GPU/CPU MPI分布式/'
                   '多GPU分布式全部支持')
@click.option('--residual-drop-threshold', type=float, default=100.0,
              help='Order Continuation 单个非最终阶段判定"可以提前升阶"的残差下降倍数，'
                   '默认100(降2个数量级)。CPU/单GPU/CPU MPI分布式/多GPU分布式全部支持')
@click.option('--n-ranks', type=int, default=1,
              help='MPI rank 总数（>1 时重建为分布式求解器——CPU MPI"传统模式"，'
                   '或配合 --multi-gpu/--fully-distributed 走对应的分布式构造入口）。'
                   '必须与本次 mpirun -np 启动的进程数一致。2026-09-02 补齐：此前分布式'
                   '路径只能保存一次最终 checkpoint，完全没有续算机制')
@click.option('--multi-gpu', is_flag=True,
              help='重建为多GPU分布式求解器（需要 --n-ranks>1）')
@click.option('--fully-distributed', is_flag=True,
              help='走"完全分布式加载"重建（只有 root rank 加载完整网格，需要 --n-ranks>1，'
                   '与 --multi-gpu 互斥）')
@click.option('--gpu-device', type=int, default=None, help='--multi-gpu 时的 GPU 设备号')
def resume(checkpoint_file: str, max_iter: int, backend: Optional[str],
           surface_mesh: Optional[str], reference_area: Optional[float], threads: int,
           skip_quality_check: bool, checkpoint_interval: int,
           cfl_start: float, cfl_max: float, cfl_min: float,
           phase_max_iter: Optional[int], residual_drop_threshold: float,
           n_ranks: int, multi_gpu: bool, fully_distributed: bool,
           gpu_device: Optional[int]) -> None:
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
        cfl_start, cfl_max, cfl_min: 自适应 CFL 的初始值/上限/下限（纯数值加速参数，不影响
            物理解，不从 checkpoint 恢复——每次 resume 由本次命令行重新指定，
            方便根据上一段收敛表现调整）。仅单机 CPU 路径生效。
        phase_max_iter: Order Continuation 非最终阶段最大步数上限，None 时取
            max_iter // len(orders) 作为这个数字的默认值——不论默认还是
            显式值，目标阶数都吃掉剩余全部步数，见同名 CLI 选项帮助文本
        residual_drop_threshold: Order Continuation 单阶段提前升阶所需的残差下降倍数
    """
    logger.info(f"Resuming simulation from checkpoint: {checkpoint_file}")

    if n_ranks > 1 or multi_gpu:
        _resume_distributed(
            checkpoint_file, max_iter, n_ranks, multi_gpu, fully_distributed,
            gpu_device, backend, surface_mesh, threads, skip_quality_check,
            checkpoint_interval, phase_max_iter, residual_drop_threshold,
            cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min,
        )
        return

    solver, iteration, metadata = rebuild_solver_from_checkpoint(
        checkpoint_file, backend=backend, surface_mesh=surface_mesh, threads=threads,
        reference_area=reference_area,
        skip_quality_check=skip_quality_check,
        cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min,
    )
    input_file = metadata["input_file"]
    order = metadata["order"]
    turbulence_model = metadata["turbulence_model"]
    target_backend = metadata["backend"]
    resolved_surface_mesh = metadata.get("surface_mesh")
    output_dir = str(Path(checkpoint_file).parent.parent)

    # 真实 bug 更正（2026-09-02）：上面这两个参数此前被无条件拒绝
    # `target_backend == 'gpu'`——但 `rebuild_solver_from_checkpoint`
    # 这条非分布式 resume 路径构造的是单机 `FRSolver(backend='gpu',
    # ...)`（`core/fr_solver/solver.py` 的 GPU 加速内核分支），不是
    # `solve steady` 单 GPU 分支专用的独立 `GPUFRSolver` 类——`FRSolver.
    # solve()` 的 Order Continuation 分派（`self.order>=2` 时自动
    # 逐阶爬坡）与 `backend_type` 无关，这个拒绝从一开始就是错的、
    # 不必要地拒绝了本来就能工作的组合，不是"待补齐"的功能缺口。
    # （2026-09-02 同一批还真正给了 `GPUFRSolver` 本身独立的 Order
    # Continuation 机制，见 core/gpu/solver/gpu_solver_order_
    # continuation.py，所以即便按原先的假设也不再需要这个拒绝。）

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
                           checkpoint_callback=_checkpoint_cb,
                           phase_max_iter=phase_max_iter,
                           residual_drop_threshold=residual_drop_threshold)
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


def _resume_distributed(
    checkpoint_file: str, max_iter: int, n_ranks: int, multi_gpu: bool,
    fully_distributed: bool, gpu_device: Optional[int], backend: Optional[str],
    surface_mesh: Optional[str], threads: int, skip_quality_check: bool,
    checkpoint_interval: int,
    phase_max_iter: Optional[int] = None,
    residual_drop_threshold: float = 100.0,
    cfl_start: float = 0.1,
    cfl_max: float = 0.5,
    cfl_min: float = 0.01,
) -> None:
    """`resume` 的分布式分支（2026-09-02 补齐，见 `resume` 文档"完成度"
    一节）——CPU MPI"传统模式"/"完全分布式加载"/多GPU 三条路径共用同一个
    重建入口 `rebuild_distributed_solver_from_checkpoint`，之后的续算
    循环 + 中途 checkpoint 保存 + 最终保存与单机分支同一个设计
    （`checkpoint_callback`），只是三种求解器的 `solve()` 返回值/最终
    保存调用各自的真实签名不同，分别处理。

    真实 bug 修复（2026-09-05，用户直接问"--residual-drop-threshold
    phase_max_iter 可以在 resume 重置吗"发现）：`resume()` 命令顶层
    确实解析了这两个 CLI 选项，但此前本函数的签名根本不接收它们，两处
    `solver.solve(...)` 调用（多GPU分支/CPU MPI分支）也完全没有传递
    ——`DistributedFRSolver.solve`/`MultiGPUDistributedSolver.solve`
    早在 2026-09-02（Order Continuation 分布式移植当天）就已经真正
    支持这两个参数（签名完全对应单机 `run_order_continuation`），只是
    CLI 这一层从未把用户在命令行传的值接力传下去——用户传了
    `--phase-max-iter`/`--residual-drop-threshold` 也会被静默忽略，
    分布式/多GPU resume 时这两个 Order Continuation 参数永远等于
    `DistributedFRSolver.solve`/`MultiGPUDistributedSolver.solve` 自己
    的函数签名默认值（`None`/`100.0`），不是用户的真实意图。CLI
    帮助文本此前写的"仅 CPU 后端支持"因此也是过时/错误的说法，一并
    改正。

    不做气动系数报告（`_report_aerodynamic_coefficients` 假设单机
    `FRSolver` 的 `.state`/`.mesh` 布局，与分布式求解器的 local+halo
    布局不兼容）——`solve_steady_command.py` 的分布式分支本身同样不
    调用它，这里保持同一个范围边界。
    """
    from autoflowcfd.core.mpi import is_root
    from autoflowcfd.cli.solve.distributed_checkpoint_io import (
        rebuild_distributed_solver_from_checkpoint,
    )

    solver, iteration, metadata = rebuild_distributed_solver_from_checkpoint(
        checkpoint_file, n_ranks=n_ranks, multi_gpu=multi_gpu,
        fully_distributed=fully_distributed, gpu_device=gpu_device,
        backend=backend, surface_mesh=surface_mesh, threads=threads,
        cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min,
        skip_quality_check=skip_quality_check,
    )
    input_file = metadata["input_file"]
    order = metadata["order"]
    turbulence_model = metadata["turbulence_model"]
    target_backend = "gpu" if multi_gpu else (backend or "cpu")
    resolved_surface_mesh = metadata.get("surface_mesh")
    output_dir = str(Path(checkpoint_file).parent.parent)

    if is_root():
        logger.info(
            f"[Distributed resume] State restored from checkpoint (iter={iteration}), "
            f"continuing for {max_iter} more iterations..."
        )

    if multi_gpu:
        def _checkpoint_cb(solver_ref, local_iteration):
            if local_iteration % checkpoint_interval != 0:
                return
            absolute_iteration = iteration + local_iteration
            try:
                # order/target_order 分离（2026-09-02）：见
                # solve_steady_command.py 的 _multi_gpu_checkpoint_cb
                # 同一处修复文档——resume 之后若仍在 Order Continuation
                # 爬坡中途，必须记录 solver_ref.current_order。
                saved_path = solver_ref.save_checkpoint_distributed(
                    output_dir, absolute_iteration, input_file,
                    solver_ref.current_order, turbulence_model, backend="gpu",
                    target_order=solver_ref.order, surface_mesh=resolved_surface_mesh,
                )
                if saved_path and is_root():
                    print(f"   [Checkpoint] iter {absolute_iteration} saved: {saved_path}")
            except Exception as e:
                if is_root():
                    print(f"   [Checkpoint] Warning: save failed at iter {absolute_iteration}: {e}")

        result = solver.solve(
            max_iter=max_iter, dt=1e-3, tol=1e-6, checkpoint_callback=_checkpoint_cb,
            phase_max_iter=phase_max_iter, residual_drop_threshold=residual_drop_threshold,
        )
        if is_root():
            print(
                f"\n✅ Resumed multi-GPU simulation finished: "
                f"total_iterations~={iteration + max_iter}, "
                f"Residual={result['final_residual']:.6e}"
            )
        saved_path = solver.save_checkpoint_distributed(
            output_dir, iteration + max_iter, input_file,
            solver.current_order, turbulence_model, backend="gpu",
            target_order=solver.order, surface_mesh=resolved_surface_mesh,
        )
        if saved_path and is_root():
            print(f"   Checkpoint saved: {saved_path}")
        solver.cleanup()
        return

    # CPU MPI（"传统模式"/"完全分布式加载"共用同一个 solve()/checkpoint 格式）。
    from autoflowcfd.core.mpi.distributed_checkpoint import (
        distributed_save_results, distributed_save_checkpoint,
    )

    def _checkpoint_cb(solver_ref, local_iteration):
        if local_iteration % checkpoint_interval != 0:
            return
        absolute_iteration = iteration + local_iteration
        try:
            # order/target_order 分离（2026-09-02）：resume 之后若仍在
            # Order Continuation 爬坡中途，必须记录 solver_ref.
            # current_order（U_sps 实际形状），不是 metadata 里那个
            # "checkpoint 保存时"的旧 order 值。
            saved_path = distributed_save_checkpoint(
                solver_ref, output_dir, absolute_iteration, input_file,
                solver_ref.current_order, turbulence_model, target_backend,
                target_order=solver_ref.order, surface_mesh=resolved_surface_mesh,
            )
            if saved_path and is_root():
                print(f"   [Checkpoint] iter {absolute_iteration} saved: {saved_path}")
        except Exception as e:
            if is_root():
                print(f"   [Checkpoint] Warning: save failed at iter {absolute_iteration}: {e}")

    solver.solve(n_steps=max_iter, dt=1e-3, output_interval=checkpoint_interval,
                 checkpoint_callback=_checkpoint_cb,
                 phase_max_iter=phase_max_iter, residual_drop_threshold=residual_drop_threshold)
    if is_root():
        print(f"\n✅ Resumed distributed simulation finished: "
              f"total_iterations~={iteration + max_iter}")

    distributed_save_results(solver, output_dir)
    distributed_save_checkpoint(
        solver, output_dir, iteration + max_iter, input_file,
        solver.current_order, turbulence_model, target_backend,
        target_order=solver.order, surface_mesh=resolved_surface_mesh,
    )


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
