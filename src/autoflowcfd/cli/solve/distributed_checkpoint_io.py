"""分布式求解器（CPU MPI"传统模式"/"完全分布式加载"/多GPU）的
checkpoint 重建辅助函数——从 `solve_checkpoint_io.py` 同一套单机重建
逻辑镜像而来（2026-09-02，用户明确要求"不允许出现完成度不是100%的
功能点"后补齐——此前分布式路径只能"保存一次最终 checkpoint"，没有
任何"从分布式 checkpoint 继续跑"的机制，`solve resume` 对
`--n-ranks`/`--multi-gpu`/`--fully-distributed` 完全没有感知）。

设计与单机版 `rebuild_solver_from_checkpoint` 完全一致：checkpoint
的 metadata 记录了重建求解器所需的全部构造参数（input_file/order/
turbulence_model/backend/自由来流条件），据此重新走一遍
`load_mesh_for_solver` + 对应的分布式求解器构造函数，再用
`distributed_load_checkpoint`/`load_checkpoint_distributed` 恢复
完整的 (n_local_cells,n_sps,n_vars) 状态。
"""

from typing import Optional

import click

from autoflowcfd.core.fr_solver.boundary.constants import _SEM_DEFAULT_NUM_EDDIES
from autoflowcfd.core.time_integration.base import DEFAULT_DUAL_TIME_STEPS
from autoflowcfd.core.utils.order_continuation.checkpoint_state import restore_phase_state


def rebuild_distributed_solver_from_checkpoint(
    checkpoint_path: str,
    n_ranks: int,
    multi_gpu: bool = False,
    fully_distributed: bool = False,
    gpu_device: Optional[int] = None,
    surface_mesh: Optional[str] = None,
    threads: int = -1,
    skip_quality_check: bool = False,
    cfl_start: Optional[float] = None,
    cfl_max: Optional[float] = None,
    cfl_min: Optional[float] = None,
    time_scheme: Optional[str] = None,
    reference_area: Optional[float] = None,
    low_mach_precond: Optional[bool] = None,
):
    """从 checkpoint 完整重建一个分布式求解器（不继续迭代）。

    Args:
        checkpoint_path: checkpoint 文件路径（`solve steady --n-ranks`/
            `--multi-gpu`/`--fully-distributed` 产出）
        n_ranks: MPI rank 总数（必须与当前 `mpirun -np` 启动的进程数
            一致——本函数不做变 rank 数恢复，只有底层 `gather_global_
            state`/`scatter_local_state` 这类纯数组操作理论上支持变
            rank 数，但重建构造过程本身（分区）依赖调用方传入正确的
            `n_ranks`）
        cfl_start / cfl_max / cfl_min: 自适应 CFL 的初始值/上限/下限。
            **真实缺口修复（2026-09-17）**：本函数此前四个构造点一个都
            不传 CFL 参数，于是全部分布式 `solve resume` 都静默使用控制器
            自身的默认值（0.1 / 0.5 / 0.05）——那个下限 0.05 高于本项目在
            两张真实网格上实测稳定的 CFL（~0.03），所以一条原本固定 CFL
            0.03 稳定收敛的分布式运行，一旦 resume 就会被抬到发散。
            `solve steady` 的同一组四个构造点早在 2026-09-15 就补齐了
            （见 `solve_steady_command.py` 那处注释），resume 这条命令
            被漏掉了。
        multi_gpu: 是否重建为 `MultiGPUDistributedSolver`
        fully_distributed: 是否走"完全分布式加载"重建（只有 root 加载
            完整网格）——与 `multi_gpu` 互斥，两者都为 False 时是 CPU
            MPI"传统模式"
        gpu_device: `multi_gpu=True` 时的 GPU 设备号
        surface_mesh: 面网格路径覆盖，None 时回退到 checkpoint metadata
        threads: CPU 后端线程数
        skip_quality_check: 跳过重建时的网格质量门检查
        time_scheme: 续算用的稳态格式覆盖（`rk3`/`newton-krylov`）；`None` 时沿用
            checkpoint 记录的格式——与单机续算同一个解析函数
            （`checkpoint_io/rebuild.py::resolve_resume_time_scheme`）。
        low_mach_precond: 低马赫预处理覆盖；`None` 时沿用 checkpoint 记录（与单机续算同一个解析函数
            `resolve_resume_low_mach_precond`）。

    Returns:
        (solver, iteration, metadata): 重建好的分布式求解器实例
        （状态已从 checkpoint 恢复）、checkpoint 记录的迭代数、以及
        重建所用的完整 metadata 字典
    """
    from autoflowcfd.core.utils.checkpoint_load import load_checkpoint_header

    # 每个 rank 只读头信息（属性、收敛历史、数据集形状）：数组由 root 在 distributed_load_checkpoint 里读取后按 rank
    # 分发。2026-10-19 以前这里每个 rank 都把整个 checkpoint（全局 U_sps 与湍流场）读进内存（I-103）
    history, iteration, metadata, field_shapes = load_checkpoint_header(checkpoint_path)

    if "U_sps" not in field_shapes:
        raise click.ClickException(
            f"Checkpoint '{checkpoint_path}' 缺少 'U_sps' 字段（完整的 "
            f"(n_cells,n_sps,n_vars) 求解器状态）——不是本版本 write_checkpoint/"
            f"distributed_save_checkpoint 写出的 checkpoint，无法精确重建。"
        )

    from autoflowcfd.cli.solve.checkpoint_io.rebuild import REBUILD_KEYS, required_metadata

    required_metadata(metadata, REBUILD_KEYS)
    input_file = metadata["input_file"]
    order = int(metadata["order"])
    # target_order（Order Continuation 的最终目标阶数，solver.order）与
    # order（checkpoint 保存那一刻的 solver.current_order，决定重建
    # mesh/求解器初始状态要用哪个 n_sps 才能跟保存的 U_sps 形状对上）
    # 是两个独立的量（2026-09-02，Order Continuation 接入分布式路径后
    # 补齐，与单机 solve_checkpoint_io.py::rebuild_solver_from_checkpoint
    # 同一处修复同一个设计——见该文件同名字段文档）。缺省回退到 order
    # 本身，兼容本次修复之前保存的旧 checkpoint（没有 target_order
    # 字段，那种情况下当时 order 记的就是静态目标阶数，二者天然相等，
    # 回退安全）。
    target_order = int(metadata.get("target_order", order))
    turbulence_model = metadata["turbulence_model"]
    resolved_surface_mesh = surface_mesh or metadata.get("surface_mesh")
    # 决定物理解的参数：与单机重建共用同一个读取函数（来流缺失即报错，不猜）。
    # 攻角/侧滑角此前在这里四条构造路径上**全部**没有恢复，续算静默变成零攻角。
    from autoflowcfd.cli.solve.checkpoint_io import physics_from_metadata
    physics = physics_from_metadata(metadata)
    from autoflowcfd.cli.solve.checkpoint_io.rebuild import (
        resolve_resume_low_mach_precond, resolve_resume_time_accurate, resolve_resume_time_scheme,
    )
    scheme = resolve_resume_time_scheme(time_scheme, metadata)

    from autoflowcfd.cli.solve.distributed_build import build_distributed_solver

    # 按 checkpoint 时的阶数构造（与保存的状态形状一致），物理参数全部取自 checkpoint
    solver, volume_data = build_distributed_solver(
        multi_gpu=multi_gpu, fully_distributed=fully_distributed, input_file=input_file, order=order,
        surface_mesh=resolved_surface_mesh, skip_quality_check=skip_quality_check, n_ranks=n_ranks,
        gpu_device=gpu_device, threads=threads, turbulence_model=turbulence_model, time_scheme=scheme,
        physics=physics, cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min,
        low_mach_precond=resolve_resume_low_mach_precond(low_mach_precond, metadata),
        time_accurate=resolve_resume_time_accurate(time_scheme, metadata),
        dual_time_inner_iter=DEFAULT_DUAL_TIME_STEPS, sem_num_eddies=_SEM_DEFAULT_NUM_EDDIES)
    loaded_metadata, iteration = solver.load_checkpoint_distributed(checkpoint_path)
    metadata = loaded_metadata or metadata

    # Order Continuation resume 支持（2026-09-02）：与单机
    # rebuild_solver_from_checkpoint 同一处修复同一个理由——构造时为了
    # 让 mesh/初始状态形状匹配 checkpoint 传的是 order（checkpoint 时的
    # current_order），这里把 solver.order 单独纠正回真正的目标阶数，
    # 否则 solve() 里 `self.order_continuation_enabled and self.order
    # >= 2` 这个门槛会被错误地拿 current_order 去判断。`_resumed_from_
    # checkpoint` 告诉 `run_order_continuation` 不要把刚
    # 恢复的真实解重置回 P0 均匀流场、从 `solver.current_order`（而不是
    # 0）继续爬坡。multi_gpu 分支（"传统模式"与"完全分布式加载"均已于
    # 2026-09-02 接入，见 gpu_distributed_order_continuation.py/
    # gpu_distributed_fully_distributed.py 模块文档）同样支持 Order
    # Continuation，这两个属性对它同样生效，不是无害占位。
    solver.order = target_order
    # 收敛历史接着 checkpoint 往下记（各 rank 都读了同一份 checkpoint，历史一致）
    from autoflowcfd.core.fr_solver.convergence_history import ConvergenceHistory

    solver.convergence_history = ConvergenceHistory.from_checkpoint(history, iteration)
    solver._resumed_from_checkpoint = True
    # 湍流场被上界大面积截断时默认拒绝续算（`order_continuation/turbulence_reset.py`），`--reset-turbulence` 置真
    solver._reset_turbulence_on_resume = False
    # 收尾气动系数用的参考面积（root 上；传统模式各 rank 都加载了体网格，完全分布式在 root 的 `_root_context`）
    from autoflowcfd.cli.solve.aero_coefficients import distributed_reference_area

    solver._report_reference_area = distributed_reference_area(solver, volume_data, reference_area)
    # Order Continuation 阶段起始残差（与单机恢复端共用；2026-10-05 以前分布式不写也不读）
    restore_phase_state(solver, metadata)

    metadata["order"] = order
    metadata["target_order"] = target_order
    metadata["turbulence_model"] = turbulence_model
    metadata["surface_mesh"] = resolved_surface_mesh
    from autoflowcfd.core.utils.checkpoint_time import restore_time_state

    try:
        restore_time_state(solver, metadata)
    except ValueError as e:
        raise click.ClickException(str(e))
    return solver, iteration, metadata


def distributed_periodic_checkpoint_callback(interval: int, output_dir: str, input_file: str,
                                             turbulence_model: str, *, surface_mesh: Optional[str] = None,
                                             iteration_offset: int = 0):
    """分布式 `solve()` 的中间 checkpoint 回调（CPU MPI 与多 GPU、steady/transient/resume 共用；此前 CLI
    里 7 份拷贝）：每 `interval` 步调用 `solver.save_checkpoint_distributed`（集体操作，全部 rank 调用）。

    阶数记 `solver.current_order`（爬坡中途 `U_sps` 的实际形状）、目标阶数记 `solver.order`。
    `iteration_offset`：resume 起点的绝对迭代数（否则会用小迭代数覆盖 resume 之前的同名 checkpoint）。
    """
    from autoflowcfd.core.mpi import is_root

    def callback(solver, local_iteration):
        if local_iteration % interval != 0:
            return
        iteration = iteration_offset + local_iteration
        try:
            saved_path = solver.save_checkpoint_distributed(
                output_dir, iteration, input_file, solver.current_order, turbulence_model,
                target_order=solver.order, surface_mesh=surface_mesh)
            if saved_path and is_root():
                print(f"   [Checkpoint] iter {iteration} saved: {saved_path}")
        except Exception as e:
            if is_root():
                print(f"   [Checkpoint] Warning: save failed at iter {iteration}: {e}")

    return callback
