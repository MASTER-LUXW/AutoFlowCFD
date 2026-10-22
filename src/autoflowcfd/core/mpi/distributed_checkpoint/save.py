"""AutoFlowCFD V2.0 - 分布式落盘：checkpoint 与最终结果"""

import os

import pickle

from typing import Optional

from autoflowcfd.core.mpi import get_size

from autoflowcfd.core.mpi.comm import run_on_root
from autoflowcfd.core.utils.checkpoint_physics import physics_metadata, run_status_metadata, solve_path_metadata
from autoflowcfd.core.utils.order_continuation.checkpoint_state import phase_state_metadata
from autoflowcfd.core.fr_solver.turbulence.init import production_ramp_metadata
from autoflowcfd.core.utils.checkpoint_time import (
    PREVIOUS_LEVEL_FIELD, TURBULENCE_PREVIOUS_FIELD, previous_level_rows, time_metadata, turbulence_previous_rows,
)

from .state import gather_global_state
from .turbulence import gather_turbulence_fields


def distributed_save_checkpoint(
    solver,
    output_dir: str,
    iteration: int,
    input_file: str,
    order: int,
    turbulence_model: str,
    backend: str,
    target_order: Optional[int] = None,
    surface_mesh: Optional[str] = None,
) -> Optional[str]:
    """分布式 checkpoint 保存。

    Root rank 收集所有 rank 的 local cells 数据，组装全局状态后
    使用标准 CheckpointManager 保存为 HDF5 文件。

    Args:
        solver: DistributedFRSolver 实例
        output_dir: 输出目录
        iteration: 当前迭代数
        input_file: 原始网格文件路径
        order: checkpoint 保存那一刻 `U_sps` 字段实际对应的阶数——
            调用方必须传 `solver.current_order`（不是固定的目标阶数），否则 Order Continuation 爬坡阶段中途
            存的 checkpoint 会出现"metadata 记的阶数与 `U_sps` 实际形状不符"的错配，见单机
            `cli/solve/checkpoint_io/write.py::write_checkpoint` 同名参数文档。
        turbulence_model: 湍流模型名
        backend: 后端名
        target_order: Order Continuation 的最终目标阶数
            （`solver.order`）。None（默认，兼容旧调用方）时回退到
            `order` 本身——与单机 `write_checkpoint` 同一个"两个独立的量"
            设计，见该函数文档。

    Returns:
        checkpoint 文件路径（仅 root rank 有值）
    """
    n_ranks = get_size()

    # 1. 收集全局状态
    U_local = solver.state.U[:solver.partition.n_local_cells]  # 只取 local cells
    n_global = solver.partition.n_global_cells
    local_cells = solver.partition.local_cells  # 全局索引

    U_global = gather_global_state(U_local, local_cells, n_global)
    # 以下收集都是集体调用，必须在非 root 返回之前
    from autoflowcfd.fr.native_padding import order_from_n_sps, sp_integration_weights, weighted_cell_average

    det_local, prism_local = solver.local_cell_geometry()
    W_local = sp_integration_weights(det_local, prism_local, order_from_n_sps(int(U_local.shape[1])))
    # 单元积分平均在各 rank 本地求（本地有几何），再收集——不在 root 上为全局场重建几何
    avg_global = gather_global_state(weighted_cell_average(U_local, W_local)[:, None, :], local_cells, n_global)
    turb_fields = gather_turbulence_fields(solver, n_global, W_local)
    # dual-time 的上一物理时间层（全部 rank 的格式相同，要么都有要么都没有）
    prev_local = previous_level_rows(solver, solver.partition.n_local_cells)
    prev_global = None if prev_local is None else gather_global_state(prev_local, local_cells, n_global)
    turb_prev_local = turbulence_previous_rows(solver, solver.partition.n_local_cells)
    turb_prev_global = (None if turb_prev_local is None
                        else gather_global_state(turb_prev_local, local_cells, n_global))

    # 2. Root rank 保存 checkpoint（成功 / 失败广播给全部 rank，`comm.py::run_on_root`）
    return run_on_root(lambda: _write_checkpoint_on_root(
        solver, output_dir, iteration, input_file, turbulence_model, order, target_order, backend, n_global,
        n_ranks, surface_mesh, U_global, avg_global, turb_fields, prev_global, turb_prev_global))


def _write_checkpoint_on_root(solver, output_dir, iteration, input_file, turbulence_model, order, target_order,
                              backend, n_global, n_ranks, surface_mesh, U_global, avg_global, turb_fields,
                              prev_global, turb_prev_global):
    """`distributed_save_checkpoint` 在 root 上的写出部分，返回 checkpoint 路径。"""
    from autoflowcfd.core.fr_solver.convergence_history import checkpoint_history
    from autoflowcfd.core.utils.checkpoint import CheckpointManager

    manager = CheckpointManager(output_dir=output_dir)

    solution_cell_avg = avg_global[:, 0, :]
    extra_fields = {"U_sps": U_global, **turb_fields}
    if prev_global is not None:
        extra_fields[PREVIOUS_LEVEL_FIELD] = prev_global
    if turb_prev_global is not None:
        extra_fields[TURBULENCE_PREVIOUS_FIELD] = turb_prev_global

    metadata = {
        "input_file": input_file,
        "order": order,
        "target_order": target_order if target_order is not None else order,
        "turbulence_model": turbulence_model,
        "backend": backend,
        "n_cells_global": n_global,
        "n_ranks": n_ranks,
        "distributed": True,
        # 时间格式（2026-09-25）与低马赫预处理开关（2026-10-18）：`solve resume` 默认沿用，与单机写入端共用
        **solve_path_metadata(solver),
        **run_status_metadata(solver),
        # 决定物理解的参数（2026-09-25 补齐）：此前一个都不写，分布式 resume
        # 因此要么按默认来流静默重建另一个算例、要么（09-24 起）直接报错。
        # 与单机写入端共用同一个函数。
        **physics_metadata(solver),
    }
    # Order Continuation 阶段起始残差（与单机写入端共用，见 order_continuation/checkpoint_state.py）
    metadata.update(phase_state_metadata(solver))
    metadata.update(production_ramp_metadata(solver))
    metadata.update(time_metadata(solver))
    if surface_mesh:
        # resume 按它重新做边界归属；缺了就退回"无面网格"的几何匹配，
        # 边界组可能与原运行不同（单机写入端一直写这个键）。
        metadata["surface_mesh"] = surface_mesh

    path = manager.save(
        solution_cell_avg,
        checkpoint_history(solver, iteration),
        iteration,
        metadata=metadata,
        extra_fields=extra_fields,
    )
    return path


def distributed_save_results(
    solver,
    output_dir: str,
) -> None:
    """分布式结果保存。

    Root rank 收集所有 rank 的 local cells 数据，组装全局状态后
    保存为单个 pickle 文件（与单机格式兼容）。

    Args:
        solver: DistributedFRSolver 实例
        output_dir: 输出目录
    """
    n_ranks = get_size()

    # 1. 收集全局状态
    U_local = solver.state.U[:solver.partition.n_local_cells]
    n_global = solver.partition.n_global_cells
    local_cells = solver.partition.local_cells

    U_global = gather_global_state(U_local, local_cells, n_global)
    # 2. Root rank 保存结果（成功 / 失败广播给全部 rank）
    run_on_root(lambda: _write_results_on_root(solver, output_dir, U_global, n_global, n_ranks))


def _write_results_on_root(solver, output_dir: str, U_global, n_global: int, n_ranks: int) -> None:
    """`distributed_save_results` 在 root 上的写出部分。"""
    os.makedirs(output_dir, exist_ok=True)

    # 计算原变量
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
    Q_global = conserved_to_primitive(U_global)

    state_path = os.path.join(output_dir, "final_state.pkl")
    with open(state_path, 'wb') as f:
        pickle.dump({
            'U': U_global,
            'Q': Q_global,
            'n_cells': n_global,
            'n_sps': solver.state.n_sps,
            'n_vars': solver.state.n_vars,
            'distributed': n_ranks > 1,
            'n_ranks': n_ranks,
        }, f)

    print(f"✅ Distributed results saved to: {output_dir}")
    print(f"   - Final state: {state_path} ({n_global} global cells)")
