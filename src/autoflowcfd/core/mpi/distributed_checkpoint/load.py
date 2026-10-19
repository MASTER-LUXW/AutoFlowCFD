"""AutoFlowCFD V2.0 - 分布式读取与状态恢复（`solve resume` 与 `solve transient --init-from`，CPU MPI 与多 GPU 共用）。

root 读取 checkpoint，各逐单元数组按 rank 点对点分发（`state.py::distribute_from_root`：每个 rank 只拿自己的
local 段）；不含数组的元数据与校验结果广播。root 上任何读取 / 校验失败都把异常广播出去、全部 rank 一起抛出。

2026-10-19（I-103）以前：全局状态、上一时间层与湍流场随元数据一起用小写 `bcast`（pickle）发给每个 rank，各
rank 再切出自己那段——每个 rank 内存 O(全局)，单条消息超过 2 GiB 时多数 MPI 实现直接失败，千万单元级 P2 无法
续算；`distributed_load_checkpoint` 在 root 读失败时不传播错误，其余 rank 卡在 bcast 里。
"""

from typing import Tuple

import numpy as np

from autoflowcfd.core.mpi import get_comm, get_rank, get_size

from autoflowcfd.core.utils.order_continuation.initial_field import prolongate_checkpoint_fields
from autoflowcfd.fr.native_padding import order_from_n_sps
from autoflowcfd.core.fr_solver.turbulence.init import RAMP_COMPLETE_KEY, RAMP_STEP_KEY, RAMP_TOTAL_KEY
from autoflowcfd.core.utils.checkpoint_time import (
    PREVIOUS_LEVEL_FIELD, TURBULENCE_PREVIOUS_FIELD, restore_previous_level, restore_turbulence_previous,
)

from .state import distribute_from_root
from .turbulence import checkpoint_turbulence_names, global_cell_is_prism, restore_turbulence_fields, transported_model


def _root_read(read):
    """root 上执行 `read()`，结果（不含逐单元数组的部分）与异常广播给全部 rank；root 返回完整结果。

    `read()` 返回 `(广播部分, 只留在 root 的数组字典)`。
    """
    rank = get_rank()
    shared, arrays, error = None, {}, None
    if rank == 0:
        try:
            shared, arrays = read()
        except Exception as e:          # 全部 rank 一起抛出同一个异常（不吞掉）
            error = e
    if get_size() > 1:
        shared, error = get_comm().bcast((shared, error), root=0)
    if error is not None:
        raise error
    return shared, arrays


def distributed_load_checkpoint(checkpoint_path: str, solver) -> Tuple[np.ndarray, dict, int]:
    """分布式 checkpoint 加载（集体调用）：本 rank 的平均流 local 段返回，湍流场与 dual-time 上一时间层直接恢复
    到求解器上。

    Args:
        checkpoint_path: checkpoint 文件路径
        solver: 分布式求解器（`DistributedFRSolver` / `MultiGPUDistributedSolver`），已按 checkpoint 阶数构造

    Returns:
        `(U_local, metadata, iteration)`：本 rank local 单元的 `U_sps` 段、checkpoint 元数据（不含数组）、迭代数

    Raises:
        ValueError: 缺少 U_sps 或其形状与求解器不符；以及 `load_checkpoint` 的全部异常（全部 rank 一起抛出）
    """
    from autoflowcfd.core.utils.checkpoint import CheckpointManager

    expect = (int(solver.partition.n_global_cells), int(solver.state.n_sps))

    def read():
        _sol, _hist, iteration, metadata = CheckpointManager().load(checkpoint_path)
        fields = metadata.pop("fields", {})
        if "U_sps" not in fields:
            raise ValueError(f"Checkpoint '{checkpoint_path}' 缺少 'U_sps' 字段，不是本版本写出的 checkpoint")
        if fields["U_sps"].shape[:2] != expect:
            raise ValueError(f"Checkpoint '{checkpoint_path}' 的 U_sps 形状 {fields['U_sps'].shape} 与求解器的全局"
                             f" (单元数, 每单元解点数) {expect} 不符（网格或阶数不一致）")
        return (iteration, metadata, sorted(fields)), fields

    (iteration, metadata, available), fields = _root_read(read)
    local_cells = solver.partition.local_cells

    def local(name):
        return distribute_from_root(fields.get(name), local_cells)

    U_local = local("U_sps")
    restore_turbulence_fields(solver, {name: local(name) for name in checkpoint_turbulence_names(solver, available)},
                              metadata, source="分布式 checkpoint")
    if PREVIOUS_LEVEL_FIELD in available:
        restore_previous_level(solver, local(PREVIOUS_LEVEL_FIELD))
    if TURBULENCE_PREVIOUS_FIELD in available:
        restore_turbulence_previous(solver, local(TURBULENCE_PREVIOUS_FIELD))
    return U_local, metadata, iteration


def restore_distributed_state_from_checkpoint(checkpoint_path: str, solver) -> Tuple[int, int]:
    """`solve transient --init-from` 的分布式版本（集体调用）：用 checkpoint 的流场（典型是先收敛的稳态）作为初场。

    与单机 `checkpoint_io/initial_field.py` 同一个用途，平均流取 checkpoint `U_sps` 的前 5 个守恒变量（分布式状态
    只存平均流）；有输运湍流模型时按模型声明的输运场名（`TransportedTurbulence.TRANSPORTED_FIELDS`，写出端写的同一组
    字段）恢复，交给 `model.restore_transported`（SST 的 omega 可容许性投影、SA-neg 壁面解点置零都在里面）。checkpoint
    是层流、另一个湍流模型或旧版本写的（没有这些字段）时保留构造时的来流初值并提示。**2026-10-04 修复**：此前从
    `U_sps[...,5:7]/rho` 换算 k/omega——那是 SST 状态数组里从未更新的历史槽位，恢复出来的永远是初值。
    `MultiGPUDistributedSolver` 的湍流模型在 `turb_model_gpu`（`turbulence.py::transported_model` 两种都认）。

    checkpoint 阶数低于求解器阶数时，root 在全局索引空间先把平均流、湍流输运场与涡粘精确延拓到求解器阶数（与单机
    同一个函数，`order_continuation/initial_field.py`），再按 rank 分发；调用方随后用 `start_from_checkpoint_field`
    起步（延拓后的正性限制、不做阶数爬坡）。

    Returns:
        `(checkpoint 记录的迭代数, checkpoint 阶数)`（各 rank 一致）

    Raises:
        ValueError: checkpoint 缺少 U_sps 字段、单元数不符或阶数高于求解器阶数；以及 `load_checkpoint` 的全部异常
            （全部 rank 一起抛出；本模块是 `core/`，不依赖 `click`，由 CLI 调用方按需转换成 `click.ClickException`）
    """
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive

    model, _ = transported_model(solver)
    names = tuple(model.TRANSPORTED_FIELDS) + ("nu_t",) if model is not None else ()
    # 全局单元的棱柱标志（集体调用，root 有值）：延拓时棱柱与四面体各用自己的矩阵
    is_prism_global = global_cell_is_prism(solver, solver.partition.n_global_cells)

    def read():
        U_global, turb_global, ramp, iteration, ckpt_order = _read_initial_fields(
            checkpoint_path, solver, names, is_prism_global, model)
        return (ramp, iteration, ckpt_order, sorted(turb_global)), {"U_sps": U_global, **turb_global}

    (ramp, iteration, ckpt_order, available), arrays = _root_read(read)
    local_cells = solver.partition.local_cells
    U_local = distribute_from_root(arrays.get("U_sps"), local_cells)
    n_local = solver.partition.n_local_cells
    solver.state.U[:n_local] = U_local
    solver.state.Q[:n_local] = conserved_to_primitive(U_local[..., :5])

    if model is not None:
        restore_turbulence_fields(
            solver, {name: distribute_from_root(arrays.get(name), local_cells)
                     for name in checkpoint_turbulence_names(solver, available)},
            ramp, source="分布式 checkpoint")
    return iteration, ckpt_order


def _read_initial_fields(checkpoint_path: str, solver, names: tuple, is_prism_global, model):
    """root 上读取 checkpoint、精确延拓到求解器阶数，返回 `(U_global, turb_global, 产生项渐变进度, iteration,
    ckpt_order)`。"""

    from autoflowcfd.core.utils.checkpoint import CheckpointManager

    _solution, _history, ckpt_iter, metadata = CheckpointManager().load(checkpoint_path)
    fields = metadata.get("fields", {})
    if "U_sps" not in fields:
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' 缺少 'U_sps' 字段（完整的 (n_cells,n_sps,n_vars) 求解器状态），无法精确恢复。")
    fields, ckpt_order = prolongate_checkpoint_fields(
        {key: fields[key] for key in ("U_sps",) + names if key in fields},
        order_from_n_sps(solver.state.n_sps), is_prism_global, model)
    U_global = np.ascontiguousarray(fields["U_sps"][:, :, :5])
    turb_global = {name: fields[name] for name in names if name in fields}
    ramp = {key: metadata[key] for key in (RAMP_STEP_KEY, RAMP_TOTAL_KEY, RAMP_COMPLETE_KEY) if key in metadata}
    return U_global, turb_global, ramp, ckpt_iter, ckpt_order
