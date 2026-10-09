"""AutoFlowCFD V2.0 - 分布式读取与状态恢复

从 `src/autoflowcfd/core/mpi/distributed_checkpoint.py`(原 515 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
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

from .state import scatter_local_state
from .turbulence import global_cell_is_prism, restore_turbulence_fields, transported_model


def distributed_load_checkpoint(
    checkpoint_path: str,
    solver,
) -> Tuple[np.ndarray, dict, int]:
    """分布式 checkpoint 加载。

    Root rank 加载完整 checkpoint，然后分发到各 rank。

    Args:
        checkpoint_path: checkpoint 文件路径
        solver: DistributedFRSolver 实例

    Returns:
        (U_local, metadata, iteration): 本 rank 的 local cells 数据 + 元数据 + 迭代数
    """
    from autoflowcfd.core.utils.checkpoint import CheckpointManager
    from types import SimpleNamespace

    rank = get_rank()
    n_ranks = get_size()

    # 1. Root rank 加载 checkpoint
    if rank == 0:
        config = SimpleNamespace(
            mode="steady",
            backend="cpu",
            order=solver.mesh.n_points_1d,
            turbulence="sst_kw",
        )
        manager = CheckpointManager(config)
        solution, history, iteration, metadata = manager.load(checkpoint_path)

        # 提取完整状态
        fields = metadata.get("fields", {})
        if "U_sps" in fields:
            U_global = fields["U_sps"]
        else:
            raise ValueError(
                f"Checkpoint '{checkpoint_path}' 缺少 'U_sps' 字段，"
                f"不是本版本写出的 checkpoint"
            )
    else:
        U_global = None
        metadata = None
        iteration = 0

    # 2. 广播元数据（含全部字段：各 rank 由此恢复自己的湍流场）
    if n_ranks > 1:
        metadata = get_comm().bcast(metadata, root=0)
        iteration = get_comm().bcast(iteration, root=0)
    restore_turbulence_fields(solver, (metadata or {}).get("fields", {}), metadata or {}, source="分布式 checkpoint")

    # 3. 分发数据到各 rank
    if n_ranks > 1:
        local_cells = solver.partition.local_cells

        if rank == 0:
            # root 广播 U_global，每个 rank（包括 root 自己）各自提取
            # 自己的 local_cells——这对大网格内存开销较高，但实现简单
            # 且正确（root 也需要重新走一遍广播+提取，保证与非 root 分支
            # 走同一条代码路径，不需要额外维护一份"root 已经有数据"的
            # 特殊情况）。
            U_global = get_comm().bcast(U_global, root=0)
            U_local = scatter_local_state(U_global, local_cells)
        else:
            # 非 root: 接收广播的全局状态
            U_global = get_comm().bcast(None, root=0)
            U_local = scatter_local_state(U_global, local_cells)
    else:
        U_local = U_global

    # dual-time 的上一物理时间层：元数据已广播到全部 rank，各自切出 local 段
    prev_global = (metadata or {}).get("fields", {}).get(PREVIOUS_LEVEL_FIELD)
    if prev_global is not None:
        restore_previous_level(solver, scatter_local_state(np.asarray(prev_global), solver.partition.local_cells))
    turb_prev_global = (metadata or {}).get("fields", {}).get(TURBULENCE_PREVIOUS_FIELD)
    if turb_prev_global is not None:
        restore_turbulence_previous(
            solver, scatter_local_state(np.asarray(turb_prev_global), solver.partition.local_cells))

    return U_local, metadata or {}, iteration


def restore_distributed_state_from_checkpoint(checkpoint_path: str, solver) -> Tuple[int, int]:
    """`solve transient --init-from` 的分布式版本（2026-09-02，补齐此前
    "--init-from 目前只有单机路径支持"的真实缺口——不是设计上不支持，
    只是没人把"全局解按分区切给各 rank"这一步接上；`gather_global_
    state`/`scatter_local_state` 这套基础设施本身早就存在、被
    `distributed_save_results`/`distributed_load_checkpoint` 使用）。

    与单机 `solve_checkpoint_io.py::restore_state_from_checkpoint` 同一个
    用途（先稳态收敛，再用该流场启动 DES/LES 瞬态计算，避免从均匀流场
    直接启动需要极长的瞬态发展时间），但**不是**同一套 n_vars 处理
    机制——单机 `FRState`/`DistributedFRState` 在这一点上架构不同：
    单机 `FRState.U` 在湍流模型激活时是 7 变量（k/omega 打包进
    `U[...,5:7]`），但 `DistributedFRState.n_vars` 恒为 5（纯 Euler
    量，见 `DistributedFRSolver.__init__`）——分布式路径的 k/omega
    是独立存储在 `solver.turb_model.k_field`/`.omega_field`（形状
    `(n_local, n_sps)`，不在 `state.U` 里）。因此这里的策略是：
    1. `state.U` 恒只取 checkpoint `U_sps` 的前 5 个变量（不管
       checkpoint 本身是 5 还是 7 变量——分布式 `state.U` 从来就不
       打包湍流量，这不是"截断"，是恢复到分布式本来的数据模型）。
    2. 有输运湍流模型时，按模型声明的输运场名（`TransportedTurbulence.TRANSPORTED_FIELDS`，
       `write_checkpoint` 写的同一组字段）取全局场，各 rank 切出 local 段交给
       `model.restore_transported`（SST 的 omega 可容许性投影、SA-neg 壁面解点置零都在里面）。
       checkpoint 是层流、另一个湍流模型或旧版本写的（没有这些字段）时保留构造时的来流初值并
       提示。**2026-10-04 修复**：此前从 `U_sps[...,5:7]/rho` 换算 k/omega——那是 SST 状态
       数组里从未更新的历史槽位，恢复出来的永远是初值。

    root 读取 checkpoint 的**全局** `U_sps` 后先在全局索引空间完成上述
    换算，再 broadcast 给全部 rank，各自用 `partition.local_cells`
    （与 `distributed_load_checkpoint`/`distributed_save_results` 同一个
    既有机制）切出自己的 local 部分。

    Args:
        checkpoint_path: 稳态 checkpoint 文件路径（`solve steady` 产出）
        solver: 已构造好（对应目标瞬态阶数/湍流模型）的分布式求解器
            实例（`DistributedFRSolver`/`MultiGPUDistributedSolver`）

    checkpoint 阶数低于求解器阶数时，root 在全局索引空间先把平均流、湍流输运场与涡粘精确延拓到求解器
    阶数（与单机同一个函数，`order_continuation/initial_field.py`），再分发；调用方随后用
    `start_from_checkpoint_field` 起步（延拓后的正性限制、不做阶数爬坡）。

    Returns:
        `(checkpoint 记录的迭代数, checkpoint 阶数)`（各 rank 一致）

    Raises:
        ValueError: checkpoint 缺少 U_sps 字段、单元数不符或阶数高于求解器阶数（全部 rank 一起抛出）——与
            `distributed_load_checkpoint` 同一个既有护栏风格（这个模块是 `core/`，不依赖 `click`，
            由 CLI 调用方按需要转换成 `click.ClickException`）。
    """
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive

    rank = get_rank()
    n_ranks = get_size()
    # 真实 bug 修复（2026-09-02，实现多 GPU"完全分布式加载"+ --init-from
    # 组合时发现，与该组合本身无关——CPU MPI/多 GPU"传统模式"+
    # --init-from 早已可从 CLI 触发同一个 bug）：`MultiGPUDistributedSolver`
    # 的湍流模型持久状态在 `turb_model_gpu`（CuPy 数组），不是
    # `turb_model`（CPU `DistributedFRSolver` 的属性名）——只判断
    # `solver.turb_model` 会让任何 GPU 分布式求解器的 `has_turb_model`
    # 恒为 False，checkpoint 里的 k/omega 场从未被恢复（不报错，静默
    # 退回自由来流默认值），DDES/IDDES/SST 瞬态从"假层流"初场起步。
    model, _ = transported_model(solver)
    names = tuple(model.TRANSPORTED_FIELDS) + ("nu_t",) if model is not None else ()

    # 全局单元的棱柱标志（集体调用，root 有值）：延拓时棱柱与四面体各用自己的矩阵
    is_prism_global = global_cell_is_prism(solver, solver.partition.n_global_cells)

    # root 读取并延拓；校验失败时把错误信息广播出去、全部 rank 一起报错（此前 root 直接抛出，其余 rank
    # 停在 bcast 里等待）
    payload, error = None, None
    if rank == 0:
        try:
            payload = _read_initial_fields(checkpoint_path, solver, names, is_prism_global, model)
        except ValueError as e:
            error = str(e)
    if n_ranks > 1:
        payload, error = get_comm().bcast((payload, error), root=0)
    if error is not None:
        raise ValueError(error)
    U_global, turb_global, ramp, iteration, ckpt_order = payload

    local_cells = solver.partition.local_cells
    U_local = scatter_local_state(U_global, local_cells)

    n_local = solver.partition.n_local_cells
    solver.state.U[:n_local] = U_local
    solver.state.Q[:n_local] = conserved_to_primitive(U_local[..., :5])

    if turb_global is not None:
        restore_turbulence_fields(solver, turb_global, ramp, source="分布式 checkpoint")

    return iteration, ckpt_order


def _read_initial_fields(checkpoint_path: str, solver, names: tuple, is_prism_global, model):
    """root 上读取 checkpoint、精确延拓到求解器阶数，返回 `(U_global, turb_global, 产生项渐变进度, iteration,
    ckpt_order)`。"""
    from types import SimpleNamespace

    from autoflowcfd.core.utils.checkpoint import CheckpointManager

    config = SimpleNamespace(mode="steady", backend="cpu", order=solver.mesh.n_points_1d, turbulence="sst_kw")
    _solution, _history, ckpt_iter, metadata = CheckpointManager(config).load(checkpoint_path)
    fields = metadata.get("fields", {})
    if "U_sps" not in fields:
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' 缺少 'U_sps' 字段（完整的 (n_cells,n_sps,n_vars) 求解器状态），无法精确恢复。")
    fields, ckpt_order = prolongate_checkpoint_fields(
        {key: fields[key] for key in ("U_sps",) + names if key in fields},
        order_from_n_sps(solver.state.n_sps), is_prism_global, model)
    U_global = np.ascontiguousarray(fields["U_sps"][:, :, :5])
    # 只广播湍流恢复需要的字段（形状与缺失的判断在 restore_turbulence_fields 里）
    turb_global = {name: fields[name] for name in names if name in fields}
    ramp = {key: metadata[key] for key in (RAMP_STEP_KEY, RAMP_TOTAL_KEY, RAMP_COMPLETE_KEY) if key in metadata}
    return U_global, turb_global, ramp, ckpt_iter, ckpt_order
