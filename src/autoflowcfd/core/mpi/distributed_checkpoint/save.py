"""AutoFlowCFD V2.0 - 分布式落盘：checkpoint 与最终结果

从 `src/autoflowcfd/core/mpi/distributed_checkpoint.py`(原 515 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""

import os

import pickle


from typing import Optional



from autoflowcfd.core.mpi import is_root, get_rank, get_size

from autoflowcfd.core.mpi.comm import barrier
from autoflowcfd.core.utils.checkpoint_physics import physics_metadata

from .state import gather_global_state


def distributed_save_checkpoint(
    solver,
    output_dir: str,
    iteration: int,
    input_file: str,
    order: int,
    turbulence_model: str,
    backend: str,
    history: Optional[dict] = None,
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
            Order Continuation 接入分布式路径后（2026-09-02），调用方
            必须传 `solver.current_order`（不是固定的目标阶数），否则
            爬坡阶段中途存的 checkpoint 会出现"metadata 记的阶数与
            `U_sps` 实际形状不符"的错配，见单机 `solve_checkpoint_io.py::
            write_checkpoint` 同名参数文档、`rebuild_distributed_solver_
            from_checkpoint` 对应的恢复逻辑。
        turbulence_model: 湍流模型名
        backend: 后端名
        history: 收敛历史（可选）
        target_order: Order Continuation 的最终目标阶数
            （`solver.order`）。None（默认，兼容旧调用方）时回退到
            `order` 本身——与单机 `write_checkpoint` 同一个"两个独立的量"
            设计，见该函数文档。

    Returns:
        checkpoint 文件路径（仅 root rank 有值）
    """
    from autoflowcfd.core.utils.checkpoint import CheckpointManager, H5PY_AVAILABLE
    from types import SimpleNamespace

    if not H5PY_AVAILABLE:
        if is_root():
            print("   ⚠️  h5py not available, skipping distributed checkpoint")
        return None

    rank = get_rank()
    n_ranks = get_size()

    # 1. 收集全局状态
    U_local = solver.state.U[:solver.partition.n_local_cells]  # 只取 local cells
    n_global = solver.partition.n_global_cells
    local_cells = solver.partition.local_cells  # 全局索引

    U_global = gather_global_state(U_local, local_cells, n_global)

    if rank != 0:
        barrier()
        return None

    # 2. Root rank 保存 checkpoint
    config = SimpleNamespace(
        mode="steady" if history is None else "transient",
        backend=backend,
        order=order,
        turbulence=turbulence_model,
    )
    manager = CheckpointManager(config, output_dir=output_dir)

    # 只对**真实自由度**取平均（2026-09-15 系统性审计发现的真实缺陷）：
    # 直接 `.mean(axis=1)` 会把 native 四面体的零填充槽位一起算进去，而
    # 那些槽位按约定在初始化时复制真实 SP #0、之后残差行填零/滤波行是
    # 单位阵，**永远冻结在初值**（实测推进 10 步后与真实 SP#0 相差 3.4%，
    # order=1 下占一半槽位）。见 fr/native_padding.py::
    # reduce_per_cell_over_real_sps。
    #
    # `U_global` 是**全局**索引空间（gather_global_state 按全局 id 归位），
    # 所以需要**全局**棱柱数。两种分布式模式的 `solver.mesh` 含义不同：
    #   - 传统模式：每个 rank 持有完整网格，`mesh.n_prism_cells` 就是全局值；
    #   - 完全分布式加载：`self.mesh` 是 `PrecompactedMeshData`，它的
    #     n_prism 是本 rank 的 **compact** 值，用在全局索引上是错的。
    # 因此只在能可靠取到全局值时做掩码；完全分布式模式下退回原样的全场
    # 平均，并**显式记录**这一点——它是一处有界、已知的输出侧失真，而不是
    # 静默行为：`U_sps`（下方 extra_fields）始终是精确的逐 SP 数据，所有
    # 内部消费方（resume、气动力后处理）都强制要求 U_sps 且缺失即报错，
    # 本字段只供粗粒度外部消费方使用。要在完全分布式下也修对，需要把
    # 逐单元 is_prism 标志一起 gather（需要真实 MPI 环境验证）。
    from autoflowcfd.fr.native_padding import (
        order_from_n_sps, reduce_per_cell_over_real_sps,
    )
    _mesh_ck = getattr(solver, 'mesh', None)
    _fully_dist = bool(getattr(solver, '_is_fully_distributed', False))
    _n_prism_global = (getattr(_mesh_ck, 'n_prism_cells', None)
                       if (_mesh_ck is not None and not _fully_dist) else None)
    if _n_prism_global is not None:
        solution_cell_avg = reduce_per_cell_over_real_sps(
            U_global, int(_n_prism_global),
            order_from_n_sps(U_global.shape[1]), 'mean')
    else:
        solution_cell_avg = U_global.mean(axis=1)  # (n_global, n_vars)
    extra_fields = {"U_sps": U_global}

    metadata = {
        "input_file": input_file,
        "order": order,
        "target_order": target_order if target_order is not None else order,
        "turbulence_model": turbulence_model,
        "backend": backend,
        "n_cells_global": n_global,
        "n_ranks": n_ranks,
        "distributed": True,
        # 决定物理解的参数（2026-09-25 补齐）：此前一个都不写，分布式 resume
        # 因此要么按默认来流静默重建另一个算例、要么（09-24 起）直接报错。
        # 与单机写入端共用同一个函数。
        **physics_metadata(solver),
    }
    if surface_mesh:
        # resume 按它重新做边界归属；缺了就退回"无面网格"的几何匹配，
        # 边界组可能与原运行不同（单机写入端一直写这个键）。
        metadata["surface_mesh"] = surface_mesh

    path = manager.save(
        solution_cell_avg,
        history or {"iterations": [iteration]},
        iteration,
        metadata=metadata,
        extra_fields=extra_fields,
    )

    barrier()
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
    rank = get_rank()
    n_ranks = get_size()

    # 1. 收集全局状态
    U_local = solver.state.U[:solver.partition.n_local_cells]
    n_global = solver.partition.n_global_cells
    local_cells = solver.partition.local_cells

    U_global = gather_global_state(U_local, local_cells, n_global)

    if rank != 0:
        barrier()
        return

    # 2. Root rank 保存结果
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

    barrier()
