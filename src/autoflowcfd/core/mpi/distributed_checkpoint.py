"""
AutoFlowCFD V2.0 - 分布式 Checkpoint 保存/加载 + 结果保存

将单机 checkpoint 和结果保存扩展为分布式版本：
- Root rank 收集所有 rank 的 local cells 数据
- 组装全局状态后保存为单文件（与单机格式兼容）
- 加载时 root rank 读取后分发到各 rank

关键设计:
- 保存格式与单机完全一致（HDF5 checkpoint + pickle 结果），后处理工具无需修改
- 使用 partition.local_cells（全局索引）定位每个 rank 的数据在全局数组中的位置
- 支持变 rank 数恢复（4 ranks 保存 → 8 ranks 恢复）
"""

import os
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from loguru import logger

from autoflowcfd.core.mpi import get_comm, get_mpi, mpi_available, is_root, get_rank, get_size
from autoflowcfd.core.mpi.comm import barrier


def gather_global_state(
    U_local: np.ndarray,
    local_cells: np.ndarray,
    n_global_cells: int,
) -> Optional[np.ndarray]:
    """从所有 rank 收集 local cells 数据，在 root 组装全局状态。

    每个 rank 持有 local cells 的 (n_local, n_sps, n_vars) 数据，
    local_cells 给出这些 cell 的全局索引。Root rank 根据全局索引
    将各 rank 的数据放入全局数组的正确位置。

    Args:
        U_local: (n_local_cells, n_sps, n_vars) 本 rank 的 local cell 数据
        local_cells: (n_local_cells,) 本 rank 的 cell 全局索引
        n_global_cells: 全局 cell 总数

    Returns:
        U_global: (n_global_cells, n_sps, n_vars) 全局状态（仅 root rank 有值，
                  其他 rank 返回 None）
    """
    comm = get_comm()
    rank = get_rank()
    n_ranks = get_size()

    if n_ranks == 1:
        # 单 rank：直接按全局索引放置
        n_sps = U_local.shape[1]
        n_vars = U_local.shape[2]
        U_global = np.zeros((n_global_cells, n_sps, n_vars), dtype=np.float64)
        U_global[local_cells] = U_local
        return U_global

    # 各 rank 向 root 发送自己的 local cell 数据 + 全局索引
    if rank == 0:
        # Root: 初始化全局数组
        n_sps = U_local.shape[1]
        n_vars = U_local.shape[2]
        U_global = np.zeros((n_global_cells, n_sps, n_vars), dtype=np.float64)

        # 放入 root 自己的数据
        U_global[local_cells] = U_local

        # 接收其他 rank 的数据
        for r in range(1, n_ranks):
            # 先接收全局索引
            n_recv = np.empty(1, dtype=np.int64)
            comm.Recv(n_recv, source=r, tag=99)
            n_l = int(n_recv[0])
            idx_buf = np.empty(n_l, dtype=np.int64)
            comm.Recv(idx_buf, source=r, tag=100)

            # 接收数据
            shape_buf = np.empty(2, dtype=np.int64)
            comm.Recv(shape_buf, source=r, tag=101)
            n_s, n_v = int(shape_buf[0]), int(shape_buf[1])
            data_buf = np.empty((n_l, n_s, n_v), dtype=np.float64)
            comm.Recv(data_buf, source=r, tag=102)

            U_global[idx_buf] = data_buf
    else:
        # 非 root: 发送数据
        n_local = np.array([len(local_cells)], dtype=np.int64)
        comm.Send(n_local, dest=0, tag=99)
        comm.Send(local_cells.astype(np.int64), dest=0, tag=100)

        shape_buf = np.array([U_local.shape[1], U_local.shape[2]], dtype=np.int64)
        comm.Send(shape_buf, dest=0, tag=101)
        comm.Send(U_local, dest=0, tag=102)

        return None

    return U_global


def scatter_local_state(
    U_global: np.ndarray,
    local_cells: np.ndarray,
) -> np.ndarray:
    """从全局状态中提取本 rank 的 local cells 数据。

    Args:
        U_global: (n_global_cells, n_sps, n_vars) 全局状态
        local_cells: (n_local_cells,) 本 rank 的 cell 全局索引

    Returns:
        U_local: (n_local_cells, n_sps, n_vars) 本 rank 的 local cell 数据
    """
    return U_global[local_cells].copy()


def distributed_save_checkpoint(
    solver,
    output_dir: str,
    iteration: int,
    input_file: str,
    order: int,
    turbulence_model: str,
    backend: str,
    history: Optional[dict] = None,
) -> Optional[str]:
    """分布式 checkpoint 保存。

    Root rank 收集所有 rank 的 local cells 数据，组装全局状态后
    使用标准 CheckpointManager 保存为 HDF5 文件。

    Args:
        solver: DistributedFRSolver 实例
        output_dir: 输出目录
        iteration: 当前迭代数
        input_file: 原始网格文件路径
        order: FR 阶数
        turbulence_model: 湍流模型名
        backend: 后端名
        history: 收敛历史（可选）

    Returns:
        checkpoint 文件路径（仅 root rank 有值）
    """
    from autoflowcfd.core.checkpoint import CheckpointManager, H5PY_AVAILABLE
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

    solution_cell_avg = U_global.mean(axis=1)  # (n_global, n_vars)
    extra_fields = {"U_sps": U_global}

    metadata = {
        "input_file": input_file,
        "order": order,
        "turbulence_model": turbulence_model,
        "backend": backend,
        "n_cells_global": n_global,
        "n_ranks": n_ranks,
        "distributed": True,
    }

    path = manager.save(
        solution_cell_avg,
        history or {"iterations": [iteration]},
        iteration,
        metadata=metadata,
        extra_fields=extra_fields,
    )

    barrier()
    return path


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
    from autoflowcfd.core.checkpoint import CheckpointManager
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

    # 2. 广播元数据
    if n_ranks > 1:
        metadata = get_comm().bcast(metadata, root=0)
        iteration = get_comm().bcast(iteration, root=0)

    # 3. 分发数据到各 rank
    if n_ranks > 1:
        local_cells = solver.partition.local_cells

        if rank == 0:
            # Root: 提取自己的数据
            U_local = scatter_local_state(U_global, local_cells)
            # 分发给其他 rank
            for r in range(1, n_ranks):
                # 需要知道 rank r 的 local_cells
                # 由于所有 rank 执行相同分区算法，可以重建
                # 但更简单的方式是让每个 rank 发送自己的 local_cells 给 root
                pass  # 下面用另一种方式

            # 替代方案：root 广播 U_global，每个 rank 自己提取
            # 这对大网格内存开销较高，但实现简单且正确
            U_global = get_comm().bcast(U_global, root=0)
            U_local = scatter_local_state(U_global, local_cells)
        else:
            # 非 root: 接收广播的全局状态
            U_global = get_comm().bcast(None, root=0)
            U_local = scatter_local_state(U_global, local_cells)
    else:
        U_local = U_global

    return U_local, metadata or {}, iteration


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
    from autoflowcfd.core.fr_residual_inviscid import conserved_to_primitive
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
