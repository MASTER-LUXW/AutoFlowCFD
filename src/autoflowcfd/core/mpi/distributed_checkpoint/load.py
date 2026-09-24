"""AutoFlowCFD V2.0 - 分布式读取与状态恢复

从 `src/autoflowcfd/core/mpi/distributed_checkpoint.py`(原 515 行)拆出(2026-09-24, 项目"单文件不超 500 行"规范)。**纯搬家, 逻辑未改**。
"""




from typing import Tuple

import numpy as np


from autoflowcfd.core.mpi import get_comm, get_rank, get_size

from .state import scatter_local_state


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

    # 2. 广播元数据
    if n_ranks > 1:
        metadata = get_comm().bcast(metadata, root=0)
        iteration = get_comm().bcast(iteration, root=0)

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

    return U_local, metadata or {}, iteration


def restore_distributed_state_from_checkpoint(checkpoint_path: str, solver) -> int:
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
    2. 若 `solver.turb_model is not None`（目标瞬态是 SST/DDES/IDDES）：
       checkpoint 是 7 变量时，从 `U_sps[...,5:7]` 换算出 k/omega
       （`k=rho_k/rho`, `omega=rho_omega/rho`，与单机 `FRState.
       _update_primitives` 同一个换算公式）写入 `turb_model.k_field`/
       `.omega_field`；checkpoint 只有 5 变量（源自稳态 `none`/纯
       层流）时，退回自由来流默认值（`_set_freestream_turbulence`，
       与单机 `restore_state_from_checkpoint`"湍流量用自由来流默认值
       初始化"同一个理念，只是单机用固定常数 k=1e-6/omega=1e-2，这里
       复用分布式已有的、更物理自洽的 Tu/VR 推导公式）。

    root 读取 checkpoint 的**全局** `U_sps` 后先在全局索引空间完成上述
    换算，再 broadcast 给全部 rank，各自用 `partition.local_cells`
    （与 `distributed_load_checkpoint`/`distributed_save_results` 同一个
    既有机制）切出自己的 local 部分。

    Args:
        checkpoint_path: 稳态 checkpoint 文件路径（`solve steady` 产出）
        solver: 已构造好（对应目标瞬态阶数/湍流模型）的分布式求解器
            实例（`DistributedFRSolver`/`MultiGPUDistributedSolver`）

    Returns:
        checkpoint 记录的迭代数（供调用方打印日志）

    Raises:
        ValueError: checkpoint 缺少 U_sps 字段或形状（n_global_cells/
            n_sps）与当前求解器不兼容——与 `distributed_load_checkpoint`
            同一个既有护栏风格（这个模块是 `core/`，不依赖 `click`，
            由 CLI 调用方按需要转换成 `click.ClickException`）。
    """
    from autoflowcfd.core.utils.checkpoint import CheckpointManager
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive
    from autoflowcfd.core.fr_solver.turbulence import _set_freestream_turbulence
    from types import SimpleNamespace

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
    is_gpu_turb_model = getattr(solver, 'turb_model_gpu', None) is not None
    has_turb_model = getattr(solver, 'turb_model', None) is not None or is_gpu_turb_model

    if rank == 0:
        config = SimpleNamespace(
            mode="steady", backend="cpu", order=solver.mesh.n_points_1d, turbulence="sst_kw",
        )
        manager = CheckpointManager(config)
        _solution, _history, ckpt_iter, metadata = manager.load(checkpoint_path)

        fields = metadata.get("fields", {})
        if "U_sps" not in fields:
            raise ValueError(
                f"Checkpoint '{checkpoint_path}' 缺少 'U_sps' 字段（完整的 "
                f"(n_cells,n_sps,n_vars) 求解器状态），无法精确恢复。"
            )
        U_ckpt = fields["U_sps"]

        n_global_cells = solver.partition.n_global_cells
        n_sps_solver = solver.state.n_sps

        if U_ckpt.shape[0] != n_global_cells or U_ckpt.shape[1] != n_sps_solver:
            raise ValueError(
                f"Checkpoint 状态形状 {U_ckpt.shape} 与目标求解器的全局形状 "
                f"(n_global_cells={n_global_cells}, n_sps={n_sps_solver}) 不匹配"
                f"（网格或阶数可能已变化），拒绝恢复。"
            )

        n_vars_ckpt = U_ckpt.shape[2]
        U_global = U_ckpt[:, :, :5].copy()

        k_omega_global = None
        if has_turb_model:
            if n_vars_ckpt >= 7:
                rho = np.maximum(U_ckpt[:, :, 0], 1e-10)
                k_global = U_ckpt[:, :, 5] / rho
                omega_global = U_ckpt[:, :, 6] / rho
                k_omega_global = (k_global, omega_global)
            else:
                k_inf, omega_inf = _set_freestream_turbulence(solver)
                k_omega_global = (
                    np.full((n_global_cells, n_sps_solver), k_inf),
                    np.full((n_global_cells, n_sps_solver), omega_inf),
                )

        iteration = ckpt_iter
    else:
        U_global = None
        k_omega_global = None
        iteration = 0

    if n_ranks > 1:
        U_global = get_comm().bcast(U_global, root=0)
        k_omega_global = get_comm().bcast(k_omega_global, root=0)
        iteration = get_comm().bcast(iteration, root=0)

    local_cells = solver.partition.local_cells
    U_local = scatter_local_state(U_global, local_cells)

    n_local = solver.partition.n_local_cells
    solver.state.U[:n_local] = U_local
    solver.state.Q[:n_local] = conserved_to_primitive(U_local[..., :5])

    if has_turb_model and k_omega_global is not None:
        k_global, omega_global = k_omega_global
        k_local = scatter_local_state(k_global[:, :, None], local_cells)[:, :, 0]
        omega_local = scatter_local_state(omega_global[:, :, None], local_cells)[:, :, 0]
        if is_gpu_turb_model:
            # GPU 分布式求解器（"传统模式"/"完全分布式加载"皆可）：
            # `turb_model_gpu.k_field`/`.omega_field` 是 CuPy 数组，写入
            # 普通 numpy 会让后续 `cp.stack`/`cp.asarray` 消费点直接
            # TypeError（或更隐蔽地静默退化——取决于 CuPy 版本），必须
            # 在这里就地转成 GPU 数组，与 `save_checkpoint_distributed`/
            # `load_checkpoint_distributed` 里 U_gpu 的同步方式一致。
            from autoflowcfd.core.gpu import get_cupy
            cp = get_cupy()
            with cp.cuda.Device(solver.device_id):
                solver.turb_model_gpu.k_field = cp.asarray(k_local)
                solver.turb_model_gpu.omega_field = cp.asarray(omega_local)
        else:
            solver.turb_model.k_field = k_local
            solver.turb_model.omega_field = omega_local

    return iteration
