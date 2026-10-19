"""AutoFlowCFD V2.0 - `--init-from`：以已有 checkpoint 的场为初场启动求解（`solve steady` 与
`solve transient` 共用，四个后端）。

典型用法：
* 稳态收敛后从该流场启动瞬态（DES/LES 不必从均匀来流起步）；
* 稳态逐阶做：低阶解收敛后以它为初场直接在更高阶上继续，不重走低阶的爬坡（cube_demo P1 收敛用了
  二百多步；`solve resume` 只能沿用 checkpoint 记录的目标阶数）。

checkpoint 阶数可以低于求解器阶数（平均流与湍流场精确延拓，`order_continuation/initial_field.py`），
不能高于。起步后直接在求解器的阶数上推进，不做阶数爬坡（`start_from_checkpoint_field`）。

2026-10-14 以前这段逻辑只有 `solve transient` 有，且在单机 / CPU MPI 传统模式 / 完全分布式 / 多 GPU
四个分支里各写了一遍。
"""


from autoflowcfd.core.utils.order_continuation.initial_field import start_from_checkpoint_field

from .restore import restore_state_from_checkpoint


def start_single_node_from_checkpoint(solver, checkpoint_path: str) -> None:
    """单机（CPU `FRSolver` / 单 GPU `GPUFRSolver`）：恢复 checkpoint 的场并从它起步。"""
    from autoflowcfd.core.utils.checkpoint import CheckpointManager

    print("\n🔄 从 checkpoint 加载初场...")
    _, _, ckpt_iter, ckpt_meta = CheckpointManager().load(checkpoint_path)
    with solver.edit_host_state() as host:
        _, ckpt_order = restore_state_from_checkpoint(checkpoint_path, host, ckpt_meta)
    start_from_checkpoint_field(solver, ckpt_order)
    print(f"   源 checkpoint 迭代数: {ckpt_iter}\n")


def start_distributed_from_checkpoint(solver, checkpoint_path: str, *, sync_device: bool = False) -> None:
    """分布式（CPU MPI 传统模式 / 完全分布式加载 / 多 GPU）：各 rank 恢复自己那一段并从它起步。

    `sync_device`（多 GPU）：`restore_distributed_state_from_checkpoint` 只写主机侧的 `solver.state.U`
    （与 CPU 路径共用的接口），真正参与计算的是设备上的 `solver.U_gpu`，恢复后显式同步（与
    `load_checkpoint_distributed` 同一个模式）。
    """
    from autoflowcfd.core.mpi import is_root
    from autoflowcfd.core.mpi.distributed_checkpoint import restore_distributed_state_from_checkpoint

    if is_root():
        print("\n🔄 从 checkpoint 加载初场...")
    ckpt_iter, ckpt_order = restore_distributed_state_from_checkpoint(checkpoint_path, solver)
    if sync_device:
        from autoflowcfd.core.gpu import get_cupy

        cp = get_cupy()
        n_local = solver.partition.n_local_cells
        with cp.cuda.Device(solver.device_id):
            solver.U_gpu = cp.asarray(solver.state.U[:n_local])
    start_from_checkpoint_field(solver, ckpt_order, report=is_root())
    if is_root():
        print(f"   源 checkpoint 迭代数: {ckpt_iter}\n")
