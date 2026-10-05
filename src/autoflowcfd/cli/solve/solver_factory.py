"""AutoFlowCFD V2.0 - 单机求解器的构造（`--backend cpu` -> `FRSolver`，`--backend gpu` -> `GPUFRSolver`）。

`solve steady/transient`、`solve resume`、`post` 的 checkpoint 重建与 Python API 共用这一个入口
（2026-10-04 统一）。此前单 GPU 有两套实现：`solve steady --backend gpu` 走 `GPUFRSolver`，其余入口走
`FRSolver(backend='gpu')`——后者只在每次残差求值时把 P0 无粘项搬到 GPU（P>=1 与全部粘性/湍流仍在 CPU），
CuPy 不可用时还会静默退回 CPU。两个求解器的构造参数同名，这里只做分派与壁距准备。
"""

import click


def build_single_node_solver(backend: str, mesh, volume_data, *, gpu_device: int = 0, **solver_kwargs):
    """按后端构造单机求解器；湍流模型需要壁距时一并准备（两个后端同一个壁距来源）。

    Args:
        backend: "cpu"、"gpu" 或 "auto"（有可用 GPU 用 gpu，否则 cpu）
        mesh: 已加载的高阶网格
        volume_data: `load_mesh_for_solver` 返回的体网格数据（取 WALL 边界面算壁距）
        gpu_device: GPU 设备号（只用于 gpu）
        **solver_kwargs: 两个求解器同名的构造参数（`order`/`turb_model_name`/`time_scheme`/来流/CFL/...）

    Raises:
        click.ClickException: `--backend gpu` 但 CuPy 不可用
        click.BadParameter: 未知后端
    """
    from autoflowcfd.cli.solve.wall_distance import (
        compute_wall_distance_for_solver,
        wall_distance_source_if_needed,
    )

    from autoflowcfd.core.gpu import gpu_available

    if backend == "auto":
        # 配置层 `backend: auto`（`SolverConfig` 默认值）：有可用 GPU 用单 GPU，否则 CPU。此前 "auto" 原样传进
        # `FRSolver`，既不等于 "gpu" 也不等于 "cpu"，实际恒为 CPU
        backend = "gpu" if gpu_available else "cpu"
        print(f"   Backend auto -> {backend}")
    if backend == "cpu":
        from autoflowcfd.core import FRSolver

        solver = FRSolver(mesh=mesh, **solver_kwargs)
        compute_wall_distance_for_solver(solver, volume_data)
        return solver
    if backend != "gpu":
        raise click.BadParameter(f"未知后端 {backend!r}（cpu / gpu / auto）", param_hint="--backend")
    if not gpu_available:
        raise click.ClickException("--backend gpu 需要 CuPy 与可用的 CUDA 设备（pip install cupy-cuda12x）")
    from autoflowcfd.core.gpu.solver.gpu_solver import GPUFRSolver

    return GPUFRSolver(
        mesh=mesh, device_id=gpu_device,
        wall_distance_source=wall_distance_source_if_needed(solver_kwargs.get("turb_model_name", "NONE"),
                                                            volume_data),
        **solver_kwargs)


def validate_backend_options(backend, n_ranks: int, multi_gpu: bool) -> None:
    """`--backend/--n-ranks/--multi-gpu` 的组合校验（`solve steady/transient/resume` 共用）。

    多 rank 的 GPU 计算必须显式 `--multi-gpu`；`--multi-gpu` 要求多 rank 且后端是 GPU（`backend=None` 表示
    续算沿用 checkpoint 记录的后端）。此前 `--backend gpu --n-ranks N` 不加 `--multi-gpu` 时 steady 静默只用
    单 GPU、transient 每个 rank 构造 CPU 求解器的 GPU 分支，`--multi-gpu --n-ranks 1` 落到单机路径。
    """
    if multi_gpu and (n_ranks <= 1 or backend not in (None, "gpu")):
        raise click.BadParameter("--multi-gpu 需要 --backend gpu 且 --n-ranks > 1（单 GPU 直接用 --backend gpu）",
                                 param_hint="--multi-gpu")
    if backend == "gpu" and n_ranks > 1 and not multi_gpu:
        raise click.BadParameter("多 rank 的 GPU 计算需要 --multi-gpu", param_hint="--n-ranks")
