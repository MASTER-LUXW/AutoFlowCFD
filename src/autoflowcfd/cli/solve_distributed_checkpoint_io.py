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

from types import SimpleNamespace
from typing import Optional

import click


def rebuild_distributed_solver_from_checkpoint(
    checkpoint_path: str,
    n_ranks: int,
    multi_gpu: bool = False,
    fully_distributed: bool = False,
    gpu_device: Optional[int] = None,
    backend: Optional[str] = None,
    surface_mesh: Optional[str] = None,
    threads: int = -1,
    skip_quality_check: bool = False,
    cfl_start: float = 0.1,
    cfl_max: float = 0.5,
    cfl_min: float = 0.01,
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
        backend: 后端覆盖（`multi_gpu`/`fully_distributed` 场景下当前
            未使用，保留与单机版同名参数一致的签名）
        surface_mesh: 面网格路径覆盖，None 时回退到 checkpoint metadata
        threads: CPU 后端线程数
        skip_quality_check: 跳过重建时的网格质量门检查

    Returns:
        (solver, iteration, metadata): 重建好的分布式求解器实例
        （状态已从 checkpoint 恢复）、checkpoint 记录的迭代数、以及
        重建所用的完整 metadata 字典
    """
    from autoflowcfd.core.utils.checkpoint import CheckpointManager
    from autoflowcfd.cli.solve_mesh_loader import load_mesh_for_solver
    from autoflowcfd.core.fr_residual.inviscid import conserved_to_primitive

    _solution, _history, iteration, metadata = CheckpointManager(
        config=SimpleNamespace(), output_dir="."
    ).load(checkpoint_path)

    fields = metadata.get("fields", {})
    if "U_sps" not in fields:
        raise click.ClickException(
            f"Checkpoint '{checkpoint_path}' 缺少 'U_sps' 字段（完整的 "
            f"(n_cells,n_sps,n_vars) 求解器状态）——不是本版本 write_checkpoint/"
            f"distributed_save_checkpoint 写出的 checkpoint，无法精确重建。"
        )

    input_file = metadata.get("input_file")
    if not input_file:
        raise click.ClickException("Checkpoint metadata 缺少 'input_file'，无法重新加载网格。")

    order = int(metadata.get("order", 2))
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
    turbulence_model = metadata.get("turbulence_model", "sst")
    resolved_surface_mesh = surface_mesh or metadata.get("surface_mesh")
    # 决定物理解的参数：与单机重建共用同一个读取函数（来流缺失即报错，不猜）。
    # 攻角/侧滑角此前在这里四条构造路径上**全部**没有恢复，续算静默变成零攻角。
    from autoflowcfd.cli.solve_checkpoint_io import physics_from_metadata
    physics = physics_from_metadata(metadata)
    rho_inf, vel_inf, p_inf = physics["rho_inf"], physics["vel_inf"], physics["p_inf"]
    aoa_deg, aos_deg = physics["aoa_deg"], physics["aos_deg"]
    mu_molecular = physics["mu_molecular"]
    turbulence_intensity = physics["turbulence_intensity"]
    viscosity_ratio = physics["viscosity_ratio"]

    if multi_gpu and fully_distributed:
        # 多 GPU"完全分布式加载"（#1，2026-09-02 实现——此前这个组合被
        # 直接拒绝，见 MultiGPUDistributedSolver.from_fully_distributed_
        # package/gpu_distributed_fully_distributed.py 模块文档）。
        from autoflowcfd.core.gpu.distributed.gpu_distributed import MultiGPUDistributedSolver
        from autoflowcfd.core.mpi.distributed_mesh_loader import distributed_mesh_load_v2
        from autoflowcfd.core.fr_solver.mach_ref import resolve_mach_ref

        freestream = {"rho_inf": rho_inf, "vel_inf": vel_inf, "p_inf": p_inf,
                      "aoa_deg": aoa_deg, "aos_deg": aos_deg}
        mach_ref = resolve_mach_ref(rho_inf, vel_inf, p_inf)
        package, root_context = distributed_mesh_load_v2(
            input_file, order, resolved_surface_mesh, n_ranks,
            freestream=freestream, mu_molecular=mu_molecular, mach_ref=mach_ref,
            enable_viscous=True, skip_quality_check=skip_quality_check,
            turb_model_name=turbulence_model.upper(),
            turbulence_intensity=turbulence_intensity, viscosity_ratio=viscosity_ratio,
            cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min,
        )
        solver = MultiGPUDistributedSolver.from_fully_distributed_package(
            package, n_ranks=n_ranks, device_id=gpu_device, root_context=root_context,
        )
        loaded_metadata, loaded_iteration = solver.load_checkpoint_distributed(checkpoint_path)
        metadata = loaded_metadata or metadata
        iteration = loaded_iteration

    elif multi_gpu:
        from autoflowcfd.core.gpu.distributed.gpu_distributed import MultiGPUDistributedSolver
        from autoflowcfd.fr.operators import generate_fr_operators

        mesh, _volume_data = load_mesh_for_solver(
            input_file, order, surface_mesh=resolved_surface_mesh,
            skip_quality_check=skip_quality_check,
        )
        ops = generate_fr_operators(order)
        solver = MultiGPUDistributedSolver(
            mesh=mesh, ops=ops, n_ranks=n_ranks, device_id=gpu_device,
            mu_molecular=mu_molecular, rho_inf=rho_inf, vel_inf=vel_inf, p_inf=p_inf,
            aoa_deg=aoa_deg, aos_deg=aos_deg,
            turb_model=turbulence_model.upper(),
            turbulence_intensity=turbulence_intensity, viscosity_ratio=viscosity_ratio,
            cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min,
        )
        # GPU 版 checkpoint 加载是求解器自身方法（见
        # gpu_distributed_init.py::load_checkpoint_distributed），内部
        # 直接写回 self.state.U/self.U_gpu，不需要调用方再手动切片。
        loaded_metadata, loaded_iteration = solver.load_checkpoint_distributed(checkpoint_path)
        metadata = loaded_metadata or metadata
        iteration = loaded_iteration

    elif fully_distributed:
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
        from autoflowcfd.core.mpi.distributed_mesh_loader import distributed_mesh_load_v2
        from autoflowcfd.core.mpi.distributed_checkpoint import distributed_load_checkpoint

        freestream = {"rho_inf": rho_inf, "vel_inf": vel_inf, "p_inf": p_inf,
                      "aoa_deg": aoa_deg, "aos_deg": aos_deg}
        from autoflowcfd.core.fr_solver.mach_ref import resolve_mach_ref
        mach_ref = resolve_mach_ref(rho_inf, vel_inf, p_inf)
        package, root_context = distributed_mesh_load_v2(
            input_file, order, resolved_surface_mesh, n_ranks,
            freestream=freestream, mu_molecular=mu_molecular, mach_ref=mach_ref,
            enable_viscous=True, skip_quality_check=skip_quality_check,
            turb_model_name=turbulence_model.upper(),
            turbulence_intensity=turbulence_intensity, viscosity_ratio=viscosity_ratio,
            cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min,
        )
        solver = DistributedFRSolver.from_fully_distributed_package(
            package, n_ranks=n_ranks, root_context=root_context,
        )

        U_local, loaded_metadata, loaded_iteration = distributed_load_checkpoint(checkpoint_path, solver)
        n_local = solver.partition.n_local_cells
        solver.state.U[:n_local] = U_local
        solver.state.Q[:n_local] = conserved_to_primitive(U_local[..., :5])
        metadata = loaded_metadata or metadata
        iteration = loaded_iteration

    else:
        # CPU MPI"传统模式"：每个 rank 独立加载完整网格，与 CLI
        # `solve steady --n-ranks` 的对应分支完全一致的构造方式。
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver
        from autoflowcfd.core.mpi.distributed_checkpoint import distributed_load_checkpoint
        from autoflowcfd.core.time_integration.base import TimeIntegrationScheme
        from autoflowcfd.fr.operators import generate_fr_operators

        mesh, _volume_data = load_mesh_for_solver(
            input_file, order, surface_mesh=resolved_surface_mesh,
            skip_quality_check=skip_quality_check,
        )
        ops = generate_fr_operators(order)
        solver = DistributedFRSolver(
            mesh=mesh, ops=ops, face_connectivity=mesh.face_connectivity,
            n_ranks=n_ranks, backend=backend or "cpu", order=order,
            turb_model_name=turbulence_model, time_scheme=TimeIntegrationScheme.SSP_RK3,
            n_threads=threads, turbulence_intensity=turbulence_intensity,
            viscosity_ratio=viscosity_ratio, mu_molecular=mu_molecular,
            rho_inf=rho_inf, vel_inf=vel_inf, p_inf=p_inf,
            aoa_deg=aoa_deg, aos_deg=aos_deg,
            cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min,
        )

        U_local, loaded_metadata, loaded_iteration = distributed_load_checkpoint(checkpoint_path, solver)
        n_local = solver.partition.n_local_cells
        solver.state.U[:n_local] = U_local
        solver.state.Q[:n_local] = conserved_to_primitive(U_local[..., :5])
        metadata = loaded_metadata or metadata
        iteration = loaded_iteration

    # Order Continuation resume 支持（2026-09-02）：与单机
    # rebuild_solver_from_checkpoint 同一处修复同一个理由——构造时为了
    # 让 mesh/初始状态形状匹配 checkpoint 传的是 order（checkpoint 时的
    # current_order），这里把 solver.order 单独纠正回真正的目标阶数，
    # 否则 solve() 里 `self.order_continuation_enabled and self.order
    # >= 2` 这个门槛会被错误地拿 current_order 去判断。`_resumed_from_
    # checkpoint` 告诉 `run_distributed_order_continuation` 不要把刚
    # 恢复的真实解重置回 P0 均匀流场、从 `solver.current_order`（而不是
    # 0）继续爬坡。multi_gpu 分支（"传统模式"与"完全分布式加载"均已于
    # 2026-09-02 接入，见 gpu_distributed_order_continuation.py/
    # gpu_distributed_fully_distributed.py 模块文档）同样支持 Order
    # Continuation，这两个属性对它同样生效，不是无害占位。
    if hasattr(solver, "order"):
        solver.order = target_order
    solver._resumed_from_checkpoint = True

    metadata["order"] = order
    metadata["target_order"] = target_order
    metadata["turbulence_model"] = turbulence_model
    metadata["surface_mesh"] = resolved_surface_mesh
    return solver, iteration, metadata
