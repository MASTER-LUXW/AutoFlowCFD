"""AutoFlowCFD V2.0 - 分布式求解器的构造与收尾落盘（`solve steady/transient/resume` 共用的唯一实现）。

四种分布式组合：CPU MPI 传统模式（每个 rank 加载完整网格）、CPU 完全分布式加载（只有 root 加载，紧凑包分发，
`distributed_mesh_load_v2`）、多 GPU 传统模式、多 GPU 完全分布式加载。四个构造点接收同一组参数：物理参数字典
（键同 `checkpoint_io/rebuild.py::PHYSICS_KEYS`，checkpoint 续算与新算例同一来源）与数值参数。此前这段构造在
`solve steady` 两个分支、`solve transient` 三个分支、续算重建四个分支里各写一份（完全分布式的来流字典、mach_ref、
`distributed_mesh_load_v2` 调用共 6 份），多 GPU 的构造参数名与类型还与其余后端不同。
"""

from typing import Optional

import click


def require_distributed_runtime(multi_gpu: bool) -> None:
    """分布式运行环境检查：MPI（mpi4py 且以 mpirun 启动）；多 GPU 另需 CuPy。

    Raises:
        click.ClickException: 缺少运行环境
    """
    from autoflowcfd.core.mpi import mpi_available

    if multi_gpu:
        from autoflowcfd.core.gpu import gpu_available

        if not gpu_available:
            raise click.ClickException("--multi-gpu 需要 CuPy 与可用的 CUDA 设备（pip install cupy-cuda12x）")
    if not mpi_available(require=True):
        raise click.ClickException("分布式求解需要 MPI：pip install mpi4py，并以 mpirun -np <n_ranks> autoflowcfd ... 启动")


def build_distributed_solver(*, multi_gpu: bool, fully_distributed: bool, input_file: str, order: int,
                             surface_mesh: Optional[str], skip_quality_check: bool, n_ranks: int,
                             gpu_device: Optional[int], threads: int, turbulence_model: str, time_scheme,
                             physics: dict, cfl_start: Optional[float], cfl_max: Optional[float],
                             cfl_min: Optional[float], low_mach_precond: bool, time_accurate: bool,
                             dual_time_inner_iter: int, sem_num_eddies: Optional[int]):
    """构造分布式求解器（集体操作）；返回 `(solver, volume_data)`，完全分布式加载时 volume_data 为 None
    （只在 root 的 `solver._root_context["volume_data"]` 里）。

    Args:
        time_scheme: `TimeIntegrationScheme`
        physics: 来流、攻角 / 侧滑角、分子粘度、Tu / VR、人工粘性、近壁处理（`PHYSICS_KEYS`）
    """
    from autoflowcfd.core.fr_solver.mach_ref import resolve_mach_ref

    require_distributed_runtime(multi_gpu)
    numerics = dict(cfl_start=cfl_start, cfl_max=cfl_max, cfl_min=cfl_min, low_mach_precond=low_mach_precond,
                    time_accurate=time_accurate, sem_num_eddies=sem_num_eddies,
                    artificial_viscosity_enabled=physics["artificial_viscosity_enabled"],
                    artificial_viscosity_alpha=physics["artificial_viscosity_alpha"],
                    wall_treatment=physics["wall_treatment"], turbulence_intensity=physics["turbulence_intensity"],
                    viscosity_ratio=physics["viscosity_ratio"], mu_molecular=physics["mu_molecular"])
    if fully_distributed:
        from autoflowcfd.core.mpi.distributed_mesh_loader import distributed_mesh_load_v2

        freestream = {k: physics[k] for k in ("rho_inf", "vel_inf", "p_inf", "aoa_deg", "aos_deg")}
        package, root_context = distributed_mesh_load_v2(
            input_file, order, surface_mesh, n_ranks, freestream=freestream,
            mach_ref=resolve_mach_ref(physics["rho_inf"], physics["vel_inf"], physics["p_inf"]),
            enable_viscous=True, skip_quality_check=skip_quality_check, turb_model_name=turbulence_model.upper(),
            time_scheme=time_scheme, dual_time_inner_iter=dual_time_inner_iter, **numerics)
        if multi_gpu:
            from autoflowcfd.core.gpu.distributed.gpu_distributed import MultiGPUDistributedSolver

            solver = MultiGPUDistributedSolver.from_fully_distributed_package(
                package, n_ranks=n_ranks, device_id=gpu_device, root_context=root_context)
        else:
            from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver

            solver = DistributedFRSolver.from_fully_distributed_package(
                package, n_ranks=n_ranks, root_context=root_context, n_threads=threads)
        return solver, None

    from autoflowcfd.cli.solve.mesh_loader import load_mesh_for_solver
    from autoflowcfd.cli.solve.wall_distance import wall_distance_source_if_needed
    from autoflowcfd.fr.operators import generate_fr_operators

    mesh, volume_data = load_mesh_for_solver(input_file, order, surface_mesh=surface_mesh,
                                             skip_quality_check=skip_quality_check)
    common = dict(turb_model_name=turbulence_model, time_scheme=time_scheme,
                  dual_time_inner_iter=dual_time_inner_iter,
                  wall_distance_source=wall_distance_source_if_needed(turbulence_model, volume_data),
                  rho_inf=physics["rho_inf"], vel_inf=physics["vel_inf"], p_inf=physics["p_inf"],
                  aoa_deg=physics["aoa_deg"], aos_deg=physics["aos_deg"], **numerics)
    if multi_gpu:
        from autoflowcfd.core.gpu.distributed.gpu_distributed import MultiGPUDistributedSolver

        solver = MultiGPUDistributedSolver(mesh=mesh, ops=generate_fr_operators(order), n_ranks=n_ranks,
                                           device_id=gpu_device, **common)
    else:
        from autoflowcfd.core.mpi.distributed_solver import DistributedFRSolver

        solver = DistributedFRSolver(mesh=mesh, ops=generate_fr_operators(order),
                                     face_connectivity=mesh.face_connectivity, n_ranks=n_ranks, order=order,
                                     n_threads=threads, **common)
    return solver, volume_data


def finish_distributed(solver, output_dir: str, iterations: int, input_file: str, turbulence_model: str,
                       surface_mesh: Optional[str], reference_area: Optional[float]) -> None:
    """分布式求解收尾（集体操作，三种分布式求解器同一套）：全局状态汇总到 root 写 final_state.pkl、写结束时的
    checkpoint、报告气动系数（参考面积只需 root 给出）；多 GPU 先把设备上的状态同步到主机，最后释放设备资源。

    此前多 GPU 不写 final_state.pkl（CPU MPI 写），收尾信息每个 rank 各打印一次。
    """
    from autoflowcfd.cli.solve.aero_coefficients import report_distributed_aerodynamic_coefficients
    from autoflowcfd.core.mpi import is_root
    from autoflowcfd.core.mpi.distributed_checkpoint import distributed_save_results

    solver.pull_host_state()
    distributed_save_results(solver, output_dir)
    saved = solver.save_checkpoint_distributed(output_dir, iterations, input_file, solver.current_order,
                                               turbulence_model, target_order=solver.order,
                                               surface_mesh=surface_mesh)
    if saved and is_root():
        print(f"   Checkpoint saved: {saved}")
    report_distributed_aerodynamic_coefficients(solver, reference_area)
    solver.cleanup()
