"""AutoFlowCFD V2.0 - `solve steady` 的单机分支（CPU `FRSolver` / 单 GPU `GPUFRSolver`）。

从 `cli/solve/steady.py` 拆出（2026-09-25，项目「单文件不超 500 行」规范）；参数由
`command.py` 在公共前段（物理常数解析与范围校验）之后逐名传入。2026-10-04 起单 GPU 与 CPU 走同一个
分支（求解器由 `cli/solve/solver_factory.py` 按 `--backend` 构造）：此前单 GPU 另有一份分支，不写中间
checkpoint、最终只存一份不含湍流场的 pickle、不算参考面积与气动力系数。
"""

import click

from autoflowcfd.core.time_integration.base import scheme_from_name
from autoflowcfd.cli.solve.checkpoint_io import periodic_checkpoint_callback, write_single_node_outputs
from autoflowcfd.cli.solve.mesh_loader import load_mesh_for_solver
from autoflowcfd.cli.solve.solver_factory import build_single_node_solver
from autoflowcfd.cli.solve.aero_coefficients import (
    _report_aerodynamic_coefficients,
    resolve_reference_area,
)
from autoflowcfd.core.time_integration.base import STEADY_DT


def _run_single_node(
    *,
    aoa_deg, aos_deg, artificial_viscosity_alpha, artificial_viscosity_enabled,
    backend, cfl_max, cfl_min, cfl_start, checkpoint_interval,
    gpu_device, input_file, max_iter, mu_molecular, order,
    output_dir, p_inf, phase_max_iter, reference_area, residual_drop_threshold,
    rho_inf, sem_num_eddies, skip_quality_check, surface_mesh, threads, tol,
    time_scheme, turbulence_intensity, turbulence_model, vel_inf, viscosity_ratio,
):
    """`solve steady` 的单机路径（`--backend cpu|gpu`）。"""
    mesh, volume_data = load_mesh_for_solver(
        input_file, order, surface_mesh=surface_mesh, skip_quality_check=skip_quality_check,
    )
    solver = build_single_node_solver(
        backend, mesh, volume_data,
        gpu_device=gpu_device,
        order=order,
        turb_model_name=turbulence_model,
        time_scheme=scheme_from_name(time_scheme),
        n_threads=threads,
        cfl_start=cfl_start,
        cfl_max=cfl_max,
        cfl_min=cfl_min,
        turbulence_intensity=turbulence_intensity,
        viscosity_ratio=viscosity_ratio,
        sem_num_eddies=sem_num_eddies,
        mu_molecular=mu_molecular,
        rho_inf=rho_inf, vel_inf=vel_inf, p_inf=p_inf,
        aoa_deg=aoa_deg, aos_deg=aos_deg,
        artificial_viscosity_enabled=artificial_viscosity_enabled,
        artificial_viscosity_alpha=artificial_viscosity_alpha,
    )

    # 参考面积（未给时沿来流方向自动估算），供迭代中输出气动力系数
    reference_area = resolve_reference_area(solver, volume_data, reference_area)

    try:
        result = solver.solve(
            max_iter=max_iter, dt=STEADY_DT, tol=tol,
            checkpoint_callback=periodic_checkpoint_callback(
                checkpoint_interval, output_dir, input_file, turbulence_model, backend, surface_mesh=surface_mesh),
            phase_max_iter=phase_max_iter,
            residual_drop_threshold=residual_drop_threshold)
        print(f"\n✅ Simulation Finished: Iterations={result.iterations}, Residual={result.final_residual:.6e}")

        # 结果（.pkl 全量状态 + HDF5 checkpoint，后者供 solve resume 使用）与气动系数
        write_single_node_outputs(solver, output_dir, result.iterations, input_file, turbulence_model, backend,
                                  surface_mesh=surface_mesh)
        _report_aerodynamic_coefficients(solver.host_view(), reference_area)

    except Exception as e:
        print(f"\n❌ Simulation Failed: {str(e)}")
        raise click.Abort()
